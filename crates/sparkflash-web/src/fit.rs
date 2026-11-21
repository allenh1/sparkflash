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

//! Whether a checkpoint can be served on this cluster, judged at every tensor-parallel width.
//!
//! The model browser asks one question of every model it lists: *can this cluster serve it, in
//! which configurations, and if not, what is missing?* The answer is a [Fit]: one
//! [Configuration] a width in [WORLDS], each judged on three things in this order.
//!
//! 1. **An engine.** A family with a device arm must run at the width, its `config.json` must
//!    plan there, and every served part must resolve a sm121 kernel. A family registered with
//!    `engine = "dynamic"` is judged the same way when a device arm reads its stored formats
//!    ([sparkflash_ops::parts::dynamic_device_arm]: the Qwen3.6 FP8 release), and by the shared
//!    CPU host reference otherwise, whose summary says so; an unregistered family is missing an
//!    engine.
//! 2. **Sparks.** The width cannot exceed the Sparks the cluster has.
//! 3. **Memory.** The rank's weights plus the caches of one session must fit one node's budget,
//!    which is its unified memory less the [MEMORY_FLOOR_BYTES] guard band. The rank's weights
//!    come from the shared tensor plan where it names the family (exact), else from the
//!    checkpoint's stored bytes split evenly over the width (an estimate). The caches come
//!    from [Architecture::kv_plan], and the longest window that still fits is reported.
//!
//! Nothing here touches the network or a device; callers pass the document and byte counts in.

use std::collections::{BTreeSet, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{LazyLock, Mutex};

use serde::Serialize;
use serde_json::Value;
use sparkflash_model::{
    Architecture, MEMORY_FLOOR_BYTES, NODE_MEMORY_BUDGET_BYTES, Target, TpPlan,
    canonical_model_type, gib,
};
use sparkflash_ops::parts::serving_language;
use sparkflash_ops::tensor_plan::{ArchitectureTensorPlan, Role};
use sparkflash_ops::{CoverageState, ScanInputs, ScanReport, TensorIndex, scan};

use crate::config::{
    DEFAULT_WEIGHTS_ID, DYNAMIC_ENGINE_ID, MODEL_PROFILES, ModelProfile, profile_for_model_type,
};

/// The tensor-parallel widths a checkpoint is judged at: every width a device arm runs.
pub const WORLDS: [usize; 3] = [1, 2, 4];

/// The shortest window a configuration must hold to count as serving: 4,096 tokens, or the
/// checkpoint's whole window when that is shorter.
pub const MIN_CONTEXT_TOKENS: usize = 4096;

/// The targets the console reports the scan for: every target the shared catalogue is keyed by.
pub const SCAN_TARGETS: &[Target] = Target::ALL;

/// One row of a scan, in the form the console's model page renders.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScanRowView {
    /// The stack the instance belongs to.
    pub stack: String,
    /// The block index, or `None` for a global part.
    pub block: Option<usize>,
    /// The operation (part kind) the row is for.
    pub operation: String,
    /// The rank-local shape the plan resolved at.
    pub shape: String,
    /// The stored format, or `?` when the checkpoint does not state it.
    pub format: String,
    /// The pass this row reports.
    pub pass: String,
    /// The target this row resolves against.
    pub target: String,
    /// The status name.
    pub status: String,
    /// The config key or tensor the status refuses, when it names one.
    pub subject: Option<String>,
    /// What would fix the row, when it did not resolve.
    pub fix: Option<String>,
}

/// One unresolved row, as a work item an implementer can take as written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScanWorkItem {
    /// The stack the instance belongs to.
    pub stack: String,
    /// The block index, or `None` for a global part.
    pub block: Option<usize>,
    /// The operation (part kind) to build or port.
    pub operation: String,
    /// The rank-local shape the plan asked for.
    pub shape: String,
    /// The stored format, or `?` when the checkpoint does not state it.
    pub format: String,
    /// The pass that refused the row.
    pub pass: String,
    /// The target the row was resolved against.
    pub target: String,
    /// The refusal's status name.
    pub status: String,
    /// The config key or tensor the status names, when it names one.
    pub subject: Option<String>,
    /// What would fix the gap.
    pub fix: String,
}

/// One scan report for one target and tensor-parallel width.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScanView {
    /// The target's stable name.
    pub target: String,
    /// The tensor-parallel width.
    pub world: usize,
    /// Whether every row resolved and neither coverage check reports a gap.
    pub clean: bool,
    /// Whether the console may offer conversion for this report.
    pub offers_convert: bool,
    /// Whether the console may offer serving for this report.
    pub offers_serve: bool,
    /// A failure that stopped the scan before it produced rows.
    pub error: Option<String>,
    /// Every row, in the scan's order.
    pub rows: Vec<ScanRowView>,
    /// Every unresolved row, in the scan's order.
    pub work_items: Vec<ScanWorkItem>,
    /// The report exactly as the plain-text renderer prints it, so the page and `onboard` show
    /// one rendering and not two.
    pub plain: String,
}

impl ScanView {
    fn of(report: &ScanReport) -> Self {
        let rows = report
            .rows
            .iter()
            .map(|row| ScanRowView {
                stack: row.stack.name().to_owned(),
                block: row.block,
                operation: row.part.name().to_owned(),
                shape: row.shape.to_string(),
                format: row
                    .format
                    .map_or_else(|| "?".to_owned(), |format| format.name().to_owned()),
                pass: row.pass.name().to_owned(),
                target: row.target.name().to_owned(),
                status: row.status.name().to_owned(),
                subject: row.status.subject().map(str::to_owned),
                fix: row.status.fix().map(str::to_owned),
            })
            .collect();
        let work_items = report
            .unresolved()
            .map(|row| ScanWorkItem {
                stack: row.stack.name().to_owned(),
                block: row.block,
                operation: row.part.name().to_owned(),
                shape: row.shape.to_string(),
                format: row
                    .format
                    .map_or_else(|| "?".to_owned(), |format| format.name().to_owned()),
                pass: row.pass.name().to_owned(),
                target: row.target.name().to_owned(),
                status: row.status.name().to_owned(),
                subject: row.status.subject().map(str::to_owned),
                fix: row
                    .status
                    .fix()
                    .unwrap_or("(the scan gave no fix)")
                    .to_owned(),
            })
            .collect();
        let clean = !report_has_gaps(report);
        Self {
            target: report.target.name().to_owned(),
            world: report.world,
            clean,
            offers_convert: clean,
            offers_serve: clean,
            error: report.error.clone(),
            rows,
            work_items,
            plain: report.render(),
        }
    }
}

/// Whether a report contains a gap the console must refuse for.
///
/// A coverage check that has not run is not a gap: step 0 leaves both checks
/// [CoverageState::NotChecked] until their passes land, and a clean kernel scan must still be
/// offerable while that is true. A check that ran and found something ([CoverageState::Dirty])
/// is a gap.
fn report_has_gaps(report: &ScanReport) -> bool {
    report.error.is_some()
        || report.rows.iter().any(|row| !row.status.is_resolved())
        || matches!(report.coverage.tensors, CoverageState::Dirty)
        || matches!(report.coverage.config_keys, CoverageState::Dirty)
}

/// The scan of `config` at every target and tensor-parallel width, in target then width order.
#[must_use]
pub fn scan_reports(
    config: &str,
    tensors: Option<&TensorIndex>,
    targets: &[Target],
    worlds: &[usize],
) -> Vec<ScanReport> {
    let mut reports = Vec::with_capacity(targets.len() * worlds.len());
    for &target in targets {
        for &world in worlds {
            reports.push(scan(ScanInputs {
                config,
                tensors,
                target,
                world,
            }));
        }
    }
    reports
}

/// The reports in the form the console's model page renders.
#[must_use]
pub fn scan_views(
    config: &str,
    tensors: Option<&TensorIndex>,
    targets: &[Target],
    worlds: &[usize],
) -> Vec<ScanView> {
    scan_reports(config, tensors, targets, worlds)
        .iter()
        .map(ScanView::of)
        .collect()
}

/// The one plain-text rendering of the scans: the console's plain view and `sparkflash onboard`
/// both print this string, so the two can never drift.
#[must_use]
pub fn scan_plain(
    config: &str,
    tensors: Option<&TensorIndex>,
    targets: &[Target],
    worlds: &[usize],
) -> String {
    scan_reports(config, tensors, targets, worlds)
        .iter()
        .map(ScanReport::render)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The Sparks a model would run on, and how much of each one it may claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Cluster {
    /// The Sparks the plan, discovery or telemetry names.
    pub nodes: usize,
    /// One node's unified memory: the smallest one telemetry reports, else the GB10 ceiling.
    pub node_bytes: u64,
    /// The guard band every node leaves unallocated.
    pub floor_bytes: u64,
    /// Whether `node_bytes` came from live telemetry rather than the GB10 ceiling.
    pub measured: bool,
}

impl Cluster {
    /// `nodes` Sparks at the GB10's [NODE_MEMORY_BUDGET_BYTES] ceiling.
    #[must_use]
    pub const fn nominal(nodes: usize) -> Self {
        Self {
            nodes,
            node_bytes: NODE_MEMORY_BUDGET_BYTES,
            floor_bytes: MEMORY_FLOOR_BYTES,
            measured: false,
        }
    }

    /// The same cluster, sized by the smallest node total telemetry reports, in GiB.
    ///
    /// A zero or non-finite reading (a rank that has not reported yet) is ignored; when no
    /// reading is usable the ceiling stands.
    #[must_use]
    pub fn measured(self, totals_gib: impl IntoIterator<Item = f64>) -> Self {
        let smallest = totals_gib
            .into_iter()
            .filter(|total| total.is_finite() && *total > 0.0)
            .map(|total| (total * (1u64 << 30) as f64) as u64)
            .min();
        match smallest {
            Some(node_bytes) => Self {
                node_bytes,
                measured: true,
                ..self
            },
            None => self,
        }
    }

    /// What one rank may claim: the node's memory less its guard band.
    #[must_use]
    pub const fn budget(&self) -> u64 {
        self.node_bytes.saturating_sub(self.floor_bytes)
    }
}

/// A checkpoint's stored weight bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightBytes {
    /// The bytes.
    pub bytes: u64,
    /// True when they were summed from the safetensors files on disk; false when they were
    /// estimated from the Hub's per-dtype parameter counts.
    pub measured: bool,
}

/// What the caller knows about one checkpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct Evidence<'a> {
    /// The checkpoint's `config.json`, when it was read.
    pub config: Option<&'a str>,
    /// The `config.json` subset a Hub search returns (`model_type`, `quantization_config`),
    /// used when the whole document was not fetched.
    pub declared: Option<&'a Value>,
    /// The checkpoint's stored weights, when known.
    pub weights: Option<WeightBytes>,
    /// What the caller found the checkpoint lacks, which blocks every width: access to a gated
    /// repository, safetensors shards.
    pub missing: &'a [String],
}

/// The answer for one configuration, or for a model as a whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// A device engine serves it, and it fits.
    Runs,
    /// A device engine serves it, but the weights and one session's caches do not fit.
    NoRoom,
    /// Something it needs is absent: an engine, a kernel, a width the engine takes, Sparks.
    Missing,
    /// The document or the size could not be read, so nothing can be claimed.
    Unknown,
}

/// One tensor-parallel width of one checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Configuration {
    /// The tensor-parallel width.
    pub world: usize,
    /// The answer at this width.
    pub verdict: Verdict,
    /// Whether a device engine, the plan and every served kernel resolve at this width; `None`
    /// when the document was not available to check.
    pub engine_ready: Option<bool>,
    /// Whether this is the width the console launches the family at.
    pub launch: bool,
    /// One rank's resident weights.
    pub rank_weights_bytes: Option<u64>,
    /// True when the rank's weights come from the shared tensor plan; false for an even split
    /// of the checkpoint's bytes.
    pub rank_weights_planned: bool,
    /// Cache bytes one context position adds on one rank.
    pub kv_bytes_per_token: Option<u64>,
    /// Cache bytes a session claims on one rank whatever its window.
    pub kv_fixed_bytes: Option<u64>,
    /// What one rank may claim.
    pub budget_bytes: u64,
    /// The longest window one session can hold beside the weights, capped at the checkpoint's.
    pub max_context: Option<usize>,
    /// What is missing or does not fit, in the operator's words; empty when it runs.
    pub reasons: Vec<String>,
    /// Facts the verdict rests on that are not failures (an estimate, a deployment record).
    pub notes: Vec<String>,
}

/// Whether, where and how a checkpoint can be served on a cluster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Fit {
    /// The answer across every width.
    pub verdict: Verdict,
    /// One line the browser shows beside the model.
    pub summary: String,
    /// The checkpoint's `model_type`, when declared.
    pub model_type: Option<String>,
    /// The family label of the device arm that serves it, when one does.
    pub family: Option<String>,
    /// The engine selector of that arm.
    pub engine: Option<String>,
    /// Whether the CPU host reference, not a device arm, executes it: a family registered on the
    /// dynamic engine whose stored formats no device arm reads.
    pub host_reference: bool,
    /// The checkpoint's own window.
    pub window: Option<usize>,
    /// The checkpoint's stored weights.
    pub weights_bytes: Option<u64>,
    /// Whether `weights_bytes` was measured on disk rather than estimated.
    pub weights_measured: bool,
    /// The cluster it was judged against.
    pub cluster: Cluster,
    /// What every width is missing, before any width-specific reason.
    pub missing: Vec<String>,
    /// One entry a width in [WORLDS].
    pub configurations: Vec<Configuration>,
    /// The one scan report at every target and width, in target then width order. Empty when no
    /// `config.json` was read: a report is never invented for a document the caller lacks.
    pub scans: Vec<ScanView>,
}

impl Fit {
    /// The configuration the console launches, when the family has one.
    #[must_use]
    pub fn launch(&self) -> Option<&Configuration> {
        self.configurations.iter().find(|entry| entry.launch)
    }

    /// The widths that run, narrowest first.
    pub fn running(&self) -> impl Iterator<Item = &Configuration> {
        self.configurations
            .iter()
            .filter(|entry| entry.verdict == Verdict::Runs)
    }
}

/// Parses a document the way the serving arm for its family does.
///
/// The Qwen arm's bridge canonicalizes the text-only alias and applies its FP8 producer
/// correction ([Architecture::experts_only_fp8]); without it the plan would ask for FP8 dense
/// kernels the release does not store. On a Qwen3.5/3.6 release it applies
/// [Architecture::weight_only_fp8], the arm's W8A16 arithmetic. Every other family is read as
/// the shared parser reads it.
///
/// # Errors
/// The shared parser's message.
pub fn architecture(text: &str) -> Result<Architecture, String> {
    let mut architecture = Architecture::from_json(text).map_err(|error| error.to_string())?;
    match canonical_model_type(&architecture.family) {
        "qwen4_exp" => {
            architecture.family = "qwen4_exp".to_owned();
            architecture.experts_only_fp8();
        }
        "qwen3_5_moe" => architecture.weight_only_fp8(),
        _ => {}
    }
    Ok(architecture)
}

/// Why the device path cannot serve `config` at `world`, or `Ok` when it can.
///
/// This is the engine half of a [Configuration]. The family's arm must run at the width, and
/// the one [ScanReport] must have no unresolved row at [Target::Sm121]. A family with no arm
/// is still scanned, so the reasons say how far the kernels already reach.
///
/// # Errors
/// Each missing piece, named for the operator.
fn scan_gaps(config: &str, architecture: &Architecture, world: usize) -> Result<(), Vec<String>> {
    let profile = profile_for_model_type(&architecture.family);
    if let Some(profile) = profile
        && !profile.worlds.contains(&world)
    {
        return Err(vec![format!(
            "the {} engine at TP={world} (it runs at {})",
            profile.label,
            widths(profile.worlds)
        )]);
    }
    // A checkpoint no device arm reads may still be executable by the shared CPU host
    // reference. It is judged by that backend's own gate, not by the kernel catalogue: the
    // catalogue answers what a device kernel exists for, which is a different question.
    if on_host_reference(profile, Some(architecture)) {
        return sparkflash_ops::parts::host_reference_refusal(architecture)
            .map_or(Ok(()), |reason| Err(vec![reason]));
    }
    let report = scan(ScanInputs {
        config,
        tensors: None,
        target: Target::Sm121,
        world,
    });
    if let Some(error) = &report.error {
        return Err(vec![format!("an execution plan at TP={world}: {error}")]);
    }
    let mut gaps = Vec::new();
    let mut seen = BTreeSet::new();
    // The pack this gate describes is the language process: Qwen's and GLM's packs do not stage
    // a vision tower, so its row is reported by the page but is not an engine gap for this gate.
    // The full plan's vision constraint is still the catalogue's and the page's.
    let served = serving_language(Target::Sm121);
    for row in report.unresolved().filter(|row| served.contains(row.part)) {
        let reason = format!(
            "{} at {} on {} ({}): {}",
            row.part.name(),
            row.shape,
            row.target.name(),
            row.status.name(),
            row.status.fix().unwrap_or("the scan gave no fix")
        );
        if seen.insert(reason.clone()) {
            gaps.push(reason);
        }
    }
    if report.coverage.tensors == CoverageState::Dirty {
        gaps.push("the tensor coverage check (an indexed tensor no part claimed)".to_owned());
    }
    if report.coverage.config_keys == CoverageState::Dirty {
        gaps.push("the config-key coverage check (a key nothing consumed)".to_owned());
    }
    if gaps.is_empty() { Ok(()) } else { Err(gaps) }
}

/// What a family with no device arm is missing.
/// Whether the shared CPU host reference executes a checkpoint under its family's profile: a
/// family registered on the dynamic engine whose stored formats no device arm reads. A document
/// not read yet is judged by its profile alone.
fn on_host_reference(profile: Option<&ModelProfile>, architecture: Option<&Architecture>) -> bool {
    profile.is_some_and(|profile| profile.engine == DYNAMIC_ENGINE_ID)
        && architecture.is_none_or(|architecture| {
            sparkflash_ops::parts::dynamic_device_arm(architecture).is_err()
        })
}

fn no_engine(model_type: &str) -> String {
    let mut served: Vec<&str> = MODEL_PROFILES
        .iter()
        .map(|profile| profile.model_type)
        .collect();
    served.dedup();
    format!(
        "a device engine for model_type {model_type:?} (this build's engines serve {}; the CPU \
         host reference runs test checkpoints, not models)",
        served.join(", ")
    )
}

/// Whether a device engine serves a document at the width the console launches its family at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchCheck {
    /// The document's own window, when it parses.
    pub window: Option<usize>,
    /// What is missing; empty when the family's device arm serves it at its launch width.
    pub gaps: Vec<String>,
}

/// Checks `text` against the device path at its family's launch width, without the memory half.
///
/// This is the cheap, cluster-free check validation and the cache listing use: the document
/// parses, declares nothing the parser drops, names a family a device arm serves, and plans and
/// resolves its kernels at the width the console launches that family at.
#[must_use]
pub fn launch_check(text: &str) -> LaunchCheck {
    let mut gaps = Vec::new();
    if let Ok(root) = serde_json::from_str::<Value>(text) {
        gaps.extend(Architecture::unmodelled_features(&root));
    }
    match architecture(text) {
        Ok(architecture) => {
            gaps.extend(switch_gaps(&architecture));
            match profile_for_model_type(&architecture.family) {
                None => gaps.push(no_engine(&architecture.family)),
                Some(profile) => {
                    if let Err(more) = scan_gaps(text, &architecture, profile.world_size) {
                        gaps.extend(more);
                    }
                }
            }
            LaunchCheck {
                window: Some(architecture.max_position_embeddings).filter(|window| *window > 0),
                gaps,
            }
        }
        Err(error) => {
            gaps.push(format!("a config.json this build can read ({error})"));
            LaunchCheck { window: None, gaps }
        }
    }
}

/// The typed switches `architecture` turns on, each a gap: no arm runs one yet.
fn switch_gaps(architecture: &Architecture) -> impl Iterator<Item = String> {
    architecture
        .switches()
        .into_iter()
        .map(|switch| format!("{switch}, which no device arm runs"))
}

/// Judges one checkpoint on `cluster`.
#[must_use]
pub fn judge(evidence: &Evidence<'_>, cluster: &Cluster) -> Fit {
    // What the family lacks comes first: it is the reason no download or access can cure.
    let mut missing: Vec<String> = Vec::new();
    let document: Option<Value> = evidence
        .config
        .and_then(|text| serde_json::from_str(text).ok());
    let parsed = evidence.config.map(architecture);
    let architecture = match &parsed {
        Some(Ok(architecture)) => Some(architecture),
        Some(Err(error)) => {
            missing.push(format!("a config.json this build can read ({error})"));
            None
        }
        None => None,
    };
    let root = document.as_ref().or(evidence.declared);
    let model_type = architecture
        .map(|architecture| architecture.family.clone())
        .or_else(|| root.and_then(declared_model_type));
    if let Some(root) = root {
        missing.extend(Architecture::unmodelled_features(root));
    }
    if let Some(architecture) = architecture {
        missing.extend(switch_gaps(architecture));
    }
    let profile = model_type.as_deref().and_then(profile_for_model_type);
    if let (Some(kind), None) = (&model_type, profile) {
        missing.push(no_engine(kind));
    }
    missing.extend(evidence.missing.iter().cloned());
    if model_type.is_none() && root.is_some() {
        missing.push("a model_type in config.json".to_owned());
    }
    let window = architecture
        .map(|architecture| architecture.max_position_embeddings)
        .filter(|window| *window > 0)
        .or(profile.map(|profile| profile.context_tokens));
    // Anything the whole checkpoint lacks blocks every width; a family whose arm is known but
    // whose document was not read leaves the engine undecided rather than refused.
    let blocked = !missing.is_empty();
    let document_key = evidence.config.map(|text| {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    });
    let configurations: Vec<Configuration> = WORLDS
        .iter()
        .map(|&world| {
            configuration(
                world,
                architecture.map(|architecture| (architecture, document_key)),
                evidence.config,
                profile,
                blocked,
                evidence.weights,
                window,
                cluster,
            )
        })
        .collect();
    let verdict = overall(&configurations, blocked);
    let mut fit = Fit {
        verdict,
        summary: String::new(),
        model_type,
        family: profile.map(|profile| profile.label.to_owned()),
        engine: profile.map(|profile| profile.engine.to_owned()),
        host_reference: on_host_reference(profile, architecture),
        window,
        weights_bytes: evidence.weights.map(|weights| weights.bytes),
        weights_measured: evidence.weights.is_some_and(|weights| weights.measured),
        cluster: *cluster,
        missing,
        configurations,
        scans: evidence.config.map_or_else(Vec::new, |config| {
            scan_views(config, None, SCAN_TARGETS, &WORLDS)
        }),
    };
    fit.summary = summary(&fit);
    fit
}

fn configuration(
    world: usize,
    architecture: Option<(&Architecture, Option<u64>)>,
    config: Option<&str>,
    profile: Option<&ModelProfile>,
    blocked: bool,
    weights: Option<WeightBytes>,
    window: Option<usize>,
    cluster: &Cluster,
) -> Configuration {
    let mut reasons = Vec::new();
    let mut notes = Vec::new();
    let derived = architecture.map(|(architecture, key)| derive(config, architecture, key, world));
    let architecture = architecture.map(|(architecture, _)| architecture);
    let engine_ready = match &derived {
        Some(derived) => match &derived.gaps {
            Ok(()) => Some(profile.is_some() && !blocked),
            Err(gaps) => {
                reasons.extend(gaps.iter().cloned());
                Some(false)
            }
        },
        None if blocked || profile.is_none() => Some(false),
        None => None,
    };
    if on_host_reference(profile, architecture) {
        notes.push(
            "the shared CPU host reference executes it (f32, one rank); it is a correctness \
             bring-up path, not a throughput claim"
                .to_owned(),
        );
    } else if let Some(arm) = profile
        .filter(|profile| profile.engine == DYNAMIC_ENGINE_ID)
        .and(architecture)
        .and_then(|architecture| sparkflash_ops::parts::dynamic_device_arm(architecture).ok())
    {
        notes.push(format!(
            "{arm} executes it on the GPU; the dynamic engine launches it from the checkpoint"
        ));
    }
    if world > cluster.nodes {
        reasons.push(format!(
            "{world} Sparks (the cluster has {})",
            cluster.nodes
        ));
    }

    let budget = cluster.budget();
    let deployed = profile.is_some_and(|profile| {
        profile.weights == DEFAULT_WEIGHTS_ID && profile.world_size == world
    });
    let planned = derived.as_ref().and_then(|derived| derived.planned);
    let rank_weights = planned.or_else(|| {
        // DeepSeek's checkpoint bytes include the Engram tables, which stay on NVMe; a share of
        // them is not its resident set, so no estimate is better than a wrong one.
        let deepseek = profile.is_some_and(|profile| profile.weights == DEFAULT_WEIGHTS_ID);
        weights
            .filter(|_| !deepseek)
            .map(|weights| weights.bytes.div_ceil(world as u64))
    });
    let kv = architecture.and_then(|architecture| architecture.kv_plan(world, world > 1).ok());
    let mut max_context = None;
    // Some(true) fits, Some(false) does not, None unknown.
    let fits = match rank_weights {
        Some(rank_weights) => {
            let fixed = kv.map_or(0, |kv| kv.fixed());
            let claimed = rank_weights.saturating_add(fixed);
            if claimed > budget {
                reasons.push(format!(
                    "room for {} of weights{} a rank (the budget is {})",
                    gib(rank_weights),
                    if fixed > 0 {
                        format!(" and {} of fixed caches", gib(fixed))
                    } else {
                        String::new()
                    },
                    gib(budget)
                ));
                Some(false)
            } else {
                let room = budget - claimed;
                match (kv, window) {
                    (Some(kv), Some(window)) => {
                        let tokens = if kv.per_token() == 0 {
                            window
                        } else {
                            usize::try_from(room / kv.per_token())
                                .unwrap_or(usize::MAX)
                                .min(window)
                        };
                        max_context = Some(tokens);
                        let needed = MIN_CONTEXT_TOKENS.min(window);
                        if tokens < needed {
                            reasons.push(format!(
                                "room for a {needed}-token cache beside {} of weights (only {tokens} \
                                 tokens fit)",
                                gib(rank_weights)
                            ));
                            Some(false)
                        } else {
                            Some(true)
                        }
                    }
                    _ => {
                        notes.push("the cache size is not modelled for this document".to_owned());
                        Some(true)
                    }
                }
            }
        }
        None if deployed => {
            notes.push(
                "the released TP=4 deployment (the Engram tables stay on NVMe, so the checkpoint \
                 size is not the resident set)"
                    .to_owned(),
            );
            max_context = profile.map(|profile| profile.context_tokens);
            Some(true)
        }
        None => None,
    };
    if planned.is_none() && rank_weights.is_some() {
        notes.push(format!(
            "weights estimated as 1/{world} of the checkpoint{}",
            if weights.is_some_and(|weights| weights.measured) {
                ""
            } else {
                ", sized from the Hub's parameter counts"
            }
        ));
    }

    let verdict = if engine_ready == Some(false) || world > cluster.nodes {
        Verdict::Missing
    } else if fits == Some(false) {
        Verdict::NoRoom
    } else if engine_ready.is_none() || fits.is_none() {
        if fits.is_none() && engine_ready.is_some() {
            reasons.push("the checkpoint's size".to_owned());
        }
        Verdict::Unknown
    } else {
        Verdict::Runs
    };
    Configuration {
        world,
        verdict,
        engine_ready,
        launch: profile.is_some_and(|profile| profile.world_size == world),
        rank_weights_bytes: rank_weights,
        rank_weights_planned: planned.is_some(),
        kv_bytes_per_token: kv.map(|kv| kv.per_token()),
        kv_fixed_bytes: kv.map(|kv| kv.fixed()),
        budget_bytes: budget,
        max_context,
        reasons,
        notes,
    }
}

/// What a document implies at one width, independent of the cluster: the engine's gaps and the
/// planned resident weights.
#[derive(Debug, Clone)]
struct Derived {
    gaps: Result<(), Vec<String>>,
    planned: Option<u64>,
}

/// Entries the memo keeps before it starts over; a search page judges at most a few dozen.
const MEMO_ENTRIES: usize = 512;

/// [Derived] by document hash and width.
///
/// Planning a release's tensors is the one expensive step (Qwen3.8's 24,576 routed experts take
/// about 0.1 s a width in a release build), and Hub variants of a release usually carry its
/// `config.json` byte for byte, so one search reuses the answer across them and across
/// keystrokes.
static MEMO: LazyLock<Mutex<HashMap<(u64, usize), Derived>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn derive(
    config: Option<&str>,
    architecture: &Architecture,
    key: Option<u64>,
    world: usize,
) -> Derived {
    let compute = || Derived {
        // The architecture came from this same document; a report over no document is not a
        // claim that the checkpoint is clean.
        gaps: config.map_or(Ok(()), |config| scan_gaps(config, architecture, world)),
        planned: planned_rank_bytes(architecture, world),
    };
    let Some(key) = key else {
        return compute();
    };
    if let Some(found) = MEMO
        .lock()
        .ok()
        .and_then(|memo| memo.get(&(key, world)).cloned())
    {
        return found;
    }
    let derived = compute();
    if let Ok(mut memo) = MEMO.lock() {
        if memo.len() >= MEMO_ENTRIES {
            memo.clear();
        }
        memo.insert((key, world), derived.clone());
    }
    derived
}

/// One rank's resident weights from the shared tensor plan, for the families it names.
///
/// The served set plus the MTP stage (speculation loads it); the host-paged PLE tables are not
/// resident and are left out.
fn planned_rank_bytes(architecture: &Architecture, world: usize) -> Option<u64> {
    let tp = TpPlan::for_architecture(architecture, world, 0).ok()?;
    let plan = ArchitectureTensorPlan::served_and_mtp(architecture, tp).ok()?;
    let resident = plan
        .rank_bytes()
        .saturating_sub(plan.rank_role_bytes(Role::PleTable));
    u64::try_from(resident).ok()
}

fn declared_model_type(root: &Value) -> Option<String> {
    root.get("model_type")
        .or_else(|| {
            root.get("text_config")
                .and_then(|text| text.get("model_type"))
        })
        .and_then(Value::as_str)
        .map(|kind| canonical_model_type(kind).to_owned())
}

fn overall(configurations: &[Configuration], blocked: bool) -> Verdict {
    let any = |verdict| configurations.iter().any(|entry| entry.verdict == verdict);
    if any(Verdict::Runs) {
        Verdict::Runs
    } else if blocked {
        Verdict::Missing
    } else if any(Verdict::NoRoom) {
        Verdict::NoRoom
    } else if any(Verdict::Unknown) {
        Verdict::Unknown
    } else {
        Verdict::Missing
    }
}

/// A reason without its parenthetical detail, for the one-line summary; the reasons themselves
/// keep it.
fn headline(reason: &str) -> &str {
    reason.split_once(" (").map_or(reason, |(head, _)| head)
}

fn summary(fit: &Fit) -> String {
    match fit.verdict {
        Verdict::Runs => {
            let running: Vec<&Configuration> = fit.running().collect();
            let worlds = running
                .iter()
                .map(|entry| format!("TP={}", entry.world))
                .collect::<Vec<_>>()
                .join(", ");
            let longest = running.iter().filter_map(|entry| entry.max_context).max();
            let engine = if fit.host_reference {
                " on the CPU host reference"
            } else {
                ""
            };
            match longest {
                Some(tokens) => format!(
                    "Runs{engine} at {worlds}, up to {} tokens",
                    thousands(tokens)
                ),
                None => format!("Runs{engine} at {worlds}"),
            }
        }
        Verdict::NoRoom => {
            let widest = fit
                .configurations
                .iter()
                .rev()
                .find(|entry| entry.verdict == Verdict::NoRoom);
            match widest.and_then(|entry| entry.reasons.first()) {
                Some(reason) => format!(
                    "Too large for {} Spark{}: needs {}",
                    fit.cluster.nodes,
                    if fit.cluster.nodes == 1 { "" } else { "s" },
                    headline(reason)
                ),
                None => "Too large for this cluster".to_owned(),
            }
        }
        Verdict::Missing => {
            let first = fit.missing.first().or_else(|| {
                fit.launch()
                    .or_else(|| fit.configurations.last())
                    .and_then(|entry| entry.reasons.first())
            });
            match first {
                Some(reason) => format!("Missing {}", headline(reason)),
                None => "Missing a device engine".to_owned(),
            }
        }
        Verdict::Unknown => {
            let first = fit
                .missing
                .first()
                .or_else(|| {
                    fit.configurations
                        .iter()
                        .find_map(|entry| entry.reasons.first())
                })
                .cloned()
                .unwrap_or_else(|| "its config.json".to_owned());
            format!("Cannot judge without {first}")
        }
    }
}

fn widths(worlds: &[usize]) -> String {
    worlds
        .iter()
        .map(|world| format!("TP={world}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn thousands(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// The bytes the Hub's per-dtype parameter counts imply.
///
/// The Hub counts a packed integer tensor (AWQ/GPTQ `I32`, a packed FP4 `U8`) in logical
/// parameters, so when the document declares `quantization_config.bits` an integer count is
/// charged that many bits a parameter instead of the integer's width. The result is an
/// estimate, and [WeightBytes::measured] says so.
#[must_use]
pub fn hub_weight_bytes(parameters: &Value, declared: Option<&Value>) -> Option<u64> {
    let parameters = parameters.as_object()?;
    let bits = declared
        .and_then(|root| root.get("quantization_config"))
        .and_then(|config| config.get("bits"))
        .and_then(Value::as_u64)
        .filter(|bits| (1..=8).contains(bits));
    let mut total: u128 = 0;
    for (dtype, count) in parameters {
        let count = u128::from(count.as_u64()?);
        let packed = matches!(dtype.as_str(), "I32" | "U32" | "I8" | "U8" | "I16" | "U16");
        let bits_each: u128 = match (dtype.as_str(), bits) {
            (_, Some(bits)) if packed => u128::from(bits),
            ("F64" | "I64" | "U64", _) => 64,
            ("F32" | "I32" | "U32", _) => 32,
            ("BF16" | "F16" | "I16" | "U16", _) => 16,
            ("F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "I8" | "U8" | "BOOL", _) => 8,
            ("F4" | "F4_E2M1" | "U4" | "I4", _) => 4,
            _ => 16,
        };
        total = total.saturating_add(count.saturating_mul(bits_each).div_ceil(8));
    }
    u64::try_from(total).ok()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const QWEN: &str =
        include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json");
    const QWEN36: &str =
        include_str!("../../sparkflash-model/config/qwen36_flash_35b_a3b_fp8.hf.json");
    const GLM: &str = include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
    const DEEPSEEK: &str =
        include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");

    fn four() -> Cluster {
        Cluster::nominal(4)
    }

    fn of(fit: &Fit, world: usize) -> &Configuration {
        fit.configurations
            .iter()
            .find(|entry| entry.world == world)
            .expect("every width is judged")
    }

    #[test]
    fn the_budget_is_the_node_less_its_floor() {
        let cluster = four();
        assert_eq!(
            cluster.budget(),
            NODE_MEMORY_BUDGET_BYTES - MEMORY_FLOOR_BYTES
        );
        let measured = cluster.measured([0.0, 119.5, 121.7, f64::NAN]);
        assert!(measured.measured);
        assert_eq!(measured.node_bytes, (119.5 * (1u64 << 30) as f64) as u64);
        assert_eq!(cluster.measured([]), cluster);
    }

    #[test]
    fn glm_runs_at_four_and_is_too_large_for_one() {
        let fit = judge(
            &Evidence {
                config: Some(GLM),
                ..Evidence::default()
            },
            &four(),
        );
        assert_eq!(fit.verdict, Verdict::Runs, "{fit:#?}");
        assert_eq!(fit.family.as_deref(), Some("GLM"));
        let tp4 = of(&fit, 4);
        assert_eq!(tp4.verdict, Verdict::Runs, "{tp4:#?}");
        assert!(tp4.launch);
        assert!(tp4.rank_weights_planned);
        assert!(
            tp4.max_context
                .is_some_and(|tokens| tokens >= MIN_CONTEXT_TOKENS)
        );
        let tp1 = of(&fit, 1);
        assert_eq!(tp1.verdict, Verdict::NoRoom, "{tp1:#?}");
        assert!(tp1.reasons.iter().any(|reason| reason.contains("weights")));
        assert!(fit.summary.starts_with("Runs at"), "{}", fit.summary);
        assert!(fit.summary.contains("TP=4"), "{}", fit.summary);
    }

    #[test]
    fn qwen_runs_on_one_spark() {
        let fit = judge(
            &Evidence {
                config: Some(QWEN),
                ..Evidence::default()
            },
            &Cluster::nominal(1),
        );
        let tp1 = of(&fit, 1);
        assert_eq!(tp1.verdict, Verdict::Runs, "{tp1:#?}");
        assert!(tp1.launch);
        let tp4 = of(&fit, 4);
        assert_eq!(tp4.verdict, Verdict::Missing);
        assert!(
            tp4.reasons.iter().any(|reason| reason.contains("4 Sparks")),
            "{tp4:#?}"
        );
    }

    #[test]
    fn qwen36_fp8_runs_on_its_device_arm_at_one_spark() {
        let fit = judge(
            &Evidence {
                config: Some(QWEN36),
                weights: Some(WeightBytes {
                    bytes: 37_500_000_000,
                    measured: false,
                }),
                ..Evidence::default()
            },
            &Cluster::nominal(1),
        );
        // The FP8 release's experts and dense projections are the 128x128 E4M3 tiles the
        // Qwen3.5-MoE device arm reads, so it is judged by the kernel catalogue and runs on the
        // GPU; the dynamic engine still launches it, straight from the checkpoint.
        assert_eq!(fit.verdict, Verdict::Runs, "{fit:#?}");
        assert_eq!(fit.family.as_deref(), Some("Qwen"));
        assert_eq!(
            fit.engine.as_deref(),
            Some(crate::config::DYNAMIC_ENGINE_ID)
        );
        assert!(!fit.host_reference);
        assert!(fit.missing.is_empty(), "{:?}", fit.missing);
        let tp1 = of(&fit, 1);
        assert!(tp1.launch);
        assert!(
            tp1.notes.iter().any(|note| note.contains("device arm")),
            "{tp1:#?}"
        );
        assert!(!fit.summary.contains("host reference"), "{}", fit.summary);
        // The family's device arm runs at one rank only.
        assert_eq!(of(&fit, 2).verdict, Verdict::Missing);
    }

    #[test]
    fn a_bf16_qwen36_runs_on_the_host_reference_at_one_spark() {
        // The same family with no quantization block stores BF16 routed experts, which no
        // device kernel reads yet: the profile's dynamic engine runs it on the host reference,
        // and the browser says so.
        let mut document: Value = serde_json::from_str(QWEN36).expect("the carried document");
        document
            .as_object_mut()
            .expect("an object")
            .remove("quantization_config");
        let text = document.to_string();
        let fit = judge(
            &Evidence {
                config: Some(&text),
                weights: Some(WeightBytes {
                    bytes: 70_000_000_000,
                    measured: false,
                }),
                ..Evidence::default()
            },
            &Cluster::nominal(1),
        );
        assert_eq!(fit.verdict, Verdict::Runs, "{fit:#?}");
        assert!(fit.host_reference);
        let tp1 = of(&fit, 1);
        assert!(
            tp1.notes
                .iter()
                .any(|note| note.contains("CPU host reference")),
            "{tp1:#?}"
        );
        assert!(
            fit.summary.contains("CPU host reference"),
            "{}",
            fit.summary
        );
    }

    #[test]
    fn deepseek_runs_only_at_its_deployed_width() {
        let fit = judge(
            &Evidence {
                config: Some(DEEPSEEK),
                weights: Some(WeightBytes {
                    bytes: 510_000_000_000,
                    measured: true,
                }),
                ..Evidence::default()
            },
            &four(),
        );
        let tp4 = of(&fit, 4);
        assert_eq!(tp4.verdict, Verdict::Runs, "{tp4:#?}");
        assert!(tp4.rank_weights_bytes.is_none());
        assert!(!tp4.notes.is_empty());
        let tp1 = of(&fit, 1);
        assert_eq!(tp1.verdict, Verdict::Missing);
        assert!(
            tp1.reasons.iter().any(|reason| reason.contains("TP=4")),
            "{tp1:#?}"
        );
    }

    #[test]
    fn a_novel_family_is_missing_an_engine_however_small() {
        let llama = json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "torch_dtype": "bfloat16"
        })
        .to_string();
        let fit = judge(
            &Evidence {
                config: Some(&llama),
                weights: Some(WeightBytes {
                    bytes: 2_500_000_000,
                    measured: false,
                }),
                ..Evidence::default()
            },
            &four(),
        );
        assert_eq!(fit.verdict, Verdict::Missing, "{fit:#?}");
        assert!(fit.summary.contains("device engine"), "{}", fit.summary);
        assert!(fit.engine.is_none());
        for entry in &fit.configurations {
            assert_eq!(entry.verdict, Verdict::Missing);
            assert_eq!(entry.engine_ready, Some(false));
            assert!(!entry.launch);
        }
        // The memory half is still reported: a 2.5 GB checkpoint would fit one Spark.
        assert!(of(&fit, 1).max_context.is_some());
    }

    #[test]
    fn a_dropped_declaration_blocks_a_released_family() {
        let mut glm: Value = serde_json::from_str(GLM).expect("the GLM document parses");
        glm["final_logit_softcapping"] = json!(30.0);
        let text = glm.to_string();
        let fit = judge(
            &Evidence {
                config: Some(&text),
                ..Evidence::default()
            },
            &four(),
        );
        assert_eq!(fit.verdict, Verdict::Missing, "{fit:#?}");
        assert!(
            fit.missing
                .iter()
                .any(|reason| reason.contains("soft-capping"))
        );
        assert!(
            fit.configurations
                .iter()
                .all(|entry| entry.verdict != Verdict::Runs)
        );
    }

    #[test]
    fn a_gated_document_names_its_access_and_an_unread_one_is_unknown() {
        let declared = json!({"model_type": "glm5_next"});
        let gated = ["access to the gated repository".to_owned()];
        let fit = judge(
            &Evidence {
                declared: Some(&declared),
                missing: &gated,
                ..Evidence::default()
            },
            &four(),
        );
        assert_eq!(fit.verdict, Verdict::Missing, "{fit:#?}");
        assert!(fit.missing[0].contains("gated"));
        let without = judge(
            &Evidence {
                declared: Some(&declared),
                ..Evidence::default()
            },
            &four(),
        );
        assert_eq!(without.verdict, Verdict::Unknown, "{without:#?}");
        assert_eq!(of(&without, 4).engine_ready, None);
    }

    #[test]
    fn hub_parameter_counts_charge_packed_integers_their_declared_bits() {
        let awq = json!({"I32": 6_525_288_448u64, "F16": 1_090_328_064u64});
        let declared = json!({"quantization_config": {"bits": 4, "quant_method": "awq"}});
        let bytes = hub_weight_bytes(&awq, Some(&declared)).expect("counts");
        // 3.26 GB of 4-bit codes and 2.18 GB of F16: the stored release is 5.57 GB.
        assert_eq!(bytes, 3_262_644_224 + 2_180_656_128);
        let glm =
            json!({"BF16": 6_926_096_640u64, "F8_E4M3": 314_396_639_232u64, "F32": 295_518u64});
        assert_eq!(
            hub_weight_bytes(&glm, None),
            Some(6_926_096_640 * 2 + 314_396_639_232 + 295_518 * 4)
        );
        assert_eq!(hub_weight_bytes(&json!({"BF16": "many"}), None), None);
    }

    #[test]
    fn counts_are_grouped_by_thousands() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(4096), "4,096");
        assert_eq!(thousands(1_048_576), "1,048,576");
    }

    const SCAN_CLEAN: &str = r#"{"model_type":"","hidden_size":5120,"intermediate_size":256,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,"vocab_size":128,"max_position_embeddings":16}"#;

    #[test]
    fn a_clean_scan_offers_convert_and_serve() {
        let views = scan_views(SCAN_CLEAN, None, &[Target::Sm121], &[1]);
        assert_eq!(views.len(), 1);
        let view = &views[0];
        assert!(view.clean, "{}", view.plain);
        assert!(view.offers_convert);
        assert!(view.offers_serve);
        assert!(view.work_items.is_empty(), "{:?}", view.work_items);
        assert!(!view.rows.is_empty());
        assert!(
            view.plain.starts_with("scan report: sm121 TP1,"),
            "{}",
            view.plain
        );
    }

    #[test]
    fn a_scan_with_a_gap_offers_neither_convert_nor_serve() {
        // A 2560-wide MLA: the plan exists, but the catalogue's MLA bodies are built for other
        // widths, so the row is a named gap and not a clean report.
        let config = json!({
            "model_type": "acme_custom_flash",
            "hidden_size": 2560,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "vocab_size": 256,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "eos_token_id": [0],
            "layer_types": ["full_attention"],
            "kv_lora_rank": 16
        })
        .to_string();
        let views = scan_views(&config, None, &[Target::Sm121], &[1, 2, 4]);
        assert_eq!(views.len(), 3);
        for view in &views {
            assert!(!view.clean, "{}", view.plain);
            assert!(!view.offers_convert, "TP{} offered convert", view.world);
            assert!(!view.offers_serve, "TP{} offered serve", view.world);
            assert!(!view.work_items.is_empty(), "{}", view.plain);
        }
        assert!(
            views
                .iter()
                .flat_map(|view| &view.work_items)
                .any(|item| item.operation == "mla_attention"
                    && item.target == "sm121"
                    && item.fix.contains("build-list")),
            "{:#?}",
            views
                .iter()
                .map(|view| &view.work_items)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_plain_view_is_one_report_per_target_and_width() {
        let plain = scan_plain(SCAN_CLEAN, None, SCAN_TARGETS, &WORLDS);
        for target in SCAN_TARGETS {
            for world in WORLDS {
                assert!(
                    plain.contains(&format!("scan report: {} TP{world},", target.name())),
                    "no {} TP{world} report",
                    target.name()
                );
            }
        }
        assert_eq!(
            plain.matches("scan report:").count(),
            SCAN_TARGETS.len() * WORLDS.len()
        );
    }
}

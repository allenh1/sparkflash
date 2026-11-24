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

//! The numerics identity of a served configuration (docs/NUMERICS.md, DECISIONS.md D67).
//!
//! Nothing about a configuration's arithmetic changes unless someone declared it. The declaration
//! lives in `numerics/registry.toml`: one entry per (model, world size, numerics profile, parallel
//! plan, CUDA architecture) with its numerics version and the fingerprint of the reference that
//! version produced. This module is the serving side of that contract. It parses the registry the
//! binary embeds, finds the entry for what this process is about to serve, and reports it at boot
//! and on `/v1/models` as one of three things:
//!
//! - **registered**: the registry has an entry, and no switch that can change the arithmetic is set;
//! - **unregistered**: no reference exists for this configuration, so nothing vouches for its bits;
//! - **overridden**: an environment switch that can change the arithmetic is set, so the registered
//!   reference no longer describes this process.
//!
//! The bits themselves are checked against a running server by `scripts/numerics.py check`, and
//! the declaration rules (no changed reference without a version bump, no bump without a change)
//! by `scripts/numerics.py ci`. This module only says which version a process claims, and never
//! claims one it cannot.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// The registry the binary was built with.
pub const REGISTRY_TOML: &str = include_str!("../../../numerics/registry.toml");

/// The registry schema this module reads.
pub const SCHEMA: u32 = 1;

/// The one parallel plan the registry knows today: tensor parallelism over `world` ranks.
pub const PLAN_TP: &str = "tp";

/// The architecture key of a configuration that runs no CUDA kernel (the host reference engine).
pub const ARCH_HOST: &str = "host";

/// The profile fields ([crate::profile::ENV_OVERRIDES]) whose value can change a served
/// configuration's arithmetic: each selects a different kernel, precision, cache representation or
/// summation order from the shipped default.
///
/// A process with an override of one of them set reports itself as overridden, because the
/// registered reference was captured without it. The list is deliberately conservative - a field
/// whose two settings are claimed to give the same bits (a chunk width, say) is still here until a
/// reference proves it - and it is not the proof that nothing else matters; the captured
/// fingerprint is.
pub const ARITHMETIC_FIELDS: &[&str] = &[
    "prefill.enabled",
    "prefill.chunk",
    "prefill.drafting_prompt",
    "prefill.tensor_cores",
    "prefill.exact",
    "prefill.activation",
    "prefill.chunk_busy",
    "prefill.exact_busy",
    "prefill.bulk_ctas",
    "prefill.moe_merge",
    "prefill.bulk_min",
    "prefill.ways",
    "prefill.pair",
    "decode.tensor_cores",
    "attention.latent_cache",
    "sequence.parallel",
    "sequence.overlap",
    "plan.threads",
    "diagnostics.attn_split",
    "diagnostics.opt_moe",
    "diagnostics.opt_gemv",
    "diagnostics.opt_misc",
    "diagnostics.opt_head",
    "diagnostics.opt_router",
    "diagnostics.opt_seams",
    "diagnostics.oneshot",
    "diagnostics.opt_attn",
    "diagnostics.opt_select",
    "diagnostics.select_rank",
    "diagnostics.shared_tp",
    "diagnostics.boundary_trace",
    "diagnostics.tp4_boundary_dir",
    "diagnostics.trace_positions",
    "diagnostics.tap_dump",
    "diagnostics.route_dump",
    "diagnostics.reduce_dump",
    "diagnostics.device_diff_dump",
    "diagnostics.rows_dump",
    "diagnostics.generate",
    "diagnostics.print_argmax",
    "diagnostics.l2_prefetch",
    "diagnostics.engram_cache",
    "diagnostics.engram_keepalive",
    "diagnostics.engram_shard",
    "diagnostics.require_io_uring",
    "diagnostics.vision",
    "diagnostics.batch_trace",
    "diagnostics.device_trace",
    "diagnostics.device_trace_events",
    "diagnostics.seam_ref_host",
    "diagnostics.seam_debug",
    "diagnostics.tp4_boundary_position",
    "diagnostics.tolerance_arm",
    "diagnostics.router_twin",
    "diagnostics.head_host_assembly",
    "diagnostics.resident_attention",
    "drafter.logits",
    "drafter.head_tiled",
    "drafter.markov_mma",
    "drafter.moe_dedup",
    "drafter.head_fp8",
    "drafter.logits_rows",
    "diagnostics.forward_evidence",
    "diagnostics.layer_device",
    "diagnostics.prefix_cache",
    "diagnostics.prefix_reuse_min_rows",
];

/// The profile fields that cannot change the bits a request gets, each because an invariance gate
/// (docs/NUMERICS.md §1) holds it to the same output: context and concurrency (batch and
/// concurrency invariance), speculation (speculation never changes the output) and diagnostics.
/// Every [crate::profile::ENV_OVERRIDES] field is in exactly one of the two lists; a test keeps a
/// new override from landing unclassified.
pub const NEUTRAL_FIELDS: &[&str] = &[
    "runtime.context_tokens",
    "runtime.concurrency",
    "runtime.lane_memory_mb",
    "runtime.mem_floor_gib",
    "runtime.arena_mmap",
    "runtime.arena_hostmapped",
    "runtime.verify_weights",
    "runtime.converted_dir",
    "runtime.rows",
    "runtime.rows_from",
    "runtime.constrained",
    "prefill.read_threads",
    "prefill.warmup",
    "prefill.warmup_steps",
    "prefill.tick_ahead",
    "prefill.decode_share",
    "prefill.deadline_ms",
    "prefill.index_pack_append",
    "prefill.index_pack_dedupe",
    "prefill.index_pack_log",
    "prefill.index_pack_verify",
    "speculation.enabled",
    "speculation.depth",
    "speculation.drafts",
    "speculation.depth_policy",
    "speculation.min_depth",
    "speculation.mtp_fold_scale",
    "speculation.dspark_k",
    "speculation.dspark_steps",
    "speculation.batch_spec",
    "speculation.batch_spec_floor",
    "speculation.row_ms",
    "speculation.batch_spec_cost",
    "speculation.drafter",
    "diagnostics.profile",
    "diagnostics.probe",
    "diagnostics.trace",
    "diagnostics.spec_trace",
    "diagnostics.device_gather",
    "diagnostics.engram_direct",
    "diagnostics.device_diff",
    "diagnostics.seam_warm",
    "diagnostics.lean_head",
    "diagnostics.cuda_graph",
    "diagnostics.prefill_sp",
    "diagnostics.seg_blocks",
    "diagnostics.engram_uring",
    "diagnostics.seam_split",
    "diagnostics.engram_staged",
    "diagnostics.engram_split",
    "diagnostics.rows_ring",
    "diagnostics.rows_groups",
    "diagnostics.head_screen",
    "diagnostics.head_screen_check",
    "diagnostics.embed_replica",
    "diagnostics.prefetch_wqb",
    "diagnostics.rows_select",
    "diagnostics.select_together",
    "diagnostics.rows_indexer",
    "diagnostics.row_table",
    "diagnostics.engram_ahead",
    "diagnostics.prefill_decoder_skip",
    "diagnostics.engram_dedupe",
    "diagnostics.moe_dedup",
    "diagnostics.moe_plan_in_cta",
    "diagnostics.moe_plan_stats",
    "diagnostics.moe_ids_dump",
    "drafter.seqs",
    "drafter.seed_split",
    "drafter.row_table",
];

/// The arithmetic switches an arm still reads directly, outside the profile table, until their
/// typed field lands (docs/arm-reference.md section 3); each moves to [ARITHMETIC_FIELDS] when
/// it does.
/// The last column spells the arm's default: a switch set to one of those values selects exactly
/// what leaving it unset selects (production's `SPARK_COMM_REDUCE=nccl` on a TP world), so it
/// keeps the claim. Any other value withdraws it, including a spelling the arm happens to read as
/// its default.
pub const ARITHMETIC_ENV: &[(&str, &str, &[&str])] = &[
    (
        "SPARKFLASH_GLM_KV_CACHE",
        "the GLM attention-cache representation",
        &["fp8"],
    ),
    (
        "SPARKFLASH_ENABLE_MOE_SPLITK",
        "the split-K routed-expert kernel",
        &["0"],
    ),
    (
        "SPARK_COMM_REDUCE",
        "the collective's reduction path",
        &["nccl"],
    ),
];

/// Every environment switch that can withdraw a process's numerics claim: the arithmetic profile
/// overrides, then the direct switches.
pub fn arithmetic_switches() -> impl Iterator<Item = &'static str> {
    crate::profile::ENV_OVERRIDES
        .iter()
        .filter(|(_, field)| ARITHMETIC_FIELDS.contains(field))
        .map(|(name, _)| *name)
        .chain(ARITHMETIC_ENV.iter().map(|(name, _, _)| *name))
}

/// Whether `name=value` withdraws the claim: an arithmetic profile override always does, since its
/// default is the model's; a direct switch does unless `value` spells its arm's default.
#[must_use]
pub fn withdraws_claim(name: &str, value: &str) -> bool {
    ARITHMETIC_ENV
        .iter()
        .find(|(switch, _, _)| *switch == name)
        .is_none_or(|(_, _, defaults)| !defaults.contains(&value))
}

/// A registry failure.
#[derive(Debug, thiserror::Error)]
pub enum NumericsError {
    /// The TOML does not parse, or a key or enum value is unknown.
    #[error("numerics registry: {0}")]
    Parse(String),
    /// An entry is malformed or two entries collide.
    #[error("numerics registry: {0}")]
    Invalid(String),
    /// A numerics profile name the registry does not define.
    #[error("{value:?} is not a numerics profile; the profiles are: {accepted}")]
    UnknownProfile {
        /// The value given.
        value: String,
        /// The accepted names.
        accepted: String,
    },
}

/// A named arithmetic a configuration can serve.
///
/// `reference` is the shipped arithmetic. A profile that reduces precision is added here only as
/// an opt-in, with its measured deltas published (docs/NUMERICS.md §2); none exists yet, so any
/// other name is refused rather than served as `reference`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericsProfile {
    /// The shipped arithmetic.
    Reference,
}

impl NumericsProfile {
    /// Every profile, in the order they are listed to an operator.
    pub const ALL: &[Self] = &[Self::Reference];

    /// The profile's name, as `--numerics` and the registry spell it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
        }
    }

    /// Parse a profile name.
    ///
    /// # Errors
    /// A name no profile has. `fast` is refused too: it is the opt-in a precision change will add,
    /// and until one has been declared and measured there is nothing for it to select.
    pub fn parse(value: &str) -> Result<Self, NumericsError> {
        Self::ALL
            .iter()
            .copied()
            .find(|profile| profile.name() == value)
            .ok_or_else(|| NumericsError::UnknownProfile {
                value: value.to_owned(),
                accepted: Self::ALL
                    .iter()
                    .map(|profile| profile.name())
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }
}

impl fmt::Display for NumericsProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// What a reference fingerprints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    /// Every generated token id plus its top-k logprobs, bit for bit. A change to any logit moves
    /// the log-softmax normaliser, so this covers the whole row.
    Logprobs,
    /// Every greedy completion's text only, for a path that serves neither logprobs nor token ids
    /// (DeepSeek's TP=4 device path). Weaker: an arithmetic change that flips no greedy token is
    /// invisible to it.
    Text,
}

/// One model's committed coverage dump (docs/NUMERICS.md §5).
///
/// The bar is the registry's shared [Prerequisites::coverage_slots]; `min_inputs` is the coverage
/// suite's declared input floor. A model without a row has no coverage prerequisite, so an
/// unapproved bump on it is refused by `scripts/numerics.py ci` with the model named.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageEntry {
    /// The model the dump is for; one row a model.
    pub model: String,
    /// The committed dump, under `numerics/coverage/`.
    pub dump: String,
    /// The declared coverage suite's input floor.
    pub min_inputs: usize,
}

/// One model's committed monitor pair (docs/NUMERICS.md §5).
///
/// The two dumps are judged at the declared relative-error bars. The judge compares the experts
/// both builds routed to and reports the ones only one did, because a router may change a row's
/// expert on a different hidden state; that divergence is a measured fact, not a malformed pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonitorsEntry {
    /// The model the pair is for; one row a model.
    pub model: String,
    /// The reference build's dump, under `numerics/monitors/`.
    pub reference: String,
    /// The candidate build's dump, under `numerics/monitors/`.
    pub candidate: String,
    /// The per-layer relative-error bar.
    pub layer_bar: f64,
    /// The per-shared-expert relative-error bar.
    pub expert_bar: f64,
}

/// The prerequisites docs/NUMERICS.md §5 names before an arithmetic change may land.
///
/// `anchor`, `coverage` and `monitors` are per model: a configuration is judged against its own
/// model's rows, not against these aggregates. Each flag reports whether every captured `logprobs`
/// model has its own passing row; a false value does not block a model that has its row.
/// `declaration_check` gates every bump. [Registry::missing_for] answers the per-model question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prerequisites {
    /// The frozen anchors and teacher-forced comparisons, reported per model (see the type docs).
    pub anchor: bool,
    /// The CI declaration check.
    pub declaration_check: bool,
    /// The fraction of routed (layer, expert) slots a committed coverage dump must reach.
    ///
    /// A fraction of the layout's slots, not a count, and not 1.0 by default: real serving
    /// reached 92-93% of Qwen3.8's 48x512 slots, so the agreed bar is 0.93 (docs/NUMERICS.md §5).
    pub coverage_slots: f64,
    /// Reported per model: a [[coverage]] row that reaches [`Self::coverage_slots`].
    pub coverage: bool,
    /// Reported per model: a [[monitors]] row that meets its bars.
    pub monitors: bool,
}

impl Prerequisites {
    /// The prerequisites that are not in place yet, by registry name.
    #[must_use]
    pub fn missing(&self) -> Vec<&'static str> {
        [
            ("anchor", self.anchor),
            ("declaration_check", self.declaration_check),
            ("coverage", self.coverage),
            ("monitors", self.monitors),
        ]
        .into_iter()
        .filter_map(|(name, present)| (!present).then_some(name))
        .collect()
    }
}

/// One served configuration: the registry's lookup key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConfigKey {
    /// The served model id, as the descriptor names it.
    pub model: String,
    /// The tensor-parallel world size.
    pub world: usize,
    /// The numerics profile.
    pub profile: NumericsProfile,
    /// The parallel plan.
    pub plan: String,
    /// The CUDA architecture the kernels were built for, or [ARCH_HOST].
    pub arch: String,
}

impl ConfigKey {
    /// A tensor-parallel configuration.
    #[must_use]
    pub fn tp(
        model: impl Into<String>,
        world: usize,
        profile: NumericsProfile,
        arch: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            world,
            profile,
            plan: PLAN_TP.to_owned(),
            arch: arch.into(),
        }
    }
}

impl fmt::Display for ConfigKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} world={} profile={} plan={} arch={}",
            self.model, self.world, self.profile, self.plan, self.arch
        )
    }
}

/// An owner's recorded approval of one declared change (docs/NUMERICS.md §5).
///
/// It sits beside the version it let land and is part of the record, like the version:
/// `scripts/numerics.py ci` refuses its removal, and a bump that lands under it still needs the
/// declared change, the report and the model's anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Approval {
    /// The decision the approval records, one line naming it.
    pub decision: String,
    /// The day it was taken (YYYY-MM-DD).
    pub date: String,
    /// The report that justifies the change, relative to the repository root.
    pub report: String,
}

/// A configuration's declared numerics: the version it serves and the reference that proves it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigEntry {
    /// The served model id.
    pub model: String,
    /// The tensor-parallel world size.
    pub world: usize,
    /// The numerics profile.
    pub profile: NumericsProfile,
    /// The parallel plan.
    pub plan: String,
    /// The CUDA architecture.
    pub arch: String,
    /// The numerics version, starting at 1 and bumped by each declared arithmetic change.
    pub version: u32,
    /// What the reference fingerprints.
    pub evidence: Evidence,
    /// The reference file, relative to `numerics/`. `None` with no [Self::fingerprint] means the
    /// configuration is registered but not yet captured.
    #[serde(default)]
    pub reference: Option<String>,
    /// The sha256 of the reference's canonical capture. `None` with no [Self::reference] means
    /// the configuration is registered but not yet captured.
    #[serde(default)]
    pub fingerprint: Option<String>,
    /// The quality report a declared change committed; required from version 2 on.
    #[serde(default)]
    pub report: Option<String>,
    /// The date the reference was captured (YYYY-MM-DD).
    pub captured: String,
    /// What a reader should know about this entry.
    #[serde(default)]
    pub note: Option<String>,
    /// The owner's recorded approval of the change this version declared, if it landed under one.
    #[serde(default)]
    pub approved: Option<Approval>,
}

impl ConfigEntry {
    /// Whether a capture vouches for this entry's bits.
    ///
    /// A configuration may be registered before its first capture: `reference` and
    /// `fingerprint` are then both absent, and the entry answers
    /// [NumericsStatus::Unregistered] until `scripts/numerics.py capture` fills them in.
    #[must_use]
    pub fn is_captured(&self) -> bool {
        self.reference.is_some()
    }

    /// The entry's lookup key.
    #[must_use]
    pub fn key(&self) -> ConfigKey {
        ConfigKey {
            model: self.model.clone(),
            world: self.world,
            profile: self.profile,
            plan: self.plan.clone(),
            arch: self.arch.clone(),
        }
    }
}

/// A frozen quality reference: the arm every numerics version of a model is measured against.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorEntry {
    /// The model the anchor is for; one anchor per model.
    pub model: String,
    /// The world size the anchor was captured at.
    pub world: usize,
    /// The profile it ran.
    pub profile: NumericsProfile,
    /// The parallel plan.
    pub plan: String,
    /// The CUDA architecture.
    pub arch: String,
    /// What the anchor's capture holds.
    pub evidence: Evidence,
    /// The reference file, relative to `numerics/`.
    pub reference: String,
    /// The sha256 of its canonical capture.
    pub fingerprint: String,
    /// The capture date (YYYY-MM-DD).
    pub captured: String,
    /// What a reader should know about this anchor.
    #[serde(default)]
    pub note: Option<String>,
    /// The anchor this one replaces, when it pinned a defect: a replacement is recorded, never a
    /// silent edit (docs/NUMERICS.md §5).
    #[serde(default)]
    pub replaces: Option<Replaces>,
}

/// The card and fingerprint an anchor replaced, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Replaces {
    /// The superseded anchor's reference file, relative to `numerics/`.
    pub reference: String,
    /// The superseded anchor's fingerprint.
    pub fingerprint: String,
    /// Why the old anchor was wrong.
    pub reason: String,
}

/// A configuration that is no longer served. Re-adding it takes a version above this one, so a
/// remove-and-re-add cannot reset a version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetiredEntry {
    /// The served model id.
    pub model: String,
    /// The world size.
    pub world: usize,
    /// The profile.
    pub profile: NumericsProfile,
    /// The parallel plan.
    pub plan: String,
    /// The CUDA architecture.
    pub arch: String,
    /// The last version it served.
    pub version: u32,
    /// Why it was retired.
    pub reason: String,
    /// The capture the retired version left in history, when it had one: a path under
    /// `references/`. A retired configuration keeps its last reference named, so the declaration
    /// check does not report that capture as orphaned (docs/NUMERICS.md §5).
    #[serde(default)]
    pub reference: Option<String>,
    /// The retired capture's fingerprint, when [Self::reference] is set.
    #[serde(default)]
    pub fingerprint: Option<String>,
}

impl RetiredEntry {
    /// The retired configuration's key.
    #[must_use]
    pub fn key(&self) -> ConfigKey {
        ConfigKey {
            model: self.model.clone(),
            world: self.world,
            profile: self.profile,
            plan: self.plan.clone(),
            arch: self.arch.clone(),
        }
    }
}

/// The parsed registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    /// The schema version, [SCHEMA].
    pub schema: u32,
    /// The prerequisites for declaring a change.
    pub prerequisites: Prerequisites,
    /// The frozen anchors.
    #[serde(default)]
    pub anchor: Vec<AnchorEntry>,
    /// The per-model coverage evidence (docs/NUMERICS.md §5); one row a model.
    #[serde(default)]
    pub coverage: Vec<CoverageEntry>,
    /// The per-model monitor evidence (docs/NUMERICS.md §5); one row a model.
    #[serde(default)]
    pub monitors: Vec<MonitorsEntry>,
    /// The served configurations.
    #[serde(default)]
    pub config: Vec<ConfigEntry>,
    /// Configurations no longer served.
    #[serde(default)]
    pub retired: Vec<RetiredEntry>,
}

impl Registry {
    /// The registry this binary embeds.
    ///
    /// # Errors
    /// As [Self::from_toml]. The workspace tests parse the same text, so a build whose registry
    /// does not parse fails its gate before it can serve.
    pub fn embedded() -> Result<Self, NumericsError> {
        Self::from_toml(REGISTRY_TOML)
    }

    /// Parse and validate a registry.
    ///
    /// # Errors
    /// A syntax error, an unknown key, or an entry [Self::validate] refuses.
    pub fn from_toml(text: &str) -> Result<Self, NumericsError> {
        let registry: Self =
            toml::from_str(text).map_err(|error| NumericsError::Parse(error.to_string()))?;
        registry.validate()?;
        Ok(registry)
    }

    /// Refuse a malformed or colliding entry.
    ///
    /// # Errors
    /// A schema other than [SCHEMA]; an unknown model or plan; a zero world or version; a
    /// fingerprint that is not 64 lowercase hex digits; a reference outside `references/`; a
    /// version above 1 without a report; a duplicate configuration or anchor; or a re-added
    /// configuration whose version does not exceed the one it retired with.
    pub fn validate(&self) -> Result<(), NumericsError> {
        if self.schema != SCHEMA {
            return Err(NumericsError::Invalid(format!(
                "schema {} is not the {SCHEMA} this build reads",
                self.schema
            )));
        }
        let bar = self.prerequisites.coverage_slots;
        if !(bar > 0.0 && bar <= 1.0) {
            return Err(NumericsError::Invalid(format!(
                "prerequisites.coverage_slots {bar} is outside (0, 1]; it is the fraction of \
                 routed (layer, expert) slots a coverage dump must reach"
            )));
        }
        let mut covered = BTreeSet::new();
        for row in &self.coverage {
            check_model(&row.model)?;
            check_coverage_entry(row)?;
            if !covered.insert(row.model.clone()) {
                return Err(NumericsError::Invalid(format!(
                    "{} has more than one [[coverage]] row; one row a model",
                    row.model
                )));
            }
        }
        let mut monitored = BTreeSet::new();
        for row in &self.monitors {
            check_model(&row.model)?;
            check_monitors_entry(row)?;
            if !monitored.insert(row.model.clone()) {
                return Err(NumericsError::Invalid(format!(
                    "{} has more than one [[monitors]] row; one row a model",
                    row.model
                )));
            }
        }
        let mut seen = BTreeSet::new();
        for entry in &self.config {
            let key = entry.key();
            check_key(&key)?;
            if entry.version == 0 {
                return Err(NumericsError::Invalid(format!(
                    "{key}: version 0; versions start at 1"
                )));
            }
            check_reference(
                &key,
                entry.reference.as_deref(),
                entry.fingerprint.as_deref(),
                &entry.captured,
            )?;
            if entry.version > 1 && entry.report.is_none() {
                return Err(NumericsError::Invalid(format!(
                    "{key}: version {} declares an arithmetic change but names no report",
                    entry.version
                )));
            }
            if let Some(report) = &entry.report
                && !report.starts_with("docs/numerics/")
            {
                return Err(NumericsError::Invalid(format!(
                    "{key}: report {report:?} is not under docs/numerics/"
                )));
            }
            if let Some(approved) = &entry.approved {
                check_approval(&key, approved)?;
            }
            if let Some(retired) = self.retired.iter().find(|retired| retired.key() == key)
                && entry.version <= retired.version
            {
                return Err(NumericsError::Invalid(format!(
                    "{key}: version {} does not exceed the {} it was retired with",
                    entry.version, retired.version
                )));
            }
            if !seen.insert(key.clone()) {
                return Err(NumericsError::Invalid(format!(
                    "{key} has more than one entry"
                )));
            }
        }
        let mut anchored = BTreeSet::new();
        for anchor in &self.anchor {
            let key = ConfigKey {
                model: anchor.model.clone(),
                world: anchor.world,
                profile: anchor.profile,
                plan: anchor.plan.clone(),
                arch: anchor.arch.clone(),
            };
            check_key(&key)?;
            check_reference(
                &key,
                Some(&anchor.reference),
                Some(&anchor.fingerprint),
                &anchor.captured,
            )?;
            if let Some(replaces) = &anchor.replaces {
                check_reference_bits(&key, Some(&replaces.reference), Some(&replaces.fingerprint))?;
                if replaces.reason.trim().is_empty() {
                    return Err(NumericsError::Invalid(format!(
                        "{}: replaces.reason is empty",
                        anchor.model
                    )));
                }
            }
            if !anchored.insert(anchor.model.clone()) {
                return Err(NumericsError::Invalid(format!(
                    "{} has more than one anchor; an anchor is frozen, never replaced",
                    anchor.model
                )));
            }
        }
        for retired in &self.retired {
            check_key(&retired.key())?;
            check_reference_bits(
                &retired.key(),
                retired.reference.as_deref(),
                retired.fingerprint.as_deref(),
            )?;
        }
        Ok(())
    }

    /// The entry for a configuration, if one is registered.
    #[must_use]
    pub fn lookup(&self, key: &ConfigKey) -> Option<&ConfigEntry> {
        self.config.iter().find(|entry| &entry.key() == key)
    }

    /// The frozen anchor for a model, if one exists.
    #[must_use]
    pub fn anchor_for(&self, model: &str) -> Option<&AnchorEntry> {
        self.anchor.iter().find(|anchor| anchor.model == model)
    }

    /// The coverage row for a model, if the registry holds one.
    #[must_use]
    pub fn coverage_for(&self, model: &str) -> Option<&CoverageEntry> {
        self.coverage.iter().find(|row| row.model == model)
    }

    /// The monitors row for a model, if the registry holds one.
    #[must_use]
    pub fn monitors_for(&self, model: &str) -> Option<&MonitorsEntry> {
        self.monitors.iter().find(|row| row.model == model)
    }

    /// The per-model prerequisites a bump on `model` needs that the registry does not hold.
    ///
    /// This answers the per-model question [Prerequisites] reports the aggregate of: the model's
    /// own frozen anchor, its own [[coverage]] row and its own [[monitors]] row. The global
    /// [Prerequisites::declaration_check] is not repeated here; [Prerequisites::missing] reports
    /// it. Whether a row's dump passes its bar is `scripts/numerics.py ci`'s judgement - the
    /// embedded registry names the evidence, it does not read it - so a present row is answered as
    /// present here.
    #[must_use]
    pub fn missing_for(&self, model: &str) -> Vec<&'static str> {
        [
            ("anchor", self.anchor_for(model).is_some()),
            ("coverage", self.coverage_for(model).is_some()),
            ("monitors", self.monitors_for(model).is_some()),
        ]
        .into_iter()
        .filter_map(|(name, present)| (!present).then_some(name))
        .collect()
    }
}

fn check_coverage_entry(row: &CoverageEntry) -> Result<(), NumericsError> {
    if !row.dump.starts_with("numerics/coverage/")
        || !row.dump.ends_with(".json")
        || row.dump.split('/').any(|part| part == ".." || part == ".")
    {
        return Err(NumericsError::Invalid(format!(
            "[[coverage]] {:?}: dump {:?} is not a JSON dump under numerics/coverage/",
            row.model, row.dump
        )));
    }
    if row.min_inputs == 0 {
        return Err(NumericsError::Invalid(format!(
            "[[coverage]] {:?}: min_inputs 0; it is the declared coverage suite's input floor",
            row.model
        )));
    }
    Ok(())
}

fn check_monitors_entry(row: &MonitorsEntry) -> Result<(), NumericsError> {
    for (field, path) in [("reference", &row.reference), ("candidate", &row.candidate)] {
        if !path.starts_with("numerics/monitors/")
            || !path.ends_with(".json")
            || path.split('/').any(|part| part == ".." || part == ".")
        {
            return Err(NumericsError::Invalid(format!(
                "[[monitors]] {:?}: {field} {path:?} is not a JSON dump under numerics/monitors/",
                row.model
            )));
        }
    }
    for (field, bar) in [("layer_bar", row.layer_bar), ("expert_bar", row.expert_bar)] {
        if !(bar.is_finite() && bar > 0.0) {
            return Err(NumericsError::Invalid(format!(
                "[[monitors]] {:?}: {field} {bar} is not finite and positive; it is the \
                 relative-error bar a committed monitor row must meet",
                row.model
            )));
        }
    }
    Ok(())
}

fn check_model(model: &str) -> Result<(), NumericsError> {
    if !crate::descriptor::MODELS
        .iter()
        .any(|known| known.id == model)
    {
        return Err(NumericsError::Invalid(format!(
            "{model:?} is not a model id this build serves"
        )));
    }
    Ok(())
}

fn check_key(key: &ConfigKey) -> Result<(), NumericsError> {
    check_model(&key.model)?;
    if key.world == 0 {
        return Err(NumericsError::Invalid(format!("{key}: world 0")));
    }
    if key.plan != PLAN_TP {
        return Err(NumericsError::Invalid(format!(
            "{key}: plan {:?} is unknown; the only plan is {PLAN_TP:?}",
            key.plan
        )));
    }
    if key.arch != ARCH_HOST && !key.arch.starts_with("sm_") {
        return Err(NumericsError::Invalid(format!(
            "{key}: arch {:?} is neither {ARCH_HOST:?} nor an sm_ target",
            key.arch
        )));
    }
    Ok(())
}

fn check_reference(
    key: &ConfigKey,
    reference: Option<&str>,
    fingerprint: Option<&str>,
    captured: &str,
) -> Result<(), NumericsError> {
    check_reference_bits(key, reference, fingerprint)?;
    if !is_date(captured) {
        return Err(NumericsError::Invalid(format!(
            "{key}: captured {captured:?} is not a YYYY-MM-DD date"
        )));
    }
    Ok(())
}

/// The reference/fingerprint pair's shape, without the capture date (a retired entry has none).
fn check_reference_bits(
    key: &ConfigKey,
    reference: Option<&str>,
    fingerprint: Option<&str>,
) -> Result<(), NumericsError> {
    match (reference, fingerprint) {
        (None, None) => {}
        (Some(reference), Some(fingerprint)) => {
            if !reference.starts_with("references/")
                || !reference.ends_with(".json")
                || reference
                    .split('/')
                    .any(|part| part.is_empty() || part == "..")
            {
                return Err(NumericsError::Invalid(format!(
                    "{key}: reference {reference:?} is not a .json path under references/"
                )));
            }
            if fingerprint.len() != 64
                || !fingerprint
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(NumericsError::Invalid(format!(
                    "{key}: fingerprint {fingerprint:?} is not 64 lowercase hex digits"
                )));
            }
        }
        _ => {
            return Err(NumericsError::Invalid(format!(
                "{key}: reference and fingerprint are declared together, or both left out for an \
                 uncaptured configuration"
            )));
        }
    }
    Ok(())
}

/// Whether a string is a YYYY-MM-DD date: the shape both `captured` and an approval's `date` take.
fn is_date(text: &str) -> bool {
    let date = text.as_bytes();
    date.len() == 10
        && date[4] == b'-'
        && date[7] == b'-'
        && date
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
}

/// Refuse an approval that is not a usable record.
fn check_approval(key: &ConfigKey, approval: &Approval) -> Result<(), NumericsError> {
    if approval.decision.trim().is_empty() {
        return Err(NumericsError::Invalid(format!(
            "{key}: approved.decision is empty"
        )));
    }
    if approval.decision.contains('\n') {
        return Err(NumericsError::Invalid(format!(
            "{key}: approved.decision is not one line"
        )));
    }
    if !is_date(&approval.date) {
        return Err(NumericsError::Invalid(format!(
            "{key}: approved.date {:?} is not a YYYY-MM-DD date",
            approval.date
        )));
    }
    if !approval.report.starts_with("docs/numerics/") {
        return Err(NumericsError::Invalid(format!(
            "{key}: approved.report {:?} is not under docs/numerics/",
            approval.report
        )));
    }
    Ok(())
}

/// What a process claims about its arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericsStatus {
    /// A registered entry describes this process.
    Registered,
    /// No reference exists for this configuration.
    Unregistered,
    /// A switch that can change the arithmetic is set.
    Overridden,
}

impl NumericsStatus {
    /// The status as `/v1/models` spells it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Registered => "registered",
            Self::Unregistered => "unregistered",
            Self::Overridden => "overridden",
        }
    }
}

/// The numerics identity of one serving process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericsIdentity {
    /// The configuration this process serves.
    pub key: ConfigKey,
    /// The registered entry for it, if there is one.
    pub entry: Option<ConfigEntry>,
    /// Each arithmetic switch set at boot, as `NAME=value`.
    pub overrides: Vec<String>,
}

impl NumericsIdentity {
    /// Resolve a configuration against a registry and an environment.
    #[must_use]
    pub fn resolve(
        registry: &Registry,
        key: ConfigKey,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Self {
        let overrides = arithmetic_switches()
            .filter_map(|name| {
                lookup(name)
                    .filter(|value| withdraws_claim(name, value))
                    .map(|value| format!("{name}={value}"))
            })
            .collect();
        Self {
            entry: registry.lookup(&key).cloned(),
            key,
            overrides,
        }
    }

    /// Resolve a configuration against the embedded registry and the process environment.
    ///
    /// # Errors
    /// The embedded registry does not parse, which the workspace tests rule out.
    pub fn resolve_process(key: ConfigKey) -> Result<Self, NumericsError> {
        Ok(Self::resolve(&Registry::embedded()?, key, |name| {
            std::env::var(name).ok()
        }))
    }

    /// What this process claims.
    #[must_use]
    pub fn status(&self) -> NumericsStatus {
        if !self.overrides.is_empty() {
            NumericsStatus::Overridden
        } else if self.entry.as_ref().is_some_and(ConfigEntry::is_captured) {
            NumericsStatus::Registered
        } else {
            NumericsStatus::Unregistered
        }
    }

    /// The version a registered process serves; `None` when it cannot claim one.
    #[must_use]
    pub fn version(&self) -> Option<u32> {
        match self.status() {
            NumericsStatus::Registered => self.entry.as_ref().map(|entry| entry.version),
            NumericsStatus::Unregistered | NumericsStatus::Overridden => None,
        }
    }

    /// The one boot line that says what this process serves.
    #[must_use]
    pub fn boot_line(&self) -> String {
        let mut line = format!(
            "sparkflash numerics: {} status={}",
            self.key,
            self.status().name()
        );
        if let Some(entry) = &self.entry {
            line.push_str(&format!(" registered_version={}", entry.version));
            if let Some(fingerprint) = &entry.fingerprint {
                line.push_str(&format!(
                    " fingerprint={} evidence={}",
                    &fingerprint[..16],
                    evidence_name(entry.evidence)
                ));
            }
            // Appended: the existing fields keep their order, and a reader sees that this version
            // landed under an owner's approval and when it was taken.
            if let Some(approved) = &entry.approved {
                line.push_str(&format!(" approved={}", approved.date));
            }
        }
        if !self.overrides.is_empty() {
            line.push_str(&format!(" overrides={}", self.overrides.join(",")));
        }
        line
    }

    /// The `numerics` member of the `/v1/models` card.
    ///
    /// `version` is null unless the process is registered: an overridden process does not serve
    /// the registered version's arithmetic, and the card must not say it does.
    #[must_use]
    pub fn card(&self) -> Value {
        let entry = self.entry.as_ref();
        json!({
            "status": self.status().name(),
            "profile": self.key.profile.name(),
            "version": self.version(),
            "registered_version": entry.map(|entry| entry.version),
            "fingerprint": entry.and_then(|entry| entry.fingerprint.as_deref()),
            "evidence": entry.map(|entry| evidence_name(entry.evidence)),
            "approved": entry.and_then(|entry| entry.approved.as_ref()),
            "world": self.key.world,
            "plan": self.key.plan,
            "arch": self.key.arch,
            "overrides": self.overrides,
            "policy": "docs/NUMERICS.md",
        })
    }
}

fn evidence_name(evidence: Evidence) -> &'static str {
    match evidence {
        Evidence::Logprobs => "logprobs",
        Evidence::Text => "text",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: &str = "nvidia/Qwen3.8-Flash-Next-NVFP4";
    const GLM: &str = "zai-org/GLM-5.3-Flash";
    const INKLING: &str = "thinkingmachines/Inkling-2-Small";
    const FINGERPRINT: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const COVERAGE_ROW: &str = concat!(
        "[[coverage]]\n",
        "model = \"nvidia/Qwen3.8-Flash-Next-NVFP4\"\n",
        "dump = \"numerics/coverage/q.json\"\n",
        "min_inputs = 8\n",
    );
    const MONITORS_ROW: &str = concat!(
        "[[monitors]]\n",
        "model = \"zai-org/GLM-5.3-Flash\"\n",
        "reference = \"numerics/monitors/r.json\"\n",
        "candidate = \"numerics/monitors/c.json\"\n",
        "layer_bar = 0.5\nexpert_bar = 0.5\n",
    );

    fn registry_with(entries: &str) -> String {
        format!(
            "schema = 1\n[prerequisites]\nanchor = false\ndeclaration_check = true\n\
             coverage_slots = 0.93\ncoverage = false\nmonitors = false\n{entries}"
        )
    }

    fn entry(version: u32, extra: &str) -> String {
        format!(
            "[[config]]\nmodel = \"{QWEN}\"\nworld = 4\nprofile = \"reference\"\nplan = \"tp\"\n\
             arch = \"sm_121a\"\nversion = {version}\nevidence = \"logprobs\"\n\
             reference = \"references/qwen/tp4-v{version}.json\"\n\
             fingerprint = \"{FINGERPRINT}\"\ncaptured = \"2026-11-03\"\n{extra}"
        )
    }

    #[test]
    fn the_embedded_registry_parses() {
        let registry = Registry::embedded().expect("the embedded registry parses and validates");
        assert_eq!(registry.schema, SCHEMA);
    }

    #[test]
    fn a_registered_configuration_reports_its_version() {
        let registry = Registry::from_toml(&registry_with(&entry(1, ""))).unwrap();
        let key = ConfigKey::tp(QWEN, 4, NumericsProfile::Reference, "sm_121a");
        let identity = NumericsIdentity::resolve(&registry, key, |_| None);
        assert_eq!(identity.status(), NumericsStatus::Registered);
        assert_eq!(identity.version(), Some(1));
        let card = identity.card();
        assert_eq!(card["status"], "registered");
        assert_eq!(card["version"], 1);
        assert_eq!(card["fingerprint"], FINGERPRINT);
        assert_eq!(card["approved"], Value::Null);
        assert!(identity.boot_line().contains("status=registered"));
        assert!(!identity.boot_line().contains("approved="));
    }

    #[test]
    fn a_recorded_approval_parses_and_is_reported() {
        let approved = "approved = { decision = \"The owner's recorded decision\", \
                        date = \"2026-11-18\", report = \"docs/numerics/qwen/v2-approval.md\" }\n";
        let registry = Registry::from_toml(&registry_with(&entry(
            2,
            &format!("report = \"docs/numerics/qwen/v2.md\"\n{approved}"),
        )))
        .expect("an entry may carry its owner's approval");
        let key = ConfigKey::tp(QWEN, 4, NumericsProfile::Reference, "sm_121a");
        let identity = NumericsIdentity::resolve(&registry, key, |_| None);
        let card = identity.card();
        assert_eq!(card["approved"]["date"], "2026-11-18");
        assert_eq!(
            card["approved"]["decision"],
            "The owner's recorded decision"
        );
        assert_eq!(
            card["approved"]["report"],
            "docs/numerics/qwen/v2-approval.md"
        );
        assert!(
            identity.boot_line().ends_with(" approved=2026-11-18"),
            "{}",
            identity.boot_line()
        );
        assert!(identity.boot_line().contains("fingerprint="));
    }

    #[test]
    fn a_malformed_approval_is_refused() {
        let key = ConfigKey::tp(QWEN, 4, NumericsProfile::Reference, "sm_121a");
        let refusal = |decision: &str, date: &str, report: &str| {
            format!(
                "report = \"docs/numerics/qwen/v2.md\"\n\
                 approved = {{ decision = {decision:?}, date = {date:?}, report = {report:?} }}\n"
            )
        };
        let cases = [
            (
                refusal("  ", "2026-11-18", "docs/numerics/qwen/a.md"),
                "approved.decision is empty",
            ),
            (
                refusal("a\nb", "2026-11-18", "docs/numerics/qwen/a.md"),
                "one line",
            ),
            (
                refusal("a decision", "18-11-2026", "docs/numerics/qwen/a.md"),
                "YYYY-MM-DD",
            ),
            (
                refusal("a decision", "2026-11-18", "numerics/a.md"),
                "under docs/numerics/",
            ),
        ];
        for (extra, needle) in cases {
            let error = Registry::from_toml(&registry_with(&entry(2, &extra)))
                .unwrap_err()
                .to_string();
            assert!(error.contains(needle), "{needle}: {error}");
            assert!(error.contains(&key.to_string()), "{error}");
        }
    }

    #[test]
    fn an_unregistered_configuration_claims_no_version() {
        let registry = Registry::from_toml(&registry_with(&entry(1, ""))).unwrap();
        let key = ConfigKey::tp(QWEN, 2, NumericsProfile::Reference, "sm_121a");
        let identity = NumericsIdentity::resolve(&registry, key, |_| None);
        assert_eq!(identity.status(), NumericsStatus::Unregistered);
        assert_eq!(identity.version(), None);
        assert_eq!(identity.card()["fingerprint"], Value::Null);
    }

    #[test]
    fn an_arithmetic_override_withdraws_the_claim() {
        let registry = Registry::from_toml(&registry_with(&entry(1, ""))).unwrap();
        let key = ConfigKey::tp(QWEN, 4, NumericsProfile::Reference, "sm_121a");
        let identity = NumericsIdentity::resolve(&registry, key, |name| {
            (name == "SPARKFLASH_QWEN4_PREFILL_MMA").then(|| "0".to_owned())
        });
        assert_eq!(identity.status(), NumericsStatus::Overridden);
        assert_eq!(identity.version(), None);
        let card = identity.card();
        assert_eq!(card["version"], Value::Null);
        assert_eq!(card["registered_version"], 1);
        assert_eq!(card["overrides"][0], "SPARKFLASH_QWEN4_PREFILL_MMA=0");
    }

    #[test]
    fn a_direct_switch_spelling_its_default_keeps_the_claim() {
        let registry = Registry::from_toml(&registry_with(&entry(1, ""))).unwrap();
        let key = ConfigKey::tp(QWEN, 4, NumericsProfile::Reference, "sm_121a");
        let with = |value: &'static str| {
            NumericsIdentity::resolve(&registry, key.clone(), move |name| {
                (name == "SPARK_COMM_REDUCE").then(|| value.to_owned())
            })
        };
        assert_eq!(with("nccl").status(), NumericsStatus::Registered);
        assert_eq!(with("nccl").version(), Some(1));
        let tcp = with("tcp");
        assert_eq!(tcp.status(), NumericsStatus::Overridden);
        assert_eq!(tcp.card()["overrides"][0], "SPARK_COMM_REDUCE=tcp");
        // A profile override's default is the model's, so any value withdraws the claim.
        assert!(withdraws_claim("SPARKFLASH_QWEN4_PREFILL_MMA", "1"));
    }

    #[test]
    fn every_profile_override_is_classified_once() {
        for (name, field) in crate::profile::ENV_OVERRIDES {
            let arithmetic = ARITHMETIC_FIELDS.contains(field);
            let neutral = NEUTRAL_FIELDS.contains(field);
            assert!(
                arithmetic != neutral,
                "{name} sets {field}, which must be in exactly one of ARITHMETIC_FIELDS and \
                 NEUTRAL_FIELDS"
            );
        }
        for (name, _, _) in ARITHMETIC_ENV {
            assert!(
                !crate::profile::ENV_OVERRIDES
                    .iter()
                    .any(|(owned, _)| owned == name),
                "{name} has a typed field now; classify the field instead"
            );
        }
        assert!(arithmetic_switches().any(|name| name == "SPARKFLASH_QWEN4_PREFILL_MMA"));
        assert!(!arithmetic_switches().any(|name| name == "SPARKFLASH_LANES"));
    }

    #[test]
    fn an_uncaptured_configuration_parses_and_claims_no_version() {
        let uncaptured = "[[config]]\nmodel = \"nvidia/Qwen3.8-Flash-Next-NVFP4\"\nworld = 4\n\
             profile = \"reference\"\nplan = \"tp\"\narch = \"sm_121a\"\nversion = 1\n\
             evidence = \"logprobs\"\ncaptured = \"2026-11-03\"\n\
             note = \"registered, not yet captured\"\n";
        let registry =
            Registry::from_toml(&registry_with(uncaptured)).expect("no capture is legal");
        let key = ConfigKey::tp(QWEN, 4, NumericsProfile::Reference, "sm_121a");
        let identity = NumericsIdentity::resolve(&registry, key, |_| None);
        assert_eq!(identity.status(), NumericsStatus::Unregistered);
        assert_eq!(identity.version(), None);
        assert_eq!(identity.entry.as_ref().map(|entry| entry.version), Some(1));
        assert_eq!(identity.card()["fingerprint"], Value::Null);
        assert!(!identity.boot_line().contains("fingerprint="));

        // Both present is a capture; one present without the other is refused.
        Registry::from_toml(&registry_with(&entry(1, ""))).expect("a capture is legal");
        for lone in [
            uncaptured.to_owned() + "reference = \"references/q.json\"\n",
            uncaptured.to_owned()
                + "fingerprint = \"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"\n",
        ] {
            let error = Registry::from_toml(&registry_with(&lone))
                .expect_err("one half of a capture is refused")
                .to_string();
            assert!(error.contains("reference and fingerprint"), "{error}");
        }
    }

    #[test]
    fn unknown_profiles_are_refused() {
        assert_eq!(
            NumericsProfile::parse("reference").unwrap(),
            NumericsProfile::Reference
        );
        let error = NumericsProfile::parse("fast").unwrap_err().to_string();
        assert!(error.contains("reference"), "{error}");
    }

    #[test]
    fn a_change_without_a_report_is_refused() {
        let error = Registry::from_toml(&registry_with(&entry(2, "")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("names no report"), "{error}");
        Registry::from_toml(&registry_with(&entry(
            2,
            "report = \"docs/numerics/qwen/v2.md\"\n",
        )))
        .unwrap();
    }

    #[test]
    fn malformed_entries_are_refused() {
        let cases = [
            (entry(1, "").replace("tp\"", "ep\""), "plan"),
            (entry(1, "").replace(FINGERPRINT, "ABC"), "fingerprint"),
            (
                entry(1, "").replace("references/qwen", "../qwen"),
                "reference",
            ),
            (entry(1, "").replace(QWEN, "acme/unknown"), "not a model id"),
            (entry(1, "").replace("2026-11-03", "yesterday"), "captured"),
            (format!("{}{}", entry(1, ""), entry(1, "")), "more than one"),
            (entry(1, "unknown_key = 1\n"), "unknown"),
        ];
        for (text, needle) in cases {
            let error = Registry::from_toml(&registry_with(&text))
                .unwrap_err()
                .to_string();
            assert!(error.contains(needle), "{needle}: {error}");
        }
    }

    #[test]
    fn a_retired_configuration_may_name_the_capture_it_kept() {
        let retired = format!(
            "[[retired]]\nmodel = \"{QWEN}\"\nworld = 4\nprofile = \"reference\"\n\
             plan = \"tp\"\narch = \"sm_121a\"\nversion = 2\n\
             reference = \"references/qwen38-tp4-sm121a.json\"\n\
             fingerprint = \"{FINGERPRINT}\"\nreason = \"test\"\n"
        );
        Registry::from_toml(&registry_with(&retired)).unwrap();
        let without = retired.replace(&format!("fingerprint = \"{FINGERPRINT}\"\n"), "");
        let error = Registry::from_toml(&registry_with(&without))
            .unwrap_err()
            .to_string();
        assert!(error.contains("fingerprint"), "{error}");
    }

    #[test]
    fn a_re_added_configuration_must_exceed_its_retired_version() {
        let retired = "[[retired]]\nmodel = \"nvidia/Qwen3.8-Flash-Next-NVFP4\"\nworld = 4\n\
                       profile = \"reference\"\nplan = \"tp\"\narch = \"sm_121a\"\nversion = 2\n\
                       reason = \"test\"\n";
        let error = Registry::from_toml(&registry_with(&format!(
            "{}{retired}",
            entry(2, "report = \"docs/numerics/qwen/v2.md\"\n")
        )))
        .unwrap_err()
        .to_string();
        assert!(error.contains("retired"), "{error}");
    }

    #[test]
    fn a_coverage_bar_outside_the_unit_interval_is_refused() {
        let text =
            registry_with(&entry(1, "")).replace("coverage_slots = 0.93", "coverage_slots = 1.5");
        let error = Registry::from_toml(&text).unwrap_err().to_string();
        assert!(error.contains("coverage_slots"), "{error}");
    }

    #[test]
    fn prerequisites_name_what_is_missing() {
        // The aggregate flags are reports of the per-model rows; derived from the registry rather
        // than spelled out, so turning one on is one reviewed edit and not two.
        let registry = Registry::embedded().unwrap();
        let expected: Vec<&str> = [
            ("anchor", registry.prerequisites.anchor),
            (
                "declaration_check",
                registry.prerequisites.declaration_check,
            ),
            ("coverage", registry.prerequisites.coverage),
            ("monitors", registry.prerequisites.monitors),
        ]
        .iter()
        .filter(|(_, on)| !on)
        .map(|(name, _)| *name)
        .collect();
        assert_eq!(registry.prerequisites.missing(), expected);
    }

    #[test]
    fn the_per_model_rows_answer_for_their_model() {
        let registry = Registry::embedded().unwrap();
        // Qwen3.8 and GLM-5.3 have coverage rows (Qwen's world-1 sweep, GLM's world-4 sweep);
        // only GLM has a monitors row, because its pair is the first judged one. Inkling-2 and
        // Dory1.0 have neither, which is why both aggregate flags are false and an unapproved
        // bump on them is refused today.
        assert!(registry.coverage_for(QWEN).is_some());
        assert!(registry.coverage_for(GLM).is_some());
        assert!(registry.monitors_for(QWEN).is_none());
        assert!(registry.monitors_for(GLM).is_some());
        assert_eq!(registry.coverage_for(QWEN).unwrap().min_inputs, 8);
        assert_eq!(registry.missing_for(QWEN), ["monitors"]);
        assert!(
            registry.missing_for(GLM).is_empty(),
            "GLM has both rows now"
        );
        assert_eq!(
            registry.missing_for(INKLING),
            ["coverage", "monitors"].as_slice()
        );
        assert_eq!(
            registry.missing_for("acme/unknown"),
            ["anchor", "coverage", "monitors"]
        );
    }

    #[test]
    fn the_per_model_tables_parse() {
        let text = registry_with(&format!("{COVERAGE_ROW}{MONITORS_ROW}"));
        let registry = Registry::from_toml(&text).expect("well-formed per-model rows parse");
        assert_eq!(registry.coverage.len(), 1);
        assert_eq!(registry.coverage[0].dump, "numerics/coverage/q.json");
        assert_eq!(registry.coverage[0].min_inputs, 8);
        assert_eq!(registry.monitors.len(), 1);
        assert_eq!(registry.monitors[0].model, GLM);
        assert_eq!(registry.monitors[0].layer_bar, 0.5);
    }

    #[test]
    fn the_aggregate_flags_are_reports_not_rows() {
        // A true flag makes a claim about every captured logprobs model; `scripts/numerics.py ci`
        // judges it against the rows and their dumps. The parser only reads the rows it is given.
        let text = registry_with(COVERAGE_ROW)
            .replace("coverage = false", "coverage = true")
            .replace("monitors = false", "monitors = true");
        let registry = Registry::from_toml(&text).expect("the aggregate flags are reports");
        assert!(registry.prerequisites.coverage);
        assert!(registry.prerequisites.monitors);
    }

    #[test]
    fn a_malformed_coverage_row_is_refused() {
        Registry::from_toml(&registry_with(COVERAGE_ROW)).expect("a well-formed row parses");
        for (bad, needle) in [
            (
                COVERAGE_ROW.replace("numerics/coverage/q.json", "docs/numerics/q.json"),
                "coverage",
            ),
            (
                COVERAGE_ROW.replace("numerics/coverage/q.json", "numerics/coverage/../q.json"),
                "coverage",
            ),
            (
                COVERAGE_ROW.replace("numerics/coverage/q.json", "numerics/coverage/q.txt"),
                "coverage",
            ),
            (
                COVERAGE_ROW.replace("min_inputs = 8", "min_inputs = 0"),
                "coverage",
            ),
            (
                COVERAGE_ROW.replace("min_inputs = 8", "min_inputs = -1"),
                "min_inputs",
            ),
            (COVERAGE_ROW.replace("min_inputs = 8\n", ""), "min_inputs"),
            (
                COVERAGE_ROW.replace("model = \"nvidia/Qwen3.8-Flash-Next-NVFP4\"\n", ""),
                "model",
            ),
            (
                COVERAGE_ROW.replace(
                    "model = \"nvidia/Qwen3.8-Flash-Next-NVFP4\"",
                    "model = \"nvidia/Qwen3.8-Flash-Next-NVFP4\"\nunknown_key = 1",
                ),
                "unknown",
            ),
            (COVERAGE_ROW.replace(QWEN, "acme/unknown"), "not a model id"),
            (format!("{COVERAGE_ROW}{COVERAGE_ROW}"), "more than one"),
        ] {
            let error = Registry::from_toml(&registry_with(&bad))
                .unwrap_err()
                .to_string();
            assert!(error.contains(needle), "{needle}: {error}");
        }
    }

    #[test]
    fn a_malformed_monitors_row_is_refused() {
        Registry::from_toml(&registry_with(MONITORS_ROW)).expect("a well-formed row parses");
        for (bad, needle) in [
            (
                MONITORS_ROW.replace("numerics/monitors/r.json", "docs/numerics/r.json"),
                "monitors",
            ),
            (
                MONITORS_ROW.replace("numerics/monitors/r.json", "numerics/monitors/../r.json"),
                "monitors",
            ),
            (
                MONITORS_ROW.replace("numerics/monitors/c.json", "numerics/monitors/c.txt"),
                "monitors",
            ),
            (
                MONITORS_ROW.replace("layer_bar = 0.5", "layer_bar = 0.0"),
                "monitors",
            ),
            (
                MONITORS_ROW.replace("expert_bar = 0.5", "expert_bar = -1.0"),
                "monitors",
            ),
            (
                MONITORS_ROW.replace("expert_bar = 0.5", "expert_bar = nan"),
                "monitors",
            ),
            (
                MONITORS_ROW.replace("model = \"zai-org/GLM-5.3-Flash\"\n", ""),
                "model",
            ),
            (
                MONITORS_ROW.replace("layer_bar = 0.5", "surprise = 1"),
                "unknown",
            ),
            (MONITORS_ROW.replace(GLM, "acme/unknown"), "not a model id"),
            (format!("{MONITORS_ROW}{MONITORS_ROW}"), "more than one"),
        ] {
            let error = Registry::from_toml(&registry_with(&bad))
                .unwrap_err()
                .to_string();
            assert!(error.contains(needle), "{needle}: {error}");
        }
    }
}

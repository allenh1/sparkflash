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
/// typed field lands (docs/adding-a-model.md); each moves to [ARITHMETIC_FIELDS] when it does.
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

/// The four prerequisites docs/NUMERICS.md §5 names before any arithmetic change may land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prerequisites {
    /// The frozen anchors and the teacher-forced comparison.
    pub anchor: bool,
    /// The CI declaration check.
    pub declaration_check: bool,
    /// Coverage inputs that reach every routed expert.
    pub coverage: bool,
    /// Per-layer and per-expert error monitors.
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
    /// The reference file, relative to `numerics/`.
    pub reference: String,
    /// The sha256 of the reference's canonical capture.
    pub fingerprint: String,
    /// The quality report a declared change committed; required from version 2 on.
    #[serde(default)]
    pub report: Option<String>,
    /// The date the reference was captured (YYYY-MM-DD).
    pub captured: String,
    /// What a reader should know about this entry.
    #[serde(default)]
    pub note: Option<String>,
}

impl ConfigEntry {
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    /// The schema version, [SCHEMA].
    pub schema: u32,
    /// The prerequisites for declaring a change.
    pub prerequisites: Prerequisites,
    /// The frozen anchors.
    #[serde(default)]
    pub anchor: Vec<AnchorEntry>,
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
        let mut seen = BTreeSet::new();
        for entry in &self.config {
            let key = entry.key();
            check_key(&key)?;
            if entry.version == 0 {
                return Err(NumericsError::Invalid(format!(
                    "{key}: version 0; versions start at 1"
                )));
            }
            check_reference(&key, &entry.reference, &entry.fingerprint, &entry.captured)?;
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
                &anchor.reference,
                &anchor.fingerprint,
                &anchor.captured,
            )?;
            if !anchored.insert(anchor.model.clone()) {
                return Err(NumericsError::Invalid(format!(
                    "{} has more than one anchor; an anchor is frozen, never replaced",
                    anchor.model
                )));
            }
        }
        for retired in &self.retired {
            check_key(&retired.key())?;
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
}

fn check_key(key: &ConfigKey) -> Result<(), NumericsError> {
    if !crate::descriptor::MODELS
        .iter()
        .any(|model| model.id == key.model)
    {
        return Err(NumericsError::Invalid(format!(
            "{key}: {:?} is not a model id this build serves",
            key.model
        )));
    }
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
    reference: &str,
    fingerprint: &str,
    captured: &str,
) -> Result<(), NumericsError> {
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
    let date = captured.as_bytes();
    let well_formed = date.len() == 10
        && date[4] == b'-'
        && date[7] == b'-'
        && date
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit());
    if !well_formed {
        return Err(NumericsError::Invalid(format!(
            "{key}: captured {captured:?} is not a YYYY-MM-DD date"
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
        } else if self.entry.is_some() {
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
            line.push_str(&format!(
                " registered_version={} fingerprint={} evidence={}",
                entry.version,
                &entry.fingerprint[..16],
                evidence_name(entry.evidence)
            ));
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
            "fingerprint": entry.map(|entry| entry.fingerprint.as_str()),
            "evidence": entry.map(|entry| evidence_name(entry.evidence)),
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
    const FINGERPRINT: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn registry_with(entries: &str) -> String {
        format!(
            "schema = 1\n[prerequisites]\nanchor = false\ndeclaration_check = true\n\
             coverage = false\nmonitors = false\n{entries}"
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
        assert!(identity.boot_line().contains("status=registered"));
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
    fn prerequisites_name_what_is_missing() {
        let registry = Registry::embedded().unwrap();
        let missing = registry.prerequisites.missing();
        assert_eq!(
            missing.is_empty(),
            registry.prerequisites
                == Prerequisites {
                    anchor: true,
                    declaration_check: true,
                    coverage: true,
                    monitors: true,
                }
        );
    }
}

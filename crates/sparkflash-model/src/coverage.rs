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

//! Which keys of a checkpoint's config.json the description reader consumed.
//!
//! The plan's section 4 requires every config key to be read or listed as ignorable
//! with a written reason. This module implements that check at the one place every read
//! passes through: the config access helpers in crate::architecture. A read is recorded
//! with its full path from the document root (text_config.hidden_size,
//! quantization_config.expert_weight.quant_algo), so a key under one stack cannot hide
//! an unread key of the same name under another.
//!
//! The recorder is thread-local. Architecture::from_value_with_coverage installs a fresh
//! one for the duration of a parse; the existing from_value/from_json call the same
//! parser and discard the report, so their results are unchanged.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

use serde_json::Value;

/// The path separator: member names from the document root, joined by a dot.
const SEPARATOR: char = '.';

/// One key present in the document that no read consumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownKey {
    /// The full path from the document root.
    pub path: String,
    /// What would fix the gap.
    pub fix: String,
}

impl UnknownKey {
    /// The fix this report recommends.
    #[must_use]
    pub fn fix(path: &str) -> String {
        format!(
            "state {path} in the checkpoint's layout keys entry with the fact it sets, or add \
             it to that layout's ignorable list with a written reason"
        )
    }
}

/// One key present in the document whose name is on the ignorable list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgnoredKey {
    /// The full path from the document root.
    pub path: String,
    /// Why the key can be ignored.
    pub reason: String,
}

/// The result of the config-key coverage check for one parse.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfigCoverage {
    read: BTreeSet<String>,
    unknown: Vec<UnknownKey>,
    ignored: Vec<IgnoredKey>,
    checked: bool,
}

/// Whether a config-key check ran and what it found.
///
/// This mirrors the scan report's CoverageState without making this crate depend on the
/// op library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyCoverageState {
    /// The check ran and every key was read or ignorably listed.
    Clean,
    /// The check ran and found keys to report.
    Dirty,
    /// The check has not run.
    NotChecked,
}

impl KeyCoverageState {
    /// The stable name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Dirty => "dirty",
            Self::NotChecked => "not checked",
        }
    }

    /// Whether the check ran and found nothing to report.
    #[must_use]
    pub const fn is_clean(self) -> bool {
        matches!(self, Self::Clean)
    }
}

impl ConfigCoverage {
    /// A coverage result for a parse whose check has not run.
    #[must_use]
    pub fn not_checked() -> Self {
        Self::default()
    }

    /// The keys that were read, in stable path order.
    pub fn read_keys(&self) -> impl Iterator<Item = &str> {
        self.read.iter().map(String::as_str)
    }

    /// The keys present in the document that no read consumed.
    #[must_use]
    pub fn unknown_keys(&self) -> &[UnknownKey] {
        &self.unknown
    }

    /// The keys present in the document that the ignorable list covers.
    #[must_use]
    pub fn ignored_keys(&self) -> &[IgnoredKey] {
        &self.ignored
    }

    /// Mark a path as read by a source other than the structural parser, such as a layout.
    pub(crate) fn claim_path(&mut self, path: &str) {
        self.read.insert(path.to_owned());
        self.unknown.retain(|unknown| unknown.path != path);
    }

    /// Mark a path as read and ignorable with a reason from the matching layout.
    pub(crate) fn ignore_path(&mut self, path: &str, reason: &str) {
        self.claim_path(path);
        self.ignored.push(IgnoredKey {
            path: path.to_owned(),
            reason: reason.to_owned(),
        });
    }

    /// Whether the check ran and every key was read or ignorably listed.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.checked && self.unknown.is_empty()
    }

    /// The state the scan report records for its config-key coverage.
    #[must_use]
    pub fn state(&self) -> KeyCoverageState {
        if !self.checked {
            KeyCoverageState::NotChecked
        } else if self.is_clean() {
            KeyCoverageState::Clean
        } else {
            KeyCoverageState::Dirty
        }
    }
}

/// The ignorable keys, each with the reason it carries no arithmetic.
///
/// The list is keyed by the key's own name, never by a model or a path. A key that
/// changes what the engine runs is never here; it is either read or reported.
const IGNORABLE: &[(&str, &str)] = &[
    (
        "transformers_version",
        "the Transformers release that serialized the file; it describes the writer, not the model",
    ),
    (
        "torch_dtype",
        "a producer dtype hint for newly allocated tensors; stored formats come from the quantization block and the tensor index",
    ),
    (
        "dtype",
        "a producer dtype hint; the stored weight format is read from the quantization block, not this field",
    ),
    (
        "initializer_range",
        "the training-time weight initialization range; no inference value is derived from it",
    ),
    (
        "attention_dropout",
        "the training-time attention dropout probability; inference runs dropout off",
    ),
    (
        "router_aux_loss_coef",
        "the training-time auxiliary load-balancing loss coefficient; inference computes no loss",
    ),
    (
        "output_router_logits",
        "a debug output switch for the producer's forward; serving returns logits, not router logits",
    ),
    (
        "use_cache",
        "a producer output switch for returning the KV cache; serving always keeps the cache",
    ),
    (
        "_name_or_path",
        "the local path the producer last saved from; it names no model fact",
    ),
    (
        "_commit_hash",
        "the producer repository commit that last saved the file; it names no model fact",
    ),
    (
        "library_name",
        "the producer library that wrote the file; the arithmetic is read from its typed fields",
    ),
    (
        "producer",
        "the ModelOpt producer block that names the tool and version; the arithmetic is read from quant_algo and quantized_layers",
    ),
];

/// The ignorable reason for a key name, if it has one.
#[must_use]
pub fn ignorable_reason(key: &str) -> Option<&'static str> {
    IGNORABLE
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, reason)| *reason)
}

fn child_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_owned()
    } else {
        format!("{parent}{SEPARATOR}{name}")
    }
}

/// A running parse's key bookkeeping.
#[derive(Debug)]
struct Recorder {
    paths: HashMap<usize, String>,
    reads: BTreeSet<String>,
}

impl Recorder {
    fn new(root: &Value) -> Self {
        let mut paths = HashMap::new();
        paths.insert(address(root), String::new());
        Self {
            paths,
            reads: BTreeSet::new(),
        }
    }

    fn path_of(&self, value: &Value) -> Option<String> {
        self.paths.get(&address(value)).cloned()
    }

    fn read(&mut self, value: &Value, name: &str, child: Option<&Value>) {
        let Some(parent) = self.path_of(value) else {
            // A value outside the recorded document cannot be given a truthful path.
            return;
        };
        let path = child_path(&parent, name);
        self.reads.insert(path.clone());
        if let Some(child) = child {
            self.paths.insert(address(child), path);
        }
    }

    fn read_entries(&mut self, value: &Value) {
        let Some(parent) = self.path_of(value) else {
            return;
        };
        let Some(map) = value.as_object() else {
            return;
        };
        for (name, child) in map {
            let path = child_path(&parent, name);
            self.reads.insert(path.clone());
            self.paths.insert(address(child), path);
        }
    }

    fn finish(self, root: &Value) -> ConfigCoverage {
        let mut unknown = Vec::new();
        let mut ignored = Vec::new();
        collect(root, "", &self.reads, &mut unknown, &mut ignored);
        ConfigCoverage {
            read: self.reads,
            unknown,
            ignored,
            checked: true,
        }
    }
}

fn address(value: &Value) -> usize {
    std::ptr::from_ref(value) as usize
}

fn collect(
    value: &Value,
    path: &str,
    reads: &BTreeSet<String>,
    unknown: &mut Vec<UnknownKey>,
    ignored: &mut Vec<IgnoredKey>,
) {
    match value {
        Value::Object(map) => {
            for (name, child) in map {
                let full = child_path(path, name);
                if !reads.contains(&full) {
                    if let Some(reason) = ignorable_reason(name) {
                        // An ignorable key's reason covers its value too: a producer metadata
                        // block is ignored whole, not one member at a time.
                        ignored.push(IgnoredKey {
                            path: full.clone(),
                            reason: reason.to_owned(),
                        });
                        continue;
                    }
                    unknown.push(UnknownKey {
                        fix: UnknownKey::fix(&full),
                        path: full.clone(),
                    });
                }
                collect(child, &full, reads, unknown, ignored);
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                collect(child, &format!("{path}[{index}]"), reads, unknown, ignored);
            }
        }
        _ => {}
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

/// Install a fresh recorder for root, run parse, and return its result with the coverage.
pub(crate) fn with_recording<R>(root: &Value, parse: impl FnOnce() -> R) -> (R, ConfigCoverage) {
    struct Restore(Option<Recorder>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            ACTIVE.with(|slot| *slot.borrow_mut() = previous);
        }
    }

    let previous = ACTIVE.with(|slot| slot.borrow_mut().replace(Recorder::new(root)));
    let _restore = Restore(previous);
    let result = parse();
    let recorder = ACTIVE.with(|slot| slot.borrow_mut().take());
    let coverage = recorder.map_or_else(ConfigCoverage::not_checked, |recorder| {
        recorder.finish(root)
    });
    (result, coverage)
}

/// Read name from value, recording the full key path when a parse is being recorded.
pub(crate) fn read_key<'a>(value: &'a Value, name: &str) -> Option<&'a Value> {
    let child = value.get(name);
    ACTIVE.with(|slot| {
        if let Some(recorder) = slot.borrow_mut().as_mut() {
            recorder.read(value, name, child);
        }
    });
    child
}

/// Every member of an object the parser iterates, recorded as read.
pub(crate) fn entries(value: &Value) -> Vec<(&str, &Value)> {
    let pairs: Vec<(&str, &Value)> = match value.as_object() {
        Some(map) => map
            .iter()
            .map(|(name, child)| (name.as_str(), child))
            .collect(),
        None => Vec::new(),
    };
    ACTIVE.with(|slot| {
        if let Some(recorder) = slot.borrow_mut().as_mut() {
            recorder.read_entries(value);
        }
    });
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_is_recorded_by_full_path() {
        let root: Value = serde_json::json!({
            "text_config": {"hidden_size": 8, "unread": 1},
            "vision_config": {"patch_size": 14}
        });
        let ((), coverage) = with_recording(&root, || {
            let text = read_key(&root, "text_config").expect("text_config");
            let vision = read_key(&root, "vision_config").expect("vision_config");
            read_key(text, "hidden_size").expect("hidden_size");
            read_key(vision, "patch_size").expect("patch_size");
        });
        assert!(
            coverage
                .read_keys()
                .any(|key| key == "text_config.hidden_size")
        );
        assert!(
            coverage
                .read_keys()
                .any(|key| key == "vision_config.patch_size")
        );
        assert_eq!(
            coverage
                .unknown_keys()
                .iter()
                .map(|unknown| unknown.path.as_str())
                .collect::<Vec<_>>(),
            ["text_config.unread"]
        );
        assert_eq!(coverage.state(), KeyCoverageState::Dirty);
    }

    #[test]
    fn an_ignorable_key_has_a_written_reason_and_is_not_unknown() {
        let root: Value = serde_json::json!({"transformers_version": "4.57.0"});
        let ((), coverage) = with_recording(&root, || {});
        assert!(coverage.unknown_keys().is_empty());
        assert_eq!(coverage.ignored_keys().len(), 1);
        assert_eq!(coverage.ignored_keys()[0].path, "transformers_version");
        assert!(!coverage.ignored_keys()[0].reason.is_empty());
        assert_eq!(coverage.state(), KeyCoverageState::Clean);
    }

    #[test]
    fn a_key_read_under_one_path_does_not_hide_another_path() {
        let root: Value = serde_json::json!({
            "text_config": {"hidden_size": 8},
            "vision_config": {"hidden_size": 16}
        });
        let ((), coverage) = with_recording(&root, || {
            let text = read_key(&root, "text_config").expect("text_config");
            read_key(&root, "vision_config").expect("vision_config");
            read_key(text, "hidden_size").expect("hidden_size");
        });
        assert!(
            coverage
                .read_keys()
                .any(|key| key == "text_config.hidden_size")
        );
        let unknown: Vec<&str> = coverage
            .unknown_keys()
            .iter()
            .map(|unknown| unknown.path.as_str())
            .collect();
        assert_eq!(unknown, ["vision_config.hidden_size"]);
    }

    #[test]
    fn array_elements_are_walked_with_their_index() {
        let root: Value = serde_json::json!({"groups": [{"a": 1}, {"b": 2}]});
        let ((), coverage) = with_recording(&root, || {
            read_key(&root, "groups").expect("groups");
        });
        let unknown: Vec<&str> = coverage
            .unknown_keys()
            .iter()
            .map(|unknown| unknown.path.as_str())
            .collect();
        assert_eq!(unknown, ["groups[0].a", "groups[1].b"]);
    }

    #[test]
    fn a_fresh_recorder_does_not_inherit_reads() {
        let first: Value = serde_json::json!({"a": 1});
        let second: Value = serde_json::json!({"a": 1});
        with_recording(&first, || {
            read_key(&first, "a");
        });
        let ((), coverage) = with_recording(&second, || {});
        assert!(coverage.read_keys().next().is_none());
        assert_eq!(coverage.unknown_keys().len(), 1);
    }

    /// The five carried checkpoints' config.json, copied byte-for-byte from their snapshots.
    /// A key that is neither read nor ignorable is reported by its full path below.
    const CARRIED: &[(&str, &str)] = &[
        (
            "DeepSeek-V4.1-Flash",
            include_str!(
                "../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json"
            ),
        ),
        (
            "GLM-5.3-Flash",
            include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json"),
        ),
        (
            "Qwen3.8-Flash-Next-NVFP4",
            include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json"),
        ),
        (
            "Inkling-2-Small",
            include_str!("../../sparkflash-model-inkling2/config/inkling2_small.hf.json"),
        ),
        (
            "Dory1.0-4B-BF16",
            include_str!(
                "../../sparkflash-model-dory/config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json"
            ),
        ),
    ];

    /// Keys present in a carried config that this description reader does not consume.
    ///
    /// This is the honest residual, not a permission list. A key is here because a real parse
    /// reported it unread; the test below fails when one stops being unread (remove it) or when a
    /// key outside this list becomes unread (teach the reader or pin the new owner). The owner
    /// column says where the read belongs: `model crate parser` (a model crate's typed parser
    /// reads it), `vision reader` (the vision description must learn it), or `description work`
    /// (the shared reader must learn it). A path in this table makes the config-key check dirty;
    /// it never makes it clean.
    #[rustfmt::skip]
    const REMAINING: &[(&str, &str, &str)] = &[
        ("DeepSeek-V4.1-Flash", "text_config.dspark_markov_rank", "model crate parser"),
        ("DeepSeek-V4.1-Flash", "text_config.dspark_noise_token_id", "model crate parser"),
        ("DeepSeek-V4.1-Flash", "text_config.model_type", "model crate parser"),
        ("DeepSeek-V4.1-Flash", "text_config.num_key_value_heads", "model crate parser"),
        ("DeepSeek-V4.1-Flash", "vision_config.max_image_tokens", "vision reader"),
        ("DeepSeek-V4.1-Flash", "vision_config.max_wh_ratio", "vision reader"),
        ("DeepSeek-V4.1-Flash", "vision_config.model_type", "vision reader"),
        ("Dory1.0-4B-BF16", "mlp_bias", "model crate parser"),
        ("Dory1.0-4B-BF16", "ngpt_alpha_init", "model crate parser"),
        ("Dory1.0-4B-BF16", "rope_parameters.full_attention.rope_type", "model crate parser"),
        ("Dory1.0-4B-BF16", "rope_parameters.sliding_attention.rope_type", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.head_dim", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.hidden_size", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.intermediate_size", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.local_layer_ids", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.markov_head_type", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.markov_rank", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.max_position_embeddings", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.num_attention_heads", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.num_key_value_heads", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.o_bias", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.q_bias", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.rms_norm_eps", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.rope_theta", "model crate parser"),
        ("Inkling-2-Small", "mtp_config.sliding_window_size", "model crate parser"),
        ("Inkling-2-Small", "text_config.final_norm_pre_affine", "model crate parser"),
        ("Inkling-2-Small", "text_config.o_bias", "model crate parser"),
        ("Inkling-2-Small", "text_config.pre_norm_pre_affine", "model crate parser"),
        ("Inkling-2-Small", "text_config.q_bias", "model crate parser"),
        ("Inkling-2-Small", "text_config.swa_head_dim", "model crate parser"),
        ("Inkling-2-Small", "text_config.swa_num_attention_heads", "model crate parser"),
        ("Inkling-2-Small", "text_config.swa_num_key_value_heads", "model crate parser"),
        ("Inkling-2-Small", "text_config.use_gate_bias", "model crate parser"),
        ("Inkling-2-Small", "vision_config.projector_bias", "vision reader"),
    ];

    /// The unknown keys a carried config reports, as `(config, path)`.
    fn unknown_pairs() -> std::collections::BTreeSet<(String, String)> {
        let mut pairs = std::collections::BTreeSet::new();
        for (name, text) in CARRIED {
            let (_, coverage) =
                crate::Architecture::from_json_with_coverage(text).expect("a carried config");
            for unknown in coverage.unknown_keys() {
                pairs.insert(((*name).to_owned(), unknown.path.clone()));
            }
        }
        pairs
    }

    #[test]
    fn the_five_carried_configs_leave_only_the_pinned_residual() {
        let expected: std::collections::BTreeSet<(String, String)> = REMAINING
            .iter()
            .map(|(name, path, _)| ((*name).to_owned(), (*path).to_owned()))
            .collect();
        let actual = unknown_pairs();
        let unexpected: Vec<&(String, String)> = actual.difference(&expected).collect();
        let resolved: Vec<&(String, String)> = expected.difference(&actual).collect();
        assert!(
            unexpected.is_empty(),
            "config keys no read consumed and no owner pinned: {unexpected:#?}"
        );
        assert!(
            resolved.is_empty(),
            "config keys are read now; remove them from REMAINING: {resolved:#?}"
        );
    }

    /// The same check against the config.json files installed in the Hugging Face cache.
    ///
    /// Set SPARKFLASH_CONFIG_SNAPSHOTS to a colon-separated list of config paths in CARRIED
    /// order (DeepSeek, GLM, Qwen, Inkling, Dory). This is a host check: it proves each installed
    /// snapshot reports exactly the pinned residual.
    #[test]
    fn the_installed_snapshots_report_the_pinned_residual() {
        let Ok(paths) = std::env::var("SPARKFLASH_CONFIG_SNAPSHOTS") else {
            return;
        };
        let paths: Vec<&str> = paths.split(':').filter(|path| !path.is_empty()).collect();
        assert_eq!(
            paths.len(),
            CARRIED.len(),
            "name one snapshot config per carried config, in order"
        );
        for ((_name, fixture), path) in CARRIED.iter().zip(paths) {
            let (_, fixture_coverage) =
                crate::Architecture::from_json_with_coverage(fixture).expect("a carried fixture");
            let expected: std::collections::BTreeSet<String> = fixture_coverage
                .unknown_keys()
                .iter()
                .map(|unknown| unknown.path.clone())
                .collect();
            let text = std::fs::read_to_string(path).expect("an installed config.json");
            let (_, actual) =
                crate::Architecture::from_json_with_coverage(&text).expect("an installed config");
            let actual: std::collections::BTreeSet<String> = actual
                .unknown_keys()
                .iter()
                .map(|unknown| unknown.path.clone())
                .collect();
            let unexpected: Vec<&String> = actual.difference(&expected).collect();
            let resolved: Vec<&String> = expected.difference(&actual).collect();
            assert!(
                unexpected.is_empty(),
                "{path} has config keys no read consumed and no owner pinned: {unexpected:#?}"
            );
            assert!(
                resolved.is_empty(),
                "{path} no longer reports these pinned residual paths: {resolved:#?}"
            );
        }
    }

    #[test]
    fn every_ignorable_entry_has_a_reason_and_a_name_only_rule() {
        for (name, reason) in IGNORABLE {
            assert!(!name.is_empty());
            assert!(
                !name.contains('.'),
                "the ignore list is keyed by name, not path"
            );
            assert!(reason.len() > 20, "{name} needs a written reason");
        }
    }
}

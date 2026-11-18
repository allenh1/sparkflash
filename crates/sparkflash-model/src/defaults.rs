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

//! The serving defaults DeepSeek measured, and the citation an arm needs to depart from one.
//!
//! The numbers live in `sparkflash-deepseek-forward`; this table is their one description for
//! the rest of the tree, with the line that states each. An arm's carried profile may override
//! one only when the same TOML section (or the document root) carries
//! `measured = "docs/…"` naming where the replacement was measured. [crate::ModelProfile]
//! refuses the override otherwise, before any device is opened.

use crate::profile::{ModelProfile, ProfileError};

/// A default's value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DefaultValue {
    /// A count of rows, sequences or positions.
    Count(usize),
    /// A wall-clock cost in milliseconds.
    Milliseconds(f64),
}

impl DefaultValue {
    /// The count this value holds, when it holds one.
    #[must_use]
    pub const fn count(self) -> Option<usize> {
        match self {
            Self::Count(count) => Some(count),
            Self::Milliseconds(_) => None,
        }
    }
}

impl std::fmt::Display for DefaultValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count(count) => write!(f, "{count}"),
            Self::Milliseconds(ms) => write!(f, "{ms}"),
        }
    }
}

/// One DeepSeek default: its profile key, value, the line that states it, and the typed field
/// an arm's profile writes when it departs from it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Defaults {
    /// The typed profile key (`section.field`).
    pub key: &'static str,
    /// The measured value.
    pub value: DefaultValue,
    /// The source that states it, as `file:line`, inside this tree.
    pub site: &'static str,
    /// The section the override and its `measured` key live in.
    pub section: &'static str,
    /// What the default governs, in one sentence.
    pub note: &'static str,
}

/// DeepSeek's scheduler and serving defaults, read from `sparkflash-deepseek-forward`.
pub const DEFAULTS: &[Defaults] = &[
    Defaults {
        key: "prefill.chunk",
        value: DefaultValue::Count(4096),
        site: "crates/sparkflash-deepseek-forward/src/tp4_forward.rs:17454-17460",
        section: "prefill",
        note: "The rows one idle prompt chunk takes; 4096 by the sweep at :17455.",
    },
    Defaults {
        key: "prefill.chunk_busy",
        value: DefaultValue::Count(2048),
        site: "crates/sparkflash-deepseek-forward/src/tp4_forward/batch.rs:84-91",
        section: "prefill",
        note: "The rows one prompt chunk takes while another sequence decodes.",
    },
    Defaults {
        key: "runtime.rows",
        value: DefaultValue::Count(64),
        site: "crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:44",
        section: "runtime",
        note: "The most rows one decode or verify step takes (R_MAX).",
    },
    Defaults {
        key: "runtime.queue_depth",
        value: DefaultValue::Count(64),
        site: "crates/sparkflash-deepseek-forward/src/tp4_forward/batch.rs:79-80",
        section: "runtime",
        note: "The sequences that may wait for a lane past the lanes themselves (QUEUE).",
    },
    Defaults {
        key: "speculation.band_rows",
        value: DefaultValue::Count(16),
        site: "crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:1679-1681",
        section: "speculation",
        note: "The rows at which the shared step's cost model bends (BAND_ROWS).",
    },
    Defaults {
        key: "speculation.step_ms",
        value: DefaultValue::Milliseconds(27.0),
        site: "crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:1669-1677",
        section: "speculation",
        note: "The flat band's fixed part for a one-sequence step (STEP_COST.step_ms).",
    },
    Defaults {
        key: "speculation.row_ms",
        value: DefaultValue::Milliseconds(4.0),
        site: "crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:1669-1677",
        section: "speculation",
        note: "Each verified row's cost on the flat band (STEP_COST.row_ms).",
    },
    Defaults {
        key: "speculation.seq_ms",
        value: DefaultValue::Milliseconds(2.3),
        site: "crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:1683-1687",
        section: "speculation",
        note: "Each sequence past the first in a shared step (SEQ_MS).",
    },
];

impl Defaults {
    /// The typed field this default describes, when the profile has one.
    #[must_use]
    pub const fn field(self) -> &'static str {
        self.key
    }
}

/// The profile value for `key`, when the profile owns that field.
fn current(profile: &ModelProfile, key: &str) -> Option<usize> {
    match key {
        "prefill.chunk" => profile.prefill.chunk,
        "prefill.chunk_busy" => profile.prefill.chunk_busy,
        "runtime.rows" => profile.runtime.rows,
        _ => None,
    }
}

/// Refuse a profile override of a [DEFAULTS] entry that carries no measurement citation.
///
/// # Errors
/// An override of a table entry whose section (or the document root) has no
/// `measured = "docs/…"` string, or one that names something else.
pub fn check_profile(profile: &ModelProfile, text: &str) -> Result<(), ProfileError> {
    let document: toml::Value = toml::from_str(text)
        .map_err(|error| ProfileError::Parse(format!("the profile's TOML: {error}")))?;
    for default in DEFAULTS {
        let Some(value) = current(profile, default.key) else {
            continue;
        };
        if Some(value) == default.value.count() {
            continue;
        }
        let section = document.get(default.section);
        let citation = section
            .and_then(|section| section.get("measured"))
            .or_else(|| document.get("measured"));
        let message = match citation {
            Some(toml::Value::String(citation)) if citation.starts_with("docs/") => continue,
            Some(toml::Value::String(citation)) => format!(
                "{} = {value} overrides DeepSeek's measured default {} ({}); its measured = {citation:?} is not a docs/… path",
                default.key, default.value, default.site
            ),
            Some(_) => format!(
                "{} = {value} overrides DeepSeek's measured default {} ({}); measured must be a \"docs/…\" string",
                default.key, default.value, default.site
            ),
            None => format!(
                "{} = {value} overrides DeepSeek's measured default {} ({}); add measured = \"docs/…\" under [{}] naming the measurement",
                default.key, default.value, default.site, default.section
            ),
        };
        return Err(ProfileError::Invalid(message));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_states_the_five_deepseek_values_and_their_sites() {
        let get = |key: &str| {
            DEFAULTS
                .iter()
                .find(|default| default.key == key)
                .unwrap_or_else(|| panic!("{key} is not stated"))
        };
        assert_eq!(get("prefill.chunk").value, DefaultValue::Count(4096));
        assert_eq!(get("prefill.chunk_busy").value, DefaultValue::Count(2048));
        assert_eq!(get("runtime.rows").value, DefaultValue::Count(64));
        assert_eq!(get("runtime.queue_depth").value, DefaultValue::Count(64));
        assert_eq!(get("speculation.band_rows").value, DefaultValue::Count(16));
        for default in DEFAULTS {
            assert!(
                default
                    .site
                    .starts_with("crates/sparkflash-deepseek-forward/"),
                "{} cites {}",
                default.key,
                default.site
            );
            assert!(
                !default.note.trim().is_empty(),
                "{} has no note",
                default.key
            );
            assert_eq!(default.field(), default.key);
        }
    }

    #[test]
    fn an_override_equal_to_the_default_needs_no_citation() {
        let profile = ModelProfile::from_toml("[prefill]\nchunk = 4096\n").expect("the default");
        assert_eq!(profile.prefill.chunk, Some(4096));
    }

    #[test]
    fn an_override_without_a_measured_key_is_refused_by_name() {
        let error =
            ModelProfile::from_toml("[prefill]\nchunk = 2048\n").expect_err("an uncited override");
        let text = error.to_string();
        assert!(text.contains("prefill.chunk"), "{text}");
        assert!(text.contains("measured"), "{text}");
        assert!(text.contains("docs/"), "{text}");
        assert!(text.contains("[prefill]"), "{text}");
    }

    #[test]
    fn an_override_with_a_measured_citation_parses() {
        let profile = ModelProfile::from_toml(
            "[prefill]\nchunk = 2048\nmeasured = \"docs/performance.md#chunk\"\n",
        )
        .expect("a cited override");
        assert_eq!(profile.prefill.chunk, Some(2048));
        assert_eq!(
            profile.prefill.measured.as_deref(),
            Some("docs/performance.md#chunk")
        );
    }

    #[test]
    fn a_measured_key_that_is_not_a_docs_path_is_refused_by_name() {
        let error = ModelProfile::from_toml("[runtime]\nrows = 32\nmeasured = \"./notes.txt\"\n")
            .expect_err("a citation outside docs");
        let text = error.to_string();
        assert!(text.contains("runtime.rows"), "{text}");
        assert!(text.contains("docs/"), "{text}");
    }

    #[test]
    fn a_root_measured_key_cites_the_whole_document() {
        let profile = ModelProfile::from_toml(
            "measured = \"docs/performance.md#rows\"\n[prefill]\nchunk = 2048\n",
        )
        .expect("a root citation");
        assert_eq!(profile.prefill.chunk, Some(2048));
    }
}

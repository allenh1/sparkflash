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

//! The knobs a served process advertises, with the default and the help text beside each value.
//!
//! A knob an operator can change is a typed field of [crate::profile] with a default that is the
//! product's own choice, and a sentence that says what moving it costs. Those three facts live
//! here, once: `/v1/models` renders this table with the resolved values, so a control plane
//! cannot drift from the daemon it is configuring, and a default is changed in one place rather
//! than in a UI and a binary that disagree.
//!
//! This is not the environment surface - that is [crate::profile::ENV_OVERRIDES]. A knob whose
//! value one rank may read differently from its peers (the drafter path, a model's kernel
//! selection) stays a launch contract and is deliberately absent here.

use crate::profile::ModelProfile;

/// A knob's value: the two shapes a served setting takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value {
    /// A whole number, rendered as JSON's number.
    Count(usize),
    /// An on/off switch.
    Flag(bool),
}

impl Value {
    /// The value as JSON.
    #[must_use]
    pub fn json(self) -> serde_json::Value {
        match self {
            Self::Count(count) => serde_json::json!(count),
            Self::Flag(flag) => serde_json::json!(flag),
        }
    }
}

/// One advertised knob: its key, the default, and what moving it costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Setting {
    /// The dotted key, which is the profile field's path.
    pub key: &'static str,
    /// The default, which is what the daemon does when nothing sets the field.
    pub default: Value,
    /// One sentence: what the knob does and what its other setting costs.
    pub help: &'static str,
}

/// Every knob a served process advertises, in profile order.
pub const SETTINGS: &[Setting] = &[
    Setting {
        key: "prefill.chunk_rows",
        default: Value::Count(4096),
        help: "Rows of the prompt one step reads at once. Higher fills the prompt faster and \
               costs startup memory; a chunk that does not fit beside the weights is refused at \
               load, not at the first request.",
    },
    Setting {
        key: "prefill.drafting_prompt",
        default: Value::Flag(false),
        help: "Whether a request that drafts runs its prompt in bulk. Off, the drafter sees the \
               prompt and prompt processing runs one decode step a token.",
    },
];

/// A knob with the value this process resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved {
    /// The knob.
    pub setting: Setting,
    /// The value in force, after the environment and any profile file.
    pub value: Value,
}

impl Resolved {
    /// The card entry: the value, the default it departs from, and the help.
    #[must_use]
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "value": self.value.json(),
            "default": self.setting.default.json(),
            "help": self.setting.help,
        })
    }
}

/// Every knob with the value `profile` resolves, for the card and the boot log.
#[must_use]
pub fn resolve(profile: &ModelProfile) -> Vec<Resolved> {
    SETTINGS
        .iter()
        .map(|&setting| Resolved {
            setting,
            value: value(profile, setting.key),
        })
        .collect()
}

/// The knobs with the values this process's environment resolves, for a card built without the
/// arm's parsed profile: the environment is the one the arm read, so the answer is the same.
///
/// A value the arm refused still answers with the documented default rather than panicking; the
/// arm is what reports the refusal to the operator.
#[must_use]
pub fn resolve_process_env() -> Vec<Resolved> {
    let mut profile = ModelProfile::default();
    let _ = profile.apply_process_env();
    resolve(&profile)
}

/// The resolved value of `key` under `profile`, or the default for a key this table does not
/// own: a caller that asks about a knob no arm has still gets the documented answer.
#[must_use]
pub fn value(profile: &ModelProfile, key: &str) -> Value {
    match key {
        // A prefill that is off is the token path, which is a chunk of one row per step.
        "prefill.chunk_rows" if profile.prefill.enabled == Some(false) => Value::Count(0),
        "prefill.chunk_rows" => {
            Value::Count(
                profile
                    .prefill
                    .chunk
                    .unwrap_or_else(|| match default_of(key) {
                        Value::Count(rows) => rows,
                        Value::Flag(_) => 0,
                    }),
            )
        }
        "prefill.drafting_prompt" => Value::Flag(profile.prefill.drafting_prompt.unwrap_or(false)),
        _ => default_of(key),
    }
}

/// The default this table documents for `key`, or zero when it documents none.
#[must_use]
pub fn default_of(key: &str) -> Value {
    SETTINGS
        .iter()
        .find(|setting| setting.key == key)
        .map_or(Value::Count(0), |setting| setting.default)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The card renders every knob with its documented default when nothing is set, and the
    /// help is a sentence that ends.
    #[test]
    fn the_card_reports_the_defaults_and_their_help() {
        let profile = ModelProfile::default();
        let resolved = resolve(&profile);
        assert_eq!(resolved.len(), 2, "every documented knob is reported");
        let rows = &resolved[0];
        assert_eq!(rows.setting.key, "prefill.chunk_rows");
        assert_eq!(rows.value, Value::Count(4096));
        assert_eq!(rows.value, rows.setting.default);
        let drafting = &resolved[1];
        assert_eq!(drafting.setting.key, "prefill.drafting_prompt");
        assert_eq!(drafting.value, Value::Flag(false));
        for entry in &resolved {
            let help = entry.setting.help;
            assert!(!help.trim().is_empty(), "{} has help", entry.setting.key);
            assert!(
                help.trim_end().ends_with('.'),
                "{}'s help is a sentence: {help}",
                entry.setting.key
            );
        }
    }

    /// A value the environment sets moves the resolved value and leaves the default and the help
    /// alone, which is what makes the card a description of the daemon rather than of the UI.
    #[test]
    fn an_override_moves_the_value_and_not_the_default() {
        let mut profile = ModelProfile::default();
        profile
            .apply_env(|name| match name {
                "SPARKFLASH_GLM_PREFILL_CHUNK" => Some("512".to_owned()),
                "SPARKFLASH_GLM_PREFILL_DRAFTING" => Some("1".to_owned()),
                _ => None,
            })
            .expect("the overrides parse");
        let resolved = resolve(&profile);
        assert_eq!(resolved[0].value, Value::Count(512));
        assert_eq!(resolved[0].setting.default, Value::Count(4096));
        assert_eq!(resolved[1].value, Value::Flag(true));
        assert_eq!(resolved[1].setting.default, Value::Flag(false));
        for entry in &resolved {
            assert!(
                entry.setting.help.trim_end().ends_with('.'),
                "{} keeps its help text",
                entry.setting.key
            );
        }
        // The rendered entry carries both numbers, so a reader can say which one it departed from.
        let json = resolved[0].json();
        assert_eq!(json["value"], serde_json::json!(512));
        assert_eq!(json["default"], serde_json::json!(4096));
        assert_eq!(json["help"], serde_json::json!(resolved[0].setting.help));
    }

    /// A prefill that is switched off is the token path, which the card reports as a chunk of
    /// zero rows rather than as the default it would have used.
    #[test]
    fn a_disabled_prefill_reports_no_chunk() {
        let mut profile = ModelProfile::default();
        profile
            .apply_env(|name| (name == "SPARKFLASH_GLM_PREFILL").then(|| "0".to_owned()))
            .expect("the override parses");
        assert_eq!(value(&profile, "prefill.chunk_rows"), Value::Count(0));
    }

    /// A knob this table does not document reports the documented default rather than panicking,
    /// so a caller that asks about an arm's knob still gets an answer.
    #[test]
    fn an_unknown_key_reports_its_default() {
        assert_eq!(
            value(&ModelProfile::default(), "prefill.unknown"),
            Value::Count(0)
        );
        assert_eq!(default_of("prefill.chunk_rows"), Value::Count(4096));
    }
}

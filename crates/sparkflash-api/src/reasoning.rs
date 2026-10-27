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

//! The `reasoning_effort` / `thinking` remapper.
//!
//! DeepSeek-V4.1-Flash exposes a continuously controllable reasoning effort as an integer
//! in 1..=100. OpenAI clients instead send the discrete strings low/medium/high/max, so
//! this module is the single source of truth mapping the wire spellings onto the numeric
//! score, and resolving whether thinking is enabled at all.
//!
//! Precedence for thinking (highest first):
//! 1. `chat_template_kwargs.thinking` (bool, or "thinking" / "non-thinking")
//! 2. `enable_thinking` (bool)
//! 3. the default: thinking on.
//!
//! The numeric table is the protocol-level default (low/minimal 25, medium/high 50,
//! xhigh 75, max 100, absent 50). A model pack may expose the same aliases through its
//! encoding surface; a test pins the two tables against each other so they cannot drift.

use serde_json::Value;
use thiserror::Error;

/// Lowest accepted reasoning effort score.
pub const EFFORT_MIN: u8 = 1;
/// Highest accepted reasoning effort score.
pub const EFFORT_MAX: u8 = 100;
/// Score used when effort is absent, null or "high".
///
/// This is the released V4.1 encoder's default (`high`), and the same 25/50/75/100
/// table used by the prompt seam; recipe 0.2's separate 50/75/75/100 table is not
/// an API default. See docs/DECISIONS.md (task-63 effort-table consolidation).
pub const EFFORT_DEFAULT: u8 = 50;
/// Score for "low" and "minimal".
pub const EFFORT_LOW: u8 = 25;
/// Score for "medium" (an alias of the default high level).
pub const EFFORT_MEDIUM: u8 = EFFORT_DEFAULT;
/// Score for "high".
pub const EFFORT_HIGH: u8 = 50;
/// Score for "xhigh".
pub const EFFORT_XHIGH: u8 = 75;
/// Score for "max".
pub const EFFORT_MAX_SCORE: u8 = 100;

/// Resolved reasoning configuration for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningSpec {
    /// Whether the reasoning channel is enabled.
    pub thinking: bool,
    /// Reasoning effort score in 1..=100, or `None` when thinking is disabled.
    pub effort: Option<u8>,
}

impl Default for ReasoningSpec {
    fn default() -> Self {
        Self {
            thinking: true,
            effort: Some(EFFORT_DEFAULT),
        }
    }
}

/// Why a reasoning request could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ReasoningError {
    /// The integer score was outside 1..=100.
    #[error("reasoning_effort must be an integer in 1..=100 (got {value})")]
    EffortOutOfRange {
        /// The offending score.
        value: i64,
    },
    /// The JSON value was neither an integer nor a recognised string.
    #[error(
        "reasoning_effort must be an integer in 1..=100 or one of \"low\", \"minimal\", \"medium\", \"high\", \"xhigh\", \"max\" (got a JSON {kind})"
    )]
    InvalidEffortType {
        /// JSON kind of the offending value.
        kind: &'static str,
    },
    /// The string was not a recognised effort spelling.
    #[error("reasoning_effort string {value:?} is not recognised")]
    UnknownEffort {
        /// The offending string.
        value: String,
    },
    /// The thinking flag had an unsupported value.
    #[error("thinking must be a boolean or \"thinking\"/\"non-thinking\" (got {value:?})")]
    InvalidThinking {
        /// Rendered offending value.
        value: String,
    },
}

/// Maps a named effort spelling onto its numeric score.
///
/// Delegates the score to `crate::encoding::resolve_reasoning_effort`; matching
/// is case-insensitive and ignores surrounding whitespace. Returns `None` for
/// unrecognised spellings.
#[must_use]
pub fn effort_for_name(name: &str) -> Option<u8> {
    let normalized = name.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "low" | "minimal" => Some(EFFORT_LOW),
        "medium" | "high" => Some(EFFORT_HIGH),
        "xhigh" => Some(EFFORT_XHIGH),
        "max" => Some(EFFORT_MAX_SCORE),
        _ => None,
    }
}

/// Scores a string spelling through the canonical encoding table, translating its error
/// into this module's typed error.
fn alias_score(text: &str) -> Result<Option<u8>, ReasoningError> {
    effort_for_name(text)
        .map(Some)
        .ok_or_else(|| ReasoningError::UnknownEffort {
            value: text.to_owned(),
        })
}

/// Resolves the reasoning configuration of a request body.
///
/// The body is the raw OpenAI/Anthropic JSON document; unknown members are ignored.
///
/// # Errors
/// Returns ReasoningError when reasoning_effort is not an integer in 1..=100 nor a
/// recognised spelling, or when a thinking flag is neither a boolean nor
/// "thinking"/"non-thinking".
pub fn resolve(body: &Value) -> Result<ReasoningSpec, ReasoningError> {
    let effort = parse_effort(body.get("reasoning_effort"))?;
    let thinking = resolve_thinking(body)?;
    Ok(ReasoningSpec {
        thinking,
        effort: if thinking { effort } else { None },
    })
}

fn parse_effort(value: Option<&Value>) -> Result<Option<u8>, ReasoningError> {
    match value {
        None | Some(Value::Null) => Ok(Some(EFFORT_DEFAULT)),
        Some(Value::Number(number)) => {
            let score = number
                .as_i64()
                .ok_or(ReasoningError::InvalidEffortType { kind: "number" })?;
            if !(i64::from(EFFORT_MIN)..=i64::from(EFFORT_MAX)).contains(&score) {
                return Err(ReasoningError::EffortOutOfRange { value: score });
            }
            Ok(Some(score as u8))
        }
        Some(Value::String(text)) => {
            if let Ok(score) = text.trim().parse::<i64>() {
                if !(i64::from(EFFORT_MIN)..=i64::from(EFFORT_MAX)).contains(&score) {
                    return Err(ReasoningError::EffortOutOfRange { value: score });
                }
                return Ok(Some(score as u8));
            }
            alias_score(text)
        }
        Some(other) => Err(ReasoningError::InvalidEffortType {
            kind: json_kind(other),
        }),
    }
}

fn resolve_thinking(body: &Value) -> Result<bool, ReasoningError> {
    if let Some(kwargs) = body.get("chat_template_kwargs").filter(|v| !v.is_null())
        && let Some(thinking) = kwargs.get("thinking").filter(|v| !v.is_null())
    {
        return parse_thinking(thinking);
    }
    if let Some(enable) = body.get("enable_thinking").filter(|v| !v.is_null()) {
        return match enable {
            Value::Bool(flag) => Ok(*flag),
            other => Err(ReasoningError::InvalidThinking {
                value: render(other),
            }),
        };
    }
    Ok(true)
}

fn parse_thinking(value: &Value) -> Result<bool, ReasoningError> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::String(text) => {
            let normalized: String = text
                .chars()
                .filter(|c| *c != '-' && *c != '_' && !c.is_whitespace())
                .flat_map(char::to_lowercase)
                .collect();
            match normalized.as_str() {
                "thinking" => Ok(true),
                "nonthinking" => Ok(false),
                _ => Err(ReasoningError::InvalidThinking {
                    value: text.clone(),
                }),
            }
        }
        other => Err(ReasoningError::InvalidThinking {
            value: render(other),
        }),
    }
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn render(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| json_kind(value).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sparkflash_model_deepseek_v41::encoding::resolve_reasoning_effort;

    #[test]
    fn absent_and_null_effort_default_to_50() {
        assert_eq!(
            resolve(&json!({})).expect("empty body"),
            ReasoningSpec {
                thinking: true,
                effort: Some(50)
            }
        );
        assert_eq!(
            resolve(&json!({"reasoning_effort": null})).expect("null effort"),
            ReasoningSpec {
                thinking: true,
                effort: Some(50)
            }
        );
    }

    #[test]
    fn integer_effort_passes_through_and_bounds_are_inclusive() {
        assert_eq!(
            resolve(&json!({"reasoning_effort": 1}))
                .expect("min")
                .effort,
            Some(1)
        );
        assert_eq!(
            resolve(&json!({"reasoning_effort": 100}))
                .expect("max")
                .effort,
            Some(100)
        );
        assert_eq!(
            resolve(&json!({"reasoning_effort": 42}))
                .expect("mid")
                .effort,
            Some(42)
        );
        for bad in [0, -1, 101, 1000] {
            let err = resolve(&json!({"reasoning_effort": bad})).expect_err("out of range");
            assert_eq!(err, ReasoningError::EffortOutOfRange { value: bad });
        }
    }

    #[test]
    fn alias_table_matches_the_v41_encoding_surface() {
        // Drift guard: if the model pack changes a score, this fails instead of silently
        // producing two different API tables.
        for (name, score) in [
            ("low", EFFORT_LOW),
            ("minimal", EFFORT_LOW),
            ("medium", EFFORT_MEDIUM),
            ("high", EFFORT_HIGH),
            ("xhigh", EFFORT_XHIGH),
            ("max", EFFORT_MAX_SCORE),
        ] {
            let spec = resolve_reasoning_effort(Some(&json!(name)), true);
            assert_eq!(spec.score, Some(score), "encoding score for {name}");
            assert_eq!(spec.error, None, "encoding error for {name}");
        }
        let default = resolve_reasoning_effort(None, true);
        assert_eq!(default.score, Some(EFFORT_DEFAULT));
        assert_eq!(default.error, None);
        assert_eq!(resolve_reasoning_effort(None, false).score, None);
    }

    #[test]
    fn every_named_effort_maps_to_the_documented_score() {
        for (name, score) in [
            ("low", 25),
            ("minimal", 25),
            ("medium", 50),
            ("high", 50),
            ("xhigh", 75),
            ("max", 100),
        ] {
            let spec = resolve(&json!({"reasoning_effort": name})).expect("known effort");
            assert_eq!(spec.effort, Some(score), "effort {name}");
        }
        assert_eq!(effort_for_name("LOW"), Some(25));
        assert_eq!(effort_for_name(" Max "), Some(100));
        assert_eq!(effort_for_name("extreme"), None);
    }

    #[test]
    fn numeric_strings_are_accepted_and_bounds_enforced() {
        assert_eq!(
            resolve(&json!({"reasoning_effort": "100"}))
                .expect("numeric string")
                .effort,
            Some(100)
        );
        assert_eq!(
            resolve(&json!({"reasoning_effort": "0"})).expect_err("zero numeric string"),
            ReasoningError::EffortOutOfRange { value: 0 }
        );
    }

    #[test]
    fn malformed_effort_types_are_typed_errors() {
        assert_eq!(
            resolve(&json!({"reasoning_effort": "bogus"})).expect_err("unknown name"),
            ReasoningError::UnknownEffort {
                value: "bogus".to_owned()
            }
        );
        assert!(matches!(
            resolve(&json!({"reasoning_effort": true})).expect_err("bool effort"),
            ReasoningError::InvalidEffortType { kind: "boolean" }
        ));
        assert!(matches!(
            resolve(&json!({"reasoning_effort": 75.5})).expect_err("float effort"),
            ReasoningError::InvalidEffortType { kind: "number" }
        ));
        assert!(matches!(
            resolve(&json!({"reasoning_effort": [75]})).expect_err("array effort"),
            ReasoningError::InvalidEffortType { kind: "array" }
        ));
    }

    #[test]
    fn enable_thinking_is_the_second_precedence_level() {
        assert!(
            !resolve(&json!({"enable_thinking": false}))
                .expect("off")
                .thinking
        );
        assert_eq!(
            resolve(&json!({"enable_thinking": false}))
                .expect("off")
                .effort,
            None
        );
        assert_eq!(
            resolve(&json!({"enable_thinking": true}))
                .expect("on")
                .effort,
            Some(50)
        );
        assert!(resolve(&json!({"enable_thinking": "yes"})).is_err());
    }

    #[test]
    fn chat_template_kwargs_override_enable_thinking_both_directions() {
        let off = resolve(&json!({
            "chat_template_kwargs": {"thinking": false},
            "enable_thinking": true,
        }))
        .expect("kwargs off wins");
        assert_eq!(
            off,
            ReasoningSpec {
                thinking: false,
                effort: None
            }
        );

        let on = resolve(&json!({
            "chat_template_kwargs": {"thinking": true},
            "enable_thinking": false,
        }))
        .expect("kwargs on wins");
        assert_eq!(
            on,
            ReasoningSpec {
                thinking: true,
                effort: Some(50)
            }
        );

        let off_string = resolve(&json!({
            "chat_template_kwargs": {"thinking": "non-thinking"},
            "enable_thinking": true,
        }))
        .expect("non-thinking string");
        assert_eq!(
            off_string,
            ReasoningSpec {
                thinking: false,
                effort: None
            }
        );

        let on_string = resolve(&json!({
            "chat_template_kwargs": {"thinking": "thinking"},
            "enable_thinking": false,
        }))
        .expect("thinking string");
        assert_eq!(
            on_string,
            ReasoningSpec {
                thinking: true,
                effort: Some(50)
            }
        );
    }

    #[test]
    fn thinking_off_discards_the_effort_but_still_validates_it() {
        assert_eq!(
            resolve(&json!({"reasoning_effort": 100, "enable_thinking": false})).expect("off"),
            ReasoningSpec {
                thinking: false,
                effort: None
            }
        );
        assert!(resolve(&json!({"reasoning_effort": 0, "enable_thinking": false})).is_err());
    }

    #[test]
    fn invalid_thinking_values_are_typed_errors() {
        assert!(matches!(
            resolve(&json!({"chat_template_kwargs": {"thinking": 3}})).expect_err("int thinking"),
            ReasoningError::InvalidThinking { .. }
        ));
        assert!(matches!(
            resolve(&json!({"chat_template_kwargs": {"thinking": "maybe"}}))
                .expect_err("bad string"),
            ReasoningError::InvalidThinking { .. }
        ));
        assert!(matches!(
            resolve(&json!({"enable_thinking": "on"})).expect_err("bad enable flags"),
            ReasoningError::InvalidThinking { .. }
        ));
        assert_eq!(
            resolve(&json!({"chat_template_kwargs": null})).expect("null kwargs"),
            ReasoningSpec::default()
        );
    }
}

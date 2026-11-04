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

//! What every model pack's configuration parser shares.
//!
//! Every pack parses a released `config.json` and refuses the shapes it cannot serve; the
//! Qwen3.8 and GLM-5.3 packs and the DFlash 2 drafter ([crate::dflash2]) report it with one
//! error type, [ConfigError], and read the fields a release writes either way with [OneOrMany].

use serde::Deserialize;

/// The error a malformed or unsupported configuration produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// The document is not valid JSON or misses a required field.
    Json(String),
    /// The document parses but describes a shape this pack cannot serve.
    Invalid(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(message) => write!(formatter, "config JSON: {message}"),
            Self::Invalid(message) => write!(formatter, "unsupported configuration: {message}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// A `config.json` field the releases write as one value or as a list, such as `eos_token_id`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    /// The field is a single value.
    One(T),
    /// The field is a list.
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    /// The field's values, one or many.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

/// A missing field reads as one default value.
impl<T: Default> Default for OneOrMany<T> {
    fn default() -> Self {
        Self::One(T::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_field_reads_as_one_value_or_a_list() {
        let one: OneOrMany<u32> = serde_json::from_str("7").expect("one");
        assert_eq!(one.into_vec(), [7]);
        let many: OneOrMany<u32> = serde_json::from_str("[7, 9]").expect("many");
        assert_eq!(many.into_vec(), [7, 9]);
        assert_eq!(OneOrMany::<u32>::default().into_vec(), [0]);
        assert!(serde_json::from_str::<OneOrMany<u32>>("\"7\"").is_err());
    }
}

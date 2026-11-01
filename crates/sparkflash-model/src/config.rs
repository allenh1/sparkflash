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

//! The error a model pack's configuration parser returns.
//!
//! Every pack parses a released `config.json` and refuses the shapes it cannot serve; the
//! Qwen3.8 and GLM-5.3 packs and the DFlash 2 drafter ([crate::dflash2]) report it with this one
//! type.

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

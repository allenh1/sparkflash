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

//! Configuration errors shared by model packs and the generic host arithmetic.

/// What is wrong with a configuration.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// The JSON could not be parsed, or a required key was absent.
    #[error("could not parse the configuration: {0}")]
    Json(String),
    /// A parsed value contradicts another.
    #[error("{field}: {detail}")]
    Invalid {
        /// The field the complaint is about.
        field: &'static str,
        /// What is wrong with it.
        detail: String,
    },
}

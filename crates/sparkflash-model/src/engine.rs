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

//! How a model runs, independent of which weights it is.

/// The host/reference engine: correctness first, no GPU.
pub const HOST_REFERENCE: &str = "host-reference";

/// The tensor-parallel device engine used by the Spark cluster.
pub const TP4_DEVICE: &str = "tp4-device";

/// The engine kind selected when the operator does not name one.
pub const DEFAULT_ENGINE_KIND: &str = TP4_DEVICE;

/// The engine kinds this build accepts, in selector order.
pub const ENGINE_NAMES: &[&str] = &[TP4_DEVICE, HOST_REFERENCE];

/// A runtime the weights can be executed with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// The CPU reference/host path.
    HostReference,
    /// The four-rank tensor-parallel device path.
    Tp4Device,
}

impl EngineKind {
    /// The stable wire/config spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HostReference => HOST_REFERENCE,
            Self::Tp4Device => TP4_DEVICE,
        }
    }
}

impl std::fmt::Display for EngineKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Parse an engine selector from a config or CLI value.
///
/// The legacy value `deepseek-v4.1-flash` is accepted as `tp4-device` so an existing
/// `sparkflash.toml` keeps working after the model and engine selectors are separated.
#[must_use]
pub fn parse_engine_kind(value: &str) -> Option<EngineKind> {
    match value.trim() {
        HOST_REFERENCE => Some(EngineKind::HostReference),
        TP4_DEVICE | "deepseek-v4.1-flash" => Some(EngineKind::Tp4Device),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_names_are_unique_and_parse() {
        for name in ENGINE_NAMES {
            assert!(parse_engine_kind(name).is_some(), "{name} should parse");
        }
        assert_eq!(
            parse_engine_kind("deepseek-v4.1-flash"),
            Some(EngineKind::Tp4Device)
        );
        assert_eq!(parse_engine_kind("vllm"), None);
        assert_eq!(DEFAULT_ENGINE_KIND, TP4_DEVICE);
    }
}

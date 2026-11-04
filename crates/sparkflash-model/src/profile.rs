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

//! Typed per-model runtime profiles.
//!
//! A model's behaviour is configuration, not a pile of environment switches read at the point of
//! use. A profile is parsed once, validated, printed at boot, and handed to the arm. The
//! environment remains an override, never the only way to reach a behaviour: every override
//! below names a typed field with a documented value domain, an unknown value is refused, and an
//! explicit TOML value wins over the environment.
//!
//! The table [ENV_OVERRIDES] is the complete environment surface this module owns. Arms with
//! switches not listed here keep them as deprecated aliases until their typed field lands; the
//! gate in docs/adding-a-model.md says a new switch must be added here, not read directly.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// A profile failure.
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    /// The TOML does not parse, or a key or enum value is unknown.
    #[error("profile: {0}")]
    Parse(String),
    /// A typed field is out of range or internally inconsistent.
    #[error("profile: {0}")]
    Invalid(String),
    /// An environment override carries a value the field does not accept.
    #[error("{name}={value:?} is not a valid {field}: {expected}")]
    Env {
        /// The environment variable.
        name: &'static str,
        /// The value found.
        value: String,
        /// The typed field.
        field: &'static str,
        /// The accepted domain.
        expected: &'static str,
    },
    /// The file could not be read.
    #[error("profile {path}: {source}")]
    Io {
        /// The profile path.
        path: String,
        /// The read error.
        source: std::io::Error,
    },
}

/// The runtime section: sequence width and admission.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeProfile {
    /// The context window one sequence allocates.
    pub context_tokens: Option<usize>,
    /// Sequences one rank admits.
    pub concurrency: Option<usize>,
    /// The KV budget in gibibytes.
    pub kv_budget_gib: Option<usize>,
    /// Whether the device path runs instead of the host twin.
    pub device: Option<bool>,
}

/// Which projection path a bulk prefill takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationMode {
    /// Quantise activations to FP8.
    Fp8,
    /// Keep each activation as a bf16 head plus a bf16 remainder.
    Split,
    /// Keep activations in bf16.
    Bf16,
}

/// The prefill section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PrefillProfile {
    /// Whether the bulk prefill path is enabled.
    pub enabled: Option<bool>,
    /// Tokens per layer-major chunk.
    pub chunk: Option<usize>,
    /// Whether a wide chunk takes the tensor cores.
    pub tensor_cores: Option<bool>,
    /// Which activation form the tensor-core GEMMs take.
    pub activation: Option<ActivationMode>,
    /// Whether prefill is exact against the token path.
    pub exact: Option<bool>,
}

/// The decode section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecodeProfile {
    /// Whether a decode step and a speculative verify take the tensor cores, in the form that
    /// computes a row to the same bits at every row count.
    pub tensor_cores: Option<bool>,
}

/// The attention-cache representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentCacheMode {
    /// Pick from the model's release.
    Auto,
    /// Keep the latent in f32.
    F32,
    /// Keep the released E4M3 latent.
    Fp8,
}

/// The attention section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AttentionProfile {
    /// The latent-cache representation.
    pub latent_cache: Option<LatentCacheMode>,
    /// Selected tokens per sparse query.
    pub selected_tokens: Option<usize>,
}

/// The speculation section.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeculationProfile {
    /// Whether speculation is enabled.
    pub enabled: Option<bool>,
    /// The fixed draft depth, when the request does not choose one.
    pub depth: Option<usize>,
    /// The MTP fold's scale.
    pub mtp_fold_scale: Option<f32>,
    /// The drafter directory or repository id.
    pub drafter: Option<String>,
}

/// The sequence-parallel section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SequenceProfile {
    /// Whether a prefill chunk's hyper-connections run sequence-parallel.
    pub parallel: Option<bool>,
    /// Whether such a chunk overlaps its collectives with the other half.
    pub overlap: Option<bool>,
}

/// The probe the device path runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeMode {
    /// No probe.
    Off,
    /// Report non-finite buffers.
    Finite,
    /// Report per-buffer hashes for a line-by-line diff.
    Hash,
    /// Dump the final rows.
    Dump,
}

/// The diagnostics section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiagnosticsProfile {
    /// Per-stage device timing.
    pub profile: Option<bool>,
    /// The device probe.
    pub probe: Option<ProbeMode>,
    /// The per-step trace.
    pub trace: Option<bool>,
    /// A trace of the speculative choices.
    pub spec_trace: Option<bool>,
}

/// One model's typed runtime profile.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelProfile {
    /// The registry model the profile configures (`numerics/registry.toml`). The declaration
    /// check (`scripts/numerics.py ci`) reads it to know whose references a changed setting needs
    /// recaptured, and a test holds it to the descriptor `weights` selects.
    pub model: Option<String>,
    /// The model family the profile belongs to, for a mismatch check.
    pub family: Option<String>,
    /// The weights selector.
    pub weights: Option<String>,
    /// Runtime width and admission.
    pub runtime: RuntimeProfile,
    /// Bulk prefill.
    pub prefill: PrefillProfile,
    /// Decode steps and speculative verifies.
    pub decode: DecodeProfile,
    /// Attention caches.
    pub attention: AttentionProfile,
    /// Speculation.
    pub speculation: SpeculationProfile,
    /// Sequence parallelism.
    pub sequence: SequenceProfile,
    /// Diagnostics.
    pub diagnostics: DiagnosticsProfile,
}

/// The environment overrides this module owns, with the typed field each sets.
///
/// A value the field does not accept is refused rather than silently ignored. These names are
/// aliases: the TOML field is the source of truth and the override exists for an operator's A/B
/// run, not for behaviour a normal boot depends on.
pub const ENV_OVERRIDES: &[(&str, &str)] = &[
    ("SPARK_MAX_SEQ_LEN", "runtime.context_tokens"),
    ("SPARKFLASH_LANES", "runtime.concurrency"),
    ("SPARKFLASH_PREFILL", "prefill.enabled"),
    ("SPARKFLASH_GLM_PREFILL", "prefill.enabled"),
    ("SPARKFLASH_QWEN4_PREFILL_CHUNK", "prefill.chunk"),
    ("SPARKFLASH_GLM_PREFILL_CHUNK", "prefill.chunk"),
    ("SPARKFLASH_QWEN4_PREFILL_MMA", "prefill.tensor_cores"),
    ("SPARKFLASH_GLM_PREFILL_EXACT", "prefill.exact"),
    ("SPARKFLASH_GLM_PREFILL_ACTIVATION", "prefill.activation"),
    ("SPARKFLASH_DECODE_MMA", "decode.tensor_cores"),
    ("SPARKFLASH_GLM_LATENT_CACHE", "attention.latent_cache"),
    ("SPARKFLASH_QWEN4_SPEC", "speculation.enabled"),
    ("SPARKFLASH_QWEN4_SPEC_DEPTH", "speculation.depth"),
    ("SPARKFLASH_QWEN4_MTP_FOLD", "speculation.mtp_fold_scale"),
    ("SPARKFLASH_QWEN4_DFLASH2", "speculation.drafter"),
    ("SPARKFLASH_SEQUENCE_PARALLEL", "sequence.parallel"),
    ("SPARKFLASH_SEQUENCE_OVERLAP", "sequence.overlap"),
    ("SPARKFLASH_QWEN4_PROF", "diagnostics.profile"),
    ("SPARKFLASH_QWEN4_PROBE", "diagnostics.probe"),
    ("SPARKFLASH_QWEN4_TRACE", "diagnostics.trace"),
    ("SPARKFLASH_QWEN4_SPEC_TRACE", "diagnostics.spec_trace"),
];

impl ModelProfile {
    /// Parse a profile from TOML, refusing unknown keys.
    ///
    /// # Errors
    /// A syntax error, an unknown key or an unknown enum value.
    pub fn from_toml(text: &str) -> Result<Self, ProfileError> {
        let profile: Self =
            toml::from_str(text).map_err(|error| ProfileError::Parse(error.to_string()))?;
        profile.validate()?;
        Ok(profile)
    }

    /// Read a profile from a file.
    ///
    /// # Errors
    /// A read error, or as [Self::from_toml].
    pub fn load(path: &Path) -> Result<Self, ProfileError> {
        let text = std::fs::read_to_string(path).map_err(|source| ProfileError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml(&text)
    }

    /// Refuse an internally inconsistent profile.
    ///
    /// # Errors
    /// A zero width, a chunk that is not a multiple of the 64-token granularity, a draft depth
    /// of zero, or a non-positive fold scale.
    pub fn validate(&self) -> Result<(), ProfileError> {
        if self.runtime.context_tokens == Some(0) {
            return Err(ProfileError::Invalid(
                "runtime.context_tokens is zero".to_owned(),
            ));
        }
        if self.runtime.concurrency == Some(0) {
            return Err(ProfileError::Invalid(
                "runtime.concurrency is zero".to_owned(),
            ));
        }
        if let Some(chunk) = self.prefill.chunk
            && (chunk == 0 || !chunk.is_multiple_of(64))
        {
            return Err(ProfileError::Invalid(format!(
                "prefill.chunk {chunk} is not a positive multiple of 64"
            )));
        }
        if self.speculation.depth == Some(0) {
            return Err(ProfileError::Invalid(
                "speculation.depth is zero".to_owned(),
            ));
        }
        if let Some(scale) = self.speculation.mtp_fold_scale
            && (!scale.is_finite() || scale <= 0.0)
        {
            return Err(ProfileError::Invalid(format!(
                "speculation.mtp_fold_scale {scale} is not a positive finite number"
            )));
        }
        Ok(())
    }

    /// Apply environment overrides, refusing values outside a field's domain.
    ///
    /// # Errors
    /// A malformed number, a switch other than 0, 1, off or on, or an unknown probe or activation
    /// name.
    pub fn apply_env(
        &mut self,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<(), ProfileError> {
        for (name, _) in ENV_OVERRIDES {
            if let Some(value) = lookup(name) {
                self.apply_one(name, &value)?;
            }
        }
        self.validate()
    }

    /// Apply the process environment.
    ///
    /// # Errors
    /// As [Self::apply_env].
    pub fn apply_process_env(&mut self) -> Result<(), ProfileError> {
        self.apply_env(|name| std::env::var(name).ok())
    }

    fn apply_one(&mut self, name: &'static str, value: &str) -> Result<(), ProfileError> {
        let field = ENV_OVERRIDES
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map_or("unknown", |(_, field)| *field);
        match field {
            "runtime.context_tokens" => {
                self.runtime.context_tokens = Some(parse_usize(name, value, field)?)
            }
            "runtime.concurrency" => {
                self.runtime.concurrency = Some(parse_usize(name, value, field)?)
            }
            "prefill.enabled" => self.prefill.enabled = Some(parse_bool(name, value, field)?),
            "prefill.chunk" => self.prefill.chunk = Some(parse_usize(name, value, field)?),
            "prefill.tensor_cores" => {
                self.prefill.tensor_cores = Some(parse_bool(name, value, field)?)
            }
            "prefill.exact" => self.prefill.exact = Some(parse_bool(name, value, field)?),
            "decode.tensor_cores" => {
                self.decode.tensor_cores = Some(parse_bool(name, value, field)?)
            }
            "prefill.activation" => {
                self.prefill.activation = Some(match value {
                    "fp8" => ActivationMode::Fp8,
                    "split" => ActivationMode::Split,
                    "bf16" => ActivationMode::Bf16,
                    other => return Err(env_error(name, other, field, "fp8, split or bf16")),
                });
            }
            "attention.latent_cache" => {
                self.attention.latent_cache = Some(match value {
                    "auto" => LatentCacheMode::Auto,
                    "f32" => LatentCacheMode::F32,
                    "fp8" => LatentCacheMode::Fp8,
                    other => return Err(env_error(name, other, field, "auto, f32 or fp8")),
                });
            }
            "speculation.enabled" => {
                self.speculation.enabled = Some(parse_bool(name, value, field)?)
            }
            "speculation.depth" => self.speculation.depth = Some(parse_usize(name, value, field)?),
            "speculation.mtp_fold_scale" => {
                self.speculation.mtp_fold_scale = Some(
                    value
                        .parse()
                        .map_err(|_| env_error(name, value, field, "a positive number"))?,
                );
            }
            "speculation.drafter" => self.speculation.drafter = Some(value.to_owned()),
            "sequence.parallel" => self.sequence.parallel = Some(parse_bool(name, value, field)?),
            "sequence.overlap" => self.sequence.overlap = Some(parse_bool(name, value, field)?),
            "diagnostics.profile" => {
                self.diagnostics.profile = Some(parse_bool(name, value, field)?)
            }
            "diagnostics.probe" => {
                self.diagnostics.probe = Some(match value {
                    "0" | "off" => ProbeMode::Off,
                    "1" | "finite" => ProbeMode::Finite,
                    "hash" => ProbeMode::Hash,
                    "dump" => ProbeMode::Dump,
                    other => return Err(env_error(name, other, field, "0, 1, hash or dump")),
                });
            }
            "diagnostics.trace" => self.diagnostics.trace = Some(parse_bool(name, value, field)?),
            "diagnostics.spec_trace" => {
                self.diagnostics.spec_trace = Some(parse_bool(name, value, field)?)
            }
            other => return Err(ProfileError::Invalid(format!("no override for {other}"))),
        }
        Ok(())
    }

    /// Render the resolved profile as TOML for the boot log.
    ///
    /// # Errors
    /// A serialization failure, which cannot happen for this plain struct.
    pub fn to_toml(&self) -> Result<String, ProfileError> {
        toml::to_string_pretty(self).map_err(|error| ProfileError::Parse(error.to_string()))
    }

    /// A one-block boot summary: every set field, with its typed value.
    #[must_use]
    pub fn boot_summary(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "sparkflash profile (model={}, family={}, weights={})",
            self.model.as_deref().unwrap_or("-"),
            self.family.as_deref().unwrap_or("-"),
            self.weights.as_deref().unwrap_or("-")
        );
        let mut rows = BTreeMap::new();
        macro_rules! row {
            ($key:literal, $value:expr) => {
                if let Some(value) = $value {
                    rows.insert($key, format!("{value:?}"));
                }
            };
        }
        row!("runtime.context_tokens", self.runtime.context_tokens);
        row!("runtime.concurrency", self.runtime.concurrency);
        row!("runtime.kv_budget_gib", self.runtime.kv_budget_gib);
        row!("runtime.device", self.runtime.device);
        row!("prefill.enabled", self.prefill.enabled);
        row!("prefill.chunk", self.prefill.chunk);
        row!("prefill.tensor_cores", self.prefill.tensor_cores);
        row!("prefill.activation", self.prefill.activation);
        row!("prefill.exact", self.prefill.exact);
        row!("decode.tensor_cores", self.decode.tensor_cores);
        row!("attention.latent_cache", self.attention.latent_cache);
        row!("attention.selected_tokens", self.attention.selected_tokens);
        row!("speculation.enabled", self.speculation.enabled);
        row!("speculation.depth", self.speculation.depth);
        row!(
            "speculation.mtp_fold_scale",
            self.speculation.mtp_fold_scale
        );
        row!("speculation.drafter", self.speculation.drafter.as_deref());
        row!("sequence.parallel", self.sequence.parallel);
        row!("sequence.overlap", self.sequence.overlap);
        row!("diagnostics.profile", self.diagnostics.profile);
        row!("diagnostics.probe", self.diagnostics.probe);
        row!("diagnostics.trace", self.diagnostics.trace);
        row!("diagnostics.spec_trace", self.diagnostics.spec_trace);
        for (key, value) in rows {
            let _ = writeln!(out, "  {key} = {value}");
        }
        out
    }
}

fn parse_usize(
    name: &'static str,
    value: &str,
    field: &'static str,
) -> Result<usize, ProfileError> {
    value
        .parse()
        .map_err(|_| env_error(name, value, field, "a non-negative integer"))
}

fn parse_bool(name: &'static str, value: &str, field: &'static str) -> Result<bool, ProfileError> {
    match value {
        "1" | "true" | "on" => Ok(true),
        "0" | "false" | "off" => Ok(false),
        other => Err(env_error(name, other, field, "0 or 1 (off or on)")),
    }
}

fn env_error(
    name: &'static str,
    value: &str,
    field: &'static str,
    expected: &'static str,
) -> ProfileError {
    ProfileError::Env {
        name,
        value: value.to_owned(),
        field,
        expected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE: &str = r#"
family = "qwen4-exp"
weights = "qwen3.8-flash-next-nvfp4"

[runtime]
context_tokens = 262144
concurrency = 2

[prefill]
enabled = true
chunk = 4096
tensor_cores = true
activation = "split"

[attention]
latent_cache = "auto"

[diagnostics]
probe = "hash"
"#;

    #[test]
    fn a_typed_profile_parses_and_prints_every_set_field() {
        let profile = ModelProfile::from_toml(PROFILE).expect("the profile parses");
        assert_eq!(profile.prefill.activation, Some(ActivationMode::Split));
        assert_eq!(profile.diagnostics.probe, Some(ProbeMode::Hash));
        let summary = profile.boot_summary();
        assert!(summary.contains("prefill.chunk = 4096"), "{summary}");
        assert!(summary.contains("diagnostics.probe = Hash"), "{summary}");
    }

    /// `scripts/numerics.py ci` asks each arm profile's `model` whose references a changed setting
    /// needs recaptured. This holds that answer to the descriptor the profile's weights select, so
    /// a wrong `model` cannot send the recapture to another model's references.
    #[test]
    fn every_arm_profile_names_the_model_its_weights_select() {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crates folder");
        let mut profiles = Vec::new();
        for arm in std::fs::read_dir(crates).expect("the crates folder lists") {
            let arm = arm.expect("a crate").path();
            let is_arm = arm
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("sparkflash-model-"));
            let Ok(config) = std::fs::read_dir(arm.join("config")) else {
                continue;
            };
            if !is_arm {
                continue;
            }
            profiles.extend(
                config
                    .map(|file| file.expect("a profile").path())
                    .filter(|path| {
                        path.extension()
                            .is_some_and(|extension| extension == "toml")
                    }),
            );
        }
        assert!(profiles.len() >= 3, "found only {profiles:?}");
        for path in profiles {
            let shown = path.display();
            let profile =
                ModelProfile::load(&path).unwrap_or_else(|error| panic!("{shown}: {error}"));
            let weights = profile
                .weights
                .as_deref()
                .unwrap_or_else(|| panic!("{shown} names no weights"));
            let descriptor = crate::descriptor::find_model_by_weights(weights)
                .unwrap_or_else(|| panic!("{shown}: no descriptor serves the weights {weights}"));
            assert_eq!(
                profile.model.as_deref(),
                Some(descriptor.id.as_str()),
                "{shown} must name the model its weights {weights} select"
            );
        }
    }

    #[test]
    fn an_unknown_key_is_refused() {
        let error = ModelProfile::from_toml("[runtime]\nconcurreny = 2\n").expect_err("typo");
        assert!(matches!(error, ProfileError::Parse(_)));
    }

    #[test]
    fn an_unknown_enum_value_is_refused() {
        let error = ModelProfile::from_toml("[prefill]\nactivation = \"fp4\"\n")
            .expect_err("unknown activation");
        assert!(matches!(error, ProfileError::Parse(_)));
    }

    #[test]
    fn the_environment_is_an_override_with_a_typed_domain() {
        let mut profile = ModelProfile::from_toml(PROFILE).expect("the profile parses");
        profile
            .apply_env(|name| match name {
                "SPARKFLASH_QWEN4_PREFILL_CHUNK" => Some("2048".to_owned()),
                "SPARKFLASH_QWEN4_PROBE" => Some("dump".to_owned()),
                "SPARKFLASH_QWEN4_SPEC" => Some("0".to_owned()),
                _ => None,
            })
            .expect("the overrides are valid");
        assert_eq!(profile.prefill.chunk, Some(2048));
        assert_eq!(profile.diagnostics.probe, Some(ProbeMode::Dump));
        assert_eq!(profile.speculation.enabled, Some(false));
    }

    #[test]
    fn an_environment_value_outside_the_domain_is_refused() {
        let mut profile = ModelProfile::default();
        let error = profile
            .apply_env(|name| (name == "SPARKFLASH_QWEN4_PREFILL_MMA").then(|| "yes".to_owned()))
            .expect_err("yes is not a bool");
        assert!(matches!(error, ProfileError::Env { .. }));
    }

    #[test]
    fn a_switch_takes_the_deepseek_spellings_too() {
        let mut profile = ModelProfile::default();
        profile
            .apply_env(|name| (name == "SPARKFLASH_DECODE_MMA").then(|| "off".to_owned()))
            .expect("off is a switch value");
        assert_eq!(profile.decode.tensor_cores, Some(false));
        profile
            .apply_env(|name| (name == "SPARKFLASH_DECODE_MMA").then(|| "on".to_owned()))
            .expect("on is a switch value");
        assert_eq!(profile.decode.tensor_cores, Some(true));
    }

    #[test]
    fn a_zero_or_misaligned_chunk_is_refused() {
        let error = ModelProfile::from_toml("[prefill]\nchunk = 100\n").expect_err("misaligned");
        assert!(matches!(error, ProfileError::Invalid(_)));
    }

    #[test]
    fn every_documented_override_is_reachable() {
        let mut profile = ModelProfile::default();
        for (name, field) in ENV_OVERRIDES {
            assert!(!field.is_empty(), "{name} has no field");
        }
        profile
            .apply_env(|name| (name == ENV_OVERRIDES[0].0).then(|| "1024".to_owned()))
            .expect("the first override parses");
    }
}

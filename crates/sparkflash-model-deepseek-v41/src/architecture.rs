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

//! The bridge between DeepSeek's typed configuration and the shared architecture.
//!
//! The typed [ModelConfig] is the arm's validated profile; [sparkflash_model::Architecture]
//! is what the rest of the tree reads. The released HF document feeds both, so a released load
//! cross-checks them: a field that drifts between the two parsers is a refusal naming that field.
//! DeepSeek's kernel implementations remain the canonical ones; this module selects no kernel.

use sparkflash_model::Architecture;
use sparkflash_model::architecture::{AttentionKind, Ffn};

use crate::config::{ConfigError, ModelConfig};
use crate::layer_plan::{LayerMode, LayerPlan};

const CARRIED_RELEASE: &str = include_str!("../config/deepseek_v41_flash.released.json");

/// Parse the shared architecture from a released DeepSeek-V4.1 `config.json`.
///
/// # Errors
/// As [Architecture::from_json].
pub fn architecture_from_json(text: &str) -> Result<Architecture, ConfigError> {
    Architecture::from_json(text).map_err(shared_error)
}

/// The shared architecture of the released DeepSeek-V4.1 checkpoint carried in this crate.
///
/// # Errors
/// [ConfigError::Json] only if the carried document were edited into an unreadable shape.
pub fn architecture_from_carried() -> Result<Architecture, ConfigError> {
    architecture_from_json(CARRIED_RELEASE)
}

/// The typed pack and the shared model crate carry separate error types; a shared-parser failure
/// keeps its message and is attributed to the architecture field.
fn shared_error(error: sparkflash_model::ConfigError) -> ConfigError {
    match error {
        sparkflash_model::ConfigError::Json(message) => ConfigError::Json(message),
        sparkflash_model::ConfigError::Invalid(message) => ConfigError::Invalid {
            field: "architecture",
            detail: message,
        },
    }
}

impl ModelConfig {
    /// Refuse a typed configuration that disagrees with the shared architecture.
    ///
    /// # Errors
    /// [ConfigError::Invalid] naming the first field that differs.
    pub fn cross_check(&self, architecture: &Architecture) -> Result<(), ConfigError> {
        check("hidden_size", self.dim, architecture.hidden_size)?;
        check("vocab_size", self.vocab_size, architecture.vocab_size)?;
        check(
            "num_hidden_layers",
            self.n_layers,
            architecture.layers.len(),
        )?;
        check(
            "mtp stages",
            self.n_mtp_layers,
            architecture.mtp.as_ref().map_or(0, |mtp| mtp.stages),
        )?;

        let compressed = self
            .compress_ratios
            .iter()
            .take(self.n_layers)
            .filter(|ratio| **ratio > 0)
            .count();
        let arch_compressed = architecture
            .layers
            .iter()
            .filter(|layer| layer.attention.compression.is_some())
            .count();
        check("compressed layers", compressed, arch_compressed)?;

        let indexers = self
            .index_source_layers
            .iter()
            .filter(|layer| **layer < self.n_layers)
            .count();
        check(
            "indexer layers",
            indexers,
            architecture.indexer_layers().len(),
        )?;

        let mla = architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.attention.kind, AttentionKind::Mla { .. }))
            .count();
        check("MLA layers", self.n_layers, mla)?;
        let moe = architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.ffn, Ffn::Moe(_)))
            .count();
        check("MoE layers", self.n_layers, moe)?;
        Ok(())
    }

    /// Refuse a [LayerPlan] whose row modes disagree with the shared architecture.
    ///
    /// The plan is the DeepSeek-specific tensor and cache view; the architecture is the shared
    /// view. This check is the one place they meet, so a plan that grows a compressor, indexer or
    /// Engram row the canonical document does not name is refused at load.
    ///
    /// # Errors
    /// [ConfigError::Invalid] naming the first row that differs.
    pub fn cross_check_plan(
        &self,
        plan: &LayerPlan,
        architecture: &Architecture,
    ) -> Result<(), ConfigError> {
        check(
            "layer plan rows",
            plan.len(),
            self.n_layers + self.n_mtp_layers,
        )?;
        for spec in plan.layers().iter().take(self.n_layers) {
            let layer =
                architecture
                    .layers
                    .get(spec.layer)
                    .ok_or_else(|| ConfigError::Invalid {
                        field: "layer plan rows",
                        detail: format!(
                            "the plan has row {} but the architecture has {} layers",
                            spec.layer,
                            architecture.layers.len()
                        ),
                    })?;
            let compressed = layer.attention.compression.is_some();
            let plan_compressed = !matches!(spec.mode, LayerMode::SwaOnly);
            check_flag("compressed layer", spec.layer, compressed, plan_compressed)?;
            check_flag(
                "indexer layer",
                spec.layer,
                layer.attention.indexer.is_some(),
                spec.owns_index,
            )?;
            let engram = layer
                .sides
                .iter()
                .any(|side| matches!(side, sparkflash_model::architecture::Side::Engram(_)));
            check_flag("Engram layer", spec.layer, engram, spec.engram)?;
        }
        Ok(())
    }
}

fn check_flag(
    field: &'static str,
    layer: usize,
    architecture: bool,
    plan: bool,
) -> Result<(), ConfigError> {
    if architecture == plan {
        Ok(())
    } else {
        Err(ConfigError::Invalid {
            field,
            detail: format!(
                "layer {layer}: the shared architecture says {architecture}, the layer plan says {plan}"
            ),
        })
    }
}

fn check(field: &'static str, typed: usize, architecture: usize) -> Result<(), ConfigError> {
    if typed == architecture {
        Ok(())
    } else {
        Err(ConfigError::Invalid {
            field,
            detail: format!(
                "the typed configuration says {typed}, the shared architecture says {architecture}"
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_carried_release_agrees_with_the_shared_architecture() {
        let config = ModelConfig::from_released_json(CARRIED_RELEASE).expect("typed config");
        let architecture = architecture_from_carried().expect("shared architecture");
        config
            .cross_check(&architecture)
            .expect("the two parsers agree");
        assert!(
            architecture
                .missing_on(sparkflash_model::Target::Sm121)
                .is_empty()
        );
        assert!(
            architecture
                .parts()
                .contains(sparkflash_model::Part::Csa2Compression)
        );
        assert!(
            architecture
                .parts()
                .contains(sparkflash_model::Part::Engram)
        );
    }

    #[test]
    fn the_layer_plan_agrees_with_the_shared_walk() {
        let config = ModelConfig::from_released_json(CARRIED_RELEASE).expect("typed config");
        let architecture = architecture_from_carried().expect("shared architecture");
        let plan = LayerPlan::from_config(&config).expect("layer plan");
        config
            .cross_check_plan(&plan, &architecture)
            .expect("the plan follows the shared architecture");
        assert_eq!(plan.len(), config.n_layers + config.n_mtp_layers);
    }

    #[test]
    fn a_plan_that_disagrees_names_the_row() {
        let config = ModelConfig::from_released_json(CARRIED_RELEASE).expect("typed config");
        let mut architecture = architecture_from_carried().expect("shared architecture");
        let plan = LayerPlan::from_config(&config).expect("layer plan");
        // Drop layer 2's compressor; the plan still says that row compresses.
        if let Some(layer) = architecture.layers.get_mut(2) {
            layer.attention.compression = None;
        }
        let error = config
            .cross_check_plan(&plan, &architecture)
            .expect_err("drift");
        assert!(error.to_string().contains("compressed layer"), "{error}");
    }

    #[test]
    fn a_shape_drift_names_the_field() {
        let config = ModelConfig::from_released_json(CARRIED_RELEASE).expect("typed config");
        let mut architecture = architecture_from_carried().expect("shared architecture");
        architecture.hidden_size += 1;
        let error = config.cross_check(&architecture).expect_err("drift");
        assert!(error.to_string().contains("hidden_size"), "{error}");
    }

    #[test]
    fn a_compression_drift_names_the_schedule() {
        let config = ModelConfig::from_released_json(CARRIED_RELEASE).expect("typed config");
        let mut architecture = architecture_from_carried().expect("shared architecture");
        if let Some(layer) = architecture.layers.get_mut(2) {
            layer.attention.compression = None;
        }
        let error = config.cross_check(&architecture).expect_err("drift");
        assert!(error.to_string().contains("compressed layers"), "{error}");
    }
}

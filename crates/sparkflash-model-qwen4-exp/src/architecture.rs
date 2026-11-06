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

//! The bridge between Qwen's typed configuration and the shared architecture.
//!
//! The typed [Qwen4ExpConfig] is the arm's validated profile; [sparkflash_model::Architecture]
//! is what the rest of the tree reads. Both come from the same released `config.json`, so every
//! load cross-checks them: a field that drifts between the two parsers is a refusal naming that
//! field, not two silently different models.

use sparkflash_model::architecture::{
    Activation, Attention, AttentionKind, Ffn, Gdn, HyperConnection, Indexer, Layer, Moe, Mtp,
    PleGeometry, Quantization, Residual, Rope, Score, Side,
};
use sparkflash_model::{Architecture, LayerStep};

use crate::config::{ConfigError, LayerKind, Qwen4ExpConfig};

const CARRIED_CONFIG: &str = include_str!("../config/qwen38_flash_next_nvfp4.hf.json");

/// Parse the shared architecture from a released Qwen3.8 `config.json`.
///
/// # Errors
/// As [Architecture::from_json].
pub fn architecture_from_json(text: &str) -> Result<Architecture, ConfigError> {
    Architecture::from_json(text)
}

/// The shared architecture of the released Qwen3.8 checkpoint carried in this crate.
///
/// # Errors
/// [ConfigError::Json] only if the carried document were edited into an unreadable shape.
pub fn architecture_from_carried() -> Result<Architecture, ConfigError> {
    Architecture::from_json(CARRIED_CONFIG)
}

/// The shared architecture a [`Qwen4ExpConfig::tiny`] profile describes.
///
/// It is constructed field by field rather than parsed: `tiny()` is the one configuration whose
/// document is derived *from* the typed struct, so parsing that document back would test the
/// renderer instead of pinning the schedule.
pub(crate) fn tiny_architecture() -> Architecture {
    let rope = Rope {
        theta: 10_000.0,
        // `tiny()` rotates half of its eight-wide head.
        rotated_dims: 4,
        scaling: None,
        mrope_section: Some([1, 1, 0]),
    };
    let ffn = || {
        Ffn::Moe(Moe {
            routed_experts: 8,
            active_experts: 2,
            intermediate: 8,
            shared_experts: 1,
            shared_intermediate: 8,
            shared_gated: true,
            score: Score::Softmax,
            routed_scaling: 1.0,
            norm_topk: true,
            selection_bias: false,
            clamp: None,
            activation: Activation::Silu,
        })
    };
    let residual = || {
        Residual::Hyper(HyperConnection {
            streams: 2,
            lowrank: Some(4),
            manifold: false,
            sinkhorn_iters: 0,
            epsilon: 0.0,
        })
    };
    Architecture {
        family: "qwen4_exp".to_owned(),
        architectures: vec!["Qwen4ExpForConditionalGeneration".to_owned()],
        hidden_size: 32,
        vocab_size: 64,
        max_position_embeddings: 64,
        norm_epsilon: 1e-6,
        embedding_tied_to_head: false,
        attention_bias: false,
        layers: vec![
            Layer {
                index: 0,
                attention: Attention {
                    kind: AttentionKind::Gdn(Gdn {
                        conv_kernel: 4,
                        key_heads: 2,
                        value_heads: 4,
                        key_dim: 4,
                        value_dim: 4,
                        output_gate: Activation::Sigmoid,
                    }),
                    heads: 4,
                    head_dim: 8,
                    window: None,
                    rope: rope.clone(),
                    compression: None,
                    indexer: None,
                },
                ffn: ffn(),
                residual: residual(),
                sides: vec![Side::Ple(PleGeometry {
                    embed_dim: 16,
                    conv_kernel: 2,
                    ngram_size: 3,
                    heads_per_ngram: 2,
                    vocab_base: 1_000,
                    seed: 1_234,
                    split_parts: 2,
                })],
            },
            Layer {
                index: 1,
                attention: Attention {
                    kind: AttentionKind::Gqa {
                        kv_heads: 2,
                        output_gate: true,
                    },
                    heads: 4,
                    head_dim: 8,
                    window: None,
                    rope,
                    compression: None,
                    indexer: Some(Indexer {
                        heads: 2,
                        head_dim: 4,
                        topk: 4,
                        owns_keys: true,
                        reuse: false,
                        pool: 2,
                        always_select_tail: false,
                        activation: Activation::Relu,
                        candidate_pool: None,
                    }),
                },
                ffn: ffn(),
                residual: residual(),
                sides: Vec::new(),
            },
        ],
        quantization: Quantization::default(),
        vision: None,
        mtp: Some(Mtp {
            stages: 1,
            block_size: None,
            target_layers: Vec::new(),
            routed_experts: None,
            active_experts: None,
        }),
    }
}

impl Qwen4ExpConfig {
    /// The decoder layers as the shared walk, in ascending index order.
    ///
    /// [Architecture::steps] reads only the typed architecture, so an arm that selects its
    /// device layers from this list cannot disagree with the checkpoint's `config.json` about
    /// which attention or feed-forward part a layer runs.
    #[must_use]
    pub fn steps(&self) -> Vec<LayerStep> {
        self.architecture.steps()
    }

    /// Refuse a typed configuration that disagrees with the shared architecture.
    ///
    /// # Errors
    /// [ConfigError::Invalid] naming the first field that differs.
    pub fn cross_check(&self, architecture: &Architecture) -> Result<(), ConfigError> {
        check("hidden_size", self.hidden_size, architecture.hidden_size)?;
        check("vocab_size", self.vocab_size, architecture.vocab_size)?;
        check(
            "max_position_embeddings",
            self.max_position_embeddings,
            architecture.max_position_embeddings,
        )?;
        check(
            "num_hidden_layers",
            self.num_hidden_layers,
            architecture.layers.len(),
        )?;

        let linear = self
            .layer_types
            .iter()
            .filter(|kind| matches!(kind, LayerKind::Linear))
            .count();
        let arch_linear = architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.attention.kind, AttentionKind::Gdn(_)))
            .count();
        check("linear attention layers", linear, arch_linear)?;
        let sparse = self
            .layer_types
            .iter()
            .filter(|kind| matches!(kind, LayerKind::Sparse))
            .count();
        let arch_sparse = architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.attention.kind, AttentionKind::Gqa { .. }))
            .count();
        check("sparse attention layers", sparse, arch_sparse)?;

        let dense = architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.ffn, Ffn::Dense(_)))
            .count();
        check("dense FFN layers", 0, dense)?;
        let moe = architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.ffn, Ffn::Moe(_)))
            .count();
        check("MoE layers", self.num_hidden_layers, moe)?;

        let ple = architecture
            .layers
            .iter()
            .filter(|layer| layer.sides.iter().any(|side| matches!(side, Side::Ple(_))))
            .count();
        check("PLE layers", self.ple_layer_ids.len(), ple)?;
        check(
            "mtp stages",
            self.mtp_num_hidden_layers,
            architecture.mtp.as_ref().map_or(0, |mtp| mtp.stages),
        )?;
        Ok(())
    }
}

fn check(field: &str, typed: usize, architecture: usize) -> Result<(), ConfigError> {
    if typed == architecture {
        Ok(())
    } else {
        Err(ConfigError::Invalid(format!(
            "{field}: the typed configuration says {typed}, the shared architecture says {architecture}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_model::Part;

    /// The arm's layer selection is the shared walk's, not a second reading of `layer_types`.
    #[test]
    fn every_layer_selects_its_part_from_the_shared_walk() {
        for (label, config) in [
            ("tiny", Qwen4ExpConfig::tiny()),
            (
                "released",
                Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses"),
            ),
        ] {
            let steps = config.steps();
            assert_eq!(steps.len(), config.num_hidden_layers, "{label}");
            assert_eq!(steps.len(), config.layer_types.len(), "{label}");
            assert_eq!(
                steps.iter().map(|step| step.index).collect::<Vec<_>>(),
                (0..config.num_hidden_layers).collect::<Vec<_>>(),
                "{label}: the walk is in ascending layer order",
            );
            for (step, kind) in steps.iter().zip(&config.layer_types) {
                let attention = match kind {
                    LayerKind::Linear => Part::GdnScan,
                    LayerKind::Sparse => Part::GqaAttention,
                };
                assert_eq!(step.attention, attention, "layer {}", step.index);
                assert_eq!(step.ffn, Part::Moe, "layer {}", step.index);
            }
        }
    }

    #[test]
    fn the_carried_release_agrees_with_the_shared_architecture() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("typed config");
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
                .contains(sparkflash_model::Part::GdnScan)
        );
        assert!(architecture.parts().contains(sparkflash_model::Part::Ple));
        let descriptor =
            crate::device::descriptor_for(&config, architecture.max_position_embeddings);
        assert_eq!(descriptor.vocab_size, architecture.vocab_size);
        assert_eq!(
            descriptor.context_tokens,
            architecture.max_position_embeddings
        );
    }

    #[test]
    fn a_shape_drift_names_the_field() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("typed config");
        let mut architecture = architecture_from_carried().expect("shared architecture");
        architecture.hidden_size += 1;
        let error = config.cross_check(&architecture).expect_err("drift");
        assert!(error.to_string().contains("hidden_size"), "{error}");
    }

    #[test]
    fn a_layer_kind_drift_names_the_schedule() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("typed config");
        let mut architecture = architecture_from_carried().expect("shared architecture");
        // Turn the first linear layer into a sparse one; the typed split no longer matches.
        if let Some(layer) = architecture.layers.first_mut() {
            layer.attention.kind = AttentionKind::Gqa {
                kv_heads: 1,
                output_gate: true,
            };
        }
        let error = config.cross_check(&architecture).expect_err("drift");
        assert!(error.to_string().contains("attention layers"), "{error}");
    }
}

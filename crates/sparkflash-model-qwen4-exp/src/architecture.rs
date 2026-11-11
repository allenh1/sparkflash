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
    PleGeometry, Quantization, Residual, Rope, ScaleDType, Score, Side, WeightFormat,
};
use sparkflash_model::{Architecture, LayerStep};

use crate::config::{ConfigError, LayerKind, Qwen4ExpConfig};

const CARRIED_CONFIG: &str = include_str!("../config/qwen38_flash_next_nvfp4.hf.json");

/// Parse the shared architecture from a released Qwen3.8 `config.json`.
///
/// The official FP8 pack writes one global `quant_method: "fp8"` while storing only the routed
/// experts as E4M3; every other tensor is BF16. [Architecture::experts_only_fp8] applies the
/// arm's producer correction after the shared parse, so the execution plan's dense/attention
/// class is BF16 and its expert class is FP8. Only this bridge and
/// `Qwen4ExpConfig::from_json_with_architecture` call it: the shared operation is Qwen's
/// correction, not a rule that every FP8-looking document should be rewritten. The NVFP4
/// document already parses that way, and the operation is a no-op there.
///
/// # Errors
/// As [Architecture::from_json].
pub fn architecture_from_json(text: &str) -> Result<Architecture, ConfigError> {
    let mut architecture = Architecture::from_json(text)?;
    // A text-only release names its root `model_type` qwen4_exp_text while the arm and its
    // plan are qwen4_exp. Normalize through the one registry helper so the typed profile and
    // the boot plan cannot disagree about the family they execute.
    architecture.family = sparkflash_model::canonical_model_type(&architecture.family).to_owned();
    correct_release_formats(&mut architecture);
    Ok(architecture)
}

/// The arm's correction of the declared formats, by family.
///
/// Only the Qwen3.8 FP8 release declares one global `quant_method: "fp8"` while storing its dense
/// projections BF16, so only that family's dense class is rewritten to BF16. A Qwen3.5/3.6 FP8
/// release stores the dense projections as the same 128x128 tiles its document declares and keeps
/// its BF16 modules in `modules_to_not_convert`, so its stored formats are read as declared. Its
/// `activation_scheme: "dynamic"` is not what the arm computes, though: the arm runs those tiles
/// weight-only (W8A16), so [Architecture::weight_only_fp8] clears the scheme.
pub(crate) fn correct_release_formats(architecture: &mut Architecture) {
    if architecture.family == crate::config::HYPER_FAMILY {
        architecture.experts_only_fp8();
    } else {
        architecture.weight_only_fp8();
    }
}

/// The shared architecture of the released Qwen3.8 checkpoint carried in this crate.
///
/// # Errors
/// [ConfigError::Json] only if the carried document were edited into an unreadable shape.
pub fn architecture_from_carried() -> Result<Architecture, ConfigError> {
    let mut architecture = architecture_from_json(CARRIED_CONFIG)?;
    // The Qwen pack stops on <|im_end|> as well as the config's <|endoftext|>; the released
    // config names only the latter. Apply the pack's stop rule here so a caller that reads the
    // architecture directly (the codec does) sees the same list the typed config does.
    if architecture.vocab_size > crate::config::IM_END_TOKEN as usize
        && !architecture
            .tokens
            .eos_token_ids
            .contains(&crate::config::IM_END_TOKEN)
    {
        architecture
            .tokens
            .eos_token_ids
            .push(crate::config::IM_END_TOKEN);
    }
    Ok(architecture)
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
        looped: None,
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
                    vocab_multiple: 8,
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
                        kv_heads: 1,
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
        // The fixture's experts are the NVFP4 release's, declared the way the Qwen schema names
        // them, so the shared plan of its rendered document carries their scales.
        quantization: Quantization {
            expert: Some("NVFP4".to_owned()),
            expert_format: WeightFormat::NvFp4 { group_size: 16 },
            // The release's MTP experts are FP8 on 128x128 tiles, declared FP8_PB_WO; the parser
            // reads that name's scale as F32, and the Qwen plan stores the BF16 the release does.
            mtp_expert_format: Some(WeightFormat::Fp8Tile {
                block_m: 128,
                block_n: 128,
                scale_dtype: ScaleDType::F32,
                dynamic_activations: true,
            }),
            ..Quantization::default()
        },
        vision: None,
        mtp: Some(Mtp {
            stages: 1,
            block_size: None,
            target_layers: Vec::new(),
            routed_experts: None,
            active_experts: None,
            // The TPC profile drafts with DFlash above TP1, like the released rank config.
            replicated: false,
        }),
        tokens: sparkflash_model::architecture::Tokens {
            bos_token_id: None,
            eos_token_ids: vec![2],
            pad_token_id: None,
            image_token_id: Some(3),
            video_token_id: None,
            vision_start_token_id: Some(4),
            vision_end_token_id: Some(5),
        },
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

        let rope = architecture
            .layers
            .first()
            .map(|layer| &layer.attention.rope)
            .ok_or_else(|| ConfigError::Invalid("the architecture has no layers".to_owned()))?;
        if (rope.theta - self.rope_theta).abs() > f64::EPSILON {
            return Err(ConfigError::Invalid(format!(
                "rope theta: the typed configuration says {}, the shared architecture says {}",
                self.rope_theta, rope.theta
            )));
        }
        check("rope rotated width", self.rotary_dim(), rope.rotated_dims)?;

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
        // The residual width and the indexer are read by both parsers; a typed default that
        // differs from the shared walk would build the other model's blocks.
        for layer in &architecture.layers {
            let streams = match &layer.residual {
                Residual::Plain => 1,
                Residual::Hyper(hyper) => hyper.streams,
            };
            check("residual streams", self.hc_count, streams)?;
            if matches!(layer.attention.kind, AttentionKind::Gqa { .. }) {
                check(
                    "indexer heads",
                    self.indexer_n_heads,
                    layer
                        .attention
                        .indexer
                        .as_ref()
                        .map_or(0, |indexer| indexer.heads),
                )?;
            }
        }
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
            crate::device::descriptor_for(&config, architecture.max_position_embeddings)
                .expect("a registered release");
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
    fn a_rope_drift_names_the_field() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("typed config");
        let mut architecture = architecture_from_carried().expect("shared architecture");
        architecture.layers[0].attention.rope.rotated_dims += 2;
        let error = config.cross_check(&architecture).expect_err("drift");
        assert!(error.to_string().contains("rope rotated width"), "{error}");
        let mut architecture = architecture_from_carried().expect("shared architecture");
        architecture.layers[0].attention.rope.theta = 1_000.0;
        let error = config.cross_check(&architecture).expect_err("drift");
        assert!(error.to_string().contains("rope theta"), "{error}");
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

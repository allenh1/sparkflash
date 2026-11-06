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

//! The shared canonical architecture behind the typed GLM-5.3 configuration.
//!
//! [crate::config::Glm5NextConfig] reads the released document into the fields this pack serves;
//! [sparkflash_model::Architecture] reads the same document into the shared vocabulary of parts.
//! Both are descriptions of one checkpoint, so they must agree. [Glm5NextConfig::cross_check]
//! compares the two and refuses the first field that disagrees, and
//! [crate::config::Glm5NextConfig::from_json] runs that check on every load.
//!
//! The bridge is configuration only: it reports which shared parts a GLM-5.3 checkpoint names and
//! leaves kernel selection to the shared part catalogue. A part that another arm already
//! implements is the same part here, not a GLM re-registration of it.

use sparkflash_model::ConfigError;
use sparkflash_model::architecture::{
    Activation, Architecture, Attention, AttentionKind, DenseMlp, Ffn, HyperConnection, Indexer,
    Kda, Layer, Moe, Mtp, Quantization, Residual, Rope, ScaleDType, Score, WeightFormat,
};

use crate::config::{Glm5NextConfig, IndexerKind, LayerKind, MODEL_TYPE, MlpKind};

const CARRIED_CONFIG: &str = include_str!("../config/glm53_flash.hf.json");

/// Parse a released Hugging Face `config.json` into the shared canonical [Architecture].
///
/// # Errors
/// [ConfigError::Json] for a malformed document and [ConfigError::Invalid] for a document the
/// shared parser cannot describe.
pub fn architecture_from_json(text: &str) -> Result<Architecture, ConfigError> {
    Architecture::from_json(text)
}

/// Parse the released `zai-org/GLM-5.3-Flash` configuration carried in this crate.
///
/// # Errors
/// As [architecture_from_json]; the carried document is exercised by this module's tests.
pub fn architecture_from_carried() -> Result<Architecture, ConfigError> {
    architecture_from_json(CARRIED_CONFIG)
}

/// The shared canonical architecture of [Glm5NextConfig::tiny], built directly.
///
/// The tiny fixture runs before any released document is parsed, so it must not come from
/// [Architecture::from_json]: this is the same four-layer schedule written as typed data. The
/// first layer is KDA with the dense MLP, the second and third are MLA (the third reusing the
/// second's selection), and the last is KDA again.
#[must_use]
pub fn tiny_architecture() -> Architecture {
    fn rope() -> Rope {
        Rope {
            theta: 1e6,
            rotated_dims: 0,
            scaling: None,
            mrope_section: None,
        }
    }
    fn residual() -> Residual {
        Residual::Hyper(HyperConnection {
            streams: 4,
            lowrank: None,
            manifold: true,
            sinkhorn_iters: 20,
            epsilon: 1e-6,
        })
    }
    fn moe() -> Ffn {
        Ffn::Moe(Moe {
            routed_experts: 8,
            active_experts: 2,
            intermediate: 32,
            shared_experts: 1,
            shared_intermediate: 32,
            shared_gated: false,
            score: Score::Sigmoid,
            routed_scaling: 2.5,
            norm_topk: true,
            selection_bias: true,
            clamp: Some(10.0),
            activation: Activation::Silu,
        })
    }
    fn kda(index: usize) -> Layer {
        Layer {
            index,
            attention: Attention {
                kind: AttentionKind::Kda(Kda {
                    conv_kernel: 4,
                    heads: 4,
                    head_dim: 8,
                    lower_bound: -5.0,
                }),
                heads: 4,
                head_dim: 8,
                window: None,
                rope: rope(),
                compression: None,
                indexer: None,
            },
            ffn: if index == 0 {
                Ffn::Dense(DenseMlp {
                    intermediate: 64,
                    activation: Activation::Silu,
                    clamp: Some(10.0),
                })
            } else {
                moe()
            },
            residual: residual(),
            sides: Vec::new(),
        }
    }
    fn mla(index: usize, indexer: Option<Indexer>) -> Layer {
        Layer {
            index,
            attention: Attention {
                kind: AttentionKind::Mla {
                    q_lora_rank: Some(16),
                    kv_lora_rank: 16,
                    v_head_dim: 8,
                    groups: None,
                },
                heads: 4,
                head_dim: 8,
                window: None,
                rope: rope(),
                compression: None,
                indexer,
            },
            ffn: moe(),
            residual: residual(),
            sides: Vec::new(),
        }
    }
    let full_indexer = Indexer {
        heads: 2,
        head_dim: 8,
        topk: 8,
        owns_keys: true,
        reuse: false,
        pool: 2,
        always_select_tail: true,
        activation: Activation::Relu,
        candidate_pool: None,
    };
    Architecture {
        family: MODEL_TYPE.to_owned(),
        architectures: vec!["Glm5NextForConditionalGeneration".to_owned()],
        hidden_size: 32,
        vocab_size: 64,
        max_position_embeddings: 128,
        norm_epsilon: 1e-5,
        embedding_tied_to_head: false,
        attention_bias: false,
        layers: vec![kda(0), mla(1, Some(full_indexer)), mla(2, None), kda(3)],
        quantization: Quantization {
            method: Some("fp8".to_owned()),
            block: Some(8),
            scale: None,
            activation: Some("dynamic".to_owned()),
            expert: None,
            dense_format: WeightFormat::Fp8Tile {
                block_m: 8,
                block_n: 8,
                scale_dtype: ScaleDType::F32,
                dynamic_activations: true,
            },
            // GLM names no expert format; its experts are the dense FP8 tiles.
            expert_format: WeightFormat::Fp8Tile {
                block_m: 8,
                block_n: 8,
                scale_dtype: ScaleDType::F32,
                dynamic_activations: true,
            },
            mtp_expert_format: None,
        },
        vision: None,
        mtp: Some(Mtp {
            stages: 1,
            block_size: None,
            target_layers: Vec::new(),
            routed_experts: Some(8),
            active_experts: Some(2),
            replicated: true,
        }),
        tokens: sparkflash_model::architecture::Tokens::default(),
    }
}

impl Glm5NextConfig {
    /// Check that this typed configuration and the shared canonical architecture describe the
    /// same checkpoint.
    ///
    /// The fields compared are the ones both representations own: the model's width, vocabulary,
    /// window and layer count; the attention kind of every decoder layer against the
    /// architecture's KDA and MLA counts; the dense and MoE layer counts; and the number of
    /// layers that run their own indexer. A disagreement is refused with the first field named,
    /// so a released document that changes shape cannot load while only one of the two views
    /// follows it.
    ///
    /// # Errors
    /// [ConfigError::Invalid] naming the first field whose values disagree.
    pub fn cross_check(&self, architecture: &Architecture) -> Result<(), ConfigError> {
        if architecture.family != MODEL_TYPE {
            return Err(ConfigError::Invalid(format!(
                "family = {:?} in the shared architecture, expected {MODEL_TYPE:?}",
                architecture.family
            )));
        }
        same("hidden_size", self.hidden_size, architecture.hidden_size)?;
        same("vocab_size", self.vocab_size, architecture.vocab_size)?;
        same(
            "max_position_embeddings",
            self.max_position_embeddings,
            architecture.max_position_embeddings,
        )?;
        same(
            "num_hidden_layers",
            self.num_hidden_layers,
            architecture.layers.len(),
        )?;

        let kda = architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.attention.kind, AttentionKind::Kda(_)))
            .count();
        same("layer_types (linear layers)", self.linear_layers(), kda)?;
        let mla = architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.attention.kind, AttentionKind::Mla { .. }))
            .count();
        same("layer_types (sparse layers)", self.sparse_layers(), mla)?;

        let dense = architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.ffn, Ffn::Dense(_)))
            .count();
        let dense_configured = self
            .mlp_layer_types
            .iter()
            .filter(|kind| **kind == MlpKind::Dense)
            .count();
        same("mlp_layer_types (dense layers)", dense_configured, dense)?;
        let moe = architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.ffn, Ffn::Moe(_)))
            .count();
        same("mlp_layer_types (MoE layers)", self.moe_layers(), moe)?;

        let indexers_configured = self
            .layer_types
            .iter()
            .zip(&self.indexer_types)
            .filter(|(layer, indexer)| {
                **layer == LayerKind::Sparse && **indexer == IndexerKind::Full
            })
            .count();
        same(
            "indexer_layers",
            indexers_configured,
            architecture.indexer_layers().len(),
        )?;
        Ok(())
    }
}

fn same(field: &str, configured: usize, canonical: usize) -> Result<(), ConfigError> {
    if configured == canonical {
        return Ok(());
    }
    Err(ConfigError::Invalid(format!(
        "{field} = {configured} in the typed configuration but {canonical} in the shared \
         architecture"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_model::parts::{Part, Target};

    #[test]
    fn the_carried_release_parses_both_ways_and_cross_checks() {
        let config = Glm5NextConfig::glm53_flash().expect("the carried typed config parses");
        let architecture = architecture_from_carried().expect("the carried architecture parses");
        assert_eq!(
            architecture,
            architecture_from_json(CARRIED_CONFIG).expect("same text")
        );
        assert_eq!(architecture.family, MODEL_TYPE);
        config
            .cross_check(&architecture)
            .expect("the carried views agree");
        let descriptor =
            crate::serve::descriptor_for(&config, architecture.max_position_embeddings);
        assert_eq!(descriptor.vocab_size, architecture.vocab_size);
        assert_eq!(
            descriptor.context_tokens,
            architecture.max_position_embeddings
        );
    }

    #[test]
    fn a_hidden_width_disagreement_names_the_field() {
        let config = Glm5NextConfig::glm53_flash().expect("the carried config parses");
        let mut architecture =
            architecture_from_carried().expect("the carried architecture parses");
        architecture.hidden_size += 1;
        let error = config
            .cross_check(&architecture)
            .expect_err("the width disagrees");
        let message = error.to_string();
        assert!(
            message.contains("hidden_size"),
            "the refusal names hidden_size: {message}"
        );
        assert!(matches!(error, ConfigError::Invalid(_)));
    }

    #[test]
    fn a_layer_kind_split_disagreement_names_the_field() {
        let config = Glm5NextConfig::glm53_flash().expect("the carried config parses");
        let mut architecture =
            architecture_from_carried().expect("the carried architecture parses");
        architecture.layers[0].attention.kind = AttentionKind::Mla {
            q_lora_rank: Some(config.q_lora_rank),
            kv_lora_rank: config.kv_lora_rank,
            v_head_dim: config.v_head_dim,
            groups: None,
        };
        let error = config
            .cross_check(&architecture)
            .expect_err("the KDA count disagrees");
        let message = error.to_string();
        assert!(
            message.contains("layer_types"),
            "the refusal names layer_types: {message}"
        );
    }

    #[test]
    fn an_indexer_schedule_disagreement_is_refused_at_load() {
        let mut root: serde_json::Value =
            serde_json::from_str(CARRIED_CONFIG).expect("the carried config is JSON");
        root["text_config"]["index_source_layer_ids"] = serde_json::json!([3]);
        let error = Glm5NextConfig::from_json(&root.to_string())
            .expect_err("the typed view runs every sparse layer's indexer");
        let message = error.to_string();
        assert!(
            message.contains("indexer_layers"),
            "the load refusal names indexer_layers: {message}"
        );
    }

    #[test]
    fn the_architecture_names_the_shared_parts_glm_needs() {
        let architecture = architecture_from_carried().expect("the carried architecture parses");
        let parts = architecture.parts();
        for part in [
            Part::KdaScan,
            Part::MlaAttention,
            Part::SparseIndexer,
            Part::DenseMlp,
            Part::Moe,
            Part::ManifoldHyperConnection,
            Part::Mtp,
        ] {
            assert!(parts.contains(part), "GLM needs {part}");
        }
        assert!(
            architecture.missing_on(Target::Sm121).is_empty(),
            "the shared catalogue carries every part: {:?}",
            architecture.missing_on(Target::Sm121)
        );
    }
}

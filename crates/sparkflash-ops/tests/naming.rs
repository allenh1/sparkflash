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

//! The shared naming of a family no pack transcribes, and the tensor plan it drives.

use std::collections::BTreeMap;

use sparkflash_model::{Architecture, TpPlan};
use sparkflash_ops::naming::{self, AttentionNames, NamingDialect};
use sparkflash_ops::tensor_plan::{ArchitectureTensorPlan, Role};
use sparkflash_weights::{DType, TensorSpec, WeightError, WeightSource};

/// A spec-only source.
struct Specs(BTreeMap<String, TensorSpec>);

impl Specs {
    fn of(specs: impl IntoIterator<Item = TensorSpec>) -> Self {
        Self(
            specs
                .into_iter()
                .map(|spec| (spec.name.clone(), spec))
                .collect(),
        )
    }
}

impl WeightSource for Specs {
    fn names(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.0
            .get(name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_owned(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        Ok(vec![0; self.spec(name)?.storage_bytes() as usize])
    }
}

fn llama() -> Architecture {
    Architecture::from_json(
        r#"{
            "model_type": "llama",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 4096,
            "rope_theta": 500000.0,
            "eos_token_id": 2
        }"#,
    )
    .expect("a Llama-shaped config")
}

/// The dynamic host path's fixture family: GDN + MoE, GQA + dense, GQA + MoE.
fn hybrid(tied: bool) -> Architecture {
    let document = serde_json::json!({
        "model_type": "acme_custom_flash",
        "hidden_size": 32,
        "intermediate_size": 16,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "tie_word_embeddings": tied,
        "eos_token_id": [0, 1],
        "layer_types": ["linear_attention", "full_attention", "full_attention"],
        "mlp_layer_types": ["sparse", "dense", "sparse"],
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": true,
        "moe_intermediate_size": 16,
        "n_shared_experts": 1,
        "shared_expert_intermediate_size": 8
    });
    Architecture::from_json(&document.to_string()).expect("the hybrid fixture")
}

#[test]
fn a_llama_document_plans_the_canonical_backbone() {
    let architecture = llama();
    let plan = ArchitectureTensorPlan::served(&architecture, TpPlan::new(1, 0).expect("one rank"))
        .expect("a family no pack names still plans");
    // Embedding, per layer two norms + four attention + three MLP projections, final norm, head.
    assert_eq!(plan.len(), 1 + 2 * 9 + 2);
    let names: Vec<&str> = plan.names().collect();
    assert_eq!(names[0], "model.embed_tokens.weight");
    assert_eq!(names[1], "model.layers.0.input_layernorm.weight");
    assert_eq!(names[2], "model.layers.0.self_attn.q_proj.weight");
    assert_eq!(names.last(), Some(&"lm_head.weight"));
    let q = plan
        .spec("model.layers.1.self_attn.q_proj.weight")
        .expect("the query projection");
    assert_eq!(
        q.shape,
        [256, 256],
        "an ungated query is one attention width"
    );
    assert_eq!(
        plan.spec("model.layers.1.self_attn.k_proj.weight")
            .expect("the key projection")
            .shape,
        [64, 256]
    );
    let bf16 = |elements: usize| 2 * elements as u128;
    let per_layer = 2 * 256 + 2 * 256 * 256 + 2 * 64 * 256 + 3 * 512 * 256;
    assert_eq!(
        plan.storage_bytes(),
        bf16(2 * 1000 * 256 + 2 * per_layer + 256)
    );
    assert_eq!(plan.role_bytes(Role::DenseMlp), bf16(2 * 3 * 512 * 256));
}

#[test]
fn a_two_rank_plan_splits_heads_and_intermediate() {
    let architecture = llama();
    let plan = ArchitectureTensorPlan::served(&architecture, TpPlan::new(2, 1).expect("rank 1"))
        .expect("the TP2 plan");
    let rank = |name: &str| plan.rank_spec(name).expect(name).shape.clone();
    assert_eq!(rank("model.layers.0.self_attn.q_proj.weight"), [128, 256]);
    assert_eq!(rank("model.layers.0.self_attn.k_proj.weight"), [32, 256]);
    assert_eq!(rank("model.layers.0.self_attn.o_proj.weight"), [256, 128]);
    assert_eq!(rank("model.layers.0.mlp.down_proj.weight"), [256, 256]);
    assert_eq!(rank("model.embed_tokens.weight"), [1000, 256]);
    let run = plan
        .rank_run("model.layers.0.mlp.gate_proj.weight")
        .expect("a row split");
    assert_eq!((run.offset, run.count, run.total), (256, 256, 512));
}

/// The canonical plan of `architecture`, respelled in another dialect: the `language_model`
/// root, `ffn_norm`, a `router`, per-head norms on every GQA layer, and no head when tied.
fn respelled(architecture: &Architecture) -> Specs {
    let canonical =
        ArchitectureTensorPlan::served(architecture, TpPlan::new(1, 0).expect("one rank"))
            .expect("the canonical plan");
    let mut specs: Vec<TensorSpec> = canonical
        .specs()
        .iter()
        .map(|spec| TensorSpec {
            name: spec
                .name
                .replace("model.layers.", "model.language_model.layers.")
                .replace("post_attention_layernorm", "ffn_norm")
                .replace("mlp.gate.weight", "mlp.router.weight"),
            ..spec.clone()
        })
        .collect();
    for layer in &architecture.layers {
        if matches!(
            layer.attention.kind,
            sparkflash_model::architecture::AttentionKind::Gqa { .. }
        ) {
            for norm in ["q_norm", "k_norm"] {
                specs.push(TensorSpec {
                    name: format!(
                        "model.language_model.layers.{}.self_attn.{norm}.weight",
                        layer.index
                    ),
                    dtype: DType::BF16,
                    shape: vec![layer.attention.head_dim],
                });
            }
        }
    }
    Specs::of(specs)
}

#[test]
fn a_probed_dialect_plans_exactly_the_checkpoint() {
    for tied in [false, true] {
        let architecture = hybrid(tied);
        let source = respelled(&architecture);
        let names = naming::resolve(&architecture, &source).expect("every tensor resolves");
        assert_eq!(names.head_tied(), tied);
        let AttentionNames::Gqa { q_norm, .. } = &names.layers[1].attention else {
            panic!("layer 1 is grouped-query attention");
        };
        assert_eq!(
            q_norm.as_deref(),
            Some("model.language_model.layers.1.self_attn.q_norm.weight")
        );
        let plan = ArchitectureTensorPlan::named(
            &architecture,
            TpPlan::new(1, 0).expect("one rank"),
            &names,
        )
        .expect("the probed plan");
        let planned: BTreeMap<String, TensorSpec> = plan
            .specs()
            .iter()
            .map(|spec| (spec.name.clone(), spec.clone()))
            .collect();
        assert_eq!(planned, source.0, "tied={tied}: the plan is the checkpoint");
        let walked: Vec<String> = plan.names().map(str::to_owned).collect();
        assert_eq!(walked, names.walk(), "the plan is in forward order");
    }
}

#[test]
fn a_missing_tensor_is_named() {
    let architecture = hybrid(false);
    let mut source = respelled(&architecture);
    let gone = "model.language_model.layers.2.mlp.experts.3.down_proj.weight";
    source.0.remove(gone);
    let error = naming::resolve(&architecture, &source).expect_err("one expert is missing");
    assert!(error.contains(gone), "{error}");
}

#[test]
fn a_family_the_convention_does_not_spell_is_refused() {
    let mut document: serde_json::Value = serde_json::from_str(
        r#"{"model_type": "acme_latent", "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 4, "vocab_size": 32,
            "kv_lora_rank": 16}"#,
    )
    .expect("json");
    document["max_position_embeddings"] = serde_json::json!(128);
    let architecture = Architecture::from_json(&document.to_string()).expect("an MLA document");
    let error = NamingDialect::canonical(&architecture)
        .names(&architecture)
        .expect_err("MLA has no shared spelling");
    assert!(error.contains("shared naming"), "{error}");
    assert!(
        ArchitectureTensorPlan::served(&architecture, TpPlan::new(1, 0).expect("one rank"))
            .is_err()
    );
}

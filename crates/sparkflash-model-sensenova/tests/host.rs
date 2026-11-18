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

//! The host reference runs the released plan end to end on a tiny seeded checkpoint.

use std::path::Path;

use serde_json::Value;
use sparkflash_model::ModelDescriptor;
use sparkflash_weights::quant::f32_to_bf16_bits;
use sparkflash_weights::rng::TensorRng;
use sparkflash_weights::safetensors::{open_checkpoint, write_checkpoint};

use sparkflash_model_sensenova::config::SenseNovaConfig;
use sparkflash_model_sensenova::forward::SenseNovaWeights;
use sparkflash_model_sensenova::tensor::SenseNovaTensorPlan;

const CARRIED: &str = include_str!("../config/sensenova_loop_9b.hf.json");

fn tiny_config() -> String {
    let mut root: Value = serde_json::from_str(CARRIED).expect("carried config");
    root.as_object_mut()
        .expect("object")
        .remove("quantization_config");
    root.as_object_mut()
        .expect("object")
        .remove("vision_config");
    root["loop_settings"] =
        serde_json::json!({"start": 1, "end": 5, "counts": [2], "eval_count": 2});
    root["mhc"] = serde_json::json!({
        "start": 1, "end": 5, "streams": 2, "iterations": 20,
        "fixed_identity": false, "pre_scale": 1.0
    });
    let text = root["text_config"].as_object_mut().expect("text config");
    text["hidden_size"] = 64.into();
    text["intermediate_size"] = 128.into();
    text["vocab_size"] = 128.into();
    text["num_hidden_layers"] = 8.into();
    text["num_attention_heads"] = 4.into();
    text["num_key_value_heads"] = 2.into();
    text["head_dim"] = 16.into();
    text["max_position_embeddings"] = 512.into();
    text["layer_types"] = serde_json::json!([
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention"
    ]);
    text["linear_num_key_heads"] = 2.into();
    text["linear_num_value_heads"] = 4.into();
    text["linear_key_head_dim"] = 16.into();
    text["linear_value_head_dim"] = 16.into();
    text["partial_rotary_factor"] = 0.25.into();
    text["rope_parameters"] = serde_json::json!({
        "rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25,
        "mrope_interleaved": true, "mrope_section": [11, 11, 10],
        "rope_type": "default", "factor": 1.0, "original_max_position_embeddings": 262144
    });
    serde_json::to_string(&root).expect("json")
}

fn write_tiny(directory: &Path) -> SenseNovaConfig {
    let json = tiny_config();
    let config = SenseNovaConfig::from_json(&json).expect("tiny config");
    let plan = SenseNovaTensorPlan::served(&config);
    let mut tensors = Vec::with_capacity(plan.len());
    for spec in plan.specs() {
        let mut rng = TensorRng::new(7, &spec.name);
        let elements = spec.elements() as usize;
        let bytes = match spec.dtype {
            sparkflash_weights::DType::BF16 => (0..elements)
                .flat_map(|_| f32_to_bf16_bits(rng.signed() * 0.05).to_le_bytes())
                .collect(),
            sparkflash_weights::DType::F32 => (0..elements)
                .flat_map(|_| (rng.signed() * 0.05).to_le_bytes())
                .collect(),
            other => panic!("unexpected fixture dtype {other:?}"),
        };
        tensors.push((spec.clone(), bytes));
    }
    write_checkpoint(directory, &tensors, 1).expect("fixture writes");
    std::fs::write(directory.join("config.json"), json).expect("config writes");
    config
}

#[test]
fn the_host_reference_runs_the_looped_plan() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let config = write_tiny(directory.path());
    assert_eq!(config.visits(), 12);
    assert_eq!(config.full_attention_visits(), 6);
    let source = open_checkpoint(directory.path()).expect("checkpoint opens");
    let weights = SenseNovaWeights::load(&config, &source).expect("weights load");
    let descriptor = ModelDescriptor::sensenova_loop_9b();
    let backend =
        sparkflash_model_sensenova::SenseNovaHostBackend::open(descriptor, config, &source, 256)
            .expect("backend opens");
    let logits = weights
        .forward_prompt(&[1, 2, 3, 4])
        .expect("prompt forwards");
    assert_eq!(logits.len(), 128);
    assert!(logits.iter().all(|value| value.is_finite()));
    // The same prompt through the serving session returns a token.
    use sparkflash_model::backend::ModelBackend;
    let mut session = backend.open_session().expect("session");
    let step = session.prefill(&[1, 2, 3, 4]).expect("prefill");
    assert!(step.next_token.is_some());
}

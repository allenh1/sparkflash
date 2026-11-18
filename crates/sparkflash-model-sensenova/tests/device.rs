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

//! The device arm's first parity check: one token on the tiny fixture against the host.

use std::path::Path;

use serde_json::Value;
use sparkflash_model::ModelDescriptor;
use sparkflash_weights::quant::f32_to_bf16_bits;
use sparkflash_weights::rng::TensorRng;
use sparkflash_weights::safetensors::{open_checkpoint, write_checkpoint};

use sparkflash_model_sensenova::SenseNovaConfig;
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
fn the_device_arm_matches_the_host_on_one_token() {
    eprintln!(
        "images: sensenova={} qwen={} attn={} scan={}",
        sparkflash_kernels::images::Image::SenseNova.available(),
        sparkflash_kernels::images::Image::Qwen4Exp.available(),
        sparkflash_kernels::images::Image::Qwen4Attn.available(),
        sparkflash_kernels::images::Image::Qwen4Scan.available(),
    );
    if !sparkflash_model_sensenova::SenseNovaDeviceBackend::available() {
        eprintln!("no SenseNova/Qwen cubin; skipping the device parity test");
        return;
    }
    let directory = tempfile::tempdir().expect("temporary directory");
    let config = write_tiny(directory.path());
    let source = open_checkpoint(directory.path()).expect("checkpoint opens");
    let weights = SenseNovaWeights::load(&config, &source).expect("host weights");
    let host = weights.forward_prompt(&[1, 2, 3]).expect("host token");
    use sparkflash_model::backend::ModelBackend;
    // The BF16 cache is the exact-arithmetic path: nothing but the kernels' own rounding
    // separates the two. The 8-bit cache adds the E4M3 round of every cached key and value,
    // which moves the logits by about a percent, so it gets its own bound and an argmax claim.
    for (kv_fp8, bound) in [(false, 5e-3f32), (true, 2e-2f32)] {
        let backend = sparkflash_model_sensenova::SenseNovaDeviceBackend::open_with_cache(
            directory.path(),
            ModelDescriptor::sensenova_loop_9b(),
            256,
            kv_fp8,
        );
        let Ok(backend) = backend else {
            eprintln!("the device arm refused the fixture: {:?}", backend.err());
            return;
        };
        let mut session = backend.open_session().expect("device session");
        let step = session.prefill(&[1, 2, 3]).expect("device prefill");
        assert_eq!(step.logits.len(), host.len());
        let worst = step
            .logits
            .iter()
            .zip(&host)
            .map(|(device, host)| (device - host).abs())
            .fold(0.0f32, f32::max);
        let scale = host.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            worst < bound,
            "kv_fp8={kv_fp8}: device and host diverge by {worst} (host max {scale}); \
             host[0..4]={:?} device[0..4]={:?}",
            &host[..4],
            &step.logits[..4]
        );
        if kv_fp8 {
            let argmax = |values: &[f32]| {
                values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(index, _)| index)
            };
            assert_eq!(
                argmax(&step.logits),
                argmax(&host),
                "the 8-bit cache must not move the first token"
            );
        }
    }
}

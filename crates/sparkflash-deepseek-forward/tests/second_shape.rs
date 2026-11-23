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

//! The second shape opens from its own description at four ranks.
//!
//! The fixture is a DeepSeek-shaped configuration whose width, heads, experts, top-k, layer
//! count, compression ratios, Engram layers and drafter taps all differ from the released model.
//! This test checks the two halves the four-rank run needs: the forward's plan resolves every
//! part, and every tensor the plan binds is one the checkpoint writer names.

use std::path::PathBuf;

use sparkflash_model_deepseek_v41::config::ModelConfig;
use sparkflash_model_deepseek_v41::tensor::TensorPlan;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/second-shape")
}

#[test]
fn every_tensor_the_second_shape_binds_is_one_the_writer_names() {
    let dir = fixture();
    let text = std::fs::read_to_string(dir.join("config.json")).expect("the fixture config");
    let config = ModelConfig::from_json(&text).expect("the fixture parses");
    let written = TensorPlan::language_model(&config);
    for rank in 0..4 {
        let plan = sparkflash_deepseek_forward::plan::plan_from_path(&dir, 4, rank)
            .unwrap_or_else(|error| panic!("rank {rank}: {error}"));
        for tensor in plan.tensors() {
            assert!(
                written.contains(&tensor.name),
                "rank {rank} binds {} which the writer does not name",
                tensor.name
            );
        }
    }
}

#[test]
fn the_second_shape_values_reach_the_forward() {
    let dir = fixture();
    let text = std::fs::read_to_string(dir.join("config.json")).expect("the fixture config");
    let config = ModelConfig::from_json(&text).expect("the fixture parses");
    let plan = sparkflash_deepseek_forward::plan::plan_from_path(&dir, 4, 0).expect("the plan");
    let values = sparkflash_deepseek_forward::plan::values_from_plan(&plan).expect("the values");
    assert_eq!(values.dim, config.dim);
    assert_eq!(values.n_heads, config.n_heads);
    assert_eq!(values.n_layers, config.n_layers);
    assert_eq!(values.n_routed_experts, config.n_routed_experts);
    assert_eq!(values.n_activated_experts, config.n_activated_experts);
    // The plan gives the MTP stages no compressed cache of their own; the backbone pattern is
    // the checkpoint's.
    assert_eq!(
        values.compress_ratios[..config.n_layers],
        config.compress_ratios[..config.n_layers]
    );
    assert_eq!(values.engram_layer_ids, config.engram_layer_ids);
    assert_eq!(
        values.dspark_target_layer_ids,
        config.dspark_target_layer_ids
    );
    assert_ne!(values.dim, 5120, "the fixture is a second shape");
}

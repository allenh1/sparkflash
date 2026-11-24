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

//! The shared expert's stored format is stated where its tensors are claimed.
//!
//! A DeepSeek release is all-MoE: every layer's FFN is a MoE block with one shared expert, and
//! the shared expert's `ffn.shared_experts.*` tensors are claimed by the [Part::Moe] part. The
//! plan must still state their dense-class format, because the forward's shared-expert gather
//! refuses when the plan states none rather than assuming the dense or the routed class.

use sparkflash_model::Part;
use sparkflash_model::Target;
use sparkflash_model::format::{FormatValues, ScaleValues};
use sparkflash_ops::EnginePlan;

const RELEASED: &str =
    include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");
const SECOND: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sparkflash-deepseek-forward/tests/fixtures/second-shape/config.json"
);

fn second_shape() -> String {
    std::fs::read_to_string(SECOND).expect("the second-shape fixture config")
}

/// The released MXFP8 arrangement: E4M3 values, one ue8m0 scale a 32 x 32 block.
fn assert_shared_format(plan: &EnginePlan, name: &str, layers: impl IntoIterator<Item = usize>) {
    for layer in layers {
        let format = plan
            .shared_expert_format(layer)
            .unwrap_or_else(|| panic!("{name}: layer {layer} states no shared-expert format"));
        assert_eq!(
            format.values(),
            FormatValues::Fp8E4M3,
            "{name}: layer {layer} values"
        );
        assert_eq!(format.block(), 32, "{name}: layer {layer} scale block");
        assert_eq!(
            format.scale_values(),
            ScaleValues::Ue8m0,
            "{name}: layer {layer} scale values"
        );
    }
}

#[test]
fn the_released_all_moe_plan_states_the_shared_expert_format() {
    let plan = EnginePlan::from_json(RELEASED, Target::Sm121, 4, 0).expect("the released plan");
    assert_shared_format(&plan, "released", 0..plan.architecture().layers.len());
}

#[test]
fn the_second_shape_states_the_shared_expert_format_at_every_rank() {
    let text = second_shape();
    for rank in 0..4 {
        let plan = EnginePlan::from_json(&text, Target::Sm121, 4, rank)
            .unwrap_or_else(|error| panic!("rank {rank}: {error}"));
        assert_shared_format(
            &plan,
            &format!("second shape rank {rank}"),
            0..plan.architecture().layers.len(),
        );
        // The shared expert is the dense class while the routed experts beside it are FP4: the
        // MoE part's aggregate cannot state the shared format, which is the defect this pins.
        for layer in 0..plan.architecture().layers.len() {
            let moe = plan.stored_format(Part::Moe, Some(layer));
            assert_ne!(
                Some(plan.shared_expert_format(layer).expect("a shared expert")),
                moe,
                "layer {layer}: the routed-expert aggregate is not the shared expert's format"
            );
        }
    }
}

#[test]
fn a_moe_layer_with_no_shared_expert_refuses() {
    let mut value: serde_json::Value = serde_json::from_str(&second_shape()).expect("json");
    value["text_config"]["n_shared_experts"] = serde_json::json!(0);
    let plan = EnginePlan::from_json(&value.to_string(), Target::Sm121, 4, 0)
        .expect("the no-shared-expert plan");
    for layer in 0..plan.architecture().layers.len() {
        assert!(
            plan.shared_expert_format(layer).is_none(),
            "layer {layer} states a shared-expert format without a shared expert"
        );
    }
}

#[test]
fn a_dense_layer_with_no_shared_expert_refuses_beside_a_moe_layer_that_resolves() {
    let mut value: serde_json::Value = serde_json::from_str(&second_shape()).expect("json");
    value["text_config"]["first_k_dense_replace"] = serde_json::json!(1);
    value["text_config"]["intermediate_size"] = serde_json::json!(512);
    let plan = EnginePlan::from_json(&value.to_string(), Target::Sm121, 4, 0)
        .expect("the dense-first plan");
    assert!(
        plan.shared_expert_format(0).is_none(),
        "a dense layer has no shared expert to state"
    );
    // Layer 0 is dense; every later layer is the same all-MoE block the released plan has.
    assert_shared_format(&plan, "dense-first", 1..plan.architecture().layers.len());
}

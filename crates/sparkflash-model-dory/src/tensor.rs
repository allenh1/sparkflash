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

//! The tensor plan of the released Dory checkpoint.
//!
//! The released snapshot is one 7.95 GB BF16 shard with 507 tensor names: the embedding, the
//! LM head and its per-vocabulary logit scale, then one attention body or one MLP body for each
//! of the 72 stored sublayers. The recurrent sublayers are stored once; looping does not
//! duplicate a name.
//!
//! The plan is derived from the configuration rather than transcribed from the index, so the
//! count and byte total below are a real check on the release.

use sparkflash_weights::{DType, TensorPlan, TensorSpec};

use crate::config::DoryConfig;

/// The released checkpoint's storage total, from the safetensors header.
pub const RELEASED_BYTES: u128 = 7_950_458_880;

/// The released checkpoint's tensor count.
pub const RELEASED_TENSORS: usize = 507;

/// The prefix every backbone tensor carries in the released checkpoint.
pub const BACKBONE_PREFIX: &str = "backbone";

/// What part of the model a tensor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// The token embedding table.
    Embedding,
    /// The LM head and its logit scale.
    LmHead,
    /// An attention sublayer.
    Attention,
    /// An MLP sublayer.
    Mlp,
}

/// The tensor plan for one Dory configuration.
#[derive(Debug, Clone)]
pub struct DoryTensorPlan {
    plan: TensorPlan<Role>,
}

impl std::ops::Deref for DoryTensorPlan {
    type Target = TensorPlan<Role>;

    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}

impl DoryTensorPlan {
    /// Plan the whole released checkpoint.
    #[must_use]
    pub fn served(config: &DoryConfig) -> Self {
        let mut plan = TensorPlan::default();
        let hidden = config.hidden_size;
        let query_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        plan.push(
            &format!("{BACKBONE_PREFIX}.embeddings.weight"),
            DType::BF16,
            vec![config.vocab_size, hidden],
            Role::Embedding,
        );
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("{BACKBONE_PREFIX}.layers.{layer}");
            if config.layer_types[layer] == "mlp" {
                plan.push(
                    &format!("{prefix}.mlp_alpha"),
                    DType::BF16,
                    vec![hidden],
                    Role::Mlp,
                );
                plan.push(
                    &format!("{prefix}.mixer.gate_proj.weight"),
                    DType::BF16,
                    vec![intermediate, hidden],
                    Role::Mlp,
                );
                plan.push(
                    &format!("{prefix}.mixer.up_proj.weight"),
                    DType::BF16,
                    vec![intermediate, hidden],
                    Role::Mlp,
                );
                plan.push(
                    &format!("{prefix}.mixer.down_proj.weight"),
                    DType::BF16,
                    vec![hidden, intermediate],
                    Role::Mlp,
                );
                plan.push(
                    &format!("{prefix}.mixer.suv_gate"),
                    DType::BF16,
                    vec![intermediate],
                    Role::Mlp,
                );
                plan.push(
                    &format!("{prefix}.mixer.suv_up"),
                    DType::BF16,
                    vec![intermediate],
                    Role::Mlp,
                );
            } else {
                plan.push(
                    &format!("{prefix}.attn_alpha"),
                    DType::BF16,
                    vec![hidden],
                    Role::Attention,
                );
                plan.push(
                    &format!("{prefix}.mixer.q_proj.weight"),
                    DType::BF16,
                    vec![query_width, hidden],
                    Role::Attention,
                );
                plan.push(
                    &format!("{prefix}.mixer.k_proj.weight"),
                    DType::BF16,
                    vec![kv_width, hidden],
                    Role::Attention,
                );
                plan.push(
                    &format!("{prefix}.mixer.v_proj.weight"),
                    DType::BF16,
                    vec![kv_width, hidden],
                    Role::Attention,
                );
                plan.push(
                    &format!("{prefix}.mixer.g_proj.weight"),
                    DType::BF16,
                    vec![query_width, hidden],
                    Role::Attention,
                );
                plan.push(
                    &format!("{prefix}.mixer.o_proj.weight"),
                    DType::BF16,
                    vec![hidden, query_width],
                    Role::Attention,
                );
                plan.push(
                    &format!("{prefix}.mixer.sqk"),
                    DType::BF16,
                    vec![query_width],
                    Role::Attention,
                );
                plan.push(
                    &format!("{prefix}.mixer.s_gate"),
                    DType::BF16,
                    vec![query_width],
                    Role::Attention,
                );
            }
        }
        plan.push(
            "lm_head.weight",
            DType::BF16,
            vec![config.vocab_size, hidden],
            Role::LmHead,
        );
        plan.push(
            "lm_head.logit_scale",
            DType::BF16,
            vec![config.vocab_size],
            Role::LmHead,
        );
        Self { plan }
    }

    /// The spec of one tensor.
    #[must_use]
    pub fn spec(&self, name: &str) -> Option<&TensorSpec> {
        self.plan.spec(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASED: &str = include_str!("../config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json");

    #[test]
    fn the_plan_matches_the_released_header() {
        let config = DoryConfig::from_json(RELEASED).expect("the released configuration parses");
        let plan = DoryTensorPlan::served(&config);
        assert_eq!(plan.len(), RELEASED_TENSORS);
        assert_eq!(plan.storage_bytes(), RELEASED_BYTES);
        assert_eq!(
            plan.spec("backbone.layers.0.mixer.q_proj.weight")
                .expect("layer 0 query")
                .shape,
            vec![4096, 2560]
        );
        assert_eq!(
            plan.spec("backbone.layers.1.mixer.down_proj.weight")
                .expect("layer 1 down")
                .shape,
            vec![2560, 7168]
        );
        assert_eq!(
            plan.role_bytes(Role::Embedding) + plan.role_bytes(Role::LmHead),
            1_342_439_424
        );
    }
}

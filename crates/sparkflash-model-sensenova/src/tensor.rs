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

//! The tensor plan of the released SenseNova-Loop-9B checkpoints.
//!
//! The FP8 mirror this arm reads first has 1,201 names: 772 text tensors (11,002,552,832
//! bytes) and 429 vision tensors. The plan is derived from the configuration and the
//! quantisation rule, so the count and byte total are a real check on the release rather than a
//! transcription of the index.

use sparkflash_weights::{DType, TensorPlan, TensorSpec};

use crate::config::{LayerType, SenseNovaConfig, Storage};

/// The text backbone prefix in the released checkpoint.
pub const TEXT_PREFIX: &str = "model.language_model";

/// The vision tower prefix.
pub const VISION_PREFIX: &str = "model.visual";

/// The FP8 mirror's text tensor count.
pub const RELEASED_FP8_TEXT_TENSORS: usize = 772;

/// The FP8 mirror's text byte total.
pub const RELEASED_FP8_TEXT_BYTES: u128 = 11_002_552_832;

/// The FP8 mirror's total tensor count.
pub const RELEASED_FP8_TENSORS: usize = 1_201;

/// What part of the model a tensor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// The token embedding.
    Embedding,
    /// The LM head.
    LmHead,
    /// A norm, router or other small tensor.
    Auxiliary,
    /// A dense projection.
    Projection,
    /// A Gated DeltaNet convolution or recurrence parameter.
    Linear,
    /// An mHC router.
    Mhc,
    /// A vision-tower tensor.
    Vision,
}

/// The tensor plan for one SenseNova configuration.
#[derive(Debug, Clone)]
pub struct SenseNovaTensorPlan {
    plan: TensorPlan<Role>,
}

impl std::ops::Deref for SenseNovaTensorPlan {
    type Target = TensorPlan<Role>;

    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}

impl SenseNovaTensorPlan {
    /// Plan the text backbone of a release.
    #[must_use]
    pub fn served(config: &SenseNovaConfig) -> Self {
        let mut plan = TensorPlan::default();
        let hidden = config.hidden_size;
        let prefix = TEXT_PREFIX;
        plan.push(
            &format!("{prefix}.embed_tokens.weight"),
            DType::BF16,
            vec![config.vocab_size, hidden],
            Role::Embedding,
        );
        plan.push(
            &format!("{prefix}.loop_gate.logits"),
            DType::F32,
            vec![hidden],
            Role::Auxiliary,
        );
        plan.push(
            &format!("{prefix}.norm.weight"),
            DType::BF16,
            vec![hidden],
            Role::Auxiliary,
        );
        for (index, kind) in config.layer_types.iter().enumerate() {
            let layer = format!("{prefix}.layers.{index}");
            plan.push(
                &format!("{layer}.input_layernorm.weight"),
                DType::BF16,
                vec![hidden],
                Role::Auxiliary,
            );
            plan.push(
                &format!("{layer}.post_attention_layernorm.weight"),
                DType::BF16,
                vec![hidden],
                Role::Auxiliary,
            );
            if index >= config.mhc.start && index < config.mhc.end {
                let width = config.mhc.streams * hidden;
                let rows = 2 * config.mhc.streams
                    + if config.mhc.fixed_identity {
                        0
                    } else {
                        config.mhc.streams * config.mhc.streams
                    };
                for router in ["mhc_attn", "mhc_ffn"] {
                    plan.push(
                        &format!("{layer}.{router}.weight"),
                        DType::F32,
                        vec![rows, width],
                        Role::Mhc,
                    );
                    plan.push(
                        &format!("{layer}.{router}.bias"),
                        DType::F32,
                        vec![rows],
                        Role::Mhc,
                    );
                }
            }
            match kind {
                LayerType::Full => {
                    let query = 2 * config.num_attention_heads * config.head_dim;
                    let kv = config.kv_width();
                    let out = config.num_attention_heads * config.head_dim;
                    for (name, rows, columns) in [
                        ("q_proj", query, hidden),
                        ("k_proj", kv, hidden),
                        ("v_proj", kv, hidden),
                        ("o_proj", hidden, out),
                    ] {
                        projection(
                            &mut plan,
                            &format!("{layer}.self_attn.{name}.weight"),
                            rows,
                            columns,
                            config.storage,
                            Role::Projection,
                        );
                    }
                    for name in ["q_norm", "k_norm"] {
                        plan.push(
                            &format!("{layer}.self_attn.{name}.weight"),
                            DType::BF16,
                            vec![config.head_dim],
                            Role::Auxiliary,
                        );
                    }
                }
                LayerType::Linear => {
                    let fused = config.linear_projection_width();
                    let value = config.linear_value_width();
                    for (name, rows, columns) in [
                        ("in_proj_qkv", fused, hidden),
                        ("in_proj_z", value, hidden),
                        ("in_proj_b", config.linear.value_heads, hidden),
                        ("in_proj_a", config.linear.value_heads, hidden),
                        ("out_proj", hidden, value),
                    ] {
                        projection(
                            &mut plan,
                            &format!("{layer}.linear_attn.{name}.weight"),
                            rows,
                            columns,
                            config.storage,
                            Role::Projection,
                        );
                    }
                    plan.push(
                        &format!("{layer}.linear_attn.conv1d.weight"),
                        DType::BF16,
                        vec![fused, 1, config.linear.conv_kernel],
                        Role::Linear,
                    );
                    plan.push(
                        &format!("{layer}.linear_attn.A_log"),
                        DType::F32,
                        vec![config.linear.value_heads],
                        Role::Linear,
                    );
                    plan.push(
                        &format!("{layer}.linear_attn.dt_bias"),
                        DType::F32,
                        vec![config.linear.value_heads],
                        Role::Linear,
                    );
                    plan.push(
                        &format!("{layer}.linear_attn.norm.weight"),
                        DType::BF16,
                        vec![config.linear.value_dim],
                        Role::Auxiliary,
                    );
                }
            }
            for name in ["gate_proj", "up_proj"] {
                projection(
                    &mut plan,
                    &format!("{layer}.mlp.{name}.weight"),
                    config.intermediate_size,
                    hidden,
                    config.storage,
                    Role::Projection,
                );
            }
            projection(
                &mut plan,
                &format!("{layer}.mlp.down_proj.weight"),
                hidden,
                config.intermediate_size,
                config.storage,
                Role::Projection,
            );
        }
        plan.push(
            "lm_head.weight",
            DType::BF16,
            vec![config.vocab_size, hidden],
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

fn projection<Role: Ord + Copy>(
    plan: &mut TensorPlan<Role>,
    name: &str,
    rows: usize,
    columns: usize,
    storage: Storage,
    role: Role,
) {
    match storage {
        Storage::Bf16 => plan.push(name, DType::BF16, vec![rows, columns], role),
        Storage::Fp8Tile128 => {
            plan.push(name, DType::F8E4M3, vec![rows, columns], role);
            plan.push(
                &format!("{name}_scale_inv"),
                DType::BF16,
                vec![rows.div_ceil(128), columns.div_ceil(128)],
                role,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASED: &str = include_str!("../config/sensenova_loop_9b.hf.json");

    #[test]
    fn the_fp8_plan_matches_the_released_text_header() {
        let config = SenseNovaConfig::from_json(RELEASED).expect("the carried configuration");
        let plan = SenseNovaTensorPlan::served(&config);
        assert_eq!(plan.len(), RELEASED_FP8_TEXT_TENSORS);
        assert_eq!(plan.storage_bytes(), RELEASED_FP8_TEXT_BYTES);
        assert_eq!(
            plan.spec("model.language_model.layers.3.self_attn.q_proj.weight")
                .expect("q projection")
                .shape,
            vec![8192, 4096]
        );
        assert_eq!(
            plan.spec("model.language_model.layers.4.mhc_attn.weight")
                .expect("mHC router")
                .shape,
            vec![8, 8192]
        );
        assert_eq!(
            plan.spec("model.language_model.layers.27.mhc_ffn.bias")
                .expect("mHC bias")
                .shape,
            vec![8]
        );
    }

    #[test]
    fn the_execution_plan_is_fifty_six_visits() {
        let config = SenseNovaConfig::from_json(RELEASED).expect("the carried configuration");
        assert_eq!(config.visits(), 56);
        assert_eq!(config.full_attention_visits(), 14);
        assert_eq!(config.linear_attention_visits(), 42);
    }
}

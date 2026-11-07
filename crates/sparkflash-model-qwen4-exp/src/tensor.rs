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

//! The tensor plan of the released NVFP4 checkpoint.
//!
//! The plan is derived from the configuration, not transcribed from the index, so the comparison
//! is a real check: a shape this pack believes can disagree with the shard headers. The full
//! checkpoint has 299,545 tensor names; this plan describes every one of them, because the byte
//! total is only meaningful if it is complete.
//!
//! # Storage rules of this checkpoint (NVIDIA ModelOpt 0.46, `quant_algo: MIXED_PRECISION`)
//!
//! * The 48 x 512 routed experts are **NVFP4**: a logical `[out, in]` weight is stored as
//!   `U8 [out, in / 2]` (two E2M1 nibbles per byte), with `F8_E4M3 [out, in / 16]` block
//!   scales (one per 16 values along K, no blocking along out), a per-tensor `F32`
//!   `weight_scale_2` and a per-tensor static `F32` `input_scale`.
//! * The PLE n-gram table is **FP8**: 128 shards of `F8_E4M3 [2500012, 160]` with one
//!   per-tensor `BF16` scale. The shards are rows of one logical table, so runtime lookup is
//!   `row -> (row / 2500012, row % 2500012)`.
//! * Everything else (attention, shared experts, router, hyper-connections, embeddings, lm_head,
//!   MTP non-expert) is BF16.
//!
//! The plan keeps the **logical** shape and lets `DType` imply packing, exactly as the V4.1 plan
//! does, except that ModelOpt's NVFP4 weight is spelled `F4E2M1` here: the container reports
//! `U8` with a halved last dimension, and the loader maps that to this logical spec.

use sparkflash_weights::{DType, TensorPlan};

use crate::config::{ExpertQuant, LayerKind, Qwen4ExpConfig};
use crate::ngram::PleLayout;

/// The prefix every language-model tensor carries in the released checkpoint.
pub const LM_PREFIX: &str = "model.language_model";

/// The bytes of the text language model including the whole PLE table.
pub const TEXT_PLE_BYTES: u128 = 129_043_957_786;

/// The bytes of the complete checkpoint (text + PLE + MTP + vision).
pub const FULL_CHECKPOINT_BYTES: u128 = 132_639_846_394;

/// The bytes of the PLE n-gram table: 128 shards plus its constants and per-tensor scale.
pub const PLE_TABLE_BYTES: u128 = 51_200_246_042;

/// The bytes of the MTP stage, planned but not served.
pub const MTP_BYTES: u128 = 2_698_026_496;

/// The bytes of the vision tower, planned but not served.
pub const VISION_BYTES: u128 = 897_862_112;

/// The bytes of the text language model the device arm must upload: everything except the PLE
/// table, which stays host-resident and is paged with the Engram store.
pub const DEVICE_RESIDENT_BYTES: u128 = TEXT_PLE_BYTES - PLE_TABLE_BYTES;

/// The resident total of the official FP8 checkpoint: the same text model with E4M3 experts
/// (one byte a weight and two bytes a 128x128 tile) in place of the packed E2M1 ones.
pub const FP8_DEVICE_RESIDENT_BYTES: u128 = 130_706_097_920;

/// What part of the model a tensor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// The token embedding table.
    Embedding,
    /// The (untied) LM head.
    LmHead,
    /// The final hyper-connection mixer that folds four streams into one.
    HyperMixer,
    /// A layer's attention hyper-connection (norm, bottleneck mixer, injection).
    AttnHyper,
    /// A layer's MLP hyper-connection.
    MlpHyper,
    /// The routed-expert router (BF16 `[512, 2560]`).
    Router,
    /// The shared expert and its sigmoid gate.
    SharedExpert,
    /// One routed expert's NVFP4 triple (weight, block scale, per-tensor scales).
    Experts,
    /// A Gated DeltaNet branch.
    GatedDeltaNet,
    /// A Qwen Sparse Attention branch (projections and norms).
    SparseAttention,
    /// The QSA indexer.
    QsaIndexer,
    /// The PLE projections, norms, convolution and hash constants.
    PleAux,
    /// The hashed PLE embedding table shards.
    PleTable,
    /// The MTP stage (planned, not served).
    Mtp,
    /// The vision tower (planned, not served).
    Vision,
}

/// The tensor plan for one configuration.
///
/// The lookups, byte totals and conformance checks are the shared [TensorPlan]'s, reached through
/// `Deref`; this type owns only how the configuration becomes names and shapes.
#[derive(Debug, Clone, Default)]
pub struct Qwen4ExpTensorPlan {
    inner: TensorPlan<Role>,
    /// The stored format of the routed experts, which is the only format the checkpoint varies.
    expert_quant: ExpertQuant,
}

impl std::ops::Deref for Qwen4ExpTensorPlan {
    type Target = TensorPlan<Role>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Qwen4ExpTensorPlan {
    /// Plan the text language model and the PLE table: everything the device arm and the PLE
    /// reader touch.
    #[must_use]
    pub fn served(config: &Qwen4ExpConfig) -> Self {
        let mut plan = Self {
            expert_quant: config.expert_quant,
            ..Self::default()
        };
        plan.push(
            &format!("{LM_PREFIX}.embed_tokens.weight"),
            DType::BF16,
            vec![config.vocab_size, config.hidden_size],
            Role::Embedding,
        );
        plan.push(
            "lm_head.weight",
            DType::BF16,
            vec![config.vocab_size, config.hidden_size],
            Role::LmHead,
        );
        plan.push_hyper(
            &format!("{LM_PREFIX}.hyper_connection_mixer"),
            config,
            false,
            Role::HyperMixer,
        );
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("{LM_PREFIX}.layers.{layer}");
            plan.push_hyper(
                &format!("{prefix}.attn_hyper_connection"),
                config,
                true,
                Role::AttnHyper,
            );
            plan.push_hyper(
                &format!("{prefix}.mlp_hyper_connection"),
                config,
                true,
                Role::MlpHyper,
            );
            plan.push(
                &format!("{prefix}.mlp.gate.weight"),
                DType::BF16,
                vec![config.num_experts, config.hidden_size],
                Role::Router,
            );
            for name in ["gate_proj", "up_proj"] {
                plan.push(
                    &format!("{prefix}.mlp.shared_expert.{name}.weight"),
                    DType::BF16,
                    vec![config.shared_expert_intermediate_size, config.hidden_size],
                    Role::SharedExpert,
                );
            }
            plan.push(
                &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                DType::BF16,
                vec![config.hidden_size, config.shared_expert_intermediate_size],
                Role::SharedExpert,
            );
            plan.push(
                &format!("{prefix}.mlp.shared_expert_gate.weight"),
                DType::BF16,
                vec![1, config.hidden_size],
                Role::SharedExpert,
            );
            for expert in 0..config.num_experts {
                for (name, rows, columns) in [
                    (
                        "gate_proj",
                        config.moe_intermediate_size,
                        config.hidden_size,
                    ),
                    ("up_proj", config.moe_intermediate_size, config.hidden_size),
                    (
                        "down_proj",
                        config.hidden_size,
                        config.moe_intermediate_size,
                    ),
                ] {
                    plan.push_expert(
                        &format!("{prefix}.mlp.experts.{expert}.{name}"),
                        rows,
                        columns,
                    );
                }
            }
            match config.layer_types[layer] {
                LayerKind::Linear => plan.push_gated_delta_net(&prefix, config),
                LayerKind::Sparse => plan.push_sparse_attention(&prefix, config),
            }
            if config.ple_layer_ids.contains(&(layer + 1)) {
                plan.push_ple(&prefix, config);
            }
        }
        plan
    }

    /// Plan the whole checkpoint, adding the MTP stage and the vision tower.
    ///
    /// The byte total of this plan is the identity the tests pin against the released
    /// `model.safetensors.index.json`'s `metadata.total_size`.
    #[must_use]
    pub fn checkpoint(config: &Qwen4ExpConfig) -> Self {
        let mut plan = Self::served(config);
        plan.push_mtp(config);
        plan.push_vision(config);
        plan
    }

    fn push(&mut self, name: &str, dtype: DType, shape: Vec<usize>, role: Role) {
        self.inner.push(name, dtype, shape, role);
    }

    fn push_hyper(&mut self, prefix: &str, config: &Qwen4ExpConfig, inject: bool, role: Role) {
        let width = config.hyper_width();
        self.push(
            &format!("{prefix}.hc_norm.weight"),
            DType::BF16,
            vec![width],
            role,
        );
        self.push(
            &format!("{prefix}.input_mix_weight_down.weight"),
            DType::BF16,
            vec![config.hc_lowrank, width],
            role,
        );
        self.push(
            &format!("{prefix}.input_mix_weight_up.weight"),
            DType::BF16,
            vec![width, config.hc_lowrank],
            role,
        );
        if inject {
            self.push(
                &format!("{prefix}.block_inject_weight.weight"),
                DType::BF16,
                vec![config.hc_count, width],
                role,
            );
        }
    }

    fn push_expert(&mut self, prefix: &str, rows: usize, columns: usize) {
        match self.expert_quant {
            ExpertQuant::Nvfp4 => self.inner.push_nvfp4(prefix, rows, columns, Role::Experts),
            // The official FP8 release keeps one BF16 inverse scale a 128x128 tile.
            ExpertQuant::Fp8Block => {
                self.inner
                    .push_fp8_block(prefix, rows, columns, 128, DType::BF16, Role::Experts)
            }
        }
    }

    fn push_gated_delta_net(&mut self, prefix: &str, config: &Qwen4ExpConfig) {
        let prefix = format!("{prefix}.linear_attn");
        let hidden = config.hidden_size;
        self.push(
            &format!("{prefix}.in_proj_qkv.weight"),
            DType::BF16,
            vec![config.linear_conv_width(), hidden],
            Role::GatedDeltaNet,
        );
        self.push(
            &format!("{prefix}.in_proj_z.weight"),
            DType::BF16,
            vec![config.linear_value_width(), hidden],
            Role::GatedDeltaNet,
        );
        for name in ["in_proj_a", "in_proj_b"] {
            self.push(
                &format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![config.linear_num_value_heads, hidden],
                Role::GatedDeltaNet,
            );
        }
        self.push(
            &format!("{prefix}.out_proj.weight"),
            DType::BF16,
            vec![hidden, config.linear_value_width()],
            Role::GatedDeltaNet,
        );
        self.push(
            &format!("{prefix}.conv1d.weight"),
            DType::BF16,
            vec![config.linear_conv_width(), 1, config.linear_conv_kernel_dim],
            Role::GatedDeltaNet,
        );
        self.push(
            &format!("{prefix}.A_log"),
            DType::BF16,
            vec![config.linear_num_value_heads],
            Role::GatedDeltaNet,
        );
        self.push(
            &format!("{prefix}.dt_bias"),
            DType::BF16,
            vec![config.linear_num_value_heads],
            Role::GatedDeltaNet,
        );
        self.push(
            &format!("{prefix}.norm.weight"),
            DType::BF16,
            vec![config.linear_value_head_dim],
            Role::GatedDeltaNet,
        );
    }

    fn push_sparse_attention(&mut self, prefix: &str, config: &Qwen4ExpConfig) {
        let prefix = format!("{prefix}.self_attn");
        let hidden = config.hidden_size;
        self.push(
            &format!("{prefix}.q_proj.weight"),
            DType::BF16,
            vec![config.q_proj_width(), hidden],
            Role::SparseAttention,
        );
        for name in ["k_proj", "v_proj"] {
            self.push(
                &format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![config.kv_proj_width(), hidden],
                Role::SparseAttention,
            );
        }
        self.push(
            &format!("{prefix}.o_proj.weight"),
            DType::BF16,
            vec![hidden, config.attn_out_width()],
            Role::SparseAttention,
        );
        for name in ["q_norm", "k_norm"] {
            self.push(
                &format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![config.head_dim],
                Role::SparseAttention,
            );
        }
        self.push(
            &format!("{prefix}.indexer.index_qk_proj.weight"),
            DType::BF16,
            vec![
                (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim,
                hidden,
            ],
            Role::QsaIndexer,
        );
        for name in ["q_layernorm", "k_layernorm"] {
            self.push(
                &format!("{prefix}.indexer.{name}.weight"),
                DType::BF16,
                vec![config.indexer_head_dim],
                Role::QsaIndexer,
            );
        }
    }

    fn push_ple(&mut self, prefix: &str, config: &Qwen4ExpConfig) {
        let prefix = format!("{prefix}.ple");
        let width = config.hyper_width();
        self.push(
            &format!("{prefix}.key_proj.weight"),
            DType::BF16,
            vec![width, config.ple_embed_dim],
            Role::PleAux,
        );
        self.push(
            &format!("{prefix}.value_proj.weight"),
            DType::BF16,
            vec![config.hidden_size, config.ple_embed_dim],
            Role::PleAux,
        );
        for name in ["norm_key", "norm_query", "norm_conv"] {
            self.push(
                &format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![width],
                Role::PleAux,
            );
        }
        self.push(
            &format!("{prefix}.conv1d.weight"),
            DType::BF16,
            vec![width, 1, config.ple_conv_kernel_size],
            Role::PleAux,
        );
        self.push(
            &format!("{prefix}.ple_embedding.layer_multipliers"),
            DType::I64,
            vec![config.ngram_size],
            Role::PleTable,
        );
        self.push(
            &format!("{prefix}.ple_embedding.ngram_heads_offsets"),
            DType::I64,
            vec![config.ngram_heads()],
            Role::PleTable,
        );
        self.push(
            &format!("{prefix}.ple_embedding.ngram_heads_vocab_sizes"),
            DType::I64,
            vec![config.ngram_heads()],
            Role::PleTable,
        );
        self.push(
            &format!("{prefix}.ple_embedding.ngram_embedding.weight_scale"),
            DType::BF16,
            vec![1],
            Role::PleTable,
        );
        let rows = PleLayout::new(config).padded_rows;
        let per_shard = rows / config.split_ngram_parts as u64;
        for shard in 0..config.split_ngram_parts {
            self.push(
                &format!("{prefix}.ple_embedding.ngram_embedding.shard_{shard}.weight"),
                DType::F8E4M3,
                vec![per_shard as usize, config.ngram_head_dim()],
                Role::PleTable,
            );
        }
    }

    /// The served plan, plus the MTP stage when the configuration carries one.
    ///
    /// The synthetic checkpoint is written through this, so a fixture whose configuration names an
    /// MTP layer actually ships its tensors.
    #[must_use]
    pub fn served_and_mtp(config: &Qwen4ExpConfig) -> Self {
        let mut plan = Self::served(config);
        if config.mtp_num_hidden_layers > 0 {
            plan.push_mtp(config);
        }
        plan
    }

    fn push_mtp(&mut self, config: &Qwen4ExpConfig) {
        // One MTP layer, a sparse-attention + MoE block with its own hyper-connections, plus the
        // embedding/hidden fusion projections and the pre-norms.
        let hidden = config.hidden_size;
        let prefix = "mtp";
        for name in ["fc_embedding", "fc_hidden"] {
            self.push(
                &format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![hidden, hidden],
                Role::Mtp,
            );
        }
        self.push_hyper(
            &format!("{prefix}.hyper_connection_mixer"),
            config,
            false,
            Role::Mtp,
        );
        self.push(
            &format!("{prefix}.pre_fc_norm_embedding.weight"),
            DType::BF16,
            vec![hidden],
            Role::Mtp,
        );
        self.push(
            &format!("{prefix}.pre_fc_norm_hidden.weight"),
            DType::BF16,
            vec![config.hyper_width()],
            Role::Mtp,
        );
        let layer = format!("{prefix}.layers.0");
        self.push_hyper(
            &format!("{layer}.attn_hyper_connection"),
            config,
            true,
            Role::Mtp,
        );
        self.push_hyper(
            &format!("{layer}.mlp_hyper_connection"),
            config,
            true,
            Role::Mtp,
        );
        self.push(
            &format!("{layer}.self_attn.q_proj.weight"),
            DType::BF16,
            vec![config.q_proj_width(), hidden],
            Role::Mtp,
        );
        for name in ["k_proj", "v_proj"] {
            self.push(
                &format!("{layer}.self_attn.{name}.weight"),
                DType::BF16,
                vec![config.kv_proj_width(), hidden],
                Role::Mtp,
            );
        }
        self.push(
            &format!("{layer}.self_attn.o_proj.weight"),
            DType::BF16,
            vec![hidden, config.attn_out_width()],
            Role::Mtp,
        );
        for name in ["q_norm", "k_norm"] {
            self.push(
                &format!("{layer}.self_attn.{name}.weight"),
                DType::BF16,
                vec![config.head_dim],
                Role::Mtp,
            );
        }
        self.push(
            &format!("{layer}.self_attn.indexer.index_qk_proj.weight"),
            DType::BF16,
            vec![
                (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim,
                hidden,
            ],
            Role::Mtp,
        );
        for name in ["q_layernorm", "k_layernorm"] {
            self.push(
                &format!("{layer}.self_attn.indexer.{name}.weight"),
                DType::BF16,
                vec![config.indexer_head_dim],
                Role::Mtp,
            );
        }
        self.push(
            &format!("{layer}.mlp.gate.weight"),
            DType::BF16,
            vec![config.num_experts, hidden],
            Role::Mtp,
        );
        for name in ["gate_proj", "up_proj"] {
            self.push(
                &format!("{layer}.mlp.shared_expert.{name}.weight"),
                DType::BF16,
                vec![config.shared_expert_intermediate_size, hidden],
                Role::Mtp,
            );
        }
        self.push(
            &format!("{layer}.mlp.shared_expert.down_proj.weight"),
            DType::BF16,
            vec![hidden, config.shared_expert_intermediate_size],
            Role::Mtp,
        );
        self.push(
            &format!("{layer}.mlp.shared_expert_gate.weight"),
            DType::BF16,
            vec![1, hidden],
            Role::Mtp,
        );
        // The MTP routed experts are FP8 with BF16 scales on 128x128 tiles, not NVFP4.
        for expert in 0..config.num_experts {
            for (name, rows, columns) in [
                ("gate_proj", config.moe_intermediate_size, hidden),
                ("up_proj", config.moe_intermediate_size, hidden),
                ("down_proj", hidden, config.moe_intermediate_size),
            ] {
                self.inner.push_fp8_block(
                    &format!("{layer}.mlp.experts.{expert}.{name}"),
                    rows,
                    columns,
                    128,
                    DType::BF16,
                    Role::Mtp,
                );
            }
        }
    }

    fn push_vision(&mut self, config: &Qwen4ExpConfig) {
        let width = config.vision_hidden_size.unwrap_or(1152);
        let depth = config.vision_depth.unwrap_or(27);
        let prefix = "model.visual";
        self.push(
            &format!("{prefix}.patch_embed.proj.weight"),
            DType::BF16,
            vec![width, 3 * 2 * 16 * 16],
            Role::Vision,
        );
        self.push(
            &format!("{prefix}.patch_embed.proj.bias"),
            DType::BF16,
            vec![width],
            Role::Vision,
        );
        self.push(
            &format!("{prefix}.pos_embed.weight"),
            DType::BF16,
            vec![2304, width],
            Role::Vision,
        );
        for block in 0..depth {
            let prefix = format!("{prefix}.blocks.{block}");
            for name in ["norm1", "norm2"] {
                self.push(
                    &format!("{prefix}.{name}.weight"),
                    DType::BF16,
                    vec![width],
                    Role::Vision,
                );
                self.push(
                    &format!("{prefix}.{name}.bias"),
                    DType::BF16,
                    vec![width],
                    Role::Vision,
                );
            }
            self.push(
                &format!("{prefix}.attn.qkv.weight"),
                DType::BF16,
                vec![3 * width, width],
                Role::Vision,
            );
            self.push(
                &format!("{prefix}.attn.qkv.bias"),
                DType::BF16,
                vec![3 * width],
                Role::Vision,
            );
            self.push(
                &format!("{prefix}.attn.proj.weight"),
                DType::BF16,
                vec![width, width],
                Role::Vision,
            );
            self.push(
                &format!("{prefix}.attn.proj.bias"),
                DType::BF16,
                vec![width],
                Role::Vision,
            );
            self.push(
                &format!("{prefix}.mlp.linear_fc1.weight"),
                DType::BF16,
                vec![4304, width],
                Role::Vision,
            );
            self.push(
                &format!("{prefix}.mlp.linear_fc1.bias"),
                DType::BF16,
                vec![4304],
                Role::Vision,
            );
            self.push(
                &format!("{prefix}.mlp.linear_fc2.weight"),
                DType::BF16,
                vec![width, 4304],
                Role::Vision,
            );
            self.push(
                &format!("{prefix}.mlp.linear_fc2.bias"),
                DType::BF16,
                vec![width],
                Role::Vision,
            );
        }
        for (name, rows, columns) in [
            ("merger.linear_fc1", 4608, 4608),
            ("merger.linear_fc2", 2560, 4608),
        ] {
            self.push(
                &format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![rows, columns],
                Role::Vision,
            );
            self.push(
                &format!("{prefix}.{name}.bias"),
                DType::BF16,
                vec![rows],
                Role::Vision,
            );
        }
        self.push(
            &format!("{prefix}.merger.norm.weight"),
            DType::BF16,
            vec![1152],
            Role::Vision,
        );
        self.push(
            &format!("{prefix}.merger.norm.bias"),
            DType::BF16,
            vec![1152],
            Role::Vision,
        );
    }
}

#[cfg(test)]
mod tests {
    use sparkflash_model::TpPlan;
    use sparkflash_ops::tensor_plan::{ArchitectureTensorPlan, Role as TensorRole};
    use sparkflash_weights::TensorSpec;

    use super::*;

    #[test]
    fn the_served_plan_matches_the_released_text_and_ple_bytes() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let plan = Qwen4ExpTensorPlan::served(&config);
        assert_eq!(plan.storage_bytes(), TEXT_PLE_BYTES);
        assert_eq!(plan.role_bytes(Role::PleTable), PLE_TABLE_BYTES);
        assert_eq!(
            plan.storage_bytes() - plan.role_bytes(Role::PleTable),
            DEVICE_RESIDENT_BYTES
        );
        // 48 x 512 experts with 4 tensors each, plus every non-expert tensor.
        assert_eq!(
            plan.role_bytes(Role::Experts),
            48u128 * 512 * 3 * (640 * 1280 + 640 * 160 + 8)
        );
    }

    /// The shared architecture plan emits exactly the specs this family plan does.
    fn assert_shared_plan(
        shared: &ArchitectureTensorPlan,
        legacy: &Qwen4ExpTensorPlan,
        label: &str,
    ) {
        assert_eq!(shared.len(), legacy.len(), "{label}: tensor count");
        let mut mismatched = 0usize;
        for spec in legacy.specs() {
            let found = shared
                .spec(&spec.name)
                .unwrap_or_else(|| panic!("{label}: {} is missing", spec.name));
            if found.dtype != spec.dtype || found.shape != spec.shape {
                mismatched += 1;
                if mismatched <= 12 {
                    eprintln!(
                        "{label}: {} legacy {:?} {:?} shared {:?} {:?}",
                        spec.name, spec.dtype, spec.shape, found.dtype, found.shape
                    );
                }
            }
        }
        assert_eq!(mismatched, 0, "{label}: {mismatched} specs differ");
        let mut shared_only = 0usize;
        for spec in shared.specs() {
            if legacy.spec(&spec.name).is_none() {
                shared_only += 1;
                if shared_only <= 12 {
                    eprintln!(
                        "{label}: shared-only {} {:?} {:?}",
                        spec.name, spec.dtype, spec.shape
                    );
                }
            }
        }
        assert_eq!(shared_only, 0, "{label}: {shared_only} shared-only specs");
        assert_eq!(
            shared.storage_bytes(),
            legacy.storage_bytes(),
            "{label}: byte total"
        );
    }

    #[test]
    fn the_shared_architecture_plan_reproduces_the_nvfp4_checkpoint() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let tp = TpPlan::new(1, 0).expect("one rank");
        let shared = ArchitectureTensorPlan::checkpoint(&config.architecture, tp)
            .expect("the shared plan builds");
        assert_shared_plan(&shared, &Qwen4ExpTensorPlan::checkpoint(&config), "nvfp4");
        let served = ArchitectureTensorPlan::served(&config.architecture, tp)
            .expect("the shared served plan builds");
        assert_shared_plan(
            &served,
            &Qwen4ExpTensorPlan::served(&config),
            "nvfp4 served",
        );
    }

    #[test]
    fn the_shared_architecture_plan_reproduces_the_fp8_served_set() {
        let config = Qwen4ExpConfig::qwen38_flash_next_fp8().expect("the carried config parses");
        let tp = TpPlan::new(1, 0).expect("one rank");
        let shared = ArchitectureTensorPlan::served(&config.architecture, tp)
            .expect("the shared plan builds");
        assert_shared_plan(&shared, &Qwen4ExpTensorPlan::served(&config), "fp8 served");
        assert_eq!(
            shared.storage_bytes() - shared.role_bytes(TensorRole::PleTable),
            FP8_DEVICE_RESIDENT_BYTES
        );
    }

    #[test]
    fn the_full_plan_matches_the_released_index_total() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let plan = Qwen4ExpTensorPlan::checkpoint(&config);
        assert_eq!(plan.storage_bytes(), FULL_CHECKPOINT_BYTES);
        assert_eq!(plan.role_bytes(Role::Mtp), MTP_BYTES);
        assert_eq!(plan.role_bytes(Role::Vision), VISION_BYTES);
        assert_eq!(plan.len(), 299_545, "the released index has 299,545 names");
    }

    #[test]
    fn the_fp8_release_resident_total_is_measured() {
        let config = Qwen4ExpConfig::qwen38_flash_next_fp8().expect("the carried config parses");
        let plan = Qwen4ExpTensorPlan::served(&config);
        let resident = plan.storage_bytes() - plan.role_bytes(Role::PleTable);
        assert_eq!(resident, FP8_DEVICE_RESIDENT_BYTES);
    }

    #[test]
    fn the_fp8_release_plans_block_scaled_experts() {
        let config = Qwen4ExpConfig::qwen38_flash_next_fp8().expect("the carried config parses");
        let plan = Qwen4ExpTensorPlan::served(&config);
        let prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj";
        let weight = plan
            .spec(&format!("{prefix}.weight"))
            .expect("the FP8 weight is planned");
        assert_eq!(weight.dtype, DType::F8E4M3);
        assert_eq!(
            weight.shape,
            vec![config.moe_intermediate_size, config.hidden_size]
        );
        let scales = plan
            .spec(&format!("{prefix}.weight_scale_inv"))
            .expect("the tile grid is planned");
        assert_eq!(scales.dtype, DType::BF16);
        assert_eq!(
            scales.shape,
            vec![
                config.moe_intermediate_size.div_ceil(128),
                config.hidden_size.div_ceil(128)
            ]
        );
        assert!(
            plan.spec(&format!("{prefix}.weight_scale_2")).is_none(),
            "the FP8 release has no per-tensor expert scale"
        );
        // One byte a weight and two bytes a 128x128 tile, against NVFP4's half a byte.
        let experts = plan.role_bytes(Role::Experts);
        assert!(experts > 120_000_000_000, "FP8 experts are {experts} bytes");
    }

    #[test]
    fn the_served_plan_accepts_the_released_checkpoints_unserved_tensors() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let served = Qwen4ExpTensorPlan::served(&config);
        let full = Qwen4ExpTensorPlan::checkpoint(&config);
        // The released checkpoint is a superset of the served set: the MTP stage and the vision
        // tower are in the file and not in the arm. 299,545 - 296,111 = 3,434 of them.
        let accepted = served
            .conformance_with_unserved(full.names().map(str::to_owned), &full)
            .expect("the released checkpoint is a superset of the served plan");
        assert_eq!(accepted, 3_434);
        assert_eq!(full.len() - served.len(), accepted);
        assert_eq!(accepted, 3_101 + 333);
    }

    #[test]
    fn the_ple_table_role_is_the_shards_plus_their_metadata() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let plan = Qwen4ExpTensorPlan::served(&config);
        // The device arm's upload set is everything except the PLE table role, and that role is
        // more than the 128 shards: three I64 hash vectors and a BF16 scale ride with them. A
        // loader that subtracts only the shards overshoots the resident total by these 282 bytes.
        let shards: u128 = plan
            .names()
            .filter(|name| name.contains("ngram_embedding.shard_"))
            .filter_map(|name| plan.spec(name))
            .map(TensorSpec::storage_bytes)
            .sum();
        assert_eq!(plan.role_bytes(Role::PleTable) - shards, 282);
        assert_eq!(
            plan.storage_bytes() - plan.role_bytes(Role::PleTable),
            DEVICE_RESIDENT_BYTES
        );
    }

    #[test]
    fn a_tensor_neither_plan_names_is_refused() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let served = Qwen4ExpTensorPlan::served(&config);
        let full = Qwen4ExpTensorPlan::checkpoint(&config);
        let mut names: Vec<String> = full.names().map(str::to_owned).collect();
        names.push("model.visual.not_a_tensor.weight".to_owned());
        let refused = served
            .conformance_with_unserved(names, &full)
            .expect_err("a name neither plan knows is refused");
        assert!(refused.contains("not_a_tensor"), "{refused}");
    }

    #[test]
    fn a_missing_served_tensor_is_refused() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let served = Qwen4ExpTensorPlan::served(&config);
        let full = Qwen4ExpTensorPlan::checkpoint(&config);
        let dropped = "lm_head.weight";
        let names: Vec<String> = full
            .names()
            .filter(|name| *name != dropped)
            .map(str::to_owned)
            .collect();
        let refused = served
            .conformance_with_unserved(names, &full)
            .expect_err("a served tensor the checkpoint lacks is refused");
        assert!(refused.contains(dropped), "{refused}");
    }

    #[test]
    fn the_served_plan_names_the_tensors_the_device_arm_reads() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let plan = Qwen4ExpTensorPlan::served(&config);
        assert_eq!(
            plan.spec("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
                .expect("the first linear projection is planned")
                .shape,
            vec![10_240, 2560]
        );
        assert_eq!(
            plan.spec("model.language_model.layers.3.self_attn.q_proj.weight")
                .expect("the first sparse layer is planned")
                .shape,
            vec![12_288, 2560]
        );
        assert_eq!(
            plan.spec("model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight")
                .expect("the indexer is planned")
                .shape,
            vec![640, 2560]
        );
        assert_eq!(
            plan.spec("model.language_model.layers.0.mlp.experts.511.down_proj.weight_scale")
                .expect("the last expert scale is planned")
                .shape,
            vec![2560, 40]
        );
        assert_eq!(
            plan.spec(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight"
            )
            .expect("the last PLE shard is planned")
            .shape,
            vec![2_500_012, 160]
        );
        assert!(
            plan.spec("model.language_model.layers.3.linear_attn.in_proj_qkv.weight")
                .is_none(),
            "a sparse layer must not plan a Gated DeltaNet projection"
        );
    }

    #[test]
    fn the_tiny_plan_is_small_but_complete() {
        let config = Qwen4ExpConfig::tiny();
        let plan = Qwen4ExpTensorPlan::served(&config);
        assert!(plan.len() > 80);
        assert_eq!(
            plan.spec("model.language_model.layers.1.self_attn.q_proj.weight")
                .expect("the sparse layer is planned")
                .shape,
            vec![2 * 4 * 8, 32]
        );
        assert_eq!(
            plan.spec("model.language_model.layers.0.mlp.experts.0.gate_proj.weight")
                .expect("an expert is planned")
                .shape,
            vec![8, 32]
        );
    }

    #[test]
    fn the_checkpoint_plan_keeps_mtp_at_tp4() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let tp = TpPlan::new(4, 0).expect("rank 0 of four");
        let plan = ArchitectureTensorPlan::checkpoint(&config.architecture, tp)
            .expect("the TP4 checkpoint plan builds");
        let mtp: Vec<_> = plan
            .specs()
            .iter()
            .filter(|spec| {
                plan.role(plan.index_of(&spec.name).expect("a planned name")) == TensorRole::Mtp
            })
            .collect();
        assert!(!mtp.is_empty(), "the whole checkpoint keeps its MTP stage");
        assert!(plan.role_bytes(TensorRole::Mtp) > 0, "TP4 MTP bytes");
        assert!(
            plan.rank_spec("mtp.fc_embedding.weight").is_some(),
            "the MTP stage is planned at TP4, not dropped by the served-world gate"
        );
    }
}

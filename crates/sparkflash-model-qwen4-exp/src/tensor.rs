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

//! The tensor plan of a configuration this pack serves.
//!
//! The names and shapes are the shared [ArchitectureTensorPlan]'s: the Qwen3.8 (`qwen4_exp`) and
//! Qwen3.5/3.6 (`qwen3_5_moe`) walks live there beside GLM's, so the converter, the web console's
//! fit and this arm read one plan. This module owns the released Qwen3.8 byte totals the tests pin
//! and the step from a typed configuration to that plan.
//!
//! The plan is derived from the configuration, not transcribed from an index, so comparing it with
//! a checkpoint is a real check: a shape this pack believes can disagree with the shard headers.
//! The Qwen3.8 checkpoint has 299,545 tensor names and this plan describes every one of them,
//! because a byte total is only meaningful if it is complete.
//!
//! # Storage rules of the Qwen3.8 checkpoint (NVIDIA ModelOpt 0.46, `quant_algo: MIXED_PRECISION`)
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
//! The plan keeps the **logical** shape and lets `DType` imply packing, except that ModelOpt's
//! NVFP4 weight is spelled `F4E2M1`: the container reports `U8` with a halved last dimension, and
//! the loader maps that to this logical spec.

use sparkflash_model::{Architecture, TpPlan};
use sparkflash_ops::engine_plan::PlanError;
use sparkflash_ops::tensor_plan::ArchitectureTensorPlan;
pub use sparkflash_ops::tensor_plan::Role;
use sparkflash_weights::TensorPlan;

use crate::config::{ConfigError, Qwen4ExpConfig};

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

/// The single-rank tensor plan of one configuration.
///
/// The lookups, byte totals and conformance checks are the shared [TensorPlan]'s, reached through
/// `Deref`.
#[derive(Debug, Clone)]
pub struct Qwen4ExpTensorPlan {
    inner: TensorPlan<Role>,
}

impl std::ops::Deref for Qwen4ExpTensorPlan {
    type Target = TensorPlan<Role>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

type Planner = fn(&Architecture, TpPlan) -> Result<ArchitectureTensorPlan, PlanError>;

impl Qwen4ExpTensorPlan {
    /// Plan the text language model and the PLE table: everything the device arm and the PLE
    /// reader touch.
    ///
    /// # Errors
    /// A configuration the shared plan does not name.
    pub fn served(config: &Qwen4ExpConfig) -> Result<Self, ConfigError> {
        Self::build(config, ArchitectureTensorPlan::served)
    }

    /// Plan the served set plus the MTP stage, when the configuration names one.
    ///
    /// The synthetic checkpoint is written through this, so a fixture whose configuration names an
    /// MTP layer actually ships its tensors.
    ///
    /// # Errors
    /// As [Self::served].
    pub fn served_and_mtp(config: &Qwen4ExpConfig) -> Result<Self, ConfigError> {
        Self::build(config, ArchitectureTensorPlan::served_and_mtp)
    }

    /// Plan the whole checkpoint, adding the MTP stage and the vision tower.
    ///
    /// The byte total of this plan is the identity the tests pin against the released
    /// `model.safetensors.index.json`'s `metadata.total_size`.
    ///
    /// # Errors
    /// As [Self::served].
    pub fn checkpoint(config: &Qwen4ExpConfig) -> Result<Self, ConfigError> {
        Self::build(config, ArchitectureTensorPlan::checkpoint)
    }

    /// The plan of the document the typed configuration renders.
    ///
    /// The typed fields are what the arm executes, so they are what is planned: a fixture that
    /// edits a width plans the edited width, and a released configuration renders back to the
    /// architecture it was parsed from (the tests pin that for every carried release).
    fn build(config: &Qwen4ExpConfig, planner: Planner) -> Result<Self, ConfigError> {
        let architecture = crate::architecture::architecture_from_json(&config.to_json())?;
        let tp = TpPlan::new(1, 0)
            .map_err(|error| ConfigError::Invalid(format!("a one-rank plan: {error}")))?;
        let plan = planner(&architecture, tp)
            .map_err(|error| ConfigError::Invalid(format!("the tensor plan: {error}")))?;
        Ok(Self {
            inner: plan.into_plan(),
        })
    }
}

#[cfg(test)]
mod tests {
    use sparkflash_model::TpPlan;
    use sparkflash_ops::tensor_plan::{ArchitectureTensorPlan, Role as TensorRole};
    use sparkflash_weights::{DType, TensorSpec};

    use super::*;

    #[test]
    fn the_served_plan_matches_the_released_text_and_ple_bytes() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let plan = Qwen4ExpTensorPlan::served(&config).expect("the plan builds");
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

    /// The released document's plan and the plan of the document its typed configuration renders
    /// are the same specs: the arm plans what it parsed.
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
    fn the_rendered_nvfp4_document_plans_the_released_checkpoint() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let tp = TpPlan::new(1, 0).expect("one rank");
        let shared = ArchitectureTensorPlan::checkpoint(&config.architecture, tp)
            .expect("the shared plan builds");
        assert_shared_plan(
            &shared,
            &Qwen4ExpTensorPlan::checkpoint(&config).expect("the plan builds"),
            "nvfp4",
        );
        let served = ArchitectureTensorPlan::served(&config.architecture, tp)
            .expect("the shared served plan builds");
        assert_shared_plan(
            &served,
            &Qwen4ExpTensorPlan::served(&config).expect("the plan builds"),
            "nvfp4 served",
        );
    }

    #[test]
    fn the_rendered_fp8_document_plans_the_released_served_set() {
        let config = Qwen4ExpConfig::qwen38_flash_next_fp8().expect("the carried config parses");
        let tp = TpPlan::new(1, 0).expect("one rank");
        let shared = ArchitectureTensorPlan::served(&config.architecture, tp)
            .expect("the shared plan builds");
        assert_shared_plan(
            &shared,
            &Qwen4ExpTensorPlan::served(&config).expect("the plan builds"),
            "fp8 served",
        );
        assert_eq!(
            shared.storage_bytes() - shared.role_bytes(TensorRole::PleTable),
            FP8_DEVICE_RESIDENT_BYTES
        );
    }

    #[test]
    fn the_rendered_qwen36_document_plans_the_released_checkpoint() {
        let config = Qwen4ExpConfig::from_json(include_str!(
            "../../sparkflash-model/config/qwen36_flash_35b_a3b_fp8.hf.json"
        ))
        .expect("the carried Qwen3.6 document parses");
        assert_eq!(config.family(), "qwen3_5_moe");
        assert_eq!(config.dense_quant, crate::config::DenseQuant::Fp8Block);
        assert_eq!(config.expert_quant, crate::config::ExpertQuant::Fp8Block);
        let tp = TpPlan::new(1, 0).expect("one rank");
        let shared = ArchitectureTensorPlan::checkpoint(&config.architecture, tp)
            .expect("the shared plan builds");
        let rendered = Qwen4ExpTensorPlan::checkpoint(&config).expect("the plan builds");
        assert_shared_plan(&shared, &rendered, "qwen3.6 fp8");
        // The released index: 64,196 names.
        assert_eq!(rendered.len(), 64_196);
    }

    #[test]
    fn the_full_plan_matches_the_released_index_total() {
        let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        let plan = Qwen4ExpTensorPlan::checkpoint(&config).expect("the plan builds");
        assert_eq!(plan.storage_bytes(), FULL_CHECKPOINT_BYTES);
        assert_eq!(plan.role_bytes(Role::Mtp), MTP_BYTES);
        assert_eq!(plan.role_bytes(Role::Vision), VISION_BYTES);
        assert_eq!(plan.len(), 299_545, "the released index has 299,545 names");
    }

    #[test]
    fn the_fp8_release_resident_total_is_measured() {
        let config = Qwen4ExpConfig::qwen38_flash_next_fp8().expect("the carried config parses");
        let plan = Qwen4ExpTensorPlan::served(&config).expect("the plan builds");
        let resident = plan.storage_bytes() - plan.role_bytes(Role::PleTable);
        assert_eq!(resident, FP8_DEVICE_RESIDENT_BYTES);
    }

    #[test]
    fn the_fp8_release_plans_block_scaled_experts() {
        let config = Qwen4ExpConfig::qwen38_flash_next_fp8().expect("the carried config parses");
        let plan = Qwen4ExpTensorPlan::served(&config).expect("the plan builds");
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
        let served = Qwen4ExpTensorPlan::served(&config).expect("the plan builds");
        let full = Qwen4ExpTensorPlan::checkpoint(&config).expect("the plan builds");
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
        let plan = Qwen4ExpTensorPlan::served(&config).expect("the plan builds");
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
        let served = Qwen4ExpTensorPlan::served(&config).expect("the plan builds");
        let full = Qwen4ExpTensorPlan::checkpoint(&config).expect("the plan builds");
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
        let served = Qwen4ExpTensorPlan::served(&config).expect("the plan builds");
        let full = Qwen4ExpTensorPlan::checkpoint(&config).expect("the plan builds");
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
        let plan = Qwen4ExpTensorPlan::served(&config).expect("the plan builds");
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
        let plan = Qwen4ExpTensorPlan::served(&config).expect("the plan builds");
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

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

//! The tensor plan of the released checkpoint, and its tensor-parallel split.
//!
//! The plan is derived from the configuration, not transcribed from the index, so the comparison
//! is a real check. It describes every one of the official release's 76,108 tensor names - the 45
//! decoder layers, the MTP layer and the vision tower - because the byte total is only meaningful
//! if it is complete.
//!
//! # Storage rules of the official release (`quant_method: fp8`, 128x128 tiles)
//!
//! * The routed and shared experts (the MTP layer's too), the three dense MLPs, and the MLA
//!   `q_a`, `q_b`, `kv_a` and `o_proj` projections are **block-scaled FP8**: E4M3 codes and one
//!   F32 `weight_scale_inv` per 128x128 tile, the layout [TensorPlan::push_fp8_block] plans and
//!   [sparkflash_weights::linalg::ScaleGrid::Tile] scales.
//! * `A_log`, `dt_bias`, the router bias and the hyper-connection base and scale are **F32**.
//! * Everything else is **BF16**: every KDA projection and convolution, `kv_b`, the indexer, the
//!   router, the norms, the embedding and the LM head.
//!
//! The BF16 release (`zai-org/GLM-5.3-Flash-BF16`) is the same plan with every FP8 weight BF16 and
//! no scale grids; a configuration without a `quantization_config` plans it.
//!
//! # The split
//!
//! Every served tensor carries a [Split]: KDA and MLA heads and every MLP's intermediate width are
//! sliced, the branch outputs (`o_proj`, `down_proj`) are sliced along their input columns and
//! all-reduced, the LM head is sliced by rows and gathered, and everything small or needed whole is
//! replicated.

use sparkflash_weights::mhc::HcGeometry;
use sparkflash_weights::{DType, TensorPlan, TensorSpec};

use crate::config::{Glm5NextConfig, IndexerKind, LayerKind, MlpKind};

/// The prefix every language-model tensor carries in the released checkpoint.
pub const LM_PREFIX: &str = "model.language_model";

/// The prefix of the vision tower's tensors.
pub const VISION_PREFIX: &str = "model.visual";

/// The bytes of the complete checkpoint: the released index's `metadata.total_size`.
pub const FULL_CHECKPOINT_BYTES: u128 = 328_326_771_576;

/// The tensor names of the complete checkpoint.
pub const FULL_CHECKPOINT_TENSORS: usize = 76_108;

/// The bytes of the BF16 release's index, which the unquantised plan pins.
pub const BF16_CHECKPOINT_BYTES: u128 = 642_646_653_816;

/// The tensor names of the BF16 release.
pub const BF16_CHECKPOINT_TENSORS: usize = 38_770;

/// The bytes of the MTP layer, planned and served only as a drafter.
pub const MTP_BYTES: u128 = 7_493_399_168;

/// The bytes of the vision tower, planned but not served.
pub const VISION_BYTES: u128 = 1_127_254_016;

/// What part of the model a tensor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// The token embedding table.
    Embedding,
    /// The (untied) LM head.
    LmHead,
    /// The decoder layers' pre-branch norms and the final norm.
    Norm,
    /// A layer's two mHC mixers (`hc_attn_*`, `hc_ffn_*`).
    Hyper,
    /// A Kimi Delta Attention branch.
    Kda,
    /// A DeepSeek Sparse Attention branch's MLA projections and norms.
    Dsa,
    /// A sparse layer's indexer.
    Indexer,
    /// A dense SwiGLU MLP (layers 0-2).
    DenseMlp,
    /// The MoE router and its selection bias.
    Router,
    /// The shared expert.
    SharedExpert,
    /// The routed experts.
    Experts,
    /// The MTP layer.
    Mtp,
    /// The vision tower (planned, not served).
    Vision,
}

/// How a tensor is divided across a tensor-parallel world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// Every rank holds the whole tensor.
    Replicated,
    /// Rank `r` of `w` holds the `r`-th of `w` equal slices of dimension 0: output rows, or heads.
    Rows,
    /// Rank `r` of `w` holds the `r`-th of `w` equal slices of dimension 1: input columns, whose
    /// partial products the ranks all-reduce.
    Columns,
}

impl Split {
    /// The shape one rank of `world` holds.
    ///
    /// A scalar is whole on every rank whatever its split. A tile-scale grid is planned with its
    /// weight's split and slices the same way; [Glm5NextConfig::check_world] is what keeps every
    /// rank's slice of the weight in whole tiles.
    ///
    /// # Errors
    /// When the split dimension does not divide by `world`.
    pub fn rank_shape(self, shape: &[usize], world: usize) -> Result<Vec<usize>, String> {
        let axis = match self {
            Self::Replicated => return Ok(shape.to_vec()),
            _ if shape.is_empty() => return Ok(Vec::new()),
            Self::Rows => 0,
            Self::Columns => 1,
        };
        let Some(&width) = shape.get(axis) else {
            return Err(format!("a {self:?} split of a {shape:?} tensor"));
        };
        if world == 0 || !width.is_multiple_of(world) {
            return Err(format!(
                "dimension {axis} of a {shape:?} tensor does not split across {world} ranks"
            ));
        }
        let mut sliced = shape.to_vec();
        sliced[axis] = width / world;
        Ok(sliced)
    }

    /// Rank `rank` of `world`'s slice of a row-major tensor, with its shape
    /// ([Split::rank_shape]): `Rows` keeps the rank's block of dimension 0, `Columns` the rank's
    /// block of dimension 1 within every index of dimension 0. A tensor the split leaves whole is
    /// returned as it is. The elements are whatever the caller holds: decoded values for the host
    /// reference, stored codes (one `[u8; N]` an element) for a device upload.
    ///
    /// # Errors
    /// As [Split::rank_shape], or when `values` is not `shape` long or `rank` is outside `world`.
    pub fn slice<T: Copy>(
        self,
        shape: &[usize],
        values: Vec<T>,
        rank: usize,
        world: usize,
    ) -> Result<(Vec<usize>, Vec<T>), String> {
        let local = self.rank_shape(shape, world)?;
        let elements: usize = shape.iter().product();
        if values.len() != elements {
            return Err(format!("{} values for a {shape:?} tensor", values.len()));
        }
        if local == shape || elements == 0 {
            return Ok((local, values));
        }
        if rank >= world {
            return Err(format!("rank {rank} is outside a world of {world}"));
        }
        let kept = local.iter().product::<usize>() / local[0];
        let sliced = match self {
            Self::Replicated => values,
            Self::Rows => values[rank * local[0] * kept..(rank + 1) * local[0] * kept].to_vec(),
            Self::Columns => values
                .chunks(elements / shape[0])
                .flat_map(|outer| &outer[rank * kept..(rank + 1) * kept])
                .copied()
                .collect(),
        };
        Ok((local, sliced))
    }
}

/// The tensor plan for one configuration, each tensor with its tensor-parallel split.
///
/// The lookups, byte totals and conformance checks are the shared [TensorPlan]'s, reached through
/// `Deref`; this type owns how the configuration becomes names, shapes and splits.
#[derive(Debug, Clone, Default)]
pub struct Glm5NextTensorPlan {
    plan: TensorPlan<Role>,
    splits: Vec<Split>,
}

impl std::ops::Deref for Glm5NextTensorPlan {
    type Target = TensorPlan<Role>;

    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}

impl Glm5NextTensorPlan {
    /// Plan the text decoder: the embedding, the 45 layers, the final norm and the LM head.
    #[must_use]
    pub fn served(config: &Glm5NextConfig) -> Self {
        let mut plan = Self::default();
        plan.push(
            &format!("{LM_PREFIX}.embed_tokens.weight"),
            DType::BF16,
            vec![config.vocab_size, config.hidden_size],
            Role::Embedding,
            Split::Replicated,
        );
        plan.push(
            &format!("{LM_PREFIX}.norm.weight"),
            DType::BF16,
            vec![config.hidden_size],
            Role::Norm,
            Split::Replicated,
        );
        plan.push(
            "lm_head.weight",
            DType::BF16,
            vec![config.vocab_size, config.hidden_size],
            Role::LmHead,
            Split::Rows,
        );
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("{LM_PREFIX}.layers.{layer}");
            plan.push_layer_norms(&prefix, config, Role::Norm);
            for branch in ["attn", "ffn"] {
                plan.push_hyper(&format!("{prefix}.hc_{branch}"), config);
            }
            match config.layer_types[layer] {
                LayerKind::Linear => plan.push_kda(&format!("{prefix}.self_attn"), config),
                LayerKind::Sparse => plan.push_dsa(
                    &format!("{prefix}.self_attn"),
                    config,
                    config.indexer_types[layer] == IndexerKind::Full,
                    None,
                ),
            }
            match config.mlp_layer_types[layer] {
                MlpKind::Dense => plan.push_dense_mlp(&format!("{prefix}.mlp"), config),
                MlpKind::Moe => plan.push_moe(&format!("{prefix}.mlp"), config, None),
            }
        }
        plan
    }

    /// The served plan, plus the MTP layer when the configuration carries one: what a drafting
    /// deployment loads.
    #[must_use]
    pub fn served_and_mtp(config: &Glm5NextConfig) -> Self {
        let mut plan = Self::served(config);
        if config.num_nextn_predict_layers > 0 {
            plan.push_mtp(config);
        }
        plan
    }

    /// Plan the whole checkpoint, adding the MTP layer and the vision tower.
    ///
    /// The byte total of this plan is the identity the tests pin against the released
    /// `model.safetensors.index.json`'s `metadata.total_size`.
    #[must_use]
    pub fn checkpoint(config: &Glm5NextConfig) -> Self {
        let mut plan = Self::served_and_mtp(config);
        plan.push_vision(config);
        plan
    }

    /// The split of the tensor at `index`.
    #[must_use]
    pub fn split(&self, index: usize) -> Split {
        self.splits[index]
    }

    /// The split of one tensor by its released name.
    #[must_use]
    pub fn split_of(&self, name: &str) -> Option<Split> {
        self.index_of(name).map(|index| self.splits[index])
    }

    /// The spec rank `rank` of `world` holds for the tensor at `index`: the same name and dtype,
    /// the sliced shape. Which slice it is follows from the rank; the shape does not depend on it.
    ///
    /// # Errors
    /// When the split dimension does not divide by `world`.
    pub fn rank_spec(&self, index: usize, world: usize) -> Result<TensorSpec, String> {
        let spec = &self.specs()[index];
        let shape = self.splits[index]
            .rank_shape(&spec.shape, world)
            .map_err(|error| format!("{}: {error}", spec.name))?;
        Ok(TensorSpec {
            name: spec.name.clone(),
            dtype: spec.dtype,
            shape,
        })
    }

    /// The bytes one rank of `world` holds.
    ///
    /// # Errors
    /// When some split dimension does not divide by `world`.
    pub fn rank_bytes(&self, world: usize) -> Result<u128, String> {
        (0..self.len())
            .map(|index| Ok(self.rank_spec(index, world)?.storage_bytes()))
            .sum()
    }

    fn push(&mut self, name: &str, dtype: DType, shape: Vec<usize>, role: Role, split: Split) {
        self.plan.push(name, dtype, shape, role);
        self.splits.push(split);
    }

    /// One linear weight `[rows, columns]` under `prefix`: block-scaled FP8 with F32 tiles of side
    /// `fp8_block`, or BF16 when that is `None` (an unquantised module, or the BF16 release).
    fn push_linear(
        &mut self,
        prefix: &str,
        [rows, columns]: [usize; 2],
        fp8_block: Option<usize>,
        role: Role,
        split: Split,
    ) {
        match fp8_block {
            Some(block) => {
                self.plan
                    .push_fp8_block(prefix, rows, columns, block, DType::F32, role);
                self.splits.resize(self.plan.len(), split);
            }
            None => self.push(
                &format!("{prefix}.weight"),
                DType::BF16,
                vec![rows, columns],
                role,
                split,
            ),
        }
    }

    fn push_layer_norms(&mut self, prefix: &str, config: &Glm5NextConfig, role: Role) {
        for name in ["input_layernorm", "post_attention_layernorm"] {
            self.push(
                &format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![config.hidden_size],
                role,
                Split::Replicated,
            );
        }
    }

    fn push_hyper(&mut self, prefix: &str, config: &Glm5NextConfig) {
        let hc = HcGeometry::from(config);
        let rows = hc.hc_mix_count();
        self.push(
            &format!("{prefix}_fn"),
            DType::BF16,
            vec![rows, hc.hc_width()],
            Role::Hyper,
            Split::Replicated,
        );
        self.push(
            &format!("{prefix}_base"),
            DType::F32,
            vec![rows],
            Role::Hyper,
            Split::Replicated,
        );
        // One scale each for the pre, post and comb blocks of the projection.
        self.push(
            &format!("{prefix}_scale"),
            DType::F32,
            vec![3],
            Role::Hyper,
            Split::Replicated,
        );
    }

    fn push_kda(&mut self, prefix: &str, config: &Glm5NextConfig) {
        let hidden = config.hidden_size;
        let width = config.linear_width();
        let head = config.linear_head_dim;
        let role = Role::Kda;
        for name in ["q", "k", "v"] {
            self.push(
                &format!("{prefix}.{name}_proj.weight"),
                DType::BF16,
                vec![width, hidden],
                role,
                Split::Rows,
            );
            self.push(
                &format!("{prefix}.{name}_conv1d.weight"),
                DType::BF16,
                vec![width, 1, config.linear_conv_kernel_dim],
                role,
                Split::Rows,
            );
        }
        // The forget and output gates are low rank: a replicated head-wide bottleneck, then a
        // per-channel expansion sliced with the heads.
        for name in ["f", "g"] {
            self.push(
                &format!("{prefix}.{name}_a_proj.weight"),
                DType::BF16,
                vec![head, hidden],
                role,
                Split::Replicated,
            );
            self.push(
                &format!("{prefix}.{name}_b_proj.weight"),
                DType::BF16,
                vec![width, head],
                role,
                Split::Rows,
            );
        }
        self.push(
            &format!("{prefix}.b_proj.weight"),
            DType::BF16,
            vec![config.linear_num_heads, hidden],
            role,
            Split::Rows,
        );
        self.push(
            &format!("{prefix}.A_log"),
            DType::F32,
            vec![config.linear_num_heads],
            role,
            Split::Rows,
        );
        self.push(
            &format!("{prefix}.dt_bias"),
            DType::F32,
            vec![width],
            role,
            Split::Rows,
        );
        self.push(
            &format!("{prefix}.o_norm.weight"),
            DType::BF16,
            vec![head],
            role,
            Split::Replicated,
        );
        self.push(
            &format!("{prefix}.o_proj.weight"),
            DType::BF16,
            vec![hidden, width],
            role,
            Split::Columns,
        );
    }

    /// An MLA branch and, when the layer runs its own (`indexer`), its indexer; a shared layer
    /// reuses the previous layer's selection and has none. `role` overrides both roles for the MTP
    /// layer.
    fn push_dsa(
        &mut self,
        prefix: &str,
        config: &Glm5NextConfig,
        indexer: bool,
        role: Option<Role>,
    ) {
        let hidden = config.hidden_size;
        let heads = config.num_attention_heads;
        let attention = role.unwrap_or(Role::Dsa);
        let fp8 = config.fp8_block;
        self.push_linear(
            &format!("{prefix}.q_a_proj"),
            [config.q_lora_rank, hidden],
            fp8,
            attention,
            Split::Replicated,
        );
        self.push(
            &format!("{prefix}.q_a_layernorm.weight"),
            DType::BF16,
            vec![config.q_lora_rank],
            attention,
            Split::Replicated,
        );
        self.push_linear(
            &format!("{prefix}.q_b_proj"),
            [heads * config.qk_nope_head_dim, config.q_lora_rank],
            fp8,
            attention,
            Split::Rows,
        );
        // With no rotary half the MQA projection is the latent alone.
        self.push_linear(
            &format!("{prefix}.kv_a_proj_with_mqa"),
            [config.kv_lora_rank, hidden],
            fp8,
            attention,
            Split::Replicated,
        );
        self.push(
            &format!("{prefix}.kv_a_layernorm.weight"),
            DType::BF16,
            vec![config.kv_lora_rank],
            attention,
            Split::Replicated,
        );
        self.push(
            &format!("{prefix}.kv_b_proj.weight"),
            DType::BF16,
            vec![
                heads * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            ],
            attention,
            Split::Rows,
        );
        self.push_linear(
            &format!("{prefix}.o_proj"),
            [hidden, heads * config.v_head_dim],
            fp8,
            attention,
            Split::Columns,
        );
        if !indexer {
            return;
        }
        let indexer = role.unwrap_or(Role::Indexer);
        let prefix = format!("{prefix}.indexer");
        let index_width = config.index_head_dim;
        for (name, shape) in [
            (
                "wq_b.weight",
                vec![config.index_n_heads * index_width, config.q_lora_rank],
            ),
            ("wk.weight", vec![index_width, hidden]),
            ("k_norm.weight", vec![index_width]),
            ("k_norm.bias", vec![index_width]),
            ("weights_proj.weight", vec![config.index_n_heads, hidden]),
            ("index_kpool_compress_gate", vec![index_width, hidden]),
            (
                "index_kpool_compress_ape",
                vec![config.index_kpool, index_width],
            ),
        ] {
            self.push(
                &format!("{prefix}.{name}"),
                DType::BF16,
                shape,
                indexer,
                Split::Replicated,
            );
        }
    }

    fn push_dense_mlp(&mut self, prefix: &str, config: &Glm5NextConfig) {
        let (hidden, width) = (config.hidden_size, config.intermediate_size);
        for name in ["gate_proj", "up_proj"] {
            self.push_linear(
                &format!("{prefix}.{name}"),
                [width, hidden],
                config.fp8_block,
                Role::DenseMlp,
                Split::Rows,
            );
        }
        self.push_linear(
            &format!("{prefix}.down_proj"),
            [hidden, width],
            config.fp8_block,
            Role::DenseMlp,
            Split::Columns,
        );
    }

    /// A MoE block; `role` overrides every role for the MTP layer.
    fn push_moe(&mut self, prefix: &str, config: &Glm5NextConfig, role: Option<Role>) {
        let (hidden, width) = (config.hidden_size, config.moe_intermediate_size);
        let router = role.unwrap_or(Role::Router);
        self.push(
            &format!("{prefix}.gate.weight"),
            DType::BF16,
            vec![config.n_routed_experts, hidden],
            router,
            Split::Replicated,
        );
        self.push(
            &format!("{prefix}.gate.e_score_correction_bias"),
            DType::F32,
            vec![config.n_routed_experts],
            router,
            Split::Replicated,
        );
        let projections = [
            ("gate_proj", width, hidden, Split::Rows),
            ("up_proj", width, hidden, Split::Rows),
            ("down_proj", hidden, width, Split::Columns),
        ];
        let shared = role.unwrap_or(Role::SharedExpert);
        for (name, rows, columns, split) in projections {
            self.push_linear(
                &format!("{prefix}.shared_experts.{name}"),
                [rows, columns],
                config.fp8_block,
                shared,
                split,
            );
        }
        let experts = role.unwrap_or(Role::Experts);
        for expert in 0..config.n_routed_experts {
            for (name, rows, columns, split) in projections {
                self.push_linear(
                    &format!("{prefix}.experts.{expert}.{name}"),
                    [rows, columns],
                    config.fp8_block,
                    experts,
                    split,
                );
            }
        }
    }

    fn push_mtp(&mut self, config: &Glm5NextConfig) {
        // A sparse-attention + MoE layer without hyper-connections: the fused input is one stream.
        let hidden = config.hidden_size;
        let prefix = format!("{LM_PREFIX}.layers.{}", config.mtp_layer());
        for name in ["enorm", "hnorm", "shared_head.norm"] {
            self.push(
                &format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![hidden],
                Role::Mtp,
                Split::Replicated,
            );
        }
        self.push(
            &format!("{prefix}.eh_proj.weight"),
            DType::BF16,
            vec![hidden, 2 * hidden],
            Role::Mtp,
            Split::Replicated,
        );
        self.push_layer_norms(&prefix, config, Role::Mtp);
        self.push_dsa(
            &format!("{prefix}.self_attn"),
            config,
            true,
            Some(Role::Mtp),
        );
        self.push_moe(&format!("{prefix}.mlp"), config, Some(Role::Mtp));
    }

    fn push_vision(&mut self, config: &Glm5NextConfig) {
        let Some(vision) = &config.vision else {
            return;
        };
        let mut push = |name: &str, shape: Vec<usize>| {
            self.push(
                &format!("{VISION_PREFIX}.{name}"),
                DType::BF16,
                shape,
                Role::Vision,
                Split::Replicated,
            );
        };
        let (width, inner, out) = (
            vision.hidden_size,
            vision.intermediate_size,
            vision.out_hidden_size,
        );
        push(
            "patch_embed.proj.weight",
            vec![
                width,
                vision.in_channels,
                vision.temporal_patch_size,
                vision.patch_size,
                vision.patch_size,
            ],
        );
        push("patch_embed.proj.bias", vec![width]);
        for block in 0..vision.depth {
            let prefix = format!("blocks.{block}");
            push(&format!("{prefix}.norm1.weight"), vec![width]);
            push(&format!("{prefix}.norm2.weight"), vec![width]);
            push(&format!("{prefix}.attn.qkv.weight"), vec![3 * width, width]);
            push(&format!("{prefix}.attn.qkv.bias"), vec![3 * width]);
            push(
                &format!("{prefix}.attn.q_norm.weight"),
                vec![vision.head_dim()],
            );
            push(
                &format!("{prefix}.attn.k_norm.weight"),
                vec![vision.head_dim()],
            );
            push(&format!("{prefix}.attn.proj.weight"), vec![width, width]);
            push(&format!("{prefix}.attn.proj.bias"), vec![width]);
            for name in ["gate_proj", "up_proj"] {
                push(&format!("{prefix}.mlp.{name}.weight"), vec![inner, width]);
                push(&format!("{prefix}.mlp.{name}.bias"), vec![inner]);
            }
            push(
                &format!("{prefix}.mlp.down_proj.weight"),
                vec![width, inner],
            );
            push(&format!("{prefix}.mlp.down_proj.bias"), vec![width]);
        }
        push("post_layernorm.weight", vec![width]);
        let merge = vision.spatial_merge_size;
        push("downsample.weight", vec![out, width, merge, merge]);
        push("downsample.bias", vec![out]);
        push("merger.proj.weight", vec![out, out]);
        push("merger.post_projection_norm.weight", vec![out]);
        push("merger.post_projection_norm.bias", vec![out]);
        let projection = vision.projection_intermediate_size;
        push("merger.gate_proj.weight", vec![projection, out]);
        push("merger.up_proj.weight", vec![projection, out]);
        push("merger.down_proj.weight", vec![out, projection]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_ops::tensor_plan::ArchitectureTensorPlan;

    fn released() -> Glm5NextConfig {
        Glm5NextConfig::glm53_flash().expect("the carried config parses")
    }

    fn assert_shared_plan(
        shared: &ArchitectureTensorPlan,
        legacy: &Glm5NextTensorPlan,
        label: &str,
    ) {
        assert_eq!(shared.len(), legacy.len(), "{label}: tensor count");
        assert_eq!(
            shared.storage_bytes(),
            legacy.storage_bytes(),
            "{label}: byte total"
        );
        for spec in legacy.specs() {
            let found = shared
                .spec(&spec.name)
                .unwrap_or_else(|| panic!("{label}: {} is missing", spec.name));
            assert_eq!(found.dtype, spec.dtype, "{label}: {} dtype", spec.name);
            assert_eq!(found.shape, spec.shape, "{label}: {} shape", spec.name);
        }
    }

    #[test]
    fn the_shared_architecture_plan_reproduces_the_released_checkpoint() {
        let config = released();
        let architecture = crate::architecture::architecture_from_carried()
            .expect("the shared architecture parses");
        let tp = sparkflash_model::TpPlan::new(1, 0).expect("one rank");
        let shared =
            ArchitectureTensorPlan::checkpoint(&architecture, tp).expect("the shared plan builds");
        assert_shared_plan(&shared, &Glm5NextTensorPlan::checkpoint(&config), "fp8");
        let served = ArchitectureTensorPlan::served_and_mtp(&architecture, tp)
            .expect("the shared served plan builds");
        assert_shared_plan(
            &served,
            &Glm5NextTensorPlan::served_and_mtp(&config),
            "fp8 served",
        );
    }

    #[test]
    fn the_shared_architecture_plan_reproduces_every_rank_slice() {
        let config = released();
        let architecture = crate::architecture::architecture_from_carried()
            .expect("the shared architecture parses");
        let legacy = Glm5NextTensorPlan::served_and_mtp(&config);
        for world in [1usize, 2, 4] {
            for rank in 0..world {
                let tp = sparkflash_model::TpPlan::new(world, rank).expect("a rank");
                let shared = ArchitectureTensorPlan::served_and_mtp(&architecture, tp)
                    .expect("the shared plan builds");
                for index in 0..legacy.len() {
                    let expected = legacy
                        .rank_spec(index, world)
                        .expect("the released plan splits");
                    let found = shared
                        .rank_spec(&expected.name)
                        .unwrap_or_else(|| panic!("{} is missing", expected.name));
                    assert_eq!(found.dtype, expected.dtype, "{} dtype", expected.name);
                    assert_eq!(
                        found.shape, expected.shape,
                        "{} rank {rank}/{world}",
                        expected.name
                    );
                }
            }
        }
    }

    #[test]
    fn the_shared_architecture_plan_reproduces_the_bf16_checkpoint() {
        let config = Glm5NextConfig {
            fp8_block: None,
            ..released()
        };
        let mut architecture = crate::architecture::architecture_from_carried()
            .expect("the shared architecture parses");
        architecture.quantization.dense_format = sparkflash_model::architecture::WeightFormat::Bf16;
        architecture.quantization.expert_format =
            sparkflash_model::architecture::WeightFormat::Bf16;
        let tp = sparkflash_model::TpPlan::new(1, 0).expect("one rank");
        let shared =
            ArchitectureTensorPlan::checkpoint(&architecture, tp).expect("the shared plan builds");
        assert_shared_plan(&shared, &Glm5NextTensorPlan::checkpoint(&config), "bf16");
    }

    #[test]
    fn the_full_plan_matches_the_released_index_total() {
        let plan = Glm5NextTensorPlan::checkpoint(&released());
        assert_eq!(plan.len(), FULL_CHECKPOINT_TENSORS);
        assert_eq!(plan.storage_bytes(), FULL_CHECKPOINT_BYTES);
        assert_eq!(plan.role_bytes(Role::Mtp), MTP_BYTES);
        assert_eq!(plan.role_bytes(Role::Vision), VISION_BYTES);
    }

    #[test]
    fn without_tiles_the_plan_matches_the_bf16_release() {
        let config = Glm5NextConfig {
            fp8_block: None,
            ..released()
        };
        let plan = Glm5NextTensorPlan::checkpoint(&config);
        assert_eq!(plan.len(), BF16_CHECKPOINT_TENSORS);
        assert_eq!(plan.storage_bytes(), BF16_CHECKPOINT_BYTES);
        assert!(plan.specs().iter().all(|spec| spec.dtype != DType::F8E4M3));
    }

    #[test]
    fn the_routed_experts_are_the_released_fp8_bytes() {
        let plan = Glm5NextTensorPlan::served(&released());
        // 42 MoE layers x 288 experts x 3 projections, each [2048, 4096] or its transpose: one
        // E4M3 code per value and one F32 scale per 128x128 tile.
        let per_projection = 2048u128 * 4096 + 16 * 32 * 4;
        assert_eq!(
            plan.role_bytes(Role::Experts),
            42 * 288 * 3 * per_projection
        );
    }

    #[test]
    fn the_served_plan_accepts_the_released_checkpoints_unserved_tensors() {
        let config = released();
        let served = Glm5NextTensorPlan::served_and_mtp(&config);
        let full = Glm5NextTensorPlan::checkpoint(&config);
        let accepted = served
            .conformance_with_unserved(full.names().map(str::to_owned), &full)
            .expect("the released checkpoint is a superset of the served plan");
        assert_eq!(accepted, 347, "the vision tower is the only unserved part");
    }

    #[test]
    fn the_plan_names_the_released_shapes() {
        let plan = Glm5NextTensorPlan::checkpoint(&released());
        let shape = |name: &str| {
            plan.spec(&format!("{LM_PREFIX}.{name}"))
                .unwrap_or_else(|| panic!("{name} is planned"))
                .shape
                .clone()
        };
        assert_eq!(shape("layers.0.self_attn.q_proj.weight"), vec![8192, 4096]);
        assert_eq!(
            shape("layers.0.self_attn.q_conv1d.weight"),
            vec![8192, 1, 4]
        );
        assert_eq!(shape("layers.0.self_attn.f_b_proj.weight"), vec![8192, 128]);
        assert_eq!(
            shape("layers.0.mlp.down_proj.weight_scale_inv"),
            vec![32, 96]
        );
        assert_eq!(
            shape("layers.3.self_attn.q_b_proj.weight"),
            vec![16_384, 1536]
        );
        assert_eq!(
            shape("layers.3.self_attn.kv_b_proj.weight"),
            vec![32_768, 512]
        );
        assert_eq!(
            shape("layers.3.self_attn.o_proj.weight"),
            vec![4096, 16_384]
        );
        assert_eq!(
            shape("layers.3.self_attn.indexer.wq_b.weight"),
            vec![4096, 1536]
        );
        assert_eq!(shape("layers.4.hc_ffn_fn"), vec![24, 16_384]);
        assert_eq!(shape("layers.45.eh_proj.weight"), vec![4096, 8192]);
        assert_eq!(
            shape("layers.45.mlp.experts.287.down_proj.weight"),
            vec![4096, 2048]
        );
        assert!(
            plan.spec(&format!("{LM_PREFIX}.layers.3.self_attn.q_proj.weight"))
                .is_none(),
            "a sparse layer must not plan a KDA projection"
        );
        assert!(
            plan.spec(&format!("{LM_PREFIX}.layers.45.hc_attn_fn"))
                .is_none(),
            "the MTP layer has no hyper-connections"
        );
    }

    #[test]
    fn four_ranks_split_heads_and_widths_and_replicate_the_rest() {
        let plan = Glm5NextTensorPlan::served(&released());
        let rank_shape = |name: &str| {
            let name = format!("{LM_PREFIX}.{name}");
            let index = plan
                .names()
                .position(|planned| planned == name)
                .unwrap_or_else(|| panic!("{name} is planned"));
            plan.rank_spec(index, 4).expect("the split divides").shape
        };
        // 16 of 64 KDA heads.
        assert_eq!(
            rank_shape("layers.0.self_attn.q_proj.weight"),
            vec![2048, 4096]
        );
        assert_eq!(rank_shape("layers.0.self_attn.A_log"), vec![16]);
        assert_eq!(
            rank_shape("layers.0.self_attn.o_proj.weight"),
            vec![4096, 2048]
        );
        assert_eq!(
            rank_shape("layers.0.self_attn.f_a_proj.weight"),
            vec![128, 4096]
        );
        // 16 of 64 MLA heads, each 256 nope + 256 value rows of kv_b.
        assert_eq!(
            rank_shape("layers.3.self_attn.kv_b_proj.weight"),
            vec![8192, 512]
        );
        assert_eq!(
            rank_shape("layers.3.self_attn.indexer.wk.weight"),
            vec![128, 4096]
        );
        // Expert width 512 per rank: four whole 128-wide tiles of every projection.
        assert_eq!(
            rank_shape("layers.4.mlp.experts.0.gate_proj.weight_scale_inv"),
            vec![4, 32]
        );
        assert_eq!(
            rank_shape("layers.4.mlp.experts.0.down_proj.weight"),
            vec![4096, 512]
        );
        assert_eq!(
            rank_shape("layers.4.mlp.experts.0.down_proj.weight_scale_inv"),
            vec![32, 4]
        );
        // 16 of 64 MLA heads of output columns, in whole tiles.
        assert_eq!(
            rank_shape("layers.3.self_attn.o_proj.weight_scale_inv"),
            vec![32, 32]
        );
        assert!(
            plan.rank_bytes(3).is_err(),
            "64 heads do not split three ways"
        );
    }

    #[test]
    fn a_rank_of_four_holds_about_a_quarter_plus_the_replicated_parts() {
        let config = released();
        let plan = Glm5NextTensorPlan::served_and_mtp(&config);
        let whole = plan.storage_bytes();
        let rank = plan
            .rank_bytes(4)
            .expect("the released model splits four ways");
        assert_eq!(plan.rank_bytes(1), Ok(whole));
        // The replicated parts (embedding, mHC, indexers, routers, low-rank bottlenecks) put a
        // rank above a quarter; the routed experts keep it far below a half.
        assert!(rank > whole / 4 && rank < whole / 3, "{rank} of {whole}");
    }

    #[test]
    fn the_tiny_plan_is_small_but_complete() {
        let config = Glm5NextConfig::tiny();
        let plan = Glm5NextTensorPlan::served_and_mtp(&config);
        assert!(plan.len() > 100);
        assert!(plan.rank_bytes(2).is_ok());
        assert_eq!(
            plan.spec(&format!("{LM_PREFIX}.layers.1.self_attn.kv_b_proj.weight"))
                .expect("the sparse layer is planned")
                .shape,
            vec![4 * (8 + 8), 16]
        );
        // Layer 2 reuses layer 1's selection, so only layer 1 carries an indexer.
        let wk = |layer: usize| {
            plan.spec(&format!(
                "{LM_PREFIX}.layers.{layer}.self_attn.indexer.wk.weight"
            ))
        };
        assert!(wk(1).is_some());
        assert!(wk(2).is_none());
        assert!(
            plan.spec(&format!("{LM_PREFIX}.layers.2.self_attn.kv_b_proj.weight"))
                .is_some()
        );
        assert_eq!(
            plan.spec(&format!("{LM_PREFIX}.layers.4.eh_proj.weight"))
                .expect("the MTP layer is planned")
                .shape,
            vec![32, 64]
        );
    }
}

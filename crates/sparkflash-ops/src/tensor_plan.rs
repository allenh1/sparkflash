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

//! The tensor plan of an [Architecture] at one tensor-parallel rank.
//!
//! The Qwen, GLM and DeepSeek arms all read the same released `config.json` through
//! [Architecture]. This module turns that walk into names, shapes and rank slices, so a released
//! index's tensor count and byte total are checked against the shared vocabulary rather than a
//! second family parser. The family-specific names live here, not in a model crate.
//!
//! [ArchitectureTensorPlan::served] is what a rank uploads, [ArchitectureTensorPlan::checkpoint]
//! adds the unserved vision tower, and [ArchitectureTensorPlan::rank_spec_at] is the rank-local
//! shape the weight walk reads. The split rules are the ones the arms already execute: heads and
//! intermediate widths divide evenly, an FP8 expert is a run of whole `block_n`-wide tiles (D91),
//! and an NVFP4 expert is padded to the 64-column MMA tile above TP1.

use sparkflash_model::architecture::{
    Architecture, AttentionKind, DenseMlp, Ffn, Kda, Layer, Moe, PleGeometry, Residual, ScaleDType,
    Side, WeightFormat,
};
use sparkflash_model::{TpAxis, TpPlan};
use sparkflash_weights::rng::is_prime;
use sparkflash_weights::{DType, TensorPlan, TensorSpec};

use crate::engine_plan::PlanError;

/// The prefix every language-model tensor carries in the released checkpoints.
pub const LM_PREFIX: &str = "model.language_model";

/// The prefix of the vision tower's tensors.
pub const VISION_PREFIX: &str = "model.visual";

/// What part of the model a tensor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// The token embedding table.
    Embedding,
    /// The (untied) LM head.
    LmHead,
    /// A decoder or final norm.
    Norm,
    /// A hyper-connection mixer.
    Hyper,
    /// The routed-expert router and its selection bias.
    Router,
    /// The shared expert and its gate.
    SharedExpert,
    /// One routed expert's weights and scales.
    Experts,
    /// A KDA, GDN, GQA or MLA branch.
    Attention,
    /// A sparse indexer.
    Indexer,
    /// A dense SwiGLU MLP.
    DenseMlp,
    /// The PLE projections, norms and convolution.
    PleAux,
    /// The host-paged PLE table and its hash constants.
    PleTable,
    /// The MTP stage.
    Mtp,
    /// The vision tower.
    Vision,
}

/// The stored format of one expert class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertFormat {
    /// A 4-bit E2M1 expert: NVFP4's per-group scale or MXFP4's per-block scale. The two share
    /// the ops library's [Precision::NvFp4] operand class but not their scale granularity, so
    /// the released group/block is carried here and used for every scale shape.
    Nvfp4 {
        group: usize,
    },
    Fp8 {
        block: usize,
        scale: DType,
    },
    Bf16,
}

impl ExpertFormat {
    fn of(format: WeightFormat) -> Self {
        match format {
            WeightFormat::NvFp4 { group_size } => Self::Nvfp4 {
                group: group_size.max(1),
            },
            WeightFormat::MxFp4 { block } => Self::Nvfp4 {
                group: block.max(1),
            },
            WeightFormat::Fp8Tile {
                block_n,
                scale_dtype,
                ..
            } => Self::Fp8 {
                block: block_n.max(1),
                scale: scale_dtype_of(scale_dtype),
            },
            WeightFormat::MxFp8 { .. } | WeightFormat::Bf16 => Self::Bf16,
        }
    }
}

fn scale_dtype_of(scale: ScaleDType) -> DType {
    match scale {
        ScaleDType::F32 => DType::F32,
        ScaleDType::Bf16 => DType::BF16,
        ScaleDType::E4M3 => DType::F8E4M3,
    }
}

/// One tensor's rank-local run along its sharded axis, in elements.
///
/// This is the D91 whole-tile slice a loader needs: `offset` is the first element of the
/// logical axis this rank holds, `count` is how many it holds, and `total` is the logical
/// axis. Concatenating every rank's `[offset, offset + count)` in rank order reconstructs
/// `[0, total)`; for an FP8 expert those are whole `block_n` tiles, never a fractional one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankRun {
    /// The first element of the logical axis this rank holds.
    pub offset: usize,
    /// The number of elements this rank holds.
    pub count: usize,
    /// The logical length of the axis.
    pub total: usize,
}

impl RankRun {
    /// The rank's run of `total` split evenly, as [TpPlan::split] computes it.
    fn even(split: sparkflash_model::Split) -> Self {
        Self {
            offset: split.offset,
            count: split.per_rank,
            total: split.total,
        }
    }

    /// The rank's run of whole `tile`-wide blocks, as [TpPlan::split_tiles] computes it.
    fn tiled(split: sparkflash_model::Split) -> Self {
        Self {
            offset: split.offset,
            count: split.per_rank,
            total: split.total,
        }
    }
}

/// An architecture's tensors, logically and at one rank.
#[derive(Debug, Clone)]
pub struct ArchitectureTensorPlan {
    plan: TensorPlan<Role>,
    rank: Vec<TensorSpec>,
    runs: Vec<Option<RankRun>>,
    tp: TpPlan,
}

impl ArchitectureTensorPlan {
    /// Plan the served set for rank `tp.rank()` of `tp.world()`.
    ///
    /// # Errors
    /// A family this plan does not name, or a world a sharded axis does not divide.
    pub fn served(architecture: &Architecture, tp: TpPlan) -> Result<Self, PlanError> {
        Self::build(architecture, tp, false, false, false)
    }

    /// Plan the served set plus the MTP stage, when the release has one the served set omits.
    ///
    /// # Errors
    /// As [Self::served].
    pub fn served_and_mtp(architecture: &Architecture, tp: TpPlan) -> Result<Self, PlanError> {
        Self::build(architecture, tp, true, false, false)
    }

    /// Plan the whole checkpoint, including the vision tower.
    ///
    /// # Errors
    /// As [Self::served].
    pub fn checkpoint(architecture: &Architecture, tp: TpPlan) -> Result<Self, PlanError> {
        Self::build(architecture, tp, true, true, true)
    }

    fn build(
        architecture: &Architecture,
        tp: TpPlan,
        mtp: bool,
        vision: bool,
        mtp_all_worlds: bool,
    ) -> Result<Self, PlanError> {
        let mut builder = Builder::new(tp, mtp, vision, mtp_all_worlds);
        let dense = architecture.quantization.dense_format;
        let experts = architecture.quantization.expert_format;
        builder.dense_format = dense;
        builder.expert_format = if matches!(experts, WeightFormat::Bf16)
            && matches!(dense, WeightFormat::Fp8Tile { .. })
        {
            // GLM names one `fp8` format for every weight; the dense class is the fallback.
            dense
        } else {
            experts
        };
        match architecture.family.as_str() {
            "qwen4_exp" => builder.qwen(architecture, mtp)?,
            "glm5_next" => builder.glm(architecture, mtp)?,
            other => {
                return Err(PlanError::Invalid(format!(
                    "the shared tensor plan does not name the {other:?} family"
                )));
            }
        }
        Ok(builder.finish())
    }

    /// The tensor-parallel plan this plan was built at.
    #[must_use]
    pub const fn tp(&self) -> TpPlan {
        self.tp
    }

    /// The logical spec at `index`.
    #[must_use]
    pub fn spec_at(&self, index: usize) -> &TensorSpec {
        &self.plan.specs()[index]
    }

    /// The rank-local spec at `index`.
    #[must_use]
    pub fn rank_spec_at(&self, index: usize) -> &TensorSpec {
        &self.rank[index]
    }

    /// The rank-local run at `index`, when the tensor is sharded along an axis.
    ///
    /// A replicated tensor returns `None`: every rank holds the whole axis, so there is no
    /// slice to drive. The run is in elements of the tensor's sharded axis, not bytes.
    #[must_use]
    pub fn rank_run_at(&self, index: usize) -> Option<RankRun> {
        self.runs[index]
    }

    /// The rank-local run of one tensor by its released name.
    #[must_use]
    pub fn rank_run(&self, name: &str) -> Option<RankRun> {
        self.plan
            .index_of(name)
            .and_then(|index| self.rank_run_at(index))
    }

    /// The rank-local spec of one tensor by its released name.
    #[must_use]
    pub fn rank_spec(&self, name: &str) -> Option<&TensorSpec> {
        self.plan
            .index_of(name)
            .map(|index| self.rank_spec_at(index))
    }

    /// The logical bytes of the plan, as the checkpoint stores them.
    #[must_use]
    pub fn storage_bytes(&self) -> u128 {
        self.plan.storage_bytes()
    }

    /// The bytes this rank holds.
    #[must_use]
    pub fn rank_bytes(&self) -> u128 {
        self.rank.iter().map(TensorSpec::storage_bytes).sum()
    }

    /// The logical bytes of one role.
    #[must_use]
    pub fn role_bytes(&self, role: Role) -> u128 {
        self.plan.role_bytes(role)
    }

    /// The rank-local bytes of one role.
    #[must_use]
    pub fn rank_role_bytes(&self, role: Role) -> u128 {
        self.rank
            .iter()
            .zip(self.roles())
            .filter(|(_, tagged)| *tagged == role)
            .map(|(spec, _)| spec.storage_bytes())
            .sum()
    }

    /// The role of every planned tensor, in order.
    pub fn roles(&self) -> impl Iterator<Item = Role> + '_ {
        (0..self.plan.len()).map(|index| self.plan.role(index))
    }

    /// The logical plan, so callers keep the shared name and byte bookkeeping.
    #[must_use]
    pub const fn plan(&self) -> &TensorPlan<Role> {
        &self.plan
    }
}

impl std::ops::Deref for ArchitectureTensorPlan {
    type Target = TensorPlan<Role>;

    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinearSplit {
    Whole,
    Rows,
    Columns,
}

struct Builder {
    plan: TensorPlan<Role>,
    rank: Vec<TensorSpec>,
    runs: Vec<Option<RankRun>>,
    tp: TpPlan,
    mtp: bool,
    vision: bool,
    /// Whether MTP is planned at every world. The checkpoint converter needs it; the served set
    /// keeps the world gate (a non-replicated MTP stage is not loaded by a multi-rank serve).
    mtp_all_worlds: bool,
    dense_format: WeightFormat,
    expert_format: WeightFormat,
}

impl Builder {
    fn new(tp: TpPlan, mtp: bool, vision: bool, mtp_all_worlds: bool) -> Self {
        Self {
            plan: TensorPlan::default(),
            rank: Vec::new(),
            runs: Vec::new(),
            tp,
            mtp,
            vision,
            mtp_all_worlds,
            dense_format: WeightFormat::Bf16,
            expert_format: WeightFormat::Bf16,
        }
    }

    fn finish(self) -> ArchitectureTensorPlan {
        ArchitectureTensorPlan {
            plan: self.plan,
            rank: self.rank,
            runs: self.runs,
            tp: self.tp,
        }
    }

    fn push(
        &mut self,
        name: String,
        dtype: DType,
        shape: Vec<usize>,
        rank: Vec<usize>,
        run: Option<RankRun>,
        role: Role,
    ) {
        self.plan.push(&name, dtype, shape, role);
        self.rank.push(TensorSpec {
            name,
            dtype,
            shape: rank,
        });
        self.runs.push(run);
    }

    fn whole(&mut self, name: String, dtype: DType, shape: Vec<usize>, role: Role) {
        self.push(name, dtype, shape.clone(), shape, None, role);
    }

    fn rows(
        &mut self,
        name: String,
        dtype: DType,
        shape: [usize; 2],
        role: Role,
    ) -> Result<(), PlanError> {
        let split = self.tp.split(shape[0], TpAxis::Heads, 1)?;
        self.push(
            name,
            dtype,
            vec![shape[0], shape[1]],
            vec![split.padded, shape[1]],
            Some(RankRun::even(split)),
            role,
        );
        Ok(())
    }

    fn columns(
        &mut self,
        name: String,
        dtype: DType,
        shape: [usize; 2],
        role: Role,
    ) -> Result<(), PlanError> {
        let split = self.tp.split(shape[1], TpAxis::Intermediate, 1)?;
        self.push(
            name,
            dtype,
            vec![shape[0], shape[1]],
            vec![shape[0], split.padded],
            Some(RankRun::even(split)),
            role,
        );
        Ok(())
    }

    fn kv_rows(
        &mut self,
        name: String,
        dtype: DType,
        shape: [usize; 2],
        role: Role,
    ) -> Result<(), PlanError> {
        let split = self.tp.split(shape[0], TpAxis::KvHeads, 1)?;
        self.push(
            name,
            dtype,
            vec![shape[0], shape[1]],
            vec![split.padded, shape[1]],
            Some(RankRun::even(split)),
            role,
        );
        Ok(())
    }

    /// A 1-D vector split by its rows.
    fn row_vector(
        &mut self,
        name: String,
        dtype: DType,
        total: usize,
        role: Role,
    ) -> Result<(), PlanError> {
        let split = self.tp.split(total, TpAxis::Heads, 1)?;
        self.push(
            name,
            dtype,
            vec![total],
            vec![split.padded],
            Some(RankRun::even(split)),
            role,
        );
        Ok(())
    }

    /// A 3-D convolution filter `[width, 1, taps]`, split on its rows.
    fn conv_rows(
        &mut self,
        name: String,
        dtype: DType,
        width: usize,
        taps: usize,
        role: Role,
    ) -> Result<(), PlanError> {
        let split = self.tp.split(width, TpAxis::Heads, 1)?;
        self.push(
            name,
            dtype,
            vec![width, 1, taps],
            vec![split.padded, 1, taps],
            Some(RankRun::even(split)),
            role,
        );
        Ok(())
    }

    /// An expert projection: `[width, hidden]` (`rows`) or `[hidden, width]` (`columns`).
    fn expert(
        &mut self,
        prefix: &str,
        width: usize,
        hidden: usize,
        format: ExpertFormat,
        rows: bool,
        role: Role,
    ) -> Result<(), PlanError> {
        let (local, run) = self.expert_width(width, format)?;
        let shape = if rows {
            vec![width, hidden]
        } else {
            vec![hidden, width]
        };
        let rank_shape = if rows {
            vec![local, hidden]
        } else {
            vec![hidden, local]
        };
        self.push(
            format!("{prefix}.weight"),
            match format {
                ExpertFormat::Nvfp4 { .. } => DType::F4E2M1,
                ExpertFormat::Fp8 { .. } => DType::F8E4M3,
                ExpertFormat::Bf16 => DType::BF16,
            },
            shape,
            rank_shape,
            run,
            role,
        );
        match format {
            ExpertFormat::Nvfp4 { group } => {
                let scale_shape = if rows {
                    vec![width, hidden.div_ceil(group)]
                } else {
                    vec![hidden, width.div_ceil(group)]
                };
                let scale_rank = if rows {
                    vec![local, hidden.div_ceil(group)]
                } else {
                    vec![hidden, local.div_ceil(group)]
                };
                self.push(
                    format!("{prefix}.weight_scale"),
                    DType::F8E4M3,
                    scale_shape,
                    scale_rank,
                    None,
                    role,
                );
                self.whole(
                    format!("{prefix}.weight_scale_2"),
                    DType::F32,
                    Vec::new(),
                    role,
                );
                self.whole(
                    format!("{prefix}.input_scale"),
                    DType::F32,
                    Vec::new(),
                    role,
                );
            }
            ExpertFormat::Fp8 { block, scale } => {
                let scale_shape = if rows {
                    vec![width.div_ceil(block), hidden.div_ceil(block)]
                } else {
                    vec![hidden.div_ceil(block), width.div_ceil(block)]
                };
                let scale_rank = if rows {
                    vec![local.div_ceil(block), hidden.div_ceil(block)]
                } else {
                    vec![hidden.div_ceil(block), local.div_ceil(block)]
                };
                // The inverse scale covers a whole tile, so its run is the weight's run in
                // tiles: the same offset and count, divided by the block width.
                let scale_run = run.map(|run| RankRun {
                    offset: run.offset / block,
                    count: run.count / block,
                    total: run.total / block,
                });
                self.push(
                    format!("{prefix}.weight_scale_inv"),
                    scale,
                    scale_shape,
                    scale_rank,
                    scale_run,
                    role,
                );
            }
            ExpertFormat::Bf16 => {}
        }
        Ok(())
    }

    /// The rank-local width of an expert with logical `total` intermediate columns, and the
    /// whole-tile run that width is when the split is one.
    ///
    /// An NVFP4 expert at TP>1 is padded to the 64-column MMA tile above an even width, which is
    /// not a run of whole stored tiles, so it reports no run rather than a misleading one. FP8
    /// experts and even splits are exact runs.
    fn expert_width(
        &self,
        total: usize,
        format: ExpertFormat,
    ) -> Result<(usize, Option<RankRun>), PlanError> {
        Ok(match format {
            ExpertFormat::Nvfp4 { .. } => {
                if self.tp.world() == 1 {
                    (
                        total,
                        Some(RankRun::even(self.tp.split(
                            total,
                            TpAxis::Intermediate,
                            1,
                        )?)),
                    )
                } else {
                    ((total / self.tp.world()).next_multiple_of(64), None)
                }
            }
            ExpertFormat::Fp8 { block, .. } if block > 1 => {
                let split = self.tp.split_tiles(total, block, TpAxis::Intermediate)?;
                (split.padded, Some(RankRun::tiled(split)))
            }
            _ => {
                let split = self.tp.split(total, TpAxis::Intermediate, 16)?;
                (split.padded, Some(RankRun::even(split)))
            }
        })
    }

    /// A block-scaled dense linear (GLM's FP8 projections) or its BF16 form.
    fn dense_linear(
        &mut self,
        prefix: &str,
        rows: usize,
        columns: usize,
        format: WeightFormat,
        role: Role,
        split: LinearSplit,
    ) -> Result<(), PlanError> {
        match format {
            WeightFormat::Fp8Tile {
                block_n,
                scale_dtype,
                ..
            } if block_n > 0 => {
                let block = block_n.max(1);
                let scale = scale_dtype_of(scale_dtype);
                let split = match split {
                    LinearSplit::Whole => None,
                    LinearSplit::Rows => Some(self.tp.split(rows, TpAxis::Heads, 1)?),
                    LinearSplit::Columns => {
                        Some(self.tp.split(columns, TpAxis::Intermediate, 1)?)
                    }
                };
                let scale_shape = [rows.div_ceil(block), columns.div_ceil(block)];
                let weight_rank = match (split, split) {
                    (Some(split), _) if split.total == rows => vec![split.padded, columns],
                    (Some(split), _) => vec![rows, split.padded],
                    (None, _) => vec![rows, columns],
                };
                let scale_rank = match (split, split) {
                    (Some(split), _) if split.total == rows => {
                        vec![split.padded.div_ceil(block), scale_shape[1]]
                    }
                    (Some(split), _) => {
                        vec![scale_shape[0], split.padded.div_ceil(block)]
                    }
                    (None, _) => scale_shape.to_vec(),
                };
                let run = split.map(RankRun::even);
                let scale_run = split.map(|split| RankRun {
                    offset: split.offset / block,
                    count: split.per_rank.div_ceil(block),
                    total: split.total.div_ceil(block),
                });
                self.push(
                    format!("{prefix}.weight"),
                    DType::F8E4M3,
                    vec![rows, columns],
                    weight_rank,
                    run,
                    role,
                );
                self.push(
                    format!("{prefix}.weight_scale_inv"),
                    scale,
                    scale_shape.to_vec(),
                    scale_rank,
                    scale_run,
                    role,
                );
            }
            _ => match split {
                LinearSplit::Whole => {
                    self.whole(
                        format!("{prefix}.weight"),
                        DType::BF16,
                        vec![rows, columns],
                        role,
                    );
                }
                LinearSplit::Rows => {
                    self.rows(
                        format!("{prefix}.weight"),
                        DType::BF16,
                        [rows, columns],
                        role,
                    )?;
                }
                LinearSplit::Columns => {
                    self.columns(
                        format!("{prefix}.weight"),
                        DType::BF16,
                        [rows, columns],
                        role,
                    )?;
                }
            },
        }
        Ok(())
    }

    fn qwen(&mut self, arch: &Architecture, include_mtp: bool) -> Result<(), PlanError> {
        let hidden = arch.hidden_size;
        let Some(reference) = arch.layers.first() else {
            return Err(PlanError::Invalid(
                "the architecture has no layers".to_owned(),
            ));
        };
        let hyper = hyper_of(reference)?;
        let hyper_width = hyper.streams * hidden;
        self.whole(
            format!("{LM_PREFIX}.embed_tokens.weight"),
            DType::BF16,
            vec![arch.vocab_size, hidden],
            Role::Embedding,
        );
        self.whole(
            "lm_head.weight".to_owned(),
            DType::BF16,
            vec![arch.vocab_size, hidden],
            Role::LmHead,
        );
        self.qwen_hyper(
            &format!("{LM_PREFIX}.hyper_connection_mixer"),
            hyper,
            hidden,
            false,
            Role::Hyper,
        );
        for layer in &arch.layers {
            let prefix = format!("{LM_PREFIX}.layers.{}", layer.index);
            let layer_hyper = hyper_of(layer)?;
            self.qwen_hyper(
                &format!("{prefix}.attn_hyper_connection"),
                layer_hyper,
                hidden,
                true,
                Role::Hyper,
            );
            self.qwen_hyper(
                &format!("{prefix}.mlp_hyper_connection"),
                layer_hyper,
                hidden,
                true,
                Role::Hyper,
            );
            let Ffn::Moe(moe) = &layer.ffn else {
                return Err(PlanError::Invalid(format!(
                    "Qwen layer {} is not a MoE layer",
                    layer.index
                )));
            };
            self.whole(
                format!("{prefix}.mlp.gate.weight"),
                DType::BF16,
                vec![moe.routed_experts, hidden],
                Role::Router,
            );
            for name in ["gate_proj", "up_proj"] {
                self.rows(
                    format!("{prefix}.mlp.shared_expert.{name}.weight"),
                    DType::BF16,
                    [moe.shared_intermediate, hidden],
                    Role::SharedExpert,
                )?;
            }
            self.columns(
                format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                DType::BF16,
                [hidden, moe.shared_intermediate],
                Role::SharedExpert,
            )?;
            self.whole(
                format!("{prefix}.mlp.shared_expert_gate.weight"),
                DType::BF16,
                vec![1, hidden],
                Role::SharedExpert,
            );
            // Qwen's official FP8 release stores one BF16 inverse scale a 128x128 expert tile,
            // whatever scale dtype the ModelOpt table names for the same arithmetic.
            let expert_format = match ExpertFormat::of(arch.quantization.expert_format) {
                ExpertFormat::Fp8 { block, .. } => ExpertFormat::Fp8 {
                    block,
                    scale: DType::BF16,
                },
                other => other,
            };
            for expert in 0..moe.routed_experts {
                for (name, rows) in [("gate_proj", true), ("up_proj", true), ("down_proj", false)] {
                    self.expert(
                        &format!("{prefix}.mlp.experts.{expert}.{name}"),
                        moe.intermediate,
                        hidden,
                        expert_format,
                        rows,
                        Role::Experts,
                    )?;
                }
            }
            match &layer.attention.kind {
                AttentionKind::Gdn(gdn) => {
                    let p = format!("{prefix}.linear_attn");
                    let key_width = gdn.key_heads * gdn.key_dim;
                    let value_width = gdn.value_heads * gdn.value_dim;
                    let conv_width = 2 * key_width + value_width;
                    self.rows(
                        format!("{p}.in_proj_qkv.weight"),
                        DType::BF16,
                        [conv_width, hidden],
                        Role::Attention,
                    )?;
                    self.rows(
                        format!("{p}.in_proj_z.weight"),
                        DType::BF16,
                        [value_width, hidden],
                        Role::Attention,
                    )?;
                    for name in ["in_proj_a", "in_proj_b"] {
                        self.rows(
                            format!("{p}.{name}.weight"),
                            DType::BF16,
                            [gdn.value_heads, hidden],
                            Role::Attention,
                        )?;
                    }
                    self.columns(
                        format!("{p}.out_proj.weight"),
                        DType::BF16,
                        [hidden, value_width],
                        Role::Attention,
                    )?;
                    self.conv_rows(
                        format!("{p}.conv1d.weight"),
                        DType::BF16,
                        conv_width,
                        gdn.conv_kernel,
                        Role::Attention,
                    )?;
                    for name in ["A_log", "dt_bias"] {
                        self.row_vector(
                            format!("{p}.{name}"),
                            DType::BF16,
                            gdn.value_heads,
                            Role::Attention,
                        )?;
                    }
                    self.whole(
                        format!("{p}.norm.weight"),
                        DType::BF16,
                        vec![gdn.value_dim],
                        Role::Attention,
                    );
                }
                AttentionKind::Gqa { kv_heads, .. } => {
                    let p = format!("{prefix}.self_attn");
                    let q_width = layer.attention.heads * layer.attention.head_dim * 2;
                    let kv_width = kv_heads * layer.attention.head_dim;
                    self.rows(
                        format!("{p}.q_proj.weight"),
                        DType::BF16,
                        [q_width, hidden],
                        Role::Attention,
                    )?;
                    for name in ["k_proj", "v_proj"] {
                        self.kv_rows(
                            format!("{p}.{name}.weight"),
                            DType::BF16,
                            [kv_width, hidden],
                            Role::Attention,
                        )?;
                    }
                    self.columns(
                        format!("{p}.o_proj.weight"),
                        DType::BF16,
                        [hidden, layer.attention.heads * layer.attention.head_dim],
                        Role::Attention,
                    )?;
                    for name in ["q_norm", "k_norm"] {
                        self.whole(
                            format!("{p}.{name}.weight"),
                            DType::BF16,
                            vec![layer.attention.head_dim],
                            Role::Attention,
                        );
                    }
                    if let Some(indexer) = &layer.attention.indexer {
                        let width = (indexer.heads + indexer.kv_heads) * indexer.head_dim;
                        self.whole(
                            format!("{p}.indexer.index_qk_proj.weight"),
                            DType::BF16,
                            vec![width, hidden],
                            Role::Indexer,
                        );
                        for name in ["q_layernorm", "k_layernorm"] {
                            self.whole(
                                format!("{p}.indexer.{name}.weight"),
                                DType::BF16,
                                vec![indexer.head_dim],
                                Role::Indexer,
                            );
                        }
                    }
                }
                other => {
                    return Err(PlanError::Invalid(format!(
                        "Qwen layer {} runs {other:?}",
                        layer.index
                    )));
                }
            }
            if let Some(ple) = layer.sides.iter().find_map(|side| match side {
                Side::Ple(ple) => Some(ple),
                Side::Engram(_) => None,
            }) {
                let p = format!("{prefix}.ple");
                self.whole(
                    format!("{p}.key_proj.weight"),
                    DType::BF16,
                    vec![hyper_width, ple.embed_dim],
                    Role::PleAux,
                );
                self.whole(
                    format!("{p}.value_proj.weight"),
                    DType::BF16,
                    vec![hidden, ple.embed_dim],
                    Role::PleAux,
                );
                for name in ["norm_key", "norm_query", "norm_conv"] {
                    self.whole(
                        format!("{p}.{name}.weight"),
                        DType::BF16,
                        vec![hyper_width],
                        Role::PleAux,
                    );
                }
                self.whole(
                    format!("{p}.conv1d.weight"),
                    DType::BF16,
                    vec![hyper_width, 1, ple.conv_kernel],
                    Role::PleAux,
                );
                let heads = ple.ngram_size.saturating_sub(1) * ple.heads_per_ngram;
                let head_dim = ple.embed_dim / heads.max(1);
                self.whole(
                    format!("{p}.ple_embedding.layer_multipliers"),
                    DType::I64,
                    vec![ple.ngram_size],
                    Role::PleTable,
                );
                self.whole(
                    format!("{p}.ple_embedding.ngram_heads_offsets"),
                    DType::I64,
                    vec![heads],
                    Role::PleTable,
                );
                self.whole(
                    format!("{p}.ple_embedding.ngram_heads_vocab_sizes"),
                    DType::I64,
                    vec![heads],
                    Role::PleTable,
                );
                self.whole(
                    format!("{p}.ple_embedding.ngram_embedding.weight_scale"),
                    DType::BF16,
                    vec![1],
                    Role::PleTable,
                );
                let rows = ple_padded_rows(ple);
                let per_shard = rows as usize / ple.split_parts.max(1);
                for shard in 0..ple.split_parts {
                    self.whole(
                        format!("{p}.ple_embedding.ngram_embedding.shard_{shard}.weight"),
                        DType::F8E4M3,
                        vec![per_shard, head_dim],
                        Role::PleTable,
                    );
                }
            }
        }
        if include_mtp
            && self.mtp
            && (self.mtp_all_worlds
                || self.tp.world() == 1
                || arch.mtp.as_ref().is_some_and(|mtp| mtp.replicated))
        {
            self.qwen_mtp(arch, hidden, hyper_width)?;
        }
        if self.vision {
            self.qwen_vision(arch)?;
        }
        Ok(())
    }

    fn qwen_hyper(
        &mut self,
        prefix: &str,
        hyper: &sparkflash_model::architecture::HyperConnection,
        hidden: usize,
        inject: bool,
        role: Role,
    ) {
        let width = hyper.streams * hidden;
        let lowrank = hyper.lowrank.unwrap_or(0);
        self.whole(
            format!("{prefix}.hc_norm.weight"),
            DType::BF16,
            vec![width],
            role,
        );
        self.whole(
            format!("{prefix}.input_mix_weight_down.weight"),
            DType::BF16,
            vec![lowrank, width],
            role,
        );
        self.whole(
            format!("{prefix}.input_mix_weight_up.weight"),
            DType::BF16,
            vec![width, lowrank],
            role,
        );
        if inject {
            self.whole(
                format!("{prefix}.block_inject_weight.weight"),
                DType::BF16,
                vec![hyper.streams, width],
                role,
            );
        }
    }

    fn qwen_mtp(
        &mut self,
        arch: &Architecture,
        hidden: usize,
        hyper_width: usize,
    ) -> Result<(), PlanError> {
        let Some(mtp) = &arch.mtp else { return Ok(()) };
        if mtp.stages == 0 {
            return Ok(());
        }
        let reference = arch
            .layers
            .iter()
            .find(|layer| matches!(layer.attention.kind, AttentionKind::Gqa { .. }))
            .or_else(|| arch.layers.first())
            .ok_or_else(|| PlanError::Invalid("the architecture has no layers".to_owned()))?;
        let AttentionKind::Gqa { kv_heads, .. } = &reference.attention.kind else {
            return Err(PlanError::Invalid(
                "the Qwen MTP stage needs a sparse-attention layer".to_owned(),
            ));
        };
        let Some(moe) = arch.layers.iter().find_map(|layer| match &layer.ffn {
            Ffn::Moe(moe) => Some(moe),
            Ffn::Dense(_) => None,
        }) else {
            return Err(PlanError::Invalid(
                "the Qwen MTP stage needs a MoE layer".to_owned(),
            ));
        };
        let Some(indexer) = &reference.attention.indexer else {
            return Err(PlanError::Invalid(
                "the Qwen MTP stage needs a sparse indexer".to_owned(),
            ));
        };
        let hyper = hyper_of(reference)?;
        let prefix = "mtp";
        for name in ["fc_embedding", "fc_hidden"] {
            self.whole(
                format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![hidden, hidden],
                Role::Mtp,
            );
        }
        self.qwen_hyper(
            &format!("{prefix}.hyper_connection_mixer"),
            hyper,
            hidden,
            false,
            Role::Mtp,
        );
        for (name, width) in [
            ("pre_fc_norm_embedding", hidden),
            ("pre_fc_norm_hidden", hyper_width),
        ] {
            self.whole(
                format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![width],
                Role::Mtp,
            );
        }
        let layer = format!("{prefix}.layers.0");
        self.qwen_hyper(
            &format!("{layer}.attn_hyper_connection"),
            hyper,
            hidden,
            true,
            Role::Mtp,
        );
        self.qwen_hyper(
            &format!("{layer}.mlp_hyper_connection"),
            hyper,
            hidden,
            true,
            Role::Mtp,
        );
        let p = format!("{layer}.self_attn");
        let q_width = reference.attention.heads * reference.attention.head_dim * 2;
        let kv_width = kv_heads * reference.attention.head_dim;
        self.whole(
            format!("{p}.q_proj.weight"),
            DType::BF16,
            vec![q_width, hidden],
            Role::Mtp,
        );
        for name in ["k_proj", "v_proj"] {
            self.whole(
                format!("{p}.{name}.weight"),
                DType::BF16,
                vec![kv_width, hidden],
                Role::Mtp,
            );
        }
        self.whole(
            format!("{p}.o_proj.weight"),
            DType::BF16,
            vec![
                hidden,
                reference.attention.heads * reference.attention.head_dim,
            ],
            Role::Mtp,
        );
        for name in ["q_norm", "k_norm"] {
            self.whole(
                format!("{p}.{name}.weight"),
                DType::BF16,
                vec![reference.attention.head_dim],
                Role::Mtp,
            );
        }
        let width = (indexer.heads + indexer.kv_heads) * indexer.head_dim;
        self.whole(
            format!("{p}.indexer.index_qk_proj.weight"),
            DType::BF16,
            vec![width, hidden],
            Role::Mtp,
        );
        for name in ["q_layernorm", "k_layernorm"] {
            self.whole(
                format!("{p}.indexer.{name}.weight"),
                DType::BF16,
                vec![indexer.head_dim],
                Role::Mtp,
            );
        }
        let m = format!("{layer}.mlp");
        self.whole(
            format!("{m}.gate.weight"),
            DType::BF16,
            vec![moe.routed_experts, hidden],
            Role::Mtp,
        );
        for name in ["gate_proj", "up_proj"] {
            self.whole(
                format!("{m}.shared_expert.{name}.weight"),
                DType::BF16,
                vec![moe.shared_intermediate, hidden],
                Role::Mtp,
            );
        }
        self.whole(
            format!("{m}.shared_expert.down_proj.weight"),
            DType::BF16,
            vec![hidden, moe.shared_intermediate],
            Role::Mtp,
        );
        self.whole(
            format!("{m}.shared_expert_gate.weight"),
            DType::BF16,
            vec![1, hidden],
            Role::Mtp,
        );
        // The released Qwen MTP stage stores its experts as E4M3 with a BF16 inverse scale a
        // 128x128 tile, whichever scale dtype the ModelOpt table records for the backbone.
        let expert_format = match ExpertFormat::of(
            arch.quantization
                .mtp_expert_format
                .unwrap_or(arch.quantization.expert_format),
        ) {
            ExpertFormat::Fp8 { block, .. } => ExpertFormat::Fp8 {
                block,
                scale: DType::BF16,
            },
            other => other,
        };
        for expert in 0..moe.routed_experts {
            for (name, rows) in [("gate_proj", true), ("up_proj", true), ("down_proj", false)] {
                self.expert(
                    &format!("{m}.experts.{expert}.{name}"),
                    moe.intermediate,
                    hidden,
                    expert_format,
                    rows,
                    Role::Mtp,
                )?;
            }
        }
        Ok(())
    }

    fn qwen_vision(&mut self, arch: &Architecture) -> Result<(), PlanError> {
        let Some(vision) = &arch.vision else {
            return Ok(());
        };
        let width = vision.hidden_size;
        let prefix = VISION_PREFIX;
        let patch =
            vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size;
        self.whole(
            format!("{prefix}.patch_embed.proj.weight"),
            DType::BF16,
            vec![width, patch],
            Role::Vision,
        );
        self.whole(
            format!("{prefix}.patch_embed.proj.bias"),
            DType::BF16,
            vec![width],
            Role::Vision,
        );
        self.whole(
            format!("{prefix}.pos_embed.weight"),
            DType::BF16,
            vec![vision.num_position_embeddings, width],
            Role::Vision,
        );
        for block in 0..vision.depth {
            let p = format!("{prefix}.blocks.{block}");
            for name in ["norm1", "norm2"] {
                self.whole(
                    format!("{p}.{name}.weight"),
                    DType::BF16,
                    vec![width],
                    Role::Vision,
                );
                self.whole(
                    format!("{p}.{name}.bias"),
                    DType::BF16,
                    vec![width],
                    Role::Vision,
                );
            }
            self.whole(
                format!("{p}.attn.qkv.weight"),
                DType::BF16,
                vec![3 * width, width],
                Role::Vision,
            );
            self.whole(
                format!("{p}.attn.qkv.bias"),
                DType::BF16,
                vec![3 * width],
                Role::Vision,
            );
            self.whole(
                format!("{p}.attn.proj.weight"),
                DType::BF16,
                vec![width, width],
                Role::Vision,
            );
            self.whole(
                format!("{p}.attn.proj.bias"),
                DType::BF16,
                vec![width],
                Role::Vision,
            );
            self.whole(
                format!("{p}.mlp.linear_fc1.weight"),
                DType::BF16,
                vec![vision.intermediate, width],
                Role::Vision,
            );
            self.whole(
                format!("{p}.mlp.linear_fc1.bias"),
                DType::BF16,
                vec![vision.intermediate],
                Role::Vision,
            );
            self.whole(
                format!("{p}.mlp.linear_fc2.weight"),
                DType::BF16,
                vec![width, vision.intermediate],
                Role::Vision,
            );
            self.whole(
                format!("{p}.mlp.linear_fc2.bias"),
                DType::BF16,
                vec![width],
                Role::Vision,
            );
        }
        let merged = width * vision.spatial_merge * vision.spatial_merge;
        for (name, rows, columns) in [
            ("merger.linear_fc1", merged, merged),
            ("merger.linear_fc2", vision.out_hidden_size, merged),
        ] {
            self.whole(
                format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![rows, columns],
                Role::Vision,
            );
            self.whole(
                format!("{prefix}.{name}.bias"),
                DType::BF16,
                vec![rows],
                Role::Vision,
            );
        }
        self.whole(
            format!("{prefix}.merger.norm.weight"),
            DType::BF16,
            vec![width],
            Role::Vision,
        );
        self.whole(
            format!("{prefix}.merger.norm.bias"),
            DType::BF16,
            vec![width],
            Role::Vision,
        );
        Ok(())
    }

    fn glm(&mut self, arch: &Architecture, include_mtp: bool) -> Result<(), PlanError> {
        let hidden = arch.hidden_size;
        self.whole(
            format!("{LM_PREFIX}.embed_tokens.weight"),
            DType::BF16,
            vec![arch.vocab_size, hidden],
            Role::Embedding,
        );
        self.whole(
            format!("{LM_PREFIX}.norm.weight"),
            DType::BF16,
            vec![hidden],
            Role::Norm,
        );
        self.rows(
            "lm_head.weight".to_owned(),
            DType::BF16,
            [arch.vocab_size, hidden],
            Role::LmHead,
        )?;
        for layer in &arch.layers {
            let prefix = format!("{LM_PREFIX}.layers.{}", layer.index);
            for name in ["input_layernorm", "post_attention_layernorm"] {
                self.whole(
                    format!("{prefix}.{name}.weight"),
                    DType::BF16,
                    vec![hidden],
                    Role::Norm,
                );
            }
            let hyper = hyper_of(layer)?;
            for branch in ["attn", "ffn"] {
                self.glm_hyper(&format!("{prefix}.hc_{branch}"), hyper, hidden)?;
            }
            match &layer.attention.kind {
                AttentionKind::Kda(kda) => {
                    self.glm_kda(&format!("{prefix}.self_attn"), kda, hidden, Role::Attention)?;
                }
                AttentionKind::Mla { .. } => {
                    let indexer = layer
                        .attention
                        .indexer
                        .as_ref()
                        .is_some_and(|indexer| !indexer.reuse);
                    self.glm_dsa(
                        &format!("{prefix}.self_attn"),
                        layer,
                        hidden,
                        indexer,
                        Role::Attention,
                        Role::Indexer,
                    )?;
                }
                AttentionKind::Gqa { kv_heads, .. } => {
                    self.glm_gqa(
                        &format!("{prefix}.self_attn"),
                        layer,
                        *kv_heads,
                        hidden,
                        Role::Attention,
                    )?;
                }
                other => {
                    return Err(PlanError::Invalid(format!(
                        "GLM layer {} runs {other:?}",
                        layer.index
                    )));
                }
            }
            match &layer.ffn {
                Ffn::Dense(dense) => {
                    self.glm_dense_mlp(&format!("{prefix}.mlp"), dense, hidden, Role::DenseMlp)?;
                }
                Ffn::Moe(moe) => self.glm_moe(
                    &format!("{prefix}.mlp"),
                    moe,
                    hidden,
                    Role::Router,
                    Role::SharedExpert,
                    Role::Experts,
                )?,
            }
        }
        if include_mtp && self.mtp {
            self.glm_mtp(arch, hidden)?;
        }
        if self.vision {
            self.glm_vision(arch)?;
        }
        Ok(())
    }

    fn glm_hyper(
        &mut self,
        prefix: &str,
        hyper: &sparkflash_model::architecture::HyperConnection,
        hidden: usize,
    ) -> Result<(), PlanError> {
        let rows = (2 + hyper.streams) * hyper.streams;
        let width = hyper.streams * hidden;
        self.whole(
            format!("{prefix}_fn"),
            DType::BF16,
            vec![rows, width],
            Role::Hyper,
        );
        self.whole(
            format!("{prefix}_base"),
            DType::F32,
            vec![rows],
            Role::Hyper,
        );
        self.whole(format!("{prefix}_scale"), DType::F32, vec![3], Role::Hyper);
        Ok(())
    }

    fn glm_kda(
        &mut self,
        prefix: &str,
        kda: &Kda,
        hidden: usize,
        role: Role,
    ) -> Result<(), PlanError> {
        let width = kda.heads * kda.head_dim;
        for name in ["q", "k", "v"] {
            self.rows(
                format!("{prefix}.{name}_proj.weight"),
                DType::BF16,
                [width, hidden],
                role,
            )?;
            self.conv_rows(
                format!("{prefix}.{name}_conv1d.weight"),
                DType::BF16,
                width,
                kda.conv_kernel,
                role,
            )?;
        }
        for name in ["f", "g"] {
            self.whole(
                format!("{prefix}.{name}_a_proj.weight"),
                DType::BF16,
                vec![kda.head_dim, hidden],
                role,
            );
            self.rows(
                format!("{prefix}.{name}_b_proj.weight"),
                DType::BF16,
                [width, kda.head_dim],
                role,
            )?;
        }
        self.rows(
            format!("{prefix}.b_proj.weight"),
            DType::BF16,
            [kda.heads, hidden],
            role,
        )?;
        self.row_vector(format!("{prefix}.A_log"), DType::F32, kda.heads, role)?;
        self.row_vector(format!("{prefix}.dt_bias"), DType::F32, width, role)?;
        self.whole(
            format!("{prefix}.o_norm.weight"),
            DType::BF16,
            vec![kda.head_dim],
            role,
        );
        self.columns(
            format!("{prefix}.o_proj.weight"),
            DType::BF16,
            [hidden, width],
            role,
        )?;
        Ok(())
    }

    fn glm_gqa(
        &mut self,
        prefix: &str,
        layer: &Layer,
        kv_heads: usize,
        hidden: usize,
        role: Role,
    ) -> Result<(), PlanError> {
        let head_dim = layer.attention.head_dim;
        self.rows(
            format!("{prefix}.q_proj.weight"),
            DType::BF16,
            [layer.attention.heads * head_dim, hidden],
            role,
        )?;
        for name in ["k_proj", "v_proj"] {
            self.kv_rows(
                format!("{prefix}.{name}.weight"),
                DType::BF16,
                [kv_heads * head_dim, hidden],
                role,
            )?;
        }
        self.columns(
            format!("{prefix}.o_proj.weight"),
            DType::BF16,
            [hidden, layer.attention.heads * head_dim],
            role,
        )?;
        for name in ["q_norm", "k_norm"] {
            self.whole(
                format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![head_dim],
                role,
            );
        }
        Ok(())
    }

    fn glm_dsa(
        &mut self,
        prefix: &str,
        layer: &Layer,
        hidden: usize,
        indexer: bool,
        attention: Role,
        index_role: Role,
    ) -> Result<(), PlanError> {
        let heads = layer.attention.heads;
        let nope = layer
            .attention
            .head_dim
            .saturating_sub(layer.attention.rope.rotated_dims);
        let AttentionKind::Mla {
            q_lora_rank,
            kv_lora_rank,
            v_head_dim,
            ..
        } = &layer.attention.kind
        else {
            return Err(PlanError::Invalid(format!(
                "GLM layer {} is not MLA",
                layer.index
            )));
        };
        let q_lora = q_lora_rank.unwrap_or(0);
        if q_lora > 0 {
            self.dense_linear(
                &format!("{prefix}.q_a_proj"),
                q_lora,
                hidden,
                self.dense_format,
                attention,
                LinearSplit::Whole,
            )?;
            self.whole(
                format!("{prefix}.q_a_layernorm.weight"),
                DType::BF16,
                vec![q_lora],
                attention,
            );
        }
        self.dense_linear(
            &format!("{prefix}.q_b_proj"),
            heads * nope,
            q_lora,
            self.dense_format,
            attention,
            LinearSplit::Rows,
        )?;
        self.dense_linear(
            &format!("{prefix}.kv_a_proj_with_mqa"),
            *kv_lora_rank,
            hidden,
            self.dense_format,
            attention,
            LinearSplit::Whole,
        )?;
        self.whole(
            format!("{prefix}.kv_a_layernorm.weight"),
            DType::BF16,
            vec![*kv_lora_rank],
            attention,
        );
        self.rows(
            format!("{prefix}.kv_b_proj.weight"),
            DType::BF16,
            [heads * (nope + *v_head_dim), *kv_lora_rank],
            attention,
        )?;
        self.dense_linear(
            &format!("{prefix}.o_proj"),
            hidden,
            heads * *v_head_dim,
            self.dense_format,
            attention,
            LinearSplit::Columns,
        )?;
        if !indexer {
            return Ok(());
        }
        let Some(spec) = &layer.attention.indexer else {
            return Err(PlanError::Invalid(format!(
                "GLM layer {} runs an indexer the architecture does not name",
                layer.index
            )));
        };
        let p = format!("{prefix}.indexer");
        let dim = spec.head_dim;
        self.whole(
            format!("{p}.wq_b.weight"),
            DType::BF16,
            vec![spec.heads * dim, q_lora],
            index_role,
        );
        self.whole(
            format!("{p}.wk.weight"),
            DType::BF16,
            vec![dim, hidden],
            index_role,
        );
        for name in ["k_norm.weight", "k_norm.bias"] {
            self.whole(format!("{p}.{name}"), DType::BF16, vec![dim], index_role);
        }
        self.whole(
            format!("{p}.weights_proj.weight"),
            DType::BF16,
            vec![spec.heads, hidden],
            index_role,
        );
        self.whole(
            format!("{p}.index_kpool_compress_gate"),
            DType::BF16,
            vec![dim, hidden],
            index_role,
        );
        self.whole(
            format!("{p}.index_kpool_compress_ape"),
            DType::BF16,
            vec![spec.pool, dim],
            index_role,
        );
        Ok(())
    }

    fn glm_dense_mlp(
        &mut self,
        prefix: &str,
        dense: &DenseMlp,
        hidden: usize,
        role: Role,
    ) -> Result<(), PlanError> {
        let format = self.dense_format;
        for name in ["gate_proj", "up_proj"] {
            self.dense_linear(
                &format!("{prefix}.{name}"),
                dense.intermediate,
                hidden,
                format,
                role,
                LinearSplit::Rows,
            )?;
        }
        self.dense_linear(
            &format!("{prefix}.down_proj"),
            hidden,
            dense.intermediate,
            format,
            role,
            LinearSplit::Columns,
        )?;
        Ok(())
    }

    fn glm_moe(
        &mut self,
        prefix: &str,
        moe: &Moe,
        hidden: usize,
        router: Role,
        shared: Role,
        experts: Role,
    ) -> Result<(), PlanError> {
        self.whole(
            format!("{prefix}.gate.weight"),
            DType::BF16,
            vec![moe.routed_experts, hidden],
            router,
        );
        if moe.selection_bias {
            self.whole(
                format!("{prefix}.gate.e_score_correction_bias"),
                DType::F32,
                vec![moe.routed_experts],
                router,
            );
        }
        let shared_width = moe.shared_intermediate * moe.shared_experts.max(1);
        let format = self.dense_format;
        for name in ["gate_proj", "up_proj"] {
            self.dense_linear(
                &format!("{prefix}.shared_experts.{name}"),
                shared_width,
                hidden,
                format,
                shared,
                LinearSplit::Rows,
            )?;
        }
        self.dense_linear(
            &format!("{prefix}.shared_experts.down_proj"),
            hidden,
            shared_width,
            format,
            shared,
            LinearSplit::Columns,
        )?;
        let expert_format = ExpertFormat::of(self.expert_format);
        for expert in 0..moe.routed_experts {
            for (name, rows) in [("gate_proj", true), ("up_proj", true), ("down_proj", false)] {
                self.expert(
                    &format!("{prefix}.experts.{expert}.{name}"),
                    moe.intermediate,
                    hidden,
                    expert_format,
                    rows,
                    experts,
                )?;
            }
        }
        Ok(())
    }

    fn glm_mtp(&mut self, arch: &Architecture, hidden: usize) -> Result<(), PlanError> {
        let Some(mtp) = &arch.mtp else { return Ok(()) };
        if mtp.stages == 0 {
            return Ok(());
        }
        let layer_index = arch.layers.len();
        let prefix = format!("{LM_PREFIX}.layers.{layer_index}");
        let reference = arch
            .layers
            .iter()
            .find(|layer| matches!(layer.attention.kind, AttentionKind::Mla { .. }))
            .or_else(|| arch.layers.first())
            .ok_or_else(|| PlanError::Invalid("the architecture has no layers".to_owned()))?;
        for name in ["enorm", "hnorm", "shared_head.norm"] {
            self.whole(
                format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![hidden],
                Role::Mtp,
            );
        }
        self.whole(
            format!("{prefix}.eh_proj.weight"),
            DType::BF16,
            vec![hidden, 2 * hidden],
            Role::Mtp,
        );
        for name in ["input_layernorm", "post_attention_layernorm"] {
            self.whole(
                format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![hidden],
                Role::Mtp,
            );
        }
        self.glm_dsa(
            &format!("{prefix}.self_attn"),
            reference,
            hidden,
            true,
            Role::Mtp,
            Role::Mtp,
        )?;
        let Some(moe) = arch.layers.iter().find_map(|layer| match &layer.ffn {
            Ffn::Moe(moe) => Some(moe),
            Ffn::Dense(_) => None,
        }) else {
            return Err(PlanError::Invalid(
                "the GLM MTP stage needs a MoE layer".to_owned(),
            ));
        };
        self.glm_moe(
            &format!("{prefix}.mlp"),
            moe,
            hidden,
            Role::Mtp,
            Role::Mtp,
            Role::Mtp,
        )
    }

    fn glm_vision(&mut self, arch: &Architecture) -> Result<(), PlanError> {
        let Some(vision) = &arch.vision else {
            return Ok(());
        };
        let width = vision.hidden_size;
        let out = vision.out_hidden_size;
        let inner = vision.intermediate;
        let head = vision.hidden_size / vision.heads.max(1);
        macro_rules! push {
            ($name:expr, $shape:expr) => {
                self.whole(
                    format!("{VISION_PREFIX}.{}", $name),
                    DType::BF16,
                    $shape,
                    Role::Vision,
                )
            };
        }
        push!(
            "patch_embed.proj.weight",
            vec![
                width,
                vision.in_channels,
                vision.temporal_patch_size,
                vision.patch_size,
                vision.patch_size,
            ]
        );
        push!("patch_embed.proj.bias", vec![width]);
        for block in 0..vision.depth {
            let p = format!("blocks.{block}");
            for name in ["norm1", "norm2"] {
                push!(&format!("{p}.{name}.weight"), vec![width]);
            }
            push!(&format!("{p}.attn.qkv.weight"), vec![3 * width, width]);
            push!(&format!("{p}.attn.qkv.bias"), vec![3 * width]);
            push!(&format!("{p}.attn.q_norm.weight"), vec![head]);
            push!(&format!("{p}.attn.k_norm.weight"), vec![head]);
            push!(&format!("{p}.attn.proj.weight"), vec![width, width]);
            push!(&format!("{p}.attn.proj.bias"), vec![width]);
            for name in ["gate_proj", "up_proj"] {
                push!(&format!("{p}.mlp.{name}.weight"), vec![inner, width]);
                push!(&format!("{p}.mlp.{name}.bias"), vec![inner]);
            }
            push!(&format!("{p}.mlp.down_proj.weight"), vec![width, inner]);
            push!(&format!("{p}.mlp.down_proj.bias"), vec![width]);
        }
        push!("post_layernorm.weight", vec![width]);
        let merge = vision.spatial_merge;
        push!("downsample.weight", vec![out, width, merge, merge]);
        push!("downsample.bias", vec![out]);
        push!("merger.proj.weight", vec![out, out]);
        push!("merger.post_projection_norm.weight", vec![out]);
        push!("merger.post_projection_norm.bias", vec![out]);
        let projection = vision.projection_intermediate;
        push!("merger.gate_proj.weight", vec![projection, out]);
        push!("merger.up_proj.weight", vec![projection, out]);
        push!("merger.down_proj.weight", vec![out, projection]);
        Ok(())
    }
}

fn hyper_of(layer: &Layer) -> Result<&sparkflash_model::architecture::HyperConnection, PlanError> {
    match &layer.residual {
        Residual::Hyper(hyper) => Ok(hyper),
        Residual::Plain => Err(PlanError::Invalid(format!(
            "layer {} uses a plain residual where the plan expects hyper-connections",
            layer.index
        ))),
    }
}

fn ple_padded_rows(ple: &PleGeometry) -> u64 {
    let heads = ple.ngram_size.saturating_sub(1) * ple.heads_per_ngram;
    let mut offset = 0u64;
    let mut prime = ple.vocab_base;
    for _ in 0..heads {
        loop {
            prime += 1;
            if is_prime(prime) {
                break;
            }
        }
        offset += prime;
    }
    let divisor = ple.vocab_multiple.max(ple.split_parts as u64).max(1);
    offset.div_ceil(divisor) * divisor
}

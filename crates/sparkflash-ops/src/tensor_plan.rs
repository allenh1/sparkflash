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
//! The arms all read the same released `config.json` through [Architecture]. This module turns
//! that walk into names, shapes and rank slices, so a released index's tensor count and byte
//! total are checked against the shared vocabulary rather than a second family parser. A
//! checkpoint with a layout takes its names from `sparkflash_model::layout`; the planners here
//! are the structural fallback and the test oracle for the worked layout.
//!
//! [ArchitectureTensorPlan::served] is what a rank uploads, [ArchitectureTensorPlan::checkpoint]
//! adds the unserved vision tower, and [ArchitectureTensorPlan::rank_spec_at] is the rank-local
//! shape the weight walk reads. The split rules are the ones the arms already execute: heads and
//! intermediate widths divide evenly, an FP8 expert is a run of whole `block_n`-wide tiles (D91),
//! and an NVFP4 expert is padded to the 64-column MMA tile above TP1.

use std::collections::{BTreeMap, BTreeSet};

use sparkflash_model::architecture::{Architecture, AttentionKind, Ffn, ScaleDType, WeightFormat};
#[cfg(test)]
use sparkflash_model::architecture::{DenseMlp, Kda, Layer, Moe, PleGeometry, Residual, Side};
use sparkflash_model::layout::{LayoutDtype, LayoutTensor, TensorSplit};
use sparkflash_model::{TpAxis, TpPlan};
#[cfg(test)]
use sparkflash_weights::rng::is_prime;
use sparkflash_weights::{DType, TensorPlan, TensorSpec};

use crate::engine_plan::PlanError;
use crate::naming::{AttentionNames, ExpertNames, FfnNames, NamingDialect, TensorNames};
use crate::scan_report::{TensorEntry, TensorIndex};

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
    /// A 4-bit E2M1 expert. NVFP4's per-16 group and MXFP4's per-32 block are different
    /// arithmetic ([Precision::NvFp4] and [Precision::MxFp4]); the loader needs only the width
    /// one scale covers, which the release declares and this carries for every scale shape.
    Fp4 {
        /// The values one scale covers: 16 for NVFP4's group, 32 for MXFP4's block.
        block: usize,
        /// Whether the release also stores a tensor-wide factor (NVFP4's two scales).
        tensor_wide: bool,
    },
    Fp8 {
        block: usize,
        scale: DType,
    },
    /// Row-scaled MXFP8: one ue8m0 scale a row, every `block` columns.
    Fp8Row {
        /// The values one scale covers along the contraction.
        block: usize,
    },
    Bf16,
}

impl ExpertFormat {
    fn of(format: WeightFormat) -> Self {
        match format {
            WeightFormat::NvFp4 { group_size, .. } => Self::Fp4 {
                block: group_size.max(1),
                tensor_wide: true,
            },
            // MXFP4 applies one power-of-two code a block and has no tensor-wide factor: the
            // released DeepSeek checkout stores only `.weight` and `.scale` per expert.
            WeightFormat::MxFp4 { block } => Self::Fp4 {
                block: block.max(1),
                tensor_wide: false,
            },
            WeightFormat::Fp8Tile {
                block_n,
                scale_dtype,
                ..
            } => Self::Fp8 {
                block: block_n.max(1),
                scale: scale_dtype_of(scale_dtype),
            },
            WeightFormat::MxFp8Row { block } => Self::Fp8Row { block },
            WeightFormat::MxFp8 { .. } | WeightFormat::Bf16 | WeightFormat::F32 => Self::Bf16,
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
        if builder.read_layout(architecture)? {
            return Ok(builder.finish());
        }
        // A family no pack transcribes is planned in the canonical spelling; a caller
        // holding the checkpoint plans its own spelling through [Self::named].
        let names = NamingDialect::canonical(architecture)
            .names(architecture)
            .map_err(PlanError::Invalid)?;
        builder.generic(architecture, &names)?;
        Ok(builder.finish())
    }

    /// Plan a family no pack transcribes, in the names `names` spells: what
    /// [crate::naming::resolve] read from a checkpoint, or [NamingDialect::canonical]'s for a
    /// document alone. The backbone only (embedding, decoder layers, final norm and head); a
    /// tied head is the embedding and is not planned twice. The order is the forward's, as
    /// [TensorNames::walk] lists it, with an FP8 weight's tile scale after it.
    ///
    /// # Errors
    /// A layer whose attention or FFN is not the one `names` spells for it, a family the
    /// convention does not name (see [NamingDialect::names]), or a world a sharded axis does not
    /// divide.
    pub fn named(
        architecture: &Architecture,
        tp: TpPlan,
        names: &TensorNames,
    ) -> Result<Self, PlanError> {
        let mut builder = Builder::new(tp, false, false, false);
        builder.dense_format = architecture.quantization.dense_format;
        builder.expert_format = architecture.quantization.expert_format;
        builder.generic(architecture, names)?;
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

    /// The logical plan by value, for a caller that keeps only the names, shapes and roles.
    #[must_use]
    pub fn into_plan(self) -> TensorPlan<Role> {
        self.plan
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
    /// Rows split by key/value head, which a world wider than the KV heads replicates.
    KvRows,
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

    /// Whether this plan carries the MTP stage at this rank.
    fn serves_mtp(&self, architecture: &Architecture) -> bool {
        self.mtp
            && (self.mtp_all_worlds
                || self.tp.world() == 1
                || architecture.mtp.as_ref().is_some_and(|mtp| mtp.replicated))
    }

    fn finish(self) -> ArchitectureTensorPlan {
        ArchitectureTensorPlan {
            plan: self.plan,
            rank: self.rank,
            runs: self.runs,
            tp: self.tp,
        }
    }

    /// Build the checkpoint's tensors from its layout, when one describes it.
    ///
    /// Returns false when no layout matches, so a checkpoint with no file still runs the
    /// structural reader. A layout that matches but describes no tensor is a refusal: a file
    /// exists to say what the structural reader could not.
    fn read_layout(&mut self, arch: &Architecture) -> Result<bool, PlanError> {
        let set = sparkflash_model::layout::LayoutSet::load()
            .map_err(|error| PlanError::Invalid(error.to_string()))?;
        let layout = set
            .select(arch)
            .map_err(|error| PlanError::Invalid(error.to_string()))?;
        let Some(layout) = layout else {
            // A family with shipped layouts but no match is a release this tree does not
            // describe, not a model to hand to the canonical fallback.
            if set.describes_family(&arch.family) {
                return Err(PlanError::Invalid(format!(
                    "the tree carries layouts for family {:?}, but none matches this release's                      architectures or quantization",
                    arch.family
                )));
            }
            return Ok(false);
        };
        let tensors = layout
            .tensors(arch)
            .map_err(|error| PlanError::Invalid(error.to_string()))?;
        if tensors.is_empty() {
            return Err(PlanError::Invalid(format!(
                "layout {} describes no tensor for this checkpoint",
                layout.id
            )));
        }
        for tensor in tensors {
            if tensor.stack == sparkflash_model::StackKind::Vision && !self.vision {
                continue;
            }
            if tensor.stack == sparkflash_model::StackKind::Draft && !self.serves_mtp(arch) {
                continue;
            }
            if tensor.unloaded.is_some() {
                continue;
            }
            self.layout_tensor(&tensor)?;
        }
        Ok(true)
    }

    fn layout_tensor(&mut self, tensor: &LayoutTensor) -> Result<(), PlanError> {
        let role = tensor_role(&tensor.role).ok_or_else(|| {
            PlanError::Invalid(format!(
                "tensor {} names unknown role {:?}",
                tensor.name, tensor.role
            ))
        })?;
        match tensor.dtype {
            LayoutDtype::Dense => {
                let [rows, columns] = two(&tensor.name, &tensor.shape)?;
                let split = match tensor.split {
                    TensorSplit::Whole => LinearSplit::Whole,
                    TensorSplit::Rows => LinearSplit::Rows,
                    TensorSplit::Columns => LinearSplit::Columns,
                    TensorSplit::KvRows => LinearSplit::KvRows,
                    other => {
                        return Err(PlanError::Invalid(format!(
                            "tensor {} is a dense projection with split {other:?}",
                            tensor.name
                        )));
                    }
                };
                let prefix = tensor.name.strip_suffix(".weight").ok_or_else(|| {
                    PlanError::Invalid(format!(
                        "dense tensor {} does not end in .weight",
                        tensor.name
                    ))
                })?;
                self.dense_linear(prefix, rows, columns, self.dense_format, role, split)?;
            }
            LayoutDtype::Expert => {
                let format = ExpertFormat::of(self.expert_format);
                let (width, hidden, rows) = match tensor.split {
                    TensorSplit::ExpertRows => {
                        let [width, hidden] = two(&tensor.name, &tensor.shape)?;
                        (width, hidden, true)
                    }
                    TensorSplit::ExpertColumns => {
                        let [hidden, width] = two(&tensor.name, &tensor.shape)?;
                        (width, hidden, false)
                    }
                    other => {
                        return Err(PlanError::Invalid(format!(
                            "expert tensor {} has split {other:?}",
                            tensor.name
                        )));
                    }
                };
                let prefix = tensor.name.strip_suffix(".weight").ok_or_else(|| {
                    PlanError::Invalid(format!(
                        "expert tensor {} does not end in .weight",
                        tensor.name
                    ))
                })?;
                self.expert(prefix, width, hidden, format, rows, role)?;
            }
            LayoutDtype::Bf16
            | LayoutDtype::F32
            | LayoutDtype::F8E4M3
            | LayoutDtype::F8E8M0
            | LayoutDtype::F4E2M1
            | LayoutDtype::U8
            | LayoutDtype::I64 => {
                let dtype = match tensor.dtype {
                    LayoutDtype::Bf16 => DType::BF16,
                    LayoutDtype::F32 => DType::F32,
                    LayoutDtype::F8E4M3 => DType::F8E4M3,
                    LayoutDtype::F8E8M0 => DType::F8E8M0,
                    LayoutDtype::F4E2M1 => DType::F4E2M1,
                    LayoutDtype::U8 => DType::U8,
                    LayoutDtype::I64 => DType::I64,
                    LayoutDtype::Dense | LayoutDtype::Expert => unreachable!(),
                };
                self.layout_split(
                    &tensor.name,
                    dtype,
                    &tensor.shape,
                    tensor.split,
                    tensor.tile,
                    role,
                )?;
            }
        }
        Ok(())
    }

    fn layout_split(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[usize],
        split: TensorSplit,
        tile: Option<usize>,
        role: Role,
    ) -> Result<(), PlanError> {
        match split {
            TensorSplit::Whole => {
                self.whole(name.to_owned(), dtype, shape.to_vec(), role);
                Ok(())
            }
            TensorSplit::Rows => {
                let [a, b] = two(name, shape)?;
                self.rows(name.to_owned(), dtype, [a, b], role)
            }
            TensorSplit::Columns => {
                let [a, b] = two(name, shape)?;
                self.columns(name.to_owned(), dtype, [a, b], role)
            }
            TensorSplit::KvRows => {
                let [a, b] = two(name, shape)?;
                self.kv_rows(name.to_owned(), dtype, [a, b], role)
            }
            TensorSplit::Vector => {
                let [a] = one(name, shape)?;
                self.row_vector(name.to_owned(), dtype, a, role)
            }
            TensorSplit::Conv => {
                let [width, _, taps] = three(name, shape)?;
                self.conv_rows(name.to_owned(), dtype, width, taps, role)
            }
            TensorSplit::Tiles => {
                let [rows, columns] = two(name, shape)?;
                let tile = tile.ok_or_else(|| {
                    PlanError::Invalid(format!("tensor {name} has a tiles split but no tile width"))
                })?;
                let split = self
                    .tp
                    .split_tiles(columns, tile, TpAxis::Intermediate)
                    .map_err(|error| PlanError::Invalid(error.to_string()))?;
                self.push(
                    name.to_owned(),
                    dtype,
                    vec![rows, columns],
                    vec![rows, split.padded],
                    Some(RankRun::tiled(split)),
                    role,
                );
                Ok(())
            }
            TensorSplit::TileRows => {
                let [rows, columns] = two(name, shape)?;
                let tile = tile.ok_or_else(|| {
                    PlanError::Invalid(format!(
                        "tensor {name} has a tile_rows split but no tile width"
                    ))
                })?;
                let split = self
                    .tp
                    .split_tiles(rows, tile, TpAxis::Heads)
                    .map_err(|error| PlanError::Invalid(error.to_string()))?;
                self.push(
                    name.to_owned(),
                    dtype,
                    vec![rows, columns],
                    vec![split.padded, columns],
                    Some(RankRun::tiled(split)),
                    role,
                );
                Ok(())
            }
            TensorSplit::ExpertAxis => {
                if shape.is_empty() {
                    return Err(PlanError::Invalid(format!(
                        "tensor {name} has an empty expert-axis shape"
                    )));
                }
                let split = self
                    .tp
                    .split(shape[0], TpAxis::Experts, 1)
                    .map_err(|error| PlanError::Invalid(error.to_string()))?;
                let mut rank = shape.to_vec();
                rank[0] = split.padded;
                self.push(
                    name.to_owned(),
                    dtype,
                    shape.to_vec(),
                    rank,
                    Some(RankRun::even(split)),
                    role,
                );
                Ok(())
            }
            TensorSplit::ExpertRows | TensorSplit::ExpertColumns => Err(PlanError::Invalid(
                format!("tensor {name} has an expert split but no expert dtype"),
            )),
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
                ExpertFormat::Fp4 { .. } => DType::F4E2M1,
                ExpertFormat::Fp8 { .. } | ExpertFormat::Fp8Row { .. } => DType::F8E4M3,
                ExpertFormat::Bf16 => DType::BF16,
            },
            shape,
            rank_shape,
            run,
            role,
        );
        match format {
            ExpertFormat::Fp4 { block, tensor_wide } => {
                let scale_shape = if rows {
                    vec![width, hidden.div_ceil(block)]
                } else {
                    vec![hidden, width.div_ceil(block)]
                };
                let scale_rank = if rows {
                    vec![local, hidden.div_ceil(block)]
                } else {
                    vec![hidden, local.div_ceil(block)]
                };
                self.push(
                    format!("{prefix}.weight_scale"),
                    DType::F8E4M3,
                    scale_shape,
                    scale_rank,
                    None,
                    role,
                );
                if tensor_wide {
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
                // tiles: the same offset and count, divided by the block width. A rank starts on
                // a tile edge; one rank holding the whole axis may end in a partial tile, which
                // still has its scale.
                let scale_run = run.map(|run| RankRun {
                    offset: run.offset / block,
                    count: run.count.div_ceil(block),
                    total: run.total.div_ceil(block),
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
            ExpertFormat::Fp8Row { block } => {
                let scale_shape = if rows {
                    vec![width, hidden.div_ceil(block)]
                } else {
                    vec![hidden, width.div_ceil(block)]
                };
                let scale_rank = if rows {
                    vec![local, hidden.div_ceil(block)]
                } else {
                    vec![hidden, local.div_ceil(block)]
                };
                let scale_run = run.map(|run| RankRun {
                    offset: run.offset / block,
                    count: run.count.div_ceil(block),
                    total: run.total.div_ceil(block),
                });
                self.push(
                    format!("{prefix}.weight_scale"),
                    DType::F8E8M0,
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
            ExpertFormat::Fp4 { .. } => {
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
            ExpertFormat::Fp8 { block, .. } | ExpertFormat::Fp8Row { block } if block > 1 => {
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
                let axis = match split {
                    LinearSplit::Whole => None,
                    LinearSplit::Rows => Some((true, self.tp.split(rows, TpAxis::Heads, 1)?)),
                    LinearSplit::KvRows => Some((true, self.tp.split(rows, TpAxis::KvHeads, 1)?)),
                    LinearSplit::Columns => {
                        Some((false, self.tp.split(columns, TpAxis::Intermediate, 1)?))
                    }
                };
                let scale_shape = [rows.div_ceil(block), columns.div_ceil(block)];
                // The split's axis is the one the caller named, never inferred from the totals:
                // a square projection's row and column totals are equal.
                let (weight_rank, scale_rank) = match axis {
                    Some((true, split)) => (
                        vec![split.padded, columns],
                        vec![split.padded.div_ceil(block), scale_shape[1]],
                    ),
                    Some((false, split)) => (
                        vec![rows, split.padded],
                        vec![scale_shape[0], split.padded.div_ceil(block)],
                    ),
                    None => (vec![rows, columns], scale_shape.to_vec()),
                };
                let split = axis.map(|(_, split)| split);
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
                LinearSplit::KvRows => {
                    self.kv_rows(
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

    /// A family no pack transcribes: the backbone in the names `names` spells, the dense
    /// projections in the dense format and the routed experts in the expert format, split as the
    /// released families split the same projections.
    fn generic(&mut self, arch: &Architecture, names: &TensorNames) -> Result<(), PlanError> {
        let hidden = arch.hidden_size;
        if names.layers.len() != arch.layers.len() {
            return Err(PlanError::Invalid(format!(
                "the names spell {} layers for an architecture of {}",
                names.layers.len(),
                arch.layers.len()
            )));
        }
        let dense = self.dense_format;
        let experts = ExpertFormat::of(self.expert_format);
        self.whole(
            names.embed.clone(),
            DType::BF16,
            vec![arch.vocab_size, hidden],
            Role::Embedding,
        );
        for (layer, named) in arch.layers.iter().zip(&names.layers) {
            self.whole(
                named.input_norm.clone(),
                DType::BF16,
                vec![hidden],
                Role::Norm,
            );
            match (&layer.attention.kind, &named.attention) {
                (
                    AttentionKind::Gqa {
                        kv_heads,
                        output_gate,
                    },
                    AttentionNames::Gqa {
                        q,
                        k,
                        v,
                        o,
                        q_norm,
                        k_norm,
                    },
                ) => {
                    let head_dim = layer.attention.head_dim;
                    let width = layer.attention.heads * head_dim;
                    let kv_width = kv_heads * head_dim;
                    let query = if *output_gate { 2 * width } else { width };
                    let attention = Role::Attention;
                    self.dense_linear(
                        stem(q)?,
                        query,
                        hidden,
                        dense,
                        attention,
                        LinearSplit::Rows,
                    )?;
                    for name in [k, v] {
                        self.dense_linear(
                            stem(name)?,
                            kv_width,
                            hidden,
                            dense,
                            attention,
                            LinearSplit::KvRows,
                        )?;
                    }
                    self.dense_linear(
                        stem(o)?,
                        hidden,
                        width,
                        dense,
                        attention,
                        LinearSplit::Columns,
                    )?;
                    for norm in [q_norm, k_norm].into_iter().flatten() {
                        self.whole(norm.clone(), DType::BF16, vec![head_dim], attention);
                    }
                }
                (AttentionKind::Gdn(gdn), AttentionNames::Gdn(named)) => {
                    let key_width = gdn.key_heads * gdn.key_dim;
                    let value_width = gdn.value_heads * gdn.value_dim;
                    let conv_width = 2 * key_width + value_width;
                    let attention = Role::Attention;
                    self.dense_linear(
                        stem(&named.qkv)?,
                        conv_width,
                        hidden,
                        dense,
                        attention,
                        LinearSplit::Rows,
                    )?;
                    self.dense_linear(
                        stem(&named.z)?,
                        value_width,
                        hidden,
                        dense,
                        attention,
                        LinearSplit::Rows,
                    )?;
                    for name in [&named.b, &named.a] {
                        self.rows(
                            name.clone(),
                            DType::BF16,
                            [gdn.value_heads, hidden],
                            attention,
                        )?;
                    }
                    self.dense_linear(
                        stem(&named.out)?,
                        hidden,
                        value_width,
                        dense,
                        attention,
                        LinearSplit::Columns,
                    )?;
                    self.conv_rows(
                        named.conv.clone(),
                        DType::BF16,
                        conv_width,
                        gdn.conv_kernel,
                        attention,
                    )?;
                    for name in [&named.a_log, &named.dt_bias] {
                        self.row_vector(name.clone(), DType::BF16, gdn.value_heads, attention)?;
                    }
                    self.whole(
                        named.norm.clone(),
                        DType::BF16,
                        vec![gdn.value_dim],
                        attention,
                    );
                }
                (kind, _) => {
                    return Err(PlanError::Invalid(format!(
                        "layer {} runs {kind:?}, which its names do not spell",
                        layer.index
                    )));
                }
            }
            self.whole(
                named.post_norm.clone(),
                DType::BF16,
                vec![hidden],
                Role::Norm,
            );
            match (&layer.ffn, &named.ffn) {
                (Ffn::Dense(mlp), FfnNames::Dense(names)) => {
                    self.swiglu(names, mlp.intermediate, hidden, dense, Role::DenseMlp)?;
                }
                (
                    Ffn::Moe(moe),
                    FfnNames::Moe {
                        router,
                        bias,
                        experts: bodies,
                        shared,
                        shared_gate,
                    },
                ) => {
                    if bodies.len() != moe.routed_experts {
                        return Err(PlanError::Invalid(format!(
                            "layer {} names {} experts of {}",
                            layer.index,
                            bodies.len(),
                            moe.routed_experts
                        )));
                    }
                    self.whole(
                        router.clone(),
                        DType::BF16,
                        vec![moe.routed_experts, hidden],
                        Role::Router,
                    );
                    if let Some(bias) = bias {
                        // The released documents store the correction bias in f32 (GLM's own
                        // plan says the same of its gate).
                        self.whole(
                            bias.clone(),
                            DType::F32,
                            vec![moe.routed_experts],
                            Role::Router,
                        );
                    }
                    for body in bodies {
                        for (name, rows) in
                            [(&body.gate, true), (&body.up, true), (&body.down, false)]
                        {
                            self.expert(
                                stem(name)?,
                                moe.intermediate,
                                hidden,
                                experts,
                                rows,
                                Role::Experts,
                            )?;
                        }
                    }
                    if let Some(shared) = shared {
                        self.swiglu(
                            shared,
                            moe.shared_intermediate,
                            hidden,
                            dense,
                            Role::SharedExpert,
                        )?;
                    }
                    if let Some(gate) = shared_gate {
                        self.whole(
                            gate.clone(),
                            DType::BF16,
                            vec![1, hidden],
                            Role::SharedExpert,
                        );
                    }
                }
                (ffn, _) => {
                    return Err(PlanError::Invalid(format!(
                        "layer {} runs {ffn:?}, which its names do not spell",
                        layer.index
                    )));
                }
            }
        }
        self.whole(names.norm.clone(), DType::BF16, vec![hidden], Role::Norm);
        if !names.head_tied() {
            self.whole(
                names.head.clone(),
                DType::BF16,
                vec![arch.vocab_size, hidden],
                Role::LmHead,
            );
        }
        Ok(())
    }

    /// A SwiGLU's gate and up rows and its down columns.
    fn swiglu(
        &mut self,
        names: &ExpertNames,
        intermediate: usize,
        hidden: usize,
        format: WeightFormat,
        role: Role,
    ) -> Result<(), PlanError> {
        for name in [&names.gate, &names.up] {
            self.dense_linear(
                stem(name)?,
                intermediate,
                hidden,
                format,
                role,
                LinearSplit::Rows,
            )?;
        }
        self.dense_linear(
            stem(&names.down)?,
            hidden,
            intermediate,
            format,
            role,
            LinearSplit::Columns,
        )
    }

    #[cfg(test)]
    fn qwen(&mut self, arch: &Architecture, include_mtp: bool) -> Result<(), PlanError> {
        let hidden = arch.hidden_size;
        let Some(reference) = arch.layers.first() else {
            return Err(PlanError::Invalid(
                "the architecture has no layers".to_owned(),
            ));
        };
        // Qwen3.8 mixes several hyper-connection streams; Qwen3.5/3.6 keep one plain residual
        // stream with a pre-norm on each branch and a final norm. Every other tensor carries the
        // same name in the two families, so one walk plans both.
        let hyper = hyper_or_plain(reference);
        let hyper_width = hyper.map_or(hidden, |hyper| hyper.streams * hidden);
        let dense = qwen_dense_format(self.dense_format);
        let experts = qwen_expert_format(self.expert_format);
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
        match hyper {
            Some(hyper) => self.qwen_hyper(
                &format!("{LM_PREFIX}.hyper_connection_mixer"),
                hyper,
                hidden,
                false,
                Role::Hyper,
            ),
            None => self.whole(
                format!("{LM_PREFIX}.norm.weight"),
                DType::BF16,
                vec![hidden],
                Role::Norm,
            ),
        }
        for layer in &arch.layers {
            let prefix = format!("{LM_PREFIX}.layers.{}", layer.index);
            match (&layer.residual, hyper) {
                (Residual::Hyper(layer_hyper), Some(_)) => {
                    for branch in ["attn", "mlp"] {
                        self.qwen_hyper(
                            &format!("{prefix}.{branch}_hyper_connection"),
                            layer_hyper,
                            hidden,
                            true,
                            Role::Hyper,
                        );
                    }
                }
                (Residual::Plain, None) => self.qwen_branch_norms(&prefix, hidden, Role::Norm),
                _ => {
                    return Err(PlanError::Invalid(format!(
                        "Qwen layer {} changes the residual kind layer {} set",
                        layer.index, reference.index
                    )));
                }
            }
            let Ffn::Moe(moe) = &layer.ffn else {
                return Err(PlanError::Invalid(format!(
                    "Qwen layer {} is not a MoE layer",
                    layer.index
                )));
            };
            self.qwen_moe(&format!("{prefix}.mlp"), moe, hidden, dense, experts, None)?;
            match &layer.attention.kind {
                AttentionKind::Gdn(gdn) => {
                    self.qwen_gdn(&format!("{prefix}.linear_attn"), gdn, hidden, dense)?;
                }
                AttentionKind::Gqa { kv_heads, .. } => self.qwen_gqa(
                    &format!("{prefix}.self_attn"),
                    layer,
                    *kv_heads,
                    hidden,
                    dense,
                    None,
                )?,
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
        if include_mtp && self.serves_mtp(arch) {
            self.qwen_mtp(arch, hidden, hyper_width, dense)?;
        }
        if self.vision {
            self.qwen_vision(arch)?;
        }
        Ok(())
    }

    /// A plain-residual layer's two pre-norms.
    #[cfg(test)]
    fn qwen_branch_norms(&mut self, prefix: &str, hidden: usize, role: Role) {
        for name in ["input_layernorm", "post_attention_layernorm"] {
            self.whole(
                format!("{prefix}.{name}.weight"),
                DType::BF16,
                vec![hidden],
                role,
            );
        }
    }

    /// A Qwen MoE block: the router, the gated shared expert and the routed experts.
    ///
    /// `stage` is the MTP role: an MTP block keeps every tensor whole, because a multi-rank
    /// serve replicates the stage rather than splitting it.
    #[cfg(test)]
    fn qwen_moe(
        &mut self,
        prefix: &str,
        moe: &Moe,
        hidden: usize,
        dense: WeightFormat,
        experts: ExpertFormat,
        stage: Option<Role>,
    ) -> Result<(), PlanError> {
        let (router, shared, routed) = stage
            .map_or((Role::Router, Role::SharedExpert, Role::Experts), |role| {
                (role, role, role)
            });
        let (rows, columns) = if stage.is_some() {
            (LinearSplit::Whole, LinearSplit::Whole)
        } else {
            (LinearSplit::Rows, LinearSplit::Columns)
        };
        self.whole(
            format!("{prefix}.gate.weight"),
            DType::BF16,
            vec![moe.routed_experts, hidden],
            router,
        );
        for name in ["gate_proj", "up_proj"] {
            self.dense_linear(
                &format!("{prefix}.shared_expert.{name}"),
                moe.shared_intermediate,
                hidden,
                dense,
                shared,
                rows,
            )?;
        }
        self.dense_linear(
            &format!("{prefix}.shared_expert.down_proj"),
            hidden,
            moe.shared_intermediate,
            dense,
            shared,
            columns,
        )?;
        self.whole(
            format!("{prefix}.shared_expert_gate.weight"),
            DType::BF16,
            vec![1, hidden],
            shared,
        );
        if experts == ExpertFormat::Bf16 {
            return self.qwen_fused_experts(prefix, moe, hidden, routed);
        }
        for expert in 0..moe.routed_experts {
            for (name, rows) in [("gate_proj", true), ("up_proj", true), ("down_proj", false)] {
                self.expert(
                    &format!("{prefix}.experts.{expert}.{name}"),
                    moe.intermediate,
                    hidden,
                    experts,
                    rows,
                    routed,
                )?;
            }
        }
        Ok(())
    }

    /// Qwen3.5/3.6's BF16 experts: every expert's gate and up rows fused in one
    /// `[experts, 2 * intermediate, hidden]` tensor (gate first) and the down projections in one
    /// `[experts, hidden, intermediate]`, both without a `.weight` suffix.
    ///
    /// A rank's share of the fused gate/up axis is two runs (its gate rows and its up rows), not
    /// one, so that tensor reports no run; the down projection's share is one run of columns.
    #[cfg(test)]
    fn qwen_fused_experts(
        &mut self,
        prefix: &str,
        moe: &Moe,
        hidden: usize,
        role: Role,
    ) -> Result<(), PlanError> {
        let (local, run) = self.expert_width(moe.intermediate, ExpertFormat::Bf16)?;
        let experts = moe.routed_experts;
        self.push(
            format!("{prefix}.experts.gate_up_proj"),
            DType::BF16,
            vec![experts, 2 * moe.intermediate, hidden],
            vec![experts, 2 * local, hidden],
            None,
            role,
        );
        self.push(
            format!("{prefix}.experts.down_proj"),
            DType::BF16,
            vec![experts, hidden, moe.intermediate],
            vec![experts, hidden, local],
            run,
            role,
        );
        Ok(())
    }

    /// A Gated DeltaNet branch. The three wide projections take the dense format; the decay and
    /// beta projections, the convolution and the norms stay BF16 in every release.
    #[cfg(test)]
    fn qwen_gdn(
        &mut self,
        p: &str,
        gdn: &sparkflash_model::architecture::Gdn,
        hidden: usize,
        dense: WeightFormat,
    ) -> Result<(), PlanError> {
        let key_width = gdn.key_heads * gdn.key_dim;
        let value_width = gdn.value_heads * gdn.value_dim;
        let conv_width = 2 * key_width + value_width;
        self.dense_linear(
            &format!("{p}.in_proj_qkv"),
            conv_width,
            hidden,
            dense,
            Role::Attention,
            LinearSplit::Rows,
        )?;
        self.dense_linear(
            &format!("{p}.in_proj_z"),
            value_width,
            hidden,
            dense,
            Role::Attention,
            LinearSplit::Rows,
        )?;
        for name in ["in_proj_a", "in_proj_b"] {
            self.rows(
                format!("{p}.{name}.weight"),
                DType::BF16,
                [gdn.value_heads, hidden],
                Role::Attention,
            )?;
        }
        self.dense_linear(
            &format!("{p}.out_proj"),
            hidden,
            value_width,
            dense,
            Role::Attention,
            LinearSplit::Columns,
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
        Ok(())
    }

    /// A gated GQA branch: the query projection carries the sigmoid gate interleaved with each
    /// head, so it is twice the attention width. Qwen3.8's sparse layers add the QSA indexer.
    ///
    /// `stage` is the MTP role, under which every tensor stays whole.
    #[cfg(test)]
    fn qwen_gqa(
        &mut self,
        p: &str,
        layer: &Layer,
        kv_heads: usize,
        hidden: usize,
        dense: WeightFormat,
        stage: Option<Role>,
    ) -> Result<(), PlanError> {
        let (attention, indexer_role) =
            stage.map_or((Role::Attention, Role::Indexer), |role| (role, role));
        let (rows, kv_rows, columns) = if stage.is_some() {
            (LinearSplit::Whole, LinearSplit::Whole, LinearSplit::Whole)
        } else {
            (LinearSplit::Rows, LinearSplit::KvRows, LinearSplit::Columns)
        };
        let width = layer.attention.heads * layer.attention.head_dim;
        let kv_width = kv_heads * layer.attention.head_dim;
        self.dense_linear(
            &format!("{p}.q_proj"),
            2 * width,
            hidden,
            dense,
            attention,
            rows,
        )?;
        for name in ["k_proj", "v_proj"] {
            self.dense_linear(
                &format!("{p}.{name}"),
                kv_width,
                hidden,
                dense,
                attention,
                kv_rows,
            )?;
        }
        self.dense_linear(
            &format!("{p}.o_proj"),
            hidden,
            width,
            dense,
            attention,
            columns,
        )?;
        for name in ["q_norm", "k_norm"] {
            self.whole(
                format!("{p}.{name}.weight"),
                DType::BF16,
                vec![layer.attention.head_dim],
                attention,
            );
        }
        if let Some(indexer) = &layer.attention.indexer {
            let width = (indexer.heads + indexer.kv_heads) * indexer.head_dim;
            self.whole(
                format!("{p}.indexer.index_qk_proj.weight"),
                DType::BF16,
                vec![width, hidden],
                indexer_role,
            );
            for name in ["q_layernorm", "k_layernorm"] {
                self.whole(
                    format!("{p}.indexer.{name}.weight"),
                    DType::BF16,
                    vec![indexer.head_dim],
                    indexer_role,
                );
            }
        }
        Ok(())
    }
    #[cfg(test)]
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

    #[cfg(test)]
    fn qwen_mtp(
        &mut self,
        arch: &Architecture,
        hidden: usize,
        hyper_width: usize,
        dense: WeightFormat,
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
                "the Qwen MTP stage needs a full- or sparse-attention layer".to_owned(),
            ));
        };
        let Some(moe) = arch.first_moe() else {
            return Err(PlanError::Invalid(
                "the Qwen MTP stage needs a MoE layer".to_owned(),
            ));
        };
        // The released Qwen MTP stages store their experts in the format the quantization table
        // records for them, falling back to the backbone's.
        let experts = qwen_expert_format(
            arch.quantization
                .mtp_expert_format
                .unwrap_or(self.expert_format),
        );
        let prefix = "mtp";
        let layer = format!("{prefix}.layers.0");
        match &reference.residual {
            Residual::Hyper(hyper) => {
                // Qwen3.8: the embedding and the hidden streams each get a square projection,
                // and the stage carries its own mixer and per-branch hyper-connections.
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
                for branch in ["attn", "mlp"] {
                    self.qwen_hyper(
                        &format!("{layer}.{branch}_hyper_connection"),
                        hyper,
                        hidden,
                        true,
                        Role::Mtp,
                    );
                }
            }
            Residual::Plain | Residual::BlockAttention => {
                // Qwen3.5/3.6: one `[hidden, 2 * hidden]` projection of the normed embedding
                // and hidden state, concatenated, then an ordinary pre-norm layer.
                self.whole(
                    format!("{prefix}.fc.weight"),
                    DType::BF16,
                    vec![hidden, 2 * hidden],
                    Role::Mtp,
                );
                for name in ["pre_fc_norm_embedding", "pre_fc_norm_hidden"] {
                    self.whole(
                        format!("{prefix}.{name}.weight"),
                        DType::BF16,
                        vec![hidden],
                        Role::Mtp,
                    );
                }
                self.qwen_branch_norms(&layer, hidden, Role::Mtp);
            }
        }
        self.qwen_gqa(
            &format!("{layer}.self_attn"),
            reference,
            *kv_heads,
            hidden,
            dense,
            Some(Role::Mtp),
        )?;
        self.qwen_moe(
            &format!("{layer}.mlp"),
            moe,
            hidden,
            dense,
            experts,
            Some(Role::Mtp),
        )?;
        if matches!(reference.residual, Residual::Plain) {
            self.whole(
                format!("{prefix}.norm.weight"),
                DType::BF16,
                vec![hidden],
                Role::Mtp,
            );
        }
        Ok(())
    }

    #[cfg(test)]
    fn qwen_vision(&mut self, arch: &Architecture) -> Result<(), PlanError> {
        let Some(vision) = &arch.vision else {
            return Ok(());
        };
        let width = vision.hidden_size;
        let prefix = VISION_PREFIX;
        let frames = vision
            .patch_temporal
            .unwrap_or(vision.temporal_patch_size)
            .max(1);
        let patch = vision.in_channels * frames * vision.patch_size * vision.patch_size;
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

    #[cfg(test)]
    fn oracle(&mut self, arch: &Architecture, include_mtp: bool) -> Result<(), PlanError> {
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
                self.oracle_hyper(&format!("{prefix}.hc_{branch}"), hyper, hidden)?;
            }
            match &layer.attention.kind {
                AttentionKind::Kda(kda) => {
                    self.oracle_kda(&format!("{prefix}.self_attn"), kda, hidden, Role::Attention)?;
                }
                AttentionKind::Mla { .. } => {
                    let indexer = layer
                        .attention
                        .indexer
                        .as_ref()
                        .is_some_and(|indexer| !indexer.reuse);
                    self.oracle_dsa(
                        &format!("{prefix}.self_attn"),
                        layer,
                        hidden,
                        indexer,
                        Role::Attention,
                        Role::Indexer,
                    )?;
                }
                AttentionKind::Gqa { kv_heads, .. } => {
                    self.oracle_gqa(
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
                    self.oracle_dense_mlp(&format!("{prefix}.mlp"), dense, hidden, Role::DenseMlp)?;
                }
                Ffn::Moe(moe) => self.oracle_moe(
                    &format!("{prefix}.mlp"),
                    moe,
                    hidden,
                    Role::Router,
                    Role::SharedExpert,
                    Role::Experts,
                )?,
            }
        }
        if include_mtp && self.serves_mtp(arch) {
            self.oracle_mtp(arch, hidden)?;
        }
        if self.vision {
            self.oracle_vision(arch)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn oracle_hyper(
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

    #[cfg(test)]
    fn oracle_kda(
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

    #[cfg(test)]
    fn oracle_gqa(
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

    #[cfg(test)]
    fn oracle_dsa(
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

    #[cfg(test)]
    fn oracle_dense_mlp(
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

    #[cfg(test)]
    fn oracle_moe(
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

    #[cfg(test)]
    fn oracle_mtp(&mut self, arch: &Architecture, hidden: usize) -> Result<(), PlanError> {
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
        self.oracle_dsa(
            &format!("{prefix}.self_attn"),
            reference,
            hidden,
            true,
            Role::Mtp,
            Role::Mtp,
        )?;
        let Some(moe) = arch.first_moe() else {
            return Err(PlanError::Invalid(
                "the GLM MTP stage needs a MoE layer".to_owned(),
            ));
        };
        self.oracle_moe(
            &format!("{prefix}.mlp"),
            moe,
            hidden,
            Role::Mtp,
            Role::Mtp,
            Role::Mtp,
        )
    }

    #[cfg(test)]
    fn oracle_vision(&mut self, arch: &Architecture) -> Result<(), PlanError> {
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
                vision
                    .patch_temporal
                    .unwrap_or(vision.temporal_patch_size)
                    .max(1),
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

// ---------------------------------------------------------------------------------
// Tensor coverage: the claim set and the checkpoint index
// ---------------------------------------------------------------------------------

/// The tensors a model description takes, as one comparable set.
///
/// This is the seam between the scan and the description. The set is
/// [ArchitectureTensorPlan::checkpoint]'s names, which a matching layout supplies when one
/// exists. The scan compares [TensorClaims] with an index and never learns how the claims were
/// built.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TensorClaims {
    specs: Vec<TensorSpec>,
}

impl TensorClaims {
    /// The claim set of the shared checkpoint plan.
    ///
    /// # Errors
    /// As [ArchitectureTensorPlan::checkpoint].
    pub fn from_checkpoint(architecture: &Architecture, tp: TpPlan) -> Result<Self, PlanError> {
        let plan = ArchitectureTensorPlan::checkpoint(architecture, tp)?;
        let mut specs = plan.specs().to_vec();
        // A stored-but-unloaded tensor is not a plan member, but the index still holds it: the
        // claim set must name it or the scan reports it unclaimed.
        if let Some(layout) = sparkflash_model::layout::matching(architecture)
            .map_err(|error| PlanError::Invalid(error.to_string()))?
        {
            for tensor in layout
                .tensors(architecture)
                .map_err(|error| PlanError::Invalid(error.to_string()))?
            {
                if tensor.unloaded.is_none() || specs.iter().any(|spec| spec.name == tensor.name) {
                    continue;
                }
                let dtype = match tensor.dtype {
                    LayoutDtype::Bf16 => DType::BF16,
                    LayoutDtype::F32 => DType::F32,
                    LayoutDtype::F8E4M3 => DType::F8E4M3,
                    LayoutDtype::F8E8M0 => DType::F8E8M0,
                    LayoutDtype::F4E2M1 => DType::F4E2M1,
                    LayoutDtype::U8 => DType::U8,
                    LayoutDtype::I64 => DType::I64,
                    LayoutDtype::Dense | LayoutDtype::Expert => {
                        return Err(PlanError::Invalid(format!(
                            "unloaded tensor {} names a checkpoint-derived dtype; the claim set                              needs an explicit one",
                            tensor.name
                        )));
                    }
                };
                specs.push(TensorSpec {
                    name: tensor.name,
                    dtype,
                    shape: tensor.shape,
                });
            }
        }
        Ok(Self::from_specs(specs))
    }

    /// A claim set over the given specs, in the given order.
    #[must_use]
    pub fn from_specs(specs: Vec<TensorSpec>) -> Self {
        Self { specs }
    }

    /// Every claimed tensor, in construction order.
    #[must_use]
    pub fn specs(&self) -> &[TensorSpec] {
        &self.specs
    }

    /// The number of claimed tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Whether the description claims no tensor at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Whether the description claims `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.specs.iter().any(|spec| spec.name == name)
    }
}

/// One planned tensor the index does not match: absent, or stored at another shape or dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDisagreement {
    /// The planned tensor's name.
    pub tensor: String,
    /// The dtype the plan expects, as [DType::name].
    pub expected_dtype: String,
    /// The logical shape the plan expects.
    pub expected_shape: Vec<usize>,
    /// The dtype the checkpoint stores, when the name is present.
    pub found_dtype: Option<String>,
    /// The shape the checkpoint stores, when the name is present.
    pub found_shape: Option<Vec<usize>>,
    /// Why the two do not agree, in one sentence.
    pub reason: String,
}

/// A set of unclaimed tensors that shares a prefix: one row for a large group, one row a tensor
/// for a small one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorGroup {
    /// The exact common prefix, without a trailing dot.
    pub prefix: String,
    /// The names the group covers, in name order.
    pub names: Vec<String>,
}

/// Groups smaller than this are emitted tensor by tensor; a larger one is one row naming its
/// prefix and count.
const SMALL_UNCLAIMED_GROUP: usize = 8;

impl TensorGroup {
    /// How many tensors the group covers.
    #[must_use]
    pub fn count(&self) -> usize {
        self.names.len()
    }

    /// The label the report prints: the exact tensor name for a single row, or the exact prefix,
    /// a `*` and the count for a group.
    #[must_use]
    pub fn label(&self) -> String {
        if self.names.len() == 1 {
            self.prefix.clone()
        } else {
            format!("{}.* ({} tensors)", self.prefix, self.names.len())
        }
    }
}

/// The result of checking a description's claims against a checkpoint index, in both directions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TensorCoverage {
    /// How many planned tensors the index holds at the planned dtype and shape.
    pub matched: usize,
    /// Planned tensors the index does not match, in claim order.
    pub disagreements: Vec<TensorDisagreement>,
    /// Index tensors no planned part takes, grouped by common prefix.
    pub unclaimed: Vec<TensorGroup>,
}

impl TensorCoverage {
    /// Whether every planned tensor matched and every indexed tensor was claimed.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.disagreements.is_empty() && self.unclaimed.is_empty()
    }

    /// How many indexed tensors no part took.
    #[must_use]
    pub fn unclaimed_count(&self) -> usize {
        self.unclaimed.iter().map(TensorGroup::count).sum()
    }
}

/// Compare a description's claims with a checkpoint index, in both directions.
///
/// A planned tensor must be in the index at the planned dtype and shape; a container's reshape
/// (a stacked expert block, a convolution's unit axis) and NVFP4's `U8` packing are accepted by
/// [TensorSpec::is_stored_as], exactly as the loader accepts them. Every indexed tensor the claim
/// set does not name is unclaimed.
#[must_use]
pub fn compare_tensor_claims(claims: &TensorClaims, index: &TensorIndex) -> TensorCoverage {
    let mut by_name: BTreeMap<&str, &TensorEntry> = BTreeMap::new();
    for entry in &index.tensors {
        by_name.entry(entry.name.as_str()).or_insert(entry);
    }
    let mut coverage = TensorCoverage::default();
    for spec in claims.specs() {
        let Some(entry) = by_name.get(spec.name.as_str()) else {
            coverage.disagreements.push(TensorDisagreement {
                tensor: spec.name.clone(),
                expected_dtype: spec.dtype.name().to_owned(),
                expected_shape: spec.shape.clone(),
                found_dtype: None,
                found_shape: None,
                reason: "the index has no tensor with this name".to_owned(),
            });
            continue;
        };
        match entry.logical_spec() {
            Ok(found) if stored_as_plan(spec, &found) => coverage.matched += 1,
            Ok(found) => {
                let reason = if found.dtype != spec.dtype {
                    format!(
                        "the index stores {} where the plan expects {}",
                        found.dtype.name(),
                        spec.dtype.name()
                    )
                } else {
                    format!(
                        "the index stores shape {} where the plan expects {}",
                        shape_text(&found.shape),
                        shape_text(&spec.shape)
                    )
                };
                coverage.disagreements.push(TensorDisagreement {
                    tensor: spec.name.clone(),
                    expected_dtype: spec.dtype.name().to_owned(),
                    expected_shape: spec.shape.clone(),
                    found_dtype: Some(entry.dtype.clone()),
                    found_shape: Some(entry.shape.clone()),
                    reason,
                });
            }
            Err(error) => coverage.disagreements.push(TensorDisagreement {
                tensor: spec.name.clone(),
                expected_dtype: spec.dtype.name().to_owned(),
                expected_shape: spec.shape.clone(),
                found_dtype: Some(entry.dtype.clone()),
                found_shape: Some(entry.shape.clone()),
                reason: error.to_string(),
            }),
        }
    }
    let claimed: BTreeSet<&str> = claims
        .specs()
        .iter()
        .map(|spec| spec.name.as_str())
        .collect();
    let unclaimed: Vec<TensorEntry> = index
        .tensors
        .iter()
        .filter(|entry| !claimed.contains(entry.name.as_str()))
        .cloned()
        .collect();
    coverage.unclaimed = group_unclaimed(&unclaimed);
    coverage
}

/// Whether a checkpoint descriptor holds a planned tensor: [TensorSpec::is_stored_as], or a
/// stored collapse of the plan's last axis into a trailing run of dimensions.
///
/// A patch embedding is the case that needs the second rule: the plan keeps the arithmetic's
/// `[width, in * frames * patch * patch]` matrix, while a released shard stores the same values
/// as `[width, in, frames, patch, patch]`. The first axis and the total width must agree, so a
/// reshape that moves the row axis or changes the element count is still a different tensor.
fn stored_as_plan(spec: &TensorSpec, found: &TensorSpec) -> bool {
    if spec.is_stored_as(found) {
        return true;
    }
    if spec.dtype != found.dtype {
        return false;
    }
    let [rows, columns] = spec.shape.as_slice() else {
        return false;
    };
    let Some((first, rest)) = found.shape.split_first() else {
        return false;
    };
    found.shape.len() > 2
        && first == rows
        && rest.iter().map(|&dim| dim as u128).product::<u128>() == *columns as u128
}

/// Group unclaimed tensors by the prefix their names share.
///
/// The grouping key is the tower root for a tower (`vision`, `visual`, `audio`, `aligner`,
/// or `model.visual` / `model.audio`) and the first two dotted components otherwise. A group of
/// [SMALL_UNCLAIMED_GROUP] tensors or fewer is emitted one row per tensor, so a single stray
/// tensor is named exactly; a larger group is one row naming the exact common prefix and its
/// count. Nothing here knows a model or a family: the prefix comes from the names.
#[must_use]
pub fn group_unclaimed(entries: &[TensorEntry]) -> Vec<TensorGroup> {
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entry in entries {
        grouped
            .entry(group_key(&entry.name))
            .or_default()
            .push(entry.name.clone());
    }
    let mut groups = Vec::new();
    for (key, mut names) in grouped {
        names.sort();
        names.dedup();
        if names.len() <= SMALL_UNCLAIMED_GROUP {
            for name in names {
                groups.push(TensorGroup {
                    prefix: name.clone(),
                    names: vec![name],
                });
            }
            continue;
        }
        let prefix = common_prefix(&names).unwrap_or(key);
        groups.push(TensorGroup { prefix, names });
    }
    groups.sort_by(|left, right| left.prefix.cmp(&right.prefix));
    groups
}

/// Whether a tensor belongs to a tower rather than the language backbone. The scan uses this
/// only to report a checkpoint whose shared plan could not be built: its tower tensors are
/// still unclaimed, while the tensors the missing plan would have claimed are the plan error's
/// subject, not tens of thousands of rows.
#[must_use]
pub fn is_tower_tensor(name: &str) -> bool {
    name.starts_with("aligner.")
        || name
            .split('.')
            .any(|part| matches!(part, "vision" | "visual" | "audio"))
}

/// The grouping key of one tensor name.
fn group_key(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    let tower = |part: &str| matches!(part, "vision" | "visual" | "audio" | "aligner");
    match parts.as_slice() {
        [] => name.to_owned(),
        [only] => (*only).to_owned(),
        [first, second, ..] if tower(first) => (*first).to_owned(),
        [first, second, ..] if first == &"model" && tower(second) => format!("{first}.{second}"),
        [first, second, ..] => format!("{first}.{second}"),
    }
}

/// The longest dotted-component prefix every name shares.
fn common_prefix(names: &[String]) -> Option<String> {
    let first: Vec<&str> = names.first()?.split('.').collect();
    let mut length = first.len();
    for name in &names[1..] {
        let parts: Vec<&str> = name.split('.').collect();
        let mut shared = 0;
        while shared < length && shared < parts.len() && first[shared] == parts[shared] {
            shared += 1;
        }
        length = shared;
        if length == 0 {
            return Some(String::new());
        }
    }
    Some(first[..length].join("."))
}

/// A shape in the report's `128x5120` spelling.
#[must_use]
pub fn shape_text(shape: &[usize]) -> String {
    if shape.is_empty() {
        "scalar".to_owned()
    } else {
        shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("x")
    }
}

/// A layer's hyper-connection, or `None` for a plain residual stream.
/// A weight's name without its `.weight`, the prefix the linear helpers spell scales under.
fn stem(name: &str) -> Result<&str, PlanError> {
    name.strip_suffix(".weight")
        .ok_or_else(|| PlanError::Invalid(format!("{name} is not a weight the plan can scale")))
}

#[cfg(test)]
fn hyper_or_plain(layer: &Layer) -> Option<&sparkflash_model::architecture::HyperConnection> {
    match &layer.residual {
        Residual::Hyper(hyper) => Some(hyper),
        Residual::Plain | Residual::BlockAttention => None,
    }
}

/// Qwen's official FP8 releases store one BF16 inverse scale a 128x128 tile, for the dense
/// projections and the experts alike, whatever scale dtype the quantization table names for the
/// same arithmetic.
#[cfg(test)]
fn qwen_dense_format(format: WeightFormat) -> WeightFormat {
    match format {
        WeightFormat::Fp8Tile {
            block_m,
            block_n,
            dynamic_activations,
            ..
        } => WeightFormat::Fp8Tile {
            block_m,
            block_n,
            scale_dtype: ScaleDType::Bf16,
            dynamic_activations,
        },
        other => other,
    }
}

/// The routed experts' stored format, with Qwen's BF16 tile scale.
#[cfg(test)]
fn qwen_expert_format(format: WeightFormat) -> ExpertFormat {
    match ExpertFormat::of(format) {
        ExpertFormat::Fp8 { block, .. } => ExpertFormat::Fp8 {
            block,
            scale: DType::BF16,
        },
        other => other,
    }
}

#[cfg(test)]
fn hyper_of(layer: &Layer) -> Result<&sparkflash_model::architecture::HyperConnection, PlanError> {
    match &layer.residual {
        Residual::Hyper(hyper) => Ok(hyper),
        Residual::Plain | Residual::BlockAttention => Err(PlanError::Invalid(format!(
            "layer {} uses a plain residual where the plan expects hyper-connections",
            layer.index
        ))),
    }
}

#[cfg(test)]
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

fn tensor_role(name: &str) -> Option<Role> {
    Some(match name {
        "embedding" => Role::Embedding,
        "lm_head" => Role::LmHead,
        "norm" => Role::Norm,
        "hyper" => Role::Hyper,
        "router" => Role::Router,
        "shared_expert" => Role::SharedExpert,
        "experts" => Role::Experts,
        "attention" => Role::Attention,
        "indexer" => Role::Indexer,
        "dense_mlp" => Role::DenseMlp,
        "ple_aux" => Role::PleAux,
        "ple_table" => Role::PleTable,
        "mtp" => Role::Mtp,
        "vision" => Role::Vision,
        _ => return None,
    })
}

fn one(name: &str, shape: &[usize]) -> Result<[usize; 1], PlanError> {
    shape
        .try_into()
        .map_err(|_| PlanError::Invalid(format!("tensor {name} is a vector with shape {shape:?}")))
}

fn two(name: &str, shape: &[usize]) -> Result<[usize; 2], PlanError> {
    shape
        .try_into()
        .map_err(|_| PlanError::Invalid(format!("tensor {name} is not a matrix: shape {shape:?}")))
}

fn three(name: &str, shape: &[usize]) -> Result<[usize; 3], PlanError> {
    shape.try_into().map_err(|_| {
        PlanError::Invalid(format!(
            "tensor {name} is not a 3-D convolution: shape {shape:?}"
        ))
    })
}

#[cfg(test)]
mod layout_equality {
    use super::*;

    fn reference(arch: &Architecture, tp: TpPlan) -> ArchitectureTensorPlan {
        let mut builder = Builder::new(tp, true, true, true);
        let dense = arch.quantization.dense_format;
        let experts = arch.quantization.expert_format;
        builder.dense_format = dense;
        builder.expert_format = if matches!(experts, WeightFormat::Bf16)
            && matches!(dense, WeightFormat::Fp8Tile { .. })
        {
            dense
        } else {
            experts
        };
        builder.oracle(arch, true).expect("the reference planner");
        builder.finish()
    }

    type PlanRow = (String, Vec<usize>, Vec<usize>, Option<RankRun>, String);

    fn table(plan: &ArchitectureTensorPlan) -> BTreeMap<String, PlanRow> {
        let mut out = BTreeMap::new();
        for index in 0..plan.len() {
            let spec = plan.spec_at(index);
            out.insert(
                spec.name.clone(),
                (
                    format!("{:?}", spec.dtype),
                    spec.shape.clone(),
                    plan.rank_spec_at(index).shape.clone(),
                    plan.rank_run_at(index),
                    format!("{:?}", plan.plan().role(index)),
                ),
            );
        }
        out
    }

    fn family_reference(arch: &Architecture, tp: TpPlan) -> ArchitectureTensorPlan {
        let mut builder = Builder::new(tp, true, true, true);
        let dense = arch.quantization.dense_format;
        let experts = arch.quantization.expert_format;
        builder.dense_format = dense;
        builder.expert_format = if matches!(experts, WeightFormat::Bf16)
            && matches!(dense, WeightFormat::Fp8Tile { .. })
        {
            dense
        } else {
            experts
        };
        builder.qwen(arch, true).expect("the reference planner");
        builder.finish()
    }

    #[test]
    fn the_second_worked_layout_reproduces_the_family_planner() {
        let cases = [
            (
                "nvfp4",
                include_str!(
                    "../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json"
                ),
                false,
                false,
            ),
            (
                "fp8",
                include_str!(
                    "../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_fp8.hf.json"
                ),
                true,
                false,
            ),
            (
                "plain-fp8",
                include_str!("../../sparkflash-model/config/qwen36_flash_35b_a3b_fp8.hf.json"),
                false,
                true,
            ),
        ];
        for (label, text, experts_only, weight_only) in cases {
            let mut architecture = Architecture::from_json(text).expect("the carried config");
            if experts_only {
                architecture.experts_only_fp8();
            }
            if weight_only {
                architecture.weight_only_fp8();
            }
            // Both Qwen families are served at one rank, but the layout's tile splits must
            // still reproduce the reference planner at every world.
            for world in [1usize, 2, 4] {
                let tp = TpPlan::for_architecture(&architecture, world, 0).expect("a TP plan");
                let expected = family_reference(&architecture, tp);
                let actual = ArchitectureTensorPlan::checkpoint(&architecture, tp)
                    .expect("the layout checkpoint plan");
                let want = table(&expected);
                let have = table(&actual);
                let missing: Vec<&String> = want
                    .keys()
                    .filter(|name| !have.contains_key(*name))
                    .collect();
                let extra: Vec<&String> = have
                    .keys()
                    .filter(|name| !want.contains_key(*name))
                    .collect();
                assert!(
                    missing.is_empty() && extra.is_empty(),
                    "{label} TP{world}: missing {} {:?}, extra {} {:?}",
                    missing.len(),
                    &missing[..missing.len().min(8)],
                    extra.len(),
                    &extra[..extra.len().min(8)],
                );
                for (name, expected) in &want {
                    let actual = &have[name];
                    assert_eq!(
                        expected.0, actual.0,
                        "{label} TP{world}: {name} stored dtype differs"
                    );
                    assert_eq!(
                        expected.1, actual.1,
                        "{label} TP{world}: {name} logical shape differs"
                    );
                    assert_eq!(
                        expected.2, actual.2,
                        "{label} TP{world}: {name} rank shape differs"
                    );
                    assert_eq!(
                        expected.3, actual.3,
                        "{label} TP{world}: {name} rank run differs"
                    );
                    assert_eq!(
                        expected.4, actual.4,
                        "{label} TP{world}: {name} role differs"
                    );
                }
            }
        }
    }

    #[test]
    fn the_worked_layout_reproduces_the_family_planner() {
        let text = include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
        let architecture = Architecture::from_json(text).expect("the carried config");
        for world in [1usize, 2, 4] {
            let tp = TpPlan::for_architecture(&architecture, world, 0).expect("a TP plan");
            let expected = reference(&architecture, tp);
            let actual = ArchitectureTensorPlan::checkpoint(&architecture, tp)
                .expect("the layout checkpoint plan");
            let want = table(&expected);
            let have = table(&actual);
            let missing: Vec<&String> = want
                .keys()
                .filter(|name| !have.contains_key(*name))
                .collect();
            let extra: Vec<&String> = have
                .keys()
                .filter(|name| !want.contains_key(*name))
                .collect();
            assert!(
                missing.is_empty() && extra.is_empty(),
                "TP{world}: missing {missing:?}, extra {extra:?}"
            );
            for (name, expected) in &want {
                let actual = &have[name];
                assert_eq!(
                    expected.0, actual.0,
                    "TP{world}: {name} stored dtype differs"
                );
                assert_eq!(
                    expected.1, actual.1,
                    "TP{world}: {name} logical shape differs"
                );
                assert_eq!(expected.2, actual.2, "TP{world}: {name} rank shape differs");
                assert_eq!(expected.3, actual.3, "TP{world}: {name} rank run differs");
                assert_eq!(expected.4, actual.4, "TP{world}: {name} role differs");
            }
        }
    }
}

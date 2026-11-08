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

//! The model-independent execution plan.
//!
//! [EnginePlan] is the object between a released [Architecture] and a model backend. It reads
//! the typed layer walk, the tensor-parallel split and the stored precision, and produces one
//! [PartRequirement] per part the run will execute, with the rank-local geometry already
//! applied. A backend opens from this plan rather than from a family's private configuration.
//!
//! The plan deliberately owns no kernel choice. [crate::resolver::KernelResolver] maps each
//! requirement to a loadable [crate::parts::PartKernel]; keeping the two apart lets the plan be
//! inspected on a CPU host and lets a new kernel implementation land without touching the walk.
//!
//! The flow is:
//!
//! ```text
//! config.json -> Architecture -> TpPlan -> EnginePlan -> KernelResolver -> backend
//! ```
//!
//! [EnginePlan::from_json] is the short form a boot path uses; [EnginePlan::from_architecture]
//! is the form a test or a synthetic checkpoint uses when it already holds the architecture.

use sparkflash_model::architecture::{
    AttentionKind, Ffn, Quantization, Residual, Side, WeightFormat,
};
use sparkflash_model::{
    Architecture, ConfigError, ModelBackend, Part, StepRole, Target, TpAxis, TpPlan, TpPlanError,
};
use sparkflash_weights::weights::WeightSource;
use thiserror::Error;

use crate::gemv::GEMV_MAX_BATCH;
use crate::plan::{GroupedFp8Body, PartGeometry, Precision};
use crate::resolver::PlanContext;

/// The operand precision this run uses for each class of weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionPolicy {
    /// Dense projections, attention weights and the embedding/head.
    pub dense: Precision,
    /// Routed and shared experts.
    pub experts: Precision,
    /// The MTP stage's experts, when the release stores them differently.
    pub mtp: Precision,
    /// Attention operands, when they differ from the dense class.
    pub attention: Precision,
}

impl PrecisionPolicy {
    /// Read the policy the released quantization block describes.
    ///
    /// The formats are already typed by [Architecture]'s parser; this maps each one through
    /// [precision_of], so the dense, expert and MTP classes cannot disagree with the names the
    /// checkpoint declares. An MTP stage that names no separate format uses the backbone experts'
    /// format, which is what the release means by naming only one.
    #[must_use]
    pub fn from_architecture(architecture: &Architecture) -> Self {
        let quantization = &architecture.quantization;
        let dense = precision_of(quantization.dense_format);
        let experts = precision_of(quantization.expert_format);
        let mtp = quantization.mtp_expert_format.map_or(experts, precision_of);
        Self {
            dense,
            experts,
            mtp,
            attention: dense,
        }
    }
}

/// The operand precision one stored weight format uses.
///
/// The two 4-bit formats share [Precision::NvFp4]: the ops library has one 4-bit operand
/// class (its kernel is named for MXFP4's block but serves both), and the release's format
/// still selects the loader. Every other format is distinct.
#[must_use]
pub const fn precision_of(format: WeightFormat) -> Precision {
    match format {
        WeightFormat::Bf16 => Precision::Bf16,
        WeightFormat::Fp8Tile { .. } => Precision::Fp8E4M3,
        WeightFormat::MxFp8 { .. } => Precision::MxFp8,
        WeightFormat::NvFp4 { .. } | WeightFormat::MxFp4 { .. } => Precision::NvFp4,
    }
}

/// One part, at this rank's geometry, with the precision the run will use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartRequirement {
    /// The model-side part.
    pub part: Part,
    /// The rank-local shape.
    pub geometry: PartGeometry,
    /// The operand precision.
    pub precision: Precision,
}

/// One decoder layer as the execution plan sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerExecution {
    /// The checkpoint layer index.
    pub index: usize,
    /// The ordered parts of the layer, from [Architecture::steps].
    pub steps: Vec<(StepRole, Part)>,
    /// The same parts with their rank-local geometry.
    pub requirements: Vec<PartRequirement>,
}

/// A plan construction failure.
#[derive(Debug, Error)]
pub enum PlanError {
    /// The released configuration did not parse.
    #[error("the released configuration: {0}")]
    Config(#[from] ConfigError),
    /// The tensor-parallel split does not divide the architecture.
    #[error("the tensor-parallel plan: {0}")]
    Tp(#[from] TpPlanError),
    /// A part the architecture names has no geometry in this plan.
    #[error("the execution plan: {0}")]
    Invalid(String),
    /// The configuration file could not be read.
    #[error("the released configuration file: {0}")]
    Io(String),
}

/// The model-independent execution plan of one model at one target and world size.
#[derive(Debug, Clone, PartialEq)]
pub struct EnginePlan {
    architecture: Architecture,
    target: Target,
    tp: TpPlan,
    precision: PrecisionPolicy,
    globals: Vec<PartRequirement>,
    layers: Vec<LayerExecution>,
}

impl EnginePlan {
    /// Parse a released `config.json` and plan it at `rank` of `world`.
    ///
    /// # Errors
    /// A configuration the shared parser refuses, or a world size that does not divide the
    /// architecture's sharded axes.
    pub fn from_json(
        text: &str,
        target: Target,
        world: usize,
        rank: usize,
    ) -> Result<Self, PlanError> {
        let architecture = Architecture::from_json(text)?;
        let tp = TpPlan::for_architecture(&architecture, world, rank)?;
        let precision = PrecisionPolicy::from_architecture(&architecture);
        Self::from_architecture(architecture, target, tp, precision)
    }

    /// Parse a released `config.json` from `path` and plan it.
    ///
    /// # Errors
    /// An unreadable file, or as [Self::from_json].
    pub fn from_path(
        path: &std::path::Path,
        target: Target,
        world: usize,
        rank: usize,
    ) -> Result<Self, PlanError> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| PlanError::Io(format!("{}: {error}", path.display())))?;
        Self::from_json(&text, target, world, rank)
    }

    /// Plan an architecture that is already parsed, at `rank` of `world`.
    ///
    /// This is the form an arm uses after its architecture bridge has applied a release's
    /// producer correction (for example [sparkflash_model::Architecture::experts_only_fp8]):
    /// the precision policy is read from the corrected typed formats, not from a second parse of
    /// the raw document.
    ///
    /// # Errors
    /// As [Self::from_architecture].
    pub fn for_architecture(
        architecture: Architecture,
        target: Target,
        world: usize,
        rank: usize,
    ) -> Result<Self, PlanError> {
        let tp = TpPlan::for_architecture(&architecture, world, rank)?;
        let precision = PrecisionPolicy::from_architecture(&architecture);
        Self::from_architecture(architecture, target, tp, precision)
    }

    /// Plan an architecture that is already parsed.
    ///
    /// # Errors
    /// A world size that does not divide the architecture's sharded axes, or a part whose
    /// geometry the architecture does not describe.
    pub fn from_architecture(
        architecture: Architecture,
        target: Target,
        tp: TpPlan,
        precision: PrecisionPolicy,
    ) -> Result<Self, PlanError> {
        // Re-derive the plan from the architecture: this is the check that the caller's
        // world/rank is valid for this model, not merely a well-formed TpPlan.
        let tp = TpPlan::for_architecture(&architecture, tp.world(), tp.rank())?;
        let globals = global_requirements(&architecture, &tp, precision)?;
        let steps = architecture.steps();
        if steps.len() != architecture.layers.len() {
            return Err(PlanError::Invalid(format!(
                "the shared walk names {} layers and the architecture has {}",
                steps.len(),
                architecture.layers.len()
            )));
        }
        let mut layers = Vec::with_capacity(steps.len());
        for (step, layer) in steps.iter().zip(&architecture.layers) {
            let mut requirements = Vec::new();
            for (_, part) in step.roles() {
                requirements.push(PartRequirement {
                    part,
                    geometry: geometry_for(part, layer, &architecture, &tp)?,
                    precision: precision_for(part, precision),
                });
            }
            layers.push(LayerExecution {
                index: step.index,
                steps: step.roles(),
                requirements,
            });
        }
        Ok(Self {
            architecture,
            target,
            tp,
            precision,
            globals,
            layers,
        })
    }

    /// The architecture this plan executes.
    #[must_use]
    pub fn architecture(&self) -> &Architecture {
        &self.architecture
    }

    /// The released `model_type` family.
    #[must_use]
    pub fn family(&self) -> &str {
        &self.architecture.family
    }

    /// The device target.
    #[must_use]
    pub const fn target(&self) -> Target {
        self.target
    }

    /// The window a device arm opens and advertises for a requested context.
    ///
    /// The requested context is the operator's `--context` (or the pack's default); the
    /// checkpoint's own `max_position_embeddings` is the hard ceiling. A request above it is
    /// capped rather than allowed to allocate past the checkpoint's window, which is what an
    /// [EngineBuilder] arm did when it reopened from the plan and silently used the full window.
    ///
    /// # Errors
    /// A zero context, which would allocate an unusable sequence.
    pub fn admitted_context(&self, requested: usize) -> Result<usize, PlanError> {
        if requested == 0 {
            return Err(PlanError::Invalid(
                "the requested context is zero".to_owned(),
            ));
        }
        Ok(requested.min(self.architecture.max_position_embeddings))
    }

    /// The single-rank plan context a device arm's [EngineBuilder] opens.
    ///
    /// The Qwen and GLM single-node arms are compiled for [Target::Sm121] and hold one whole
    /// rank; a plan at another target or at a world larger than one belongs to the TP4 group,
    /// which opens each rank through its own pack path. Returning the context instead of a
    /// second `SM121_CATALOGUE` constant is what makes the builder read the plan's own target.
    ///
    /// # Errors
    /// A plan this build's single-rank arms cannot open, naming the target or world it does.
    pub fn single_rank_context(&self) -> Result<PlanContext, PlanError> {
        if self.target != Target::Sm121 {
            return Err(PlanError::Invalid(format!(
                "the single-rank device arms are compiled for {}; this plan targets {}",
                Target::Sm121,
                self.target
            )));
        }
        if self.tp.world() != 1 || self.tp.rank() != 0 {
            return Err(PlanError::Invalid(format!(
                "the single-rank builder opens TP1/rank0; this plan is TP{}/rank{} (the TP4 \
                 group opens each rank through PackArm::OPEN_RANK, not EngineBuilder::open)",
                self.tp.world(),
                self.tp.rank()
            )));
        }
        Ok(PlanContext::catalogue(self.target))
    }

    /// The tensor-parallel plan.
    #[must_use]
    pub const fn tp(&self) -> TpPlan {
        self.tp
    }

    /// The cache claim this plan's architecture makes at this rank's world size.
    ///
    /// The caller states the absorbed-MLA latent representation explicitly (`latent_fp8`),
    /// because the same checkpoint can be served with either cache; the architecture must not
    /// infer it from the weight format. Every indexer term still comes from the checkpoint.
    ///
    /// # Errors
    /// A checkpoint whose compressed caches [Architecture::kv_plan] does not model.
    pub fn kv_plan(&self, latent_fp8: bool) -> Result<sparkflash_model::KvPlan, ConfigError> {
        self.architecture.kv_plan(self.tp.world(), latent_fp8)
    }

    /// The precision policy.
    #[must_use]
    pub const fn precision(&self) -> PrecisionPolicy {
        self.precision
    }

    /// The parts outside the decoder layers: embedding, final norm, head, MTP, vision.
    #[must_use]
    pub fn globals(&self) -> &[PartRequirement] {
        &self.globals
    }

    /// The decoder layers in forward order.
    #[must_use]
    pub fn layers(&self) -> &[LayerExecution] {
        &self.layers
    }

    /// Every requirement, globals first, then layers in forward order.
    pub fn requirements(&self) -> impl Iterator<Item = &PartRequirement> {
        self.globals.iter().chain(
            self.layers
                .iter()
                .flat_map(|layer| layer.requirements.iter()),
        )
    }

    /// A one-line boot summary: model, target, world, layers and parts.
    #[must_use]
    pub fn boot_summary(&self) -> String {
        let parts = self.requirements().count();
        format!(
            "{} on {} TP{}/{}: {} layers, {} part instances, dense {:?}, experts {:?}",
            self.architecture.family,
            self.target,
            self.tp.rank(),
            self.tp.world(),
            self.layers.len(),
            parts,
            self.precision.dense,
            self.precision.experts
        )
    }
}

/// A backend that can open from an [EnginePlan].
///
/// A model arm implements this instead of asking the CLI to construct it. The plan is the
/// complete description of what to open; the weights are the only remaining input.
pub trait EngineBuilder {
    /// The backend this builder produces.
    type Backend: ModelBackend;

    /// Open the backend for this plan and weight source over `context` tokens.
    ///
    /// `context` is the window the serve path admitted (`--context`, capped by the checkpoint's
    /// own `max_position_embeddings`). A builder must open its sequence caches and descriptor at
    /// this window, not at the plan's full architecture window: the plan describes the
    /// checkpoint, while the operator chose how much of it to allocate.
    ///
    /// # Errors
    /// A shape, weight or device refusal, or a zero context.
    fn open(
        plan: &EnginePlan,
        weights: &dyn WeightSource,
        context: usize,
    ) -> Result<Self::Backend, PlanError>;

    /// Open the backend from a converted rank file of the same plan, over `context` tokens.
    ///
    /// A converted file is one arm-level storage optimisation, not a second plan: the default
    /// opens from the weight source exactly as [Self::open] does, and an arm that can read a
    /// converted rank file overrides this to use it. The plan gate and the context are the same
    /// call.
    ///
    /// # Errors
    /// As [Self::open], and a rank file the arm refuses.
    fn open_converted(
        plan: &EnginePlan,
        weights: &dyn WeightSource,
        converted: &std::path::Path,
        context: usize,
    ) -> Result<Self::Backend, PlanError> {
        let _ = converted;
        Self::open(plan, weights, context)
    }
}

fn precision_for(part: Part, policy: PrecisionPolicy) -> Precision {
    match part {
        Part::Moe => policy.experts,
        Part::Mtp => policy.mtp,
        Part::MlaAttention | Part::GqaAttention | Part::GdnScan | Part::KdaScan => policy.attention,
        // The embedding, both norms and the head are stored and computed at bf16 in every
        // released arm, including the DeepSeek FP8 checkpoint (its head is not converted).
        Part::Embedding | Part::RmsNorm | Part::LayerNorm | Part::LmHead => Precision::Bf16,
        _ => policy.dense,
    }
}

fn global_requirements(
    architecture: &Architecture,
    tp: &TpPlan,
    precision: PrecisionPolicy,
) -> Result<Vec<PartRequirement>, PlanError> {
    let hidden = architecture.hidden_size;
    let vocab = architecture.vocab_size;
    let mut globals = vec![
        PartRequirement {
            part: Part::Embedding,
            geometry: PartGeometry {
                hidden,
                rows: 1,
                vocab,
                world: tp.world(),
                rank: tp.rank(),
                ..PartGeometry::new()
            },
            precision: Precision::Bf16,
        },
        PartRequirement {
            part: Part::RmsNorm,
            geometry: PartGeometry {
                hidden,
                rows: 1,
                world: tp.world(),
                rank: tp.rank(),
                ..PartGeometry::new()
            },
            precision: Precision::Bf16,
        },
        PartRequirement {
            part: Part::LmHead,
            geometry: PartGeometry {
                hidden,
                rows: 1,
                vocab,
                world: tp.world(),
                rank: tp.rank(),
                ..PartGeometry::new()
            },
            precision: precision_for(Part::LmHead, precision),
        },
    ];
    // A non-replicated stage is not loaded by a multi-rank serve: the shared tensor plan's
    // `served_and_mtp` omits it at TP>1, so the boot plan must not name a part the served set
    // does not carry. Qwen4-Exp is the release that clears `replicated`; GLM and DeepSeek keep
    // the stage on every rank.
    if let Some(mtp) = &architecture.mtp
        && (mtp.replicated || tp.world() == 1)
    {
        globals.push(PartRequirement {
            part: Part::Mtp,
            geometry: PartGeometry {
                hidden,
                rows: mtp.stages,
                block_size: mtp.block_size.unwrap_or(0),
                experts: mtp.routed_experts.unwrap_or(0),
                top_k: mtp.active_experts.unwrap_or(0),
                positions: architecture.max_position_embeddings,
                world: tp.world(),
                rank: tp.rank(),
                ..PartGeometry::new()
            },
            precision: precision.mtp,
        });
    }
    if let Some(vision) = &architecture.vision {
        globals.push(PartRequirement {
            part: Part::Vision,
            geometry: PartGeometry {
                hidden: vision.hidden_size,
                rows: vision.depth,
                intermediate: vision.intermediate,
                heads: vision.heads,
                head_dim: vision.hidden_size / vision.heads.max(1),
                patch_size: vision.patch_size,
                spatial_merge: vision.spatial_merge,
                out_hidden: vision.out_hidden_size,
                positions: architecture.max_position_embeddings,
                world: tp.world(),
                rank: tp.rank(),
                ..PartGeometry::new()
            },
            precision: precision.dense,
        });
    }
    Ok(globals)
}

fn geometry_for(
    part: Part,
    layer: &sparkflash_model::architecture::Layer,
    architecture: &Architecture,
    tp: &TpPlan,
) -> Result<PartGeometry, PlanError> {
    let hidden = architecture.hidden_size;
    let positions = architecture.max_position_embeddings;
    let mut geometry = PartGeometry {
        hidden,
        rows: 1,
        positions,
        world: tp.world(),
        rank: tp.rank(),
        ..PartGeometry::new()
    };
    let attention = &layer.attention;
    match part {
        Part::MlaAttention => match &attention.kind {
            AttentionKind::Mla {
                q_lora_rank,
                kv_lora_rank,
                v_head_dim,
                groups,
            } => {
                let heads = split(tp, attention.heads, TpAxis::Heads, 1)?;
                geometry.heads = heads.padded;
                geometry.head_offset = heads.offset;
                geometry.head_dim = attention.head_dim;
                geometry.q_lora = q_lora_rank.unwrap_or(0);
                geometry.kv_lora = *kv_lora_rank;
                geometry.v_head = *v_head_dim;
                geometry.groups = groups.map_or(0, |groups| groups.groups);
                geometry.window = attention.window.unwrap_or(0);
            }
            _ => {
                return Err(PlanError::Invalid(
                    "an MLA part on a non-MLA layer".to_owned(),
                ));
            }
        },
        Part::GqaAttention => match &attention.kind {
            AttentionKind::Gqa { kv_heads, .. } => {
                let heads = split(tp, attention.heads, TpAxis::Heads, 1)?;
                let kv = split(tp, *kv_heads, TpAxis::KvHeads, 1)?;
                geometry.heads = heads.padded;
                geometry.head_offset = heads.offset;
                geometry.kv_heads = kv.padded;
                geometry.kv_head_offset = kv.offset;
                geometry.head_dim = attention.head_dim;
                geometry.window = attention.window.unwrap_or(0);
            }
            _ => {
                return Err(PlanError::Invalid(
                    "a GQA part on a non-GQA layer".to_owned(),
                ));
            }
        },
        Part::GdnScan => match &attention.kind {
            AttentionKind::Gdn(gdn) => {
                let key = split(tp, gdn.key_heads, TpAxis::Heads, 1)?;
                let value = split(tp, gdn.value_heads, TpAxis::KvHeads, 1)?;
                geometry.heads = key.padded;
                geometry.head_offset = key.offset;
                geometry.kv_heads = value.padded;
                geometry.kv_head_offset = value.offset;
                geometry.head_dim = gdn.key_dim;
                geometry.v_head = gdn.value_dim;
                geometry.window = gdn.conv_kernel;
            }
            _ => {
                return Err(PlanError::Invalid(
                    "a GDN part on a non-GDN layer".to_owned(),
                ));
            }
        },
        Part::KdaScan => match &attention.kind {
            AttentionKind::Kda(kda) => {
                let heads = split(tp, kda.heads, TpAxis::Heads, 1)?;
                geometry.heads = heads.padded;
                geometry.head_offset = heads.offset;
                geometry.head_dim = kda.head_dim;
                // The shared KDA body is compiled with a 128-wide value head, the same template
                // as GDN; record it so the head-pair rule can prove the template.
                geometry.v_head = kda.head_dim;
                geometry.window = kda.conv_kernel;
            }
            _ => {
                return Err(PlanError::Invalid(
                    "a KDA part on a non-KDA layer".to_owned(),
                ));
            }
        },
        Part::Csa2Compression => {
            geometry.ratio = attention
                .compression
                .as_ref()
                .map_or(0, |compression| compression.ratio);
        }
        Part::SparseIndexer => {
            let indexer = attention.indexer.as_ref().ok_or_else(|| {
                PlanError::Invalid("an indexer part on a layer without one".to_owned())
            })?;
            geometry.heads = indexer.heads;
            geometry.head_dim = indexer.head_dim;
            geometry.top_k = indexer.topk;
            geometry.ratio = indexer.pool;
        }
        Part::DenseMlp => match &layer.ffn {
            Ffn::Dense(dense) => {
                geometry.intermediate =
                    split(tp, dense.intermediate, TpAxis::Intermediate, 16)?.padded;
            }
            _ => {
                return Err(PlanError::Invalid(
                    "a dense MLP on a non-dense layer".to_owned(),
                ));
            }
        },
        Part::Moe => match &layer.ffn {
            Ffn::Moe(moe) => {
                geometry.intermediate = split_expert(
                    tp,
                    moe.intermediate,
                    architecture.quantization.expert_format,
                )?;
                geometry.experts = moe.routed_experts;
                geometry.top_k = moe.active_experts;
                geometry.shared_experts = moe.shared_experts;
                geometry.shared = if moe.shared_experts == 0 {
                    0
                } else {
                    split(
                        tp,
                        moe.shared_intermediate * moe.shared_experts,
                        TpAxis::Intermediate,
                        16,
                    )?
                    .padded
                };
                let (min_tokens, body) = grouped_policy(&architecture.quantization);
                geometry.grouped_min_tokens = min_tokens;
                geometry.grouped_fp8_body = body;
            }
            _ => {
                return Err(PlanError::Invalid(
                    "a MoE part on a non-MoE layer".to_owned(),
                ));
            }
        },
        Part::HyperConnection | Part::ManifoldHyperConnection => match &layer.residual {
            Residual::Hyper(hyper) => {
                geometry.streams = hyper.streams;
                geometry.intermediate = hyper.lowrank.unwrap_or(0);
            }
            Residual::Plain => {
                return Err(PlanError::Invalid(
                    "a hyper-connection part on a plain residual".to_owned(),
                ));
            }
        },
        Part::Engram => {
            let side = layer
                .sides
                .iter()
                .find_map(|side| match side {
                    Side::Engram(engram) => Some(engram),
                    Side::Ple(_) => None,
                })
                .ok_or_else(|| PlanError::Invalid("an Engram part without a side".to_owned()))?;
            let ordinal = architecture
                .layers
                .iter()
                .filter(|candidate| {
                    candidate
                        .sides
                        .iter()
                        .any(|side| matches!(side, Side::Engram(_)))
                })
                .position(|candidate| candidate.index == layer.index)
                .unwrap_or(0);
            geometry.table_index = ordinal;
            geometry.rows = side
                .tables
                .get(ordinal)
                .copied()
                .or_else(|| side.tables.first().copied())
                .unwrap_or(0);
            geometry.vocab = side.vocab_size;
            geometry.compressed_vocab = side.compressed_vocab;
            geometry.pad_id = usize::try_from(side.pad_id).unwrap_or(0);
            geometry.heads = side.heads;
            geometry.head_dim = side.head_dim;
            geometry.ratio = side.max_ngram;
        }
        Part::Ple => {
            let side = layer
                .sides
                .iter()
                .find_map(|side| match side {
                    Side::Ple(ple) => Some(ple),
                    Side::Engram(_) => None,
                })
                .ok_or_else(|| PlanError::Invalid("a PLE part without a side".to_owned()))?;
            geometry.rows = side.split_parts;
            geometry.heads = side.heads_per_ngram;
            geometry.head_dim = side.embed_dim;
            geometry.ratio = side.ngram_size;
            geometry.window = side.conv_kernel;
            geometry.vocab = usize::try_from(side.vocab_base).unwrap_or(usize::MAX);
        }
        Part::Mtp
        | Part::Vision
        | Part::Embedding
        | Part::RmsNorm
        | Part::LayerNorm
        | Part::LmHead
        | Part::DenseGemm
        | Part::Activation
        | Part::Rope
        | Part::Quantize
        | Part::MoeRouter
        | Part::MoeCombine
        | Part::GroupedMoeGemm
        | Part::MoePlan
        | Part::Collective => {
            return Err(PlanError::Invalid(format!(
                "{part} is a global part, not a layer part"
            )));
        }
    }
    Ok(geometry)
}

/// The grouped FP8 policy the released formats imply.
///
/// The FP8 packs differ in a typed fact, not a family name: whether the dense class quantizes
/// its activations. GLM's pack stores the dense and routed weights alike as E4M3, and its
/// dynamic activation scheme drives the q8 MMA; its host counts token rows, with `MAX_ROWS =
/// GEMV_MAX_BATCH = 8` and the bulk pass starting at 9 tokens, so the plan sets the grouped
/// threshold to `GEMV_MAX_BATCH + 1` (9 tokens) and names the q8 body. Every other FP8 expert
/// release is a Qwen arm's, whose chunk takes the BF16-activation grouped tile from 64 tokens.
/// Qwen3.8's pack stores only the routed experts as E4M3, so the arm's
/// [Architecture::experts_only_fp8] correction leaves the dense class BF16. Qwen3.5/3.6 stores
/// the dense projections as GLM's tiles, but the arm computes them weight-only, so its
/// [Architecture::weight_only_fp8] correction clears the dynamic scheme. [GroupedFp8Body::Bf16Activation] and
/// [GroupedFp8Body::SplitActivation] are the other activation forms a serving profile may
/// select, but [EnginePlan] names the released dynamic default; a caller constructing a
/// grouped requirement for an overriding profile must set
/// [PartGeometry::grouped_fp8_body] to the matching variant itself.
///
/// An NVFP4 expert names no FP8 body. A Qwen NVFP4 release still takes its grouped tile from
/// 64 tokens, while a DeepSeek MXFP4 release keeps the row-count pass.
fn grouped_policy(quantization: &Quantization) -> (usize, Option<GroupedFp8Body>) {
    match quantization.expert_format {
        WeightFormat::Fp8Tile { .. } => {
            if matches!(
                quantization.dense_format,
                WeightFormat::Fp8Tile {
                    dynamic_activations: true,
                    ..
                }
            ) {
                (
                    GEMV_MAX_BATCH + 1,
                    Some(GroupedFp8Body::QuantizedActivation),
                )
            } else {
                (64, Some(GroupedFp8Body::Tiled))
            }
        }
        WeightFormat::NvFp4 { .. } if matches!(quantization.dense_format, WeightFormat::Bf16) => {
            (64, None)
        }
        _ => (0, None),
    }
}

/// The rank-local routed-expert intermediate width.
///
/// An FP8 expert's inverse scale covers a whole `block_n`-column tile (128 in the released
/// arms), so the plan splits the expert into whole tiles: five 128-wide tiles over four ranks
/// are 2/1/1/1 columns, the widths the Qwen host's `rank_config` uploads. Splitting an FP8
/// width to 16 columns and rounding up would claim 160 columns for every rank, which is not a
/// run of whole tiles and does not match the weights the rank holds. Every other format keeps
/// the even split rounded to 16 columns.
fn split_expert(tp: &TpPlan, total: usize, format: WeightFormat) -> Result<usize, PlanError> {
    match format {
        WeightFormat::Fp8Tile { block_n, .. } if block_n > 1 => {
            Ok(tp.split_tiles(total, block_n, TpAxis::Intermediate)?.padded)
        }
        _ => Ok(tp.split(total, TpAxis::Intermediate, 16)?.padded),
    }
}

fn split(
    tp: &TpPlan,
    total: usize,
    axis: TpAxis,
    alignment: usize,
) -> Result<sparkflash_model::Split, PlanError> {
    Ok(tp.split(total, axis, alignment)?)
}

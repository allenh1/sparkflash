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

use sparkflash_model::architecture::{AttentionKind, Ffn, Residual, Side};
use sparkflash_model::{
    Architecture, ConfigError, ModelBackend, Part, StepRole, Target, TpAxis, TpPlan, TpPlanError,
};
use sparkflash_weights::weights::WeightSource;
use thiserror::Error;

use crate::plan::{PartGeometry, Precision};

/// The operand precision this run uses for each class of weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionPolicy {
    /// Dense projections, attention weights and the embedding/head.
    pub dense: Precision,
    /// Routed and shared experts.
    pub experts: Precision,
    /// Attention operands, when they differ from the dense class.
    pub attention: Precision,
}

impl PrecisionPolicy {
    /// Read the policy the released quantization block describes.
    ///
    /// The mapping is deliberately conservative: an unknown method falls back to BF16 rather
    /// than guessing a narrower format, and an expert format is read independently of the dense
    /// format because the two releases already differ.
    #[must_use]
    pub fn from_architecture(architecture: &Architecture) -> Self {
        let method = architecture
            .quantization
            .method
            .as_deref()
            .unwrap_or("")
            .to_ascii_lowercase();
        let scale = architecture
            .quantization
            .scale
            .as_deref()
            .unwrap_or("")
            .to_ascii_lowercase();
        let expert = architecture
            .quantization
            .expert
            .as_deref()
            .unwrap_or("")
            .to_ascii_lowercase();
        let dense = if method.contains("nvfp4") {
            Precision::NvFp4
        } else if method.contains("fp8") || method.contains("mxfp8") {
            if scale.contains("ue8m0") {
                Precision::MxFp8
            } else {
                Precision::Fp8E4M3
            }
        } else {
            Precision::Bf16
        };
        let experts = if expert.contains("fp4") {
            Precision::NvFp4
        } else {
            dense
        };
        Self {
            dense,
            experts,
            attention: dense,
        }
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

    /// The tensor-parallel plan.
    #[must_use]
    pub const fn tp(&self) -> TpPlan {
        self.tp
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

    /// Open the backend for this plan and weight source.
    ///
    /// # Errors
    /// A shape, weight or device refusal.
    fn open(plan: &EnginePlan, weights: &dyn WeightSource) -> Result<Self::Backend, PlanError>;
}

fn precision_for(part: Part, policy: PrecisionPolicy) -> Precision {
    match part {
        Part::Moe | Part::Mtp => policy.experts,
        Part::MlaAttention | Part::GqaAttention | Part::GdnScan | Part::KdaScan => policy.attention,
        Part::Embedding | Part::RmsNorm => Precision::Bf16,
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
            precision: precision.dense,
        },
    ];
    if let Some(mtp) = &architecture.mtp {
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
            precision: precision.experts,
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
                geometry.intermediate =
                    split(tp, moe.intermediate, TpAxis::Intermediate, 16)?.padded;
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
        Part::Mtp | Part::Vision | Part::Embedding | Part::RmsNorm | Part::LmHead => {
            return Err(PlanError::Invalid(format!(
                "{part} is a global part, not a layer part"
            )));
        }
    }
    Ok(geometry)
}

fn split(
    tp: &TpPlan,
    total: usize,
    axis: TpAxis,
    alignment: usize,
) -> Result<sparkflash_model::Split, PlanError> {
    Ok(tp.split(total, axis, alignment)?)
}

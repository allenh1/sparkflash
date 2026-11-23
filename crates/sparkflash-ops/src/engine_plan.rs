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

use std::collections::BTreeMap;

use sparkflash_model::architecture::{
    AttentionKind, Ffn, Quantization, Residual, ScaleDType, Side, WeightFormat,
};
use sparkflash_model::format::StoredFormat;
use sparkflash_model::layout::LayoutDtype;
use sparkflash_model::stack::StackKind;
use sparkflash_model::{
    Architecture, ConfigError, FactValue, ModelBackend, Part, Stack, StepRole, Target, TpAxis,
    TpPlan, TpPlanError,
};
use sparkflash_weights::weights::WeightSource;
use thiserror::Error;

use crate::gemv::GEMV_MAX_BATCH;
use crate::plan::{GroupedFp8Body, ModelFamily, PartGeometry, Precision};
use crate::resolver::PlanContext;

/// One part instance's position: the part kind and its layout block, or `None` for a global.
type PartKey = (Part, Option<usize>);
/// The formats and claimed element counts of one part's layout tensors.
type FormatTotals = BTreeMap<PartKey, Vec<(WeightFormat, usize)>>;
/// The stored format of each part instance.
type PartFormats = BTreeMap<PartKey, StoredFormat>;

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
        // Attention defaults to the dense class. A part whose layout claims a tensor in another
        // stored format takes that format's precision at [EnginePlan::from_architecture].
        let attention = dense;
        Self {
            dense,
            experts,
            mtp,
            attention,
        }
    }
}

/// The operand precision one stored weight format uses.
///
/// Every format is its own arithmetic. MXFP4 and NVFP4 are not one class: MXFP4 applies one
/// power-of-two code a 32-value block exactly, NVFP4 a per-16 E4M3 group and a tensor-wide
/// factor, so folding them let an MXFP4 release resolve an NVFP4 body.
#[must_use]
pub const fn precision_of(format: WeightFormat) -> Precision {
    precision_of_stored(format.stored_format())
}

/// The operand precision one format carries, scale kinds and block included.
#[must_use]
pub const fn precision_of_stored(format: StoredFormat) -> Precision {
    match format.values() {
        sparkflash_model::format::FormatValues::Bf16 => Precision::Bf16,
        sparkflash_model::format::FormatValues::F32 => Precision::F32,
        sparkflash_model::format::FormatValues::Fp8E4M3 => {
            // A tile-scaled E4M3 is FP8; one power-of-two code a block (or a row) is MXFP8.
            if format.has_scale(sparkflash_model::format::ScaleKind::Tile) {
                Precision::Fp8E4M3
            } else {
                Precision::MxFp8
            }
        }
        sparkflash_model::format::FormatValues::Fp4E2M1 => {
            if format.has_scale(sparkflash_model::format::ScaleKind::TensorWide) {
                Precision::NvFp4
            } else {
                Precision::MxFp4
            }
        }
    }
}

/// The canonical stored format for an arithmetic class.
///
/// A part whose layout claims no tensor (an elementwise body, a synthetic test) has no stored
/// dtype to read. Its arithmetic still names a format, and the resolver needs one to key the
/// catalogue; this is that format. It is never substituted for a format the layout claimed.
#[must_use]
pub const fn format_of_precision(precision: Precision) -> StoredFormat {
    match precision {
        Precision::F32 => StoredFormat::F32,
        Precision::Bf16 => StoredFormat::BF16,
        Precision::Fp8E4M3 => StoredFormat::FP8_E4M3_TILE_128,
        Precision::MxFp8 => StoredFormat::MXFP8_32,
        Precision::NvFp4 => StoredFormat::NVFP4_16,
        Precision::MxFp4 => StoredFormat::MXFP4_32,
    }
}

/// One part, at this rank's geometry, with the stored format the run will read.
///
/// The format is the checkpoint fact the plan resolved, not a second summary of it: its values,
/// block and scale kinds travel to the catalogue key and to the bind, and the arithmetic the
/// resolver matches is derived from it ([PartRequirement::precision]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartRequirement {
    /// The model-side part.
    pub part: Part,
    /// The rank-local shape.
    pub geometry: PartGeometry,
    /// The stored format this part reads.
    pub format: StoredFormat,
}

impl PartRequirement {
    /// The arithmetic class the format runs at.
    #[must_use]
    pub const fn precision(&self) -> Precision {
        precision_of_stored(self.format)
    }
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

/// One tensor a part instance claims at this rank, with the split applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanTensor {
    /// The exact checkpoint tensor name.
    pub name: String,
    /// The stored logical shape.
    pub shape: Vec<usize>,
    /// The stored format, as the checkpoint's loader names it (`BF16`, `F8_E4M3`, ...).
    pub format: String,
    /// The rank-local shape.
    pub rank_shape: Vec<usize>,
    /// The bytes this rank holds.
    pub shard_bytes: usize,
    /// The part that claims it.
    pub part: Part,
    /// The decoder block that claims it, when it belongs to one.
    pub block: Option<usize>,
    /// The layout's rank split.
    pub split: String,
    /// Whether this rank binds it on the device (`resident`) or a host seam reads it.
    pub placement: String,
}

/// The catalogue's DeepSeek symbols, grouped by the part that launches them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartSymbols {
    /// The part kind.
    pub part: Part,
    /// The decoder block, when the part is one layer's.
    pub block: Option<usize>,
    /// The image the symbols belong to.
    pub image: &'static str,
    /// The symbol names, in catalogue order.
    pub symbols: Vec<&'static str>,
}

/// The model-independent execution plan of one model at one target and world size.
#[derive(Debug, Clone, PartialEq)]
pub struct EnginePlan {
    architecture: Architecture,
    target: Target,
    tp: TpPlan,
    precision: PrecisionPolicy,
    formats: PartFormats,
    globals: Vec<PartRequirement>,
    layers: Vec<LayerExecution>,
    model_facts: BTreeMap<String, FactValue>,
    stacks: Vec<Stack>,
    tensors: Vec<PlanTensor>,
    symbols: Vec<PartSymbols>,
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
        // The layout states each part's stored dtype. That dtype, not the part's kind, decides
        // the operand precision: a block mixer stored in f32 and an attention stored in bf16 are
        // both exact even beside NVFP4 experts.
        let formats = stored_formats(&architecture);
        let model = ModelFamily::of(&architecture.family, &architecture.architectures);
        let mut globals = global_requirements(&architecture, &tp, precision, &formats)?;
        for requirement in &mut globals {
            requirement.geometry.model = Some(model);
        }
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
            let mut roles = step.roles();
            let mut requirements = Vec::new();
            for (_, part) in &roles {
                let format = format_for(&architecture, &formats, *part, Some(layer.index));
                requirements.push(PartRequirement {
                    part: *part,
                    geometry: geometry_for(*part, layer, &architecture, &tp)?,
                    format: format.unwrap_or_else(|| format_of_precision(precision.dense)),
                });
            }
            // A layout may claim layer tensors for a part the forward walk folds into another
            // step (Inkling's short convolution is launched with the attention). The part still
            // needs a row and a body, so it is planned beside the step it belongs to.
            for (part, block) in formats.keys() {
                if *block != Some(layer.index) || roles.iter().any(|(_, held)| held == part) {
                    continue;
                }
                // A layout also claims the layer's norms, which are shared global parts rather
                // than steps of the block walk. Only a part the walk can place gets a row here.
                let Ok(geometry) = geometry_for(*part, layer, &architecture, &tp) else {
                    continue;
                };
                let role = if *part == Part::ShortConv {
                    StepRole::Attention
                } else {
                    StepRole::Ffn
                };
                let format = format_for(&architecture, &formats, *part, Some(layer.index));
                roles.push((role, *part));
                requirements.push(PartRequirement {
                    part: *part,
                    geometry,
                    format: format.unwrap_or_else(|| format_of_precision(precision.dense)),
                });
            }
            for requirement in &mut requirements {
                requirement.geometry.model = Some(model);
            }
            layers.push(LayerExecution {
                index: step.index,
                steps: roles,
                requirements,
            });
        }
        let model_facts = model_facts(&architecture);
        let stacks = plan_stacks(&architecture);
        let tensors = plan_tensors(&architecture, &tp, &stacks);
        let symbols = part_symbols(&architecture);
        Ok(Self {
            architecture,
            target,
            tp,
            precision,
            formats,
            globals,
            layers,
            model_facts,
            stacks,
            tensors,
            symbols,
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

    /// The stored format the layout claims for one part instance.
    ///
    /// `block` is the layout block the part belongs to, or `None` for a global part (including
    /// the draft stage and the vision tower, whose tensors are not decoder blocks). The value is
    /// the format the layout's stored dtypes agree on; it is never inferred from the part kind.
    #[must_use]
    pub fn stored_format(&self, part: Part, block: Option<usize>) -> Option<StoredFormat> {
        self.formats.get(&(part, block)).copied()
    }

    /// The model-wide facts the plan carries, named as the open manifest names them.
    #[must_use]
    pub fn model_facts(&self) -> &BTreeMap<String, FactValue> {
        &self.model_facts
    }

    /// The description's stacks, with each part instance's facts and tensor claims.
    #[must_use]
    pub fn stacks(&self) -> &[Stack] {
        &self.stacks
    }

    /// Every tensor a part instance claims at this rank.
    #[must_use]
    pub fn tensors(&self) -> &[PlanTensor] {
        &self.tensors
    }

    /// The catalogue's DeepSeek symbols, grouped by part.
    #[must_use]
    pub fn symbols(&self) -> &[PartSymbols] {
        &self.symbols
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

/// The stored format of each part's claimed tensors, by part and block.
///
/// The layout's dtype is the checkpoint's stored dtype; `dense` and `expert` name the release's
/// quantisation classes. A part with several formats takes the one its claimed bytes mostly use,
/// so a router's bf16 row does not describe the experts beside it. A scale or index tensor that
/// names no arithmetic is skipped; an f32 part keeps f32 only when no sibling tensor states a
/// narrower format.
fn stored_formats(architecture: &Architecture) -> PartFormats {
    let mut totals = FormatTotals::new();
    let Ok(Some(layout)) = sparkflash_model::layout::matching(architecture) else {
        return BTreeMap::new();
    };
    let Ok(tensors) = layout.tensors(architecture) else {
        return BTreeMap::new();
    };
    for tensor in tensors {
        let block = match tensor.stack {
            StackKind::Draft | StackKind::Vision | StackKind::Audio => None,
            StackKind::Language => tensor.block,
        };
        // A routed expert's packed dtype is a container, not its arithmetic. The checkpoint
        // stores NVFP4 as U8 bytes with E4M3 block scales, and MXFP8 as E4M3 bytes with U8
        // `ue8m0` scales; the wider weight tensor names which one the layer uses, while the
        // smaller scale stream is the other dtype and loses the aggregate below.
        let format = if tensor.part == Part::Moe {
            match tensor.dtype {
                LayoutDtype::Expert | LayoutDtype::U8 | LayoutDtype::F4E2M1 => {
                    Some(architecture.quantization.expert_format)
                }
                LayoutDtype::F8E4M3 => Some(match architecture.quantization.expert_format {
                    format @ WeightFormat::Fp8Tile { .. } => format,
                    _ => WeightFormat::MxFp8 { block: 32 },
                }),
                _ => layout_format(tensor.dtype, architecture),
            }
        } else {
            layout_format(tensor.dtype, architecture)
        };
        let Some(format) = format else {
            continue;
        };
        let elements = tensor
            .shape
            .iter()
            .fold(1usize, |product, dim| product.saturating_mul(*dim));
        let total = totals.entry((tensor.part, block)).or_default();
        if let Some((_, weight)) = total.iter_mut().find(|(held, _)| *held == format) {
            *weight = weight.saturating_add(elements);
        } else {
            total.push((format, elements));
        }
    }
    // A part's format is the one its claimed bytes mostly use: a router's bf16 row beside 256
    // NVFP4 experts does not make the block bf16. An exact tie keeps the later declaration, so a
    // part that states two same-sized classes is reported by the last one the layout names.
    totals
        .into_iter()
        .filter_map(|(key, formats)| {
            formats
                .into_iter()
                .max_by_key(|(_, weight)| *weight)
                .map(|(format, _)| (key, format.stored_format()))
        })
        .collect()
}

/// The model-wide facts the open manifest names, read from the typed architecture and the
/// layout's config vars.
///
/// The names are the baseline manifest's names, not a second vocabulary: a value the forward
/// reads at open is carried here once, on the plan, so a backend never re-parses the checkpoint.
fn model_facts(architecture: &Architecture) -> BTreeMap<String, FactValue> {
    let mut facts = BTreeMap::new();
    let first = architecture.layers.first();
    let first_moe = architecture
        .layers
        .iter()
        .find_map(|layer| match &layer.ffn {
            sparkflash_model::architecture::Ffn::Moe(moe) => Some(moe),
            sparkflash_model::architecture::Ffn::Dense(_) => None,
        });
    let first_hyper = architecture
        .layers
        .iter()
        .find_map(|layer| match &layer.residual {
            sparkflash_model::architecture::Residual::Hyper(hyper) => Some(hyper),
            _ => None,
        });
    let first_mla = architecture
        .layers
        .iter()
        .find_map(|layer| match &layer.attention.kind {
            sparkflash_model::architecture::AttentionKind::Mla {
                q_lora_rank,
                groups,
                ..
            } => Some((*q_lora_rank, *groups)),
            _ => None,
        });
    let mut put = |name: &str, value: FactValue| {
        facts.insert(name.to_owned(), value);
    };
    put(
        "vocab_size",
        FactValue::Integer(architecture.vocab_size as i64),
    );
    put("dim", FactValue::Integer(architecture.hidden_size as i64));
    put(
        "n_layers",
        FactValue::Integer(architecture.layers.len() as i64),
    );
    put("norm_eps", FactValue::Float(architecture.norm_epsilon));
    put(
        "image_token_id",
        architecture
            .tokens
            .image_token_id
            .map_or(FactValue::Null, |id| FactValue::Integer(i64::from(id))),
    );
    put(
        "dtype",
        FactValue::Text(
            architecture
                .quantization
                .method
                .clone()
                .unwrap_or_else(|| "bf16".to_owned()),
        ),
    );
    put(
        "expert_dtype",
        FactValue::Text(
            architecture
                .quantization
                .expert
                .clone()
                .unwrap_or_else(|| "bf16".to_owned()),
        ),
    );
    if let Some(attention) = first.map(|layer| &layer.attention) {
        put("n_heads", FactValue::Integer(attention.heads as i64));
        put("head_dim", FactValue::Integer(attention.head_dim as i64));
        put(
            "rope_head_dim",
            FactValue::Integer(attention.rope.rotated_dims as i64),
        );
        put("rope_theta", FactValue::Float(attention.rope.theta));
        put(
            "window_size",
            attention
                .window
                .map_or(FactValue::Null, |window| FactValue::Integer(window as i64)),
        );
    }
    if let Some(scaling) = architecture
        .layers
        .iter()
        .find_map(|layer| layer.attention.rope.scaling.as_ref())
    {
        put(
            "original_seq_len",
            FactValue::Integer(scaling.original_seq_len as i64),
        );
        put("rope_factor", FactValue::Float(scaling.factor));
        put("beta_fast", FactValue::Float(scaling.beta_fast));
        put("beta_slow", FactValue::Float(scaling.beta_slow));
    }
    if let Some((q_lora, groups)) = first_mla {
        put(
            "q_lora_rank",
            q_lora.map_or(FactValue::Null, |rank| FactValue::Integer(rank as i64)),
        );
        if let Some(groups) = groups {
            put("o_groups", FactValue::Integer(groups.groups as i64));
            put("o_lora_rank", FactValue::Integer(groups.rank as i64));
        }
    }
    if let Some(moe) = first_moe {
        put("moe_inter_dim", FactValue::Integer(moe.intermediate as i64));
        put(
            "n_routed_experts",
            FactValue::Integer(moe.routed_experts as i64),
        );
        put(
            "n_shared_experts",
            FactValue::Integer(moe.shared_experts as i64),
        );
        put(
            "n_activated_experts",
            FactValue::Integer(moe.active_experts as i64),
        );
        put(
            "score_func",
            FactValue::Text(
                match moe.score {
                    sparkflash_model::architecture::Score::Softmax => "softmax",
                    sparkflash_model::architecture::Score::Sigmoid => "sigmoid",
                    sparkflash_model::architecture::Score::SqrtSoftPlus => "sqrtsoftplus",
                }
                .to_owned(),
            ),
        );
        put("route_scale", FactValue::Float(moe.routed_scaling));
        put("norm_topk_prob", FactValue::Bool(moe.norm_topk));
        put(
            "swiglu_limit",
            moe.clamp.map_or(FactValue::Null, FactValue::Float),
        );
    }
    if let Some(hyper) = first_hyper {
        put("hc_mult", FactValue::Integer(hyper.streams as i64));
        put(
            "hc_sinkhorn_iters",
            FactValue::Integer(hyper.sinkhorn_iters as i64),
        );
        put("hc_eps", FactValue::Float(hyper.epsilon));
    }
    let kv_sources: Vec<i64> = architecture
        .layers
        .iter()
        .filter(|layer| {
            layer
                .attention
                .compression
                .as_ref()
                .is_some_and(|compression| compression.owns_keys)
        })
        .map(|layer| layer.index as i64)
        .collect();
    let index_sources: Vec<i64> = architecture
        .layers
        .iter()
        .filter(|layer| layer.attention.indexer.is_some())
        .map(|layer| layer.index as i64)
        .collect();
    put("kv_source_layers", FactValue::Integers(kv_sources));
    put("index_source_layers", FactValue::Integers(index_sources));
    let pool = architecture
        .layers
        .iter()
        .find_map(|layer| layer.attention.indexer.as_ref()?.candidate_pool.as_ref());
    put(
        "candidate_source_layer",
        pool.map_or(FactValue::Integer(0), |pool| {
            FactValue::Integer(pool.source_layer as i64)
        }),
    );
    put(
        "candidate_topk_blocks",
        pool.map_or(FactValue::Integer(0), |pool| {
            FactValue::Integer(pool.blocks as i64)
        }),
    );
    put(
        "candidate_block_size",
        pool.map_or(FactValue::Integer(0), |pool| {
            FactValue::Integer(pool.block_size as i64)
        }),
    );
    let indexer = architecture
        .layers
        .iter()
        .find_map(|layer| layer.attention.indexer.as_ref());
    if let Some(indexer) = indexer {
        put("index_n_heads", FactValue::Integer(indexer.heads as i64));
        put(
            "index_head_dim",
            FactValue::Integer(indexer.head_dim as i64),
        );
        put("index_topk", FactValue::Integer(indexer.topk as i64));
    }
    let compressed = architecture.layers.iter().find_map(|layer| {
        layer
            .attention
            .compression
            .as_ref()
            .filter(|compression| compression.ratio > 0)
            .map(|_| layer.attention.rope.theta)
    });
    put(
        "compress_rope_theta",
        compressed.map_or(FactValue::Null, FactValue::Float),
    );
    let mut ratios = Vec::with_capacity(architecture.layers.len());
    for layer in &architecture.layers {
        ratios.push(
            layer
                .attention
                .compression
                .as_ref()
                .map_or(0, |compression| compression.ratio) as i64,
        );
    }
    if let Some(mtp) = &architecture.mtp {
        ratios.extend(std::iter::repeat_n(0, mtp.stages));
        put("n_mtp_layers", FactValue::Integer(mtp.stages as i64));
        put(
            "dspark_block_size",
            mtp.block_size
                .map_or(FactValue::Null, |size| FactValue::Integer(size as i64)),
        );
        put(
            "dspark_target_layer_ids",
            FactValue::Integers(
                mtp.target_layers
                    .iter()
                    .map(|layer| *layer as i64)
                    .collect(),
            ),
        );
        put(
            "dspark_n_routed_experts",
            mtp.routed_experts
                .map_or(FactValue::Null, |count| FactValue::Integer(count as i64)),
        );
        put(
            "dspark_n_activated_experts",
            mtp.active_experts
                .map_or(FactValue::Null, |count| FactValue::Integer(count as i64)),
        );
    }
    put("compress_ratios", FactValue::Integers(ratios));
    let engram_layers: Vec<i64> = architecture
        .layers
        .iter()
        .filter(|layer| {
            layer
                .sides
                .iter()
                .any(|side| matches!(side, sparkflash_model::architecture::Side::Engram(_)))
        })
        .map(|layer| layer.index as i64)
        .collect();
    put("engram_layer_ids", FactValue::Integers(engram_layers));
    if let Some(engram) = architecture.layers.iter().find_map(|layer| {
        layer.sides.iter().find_map(|side| match side {
            sparkflash_model::architecture::Side::Engram(engram) => Some(engram),
            sparkflash_model::architecture::Side::Ple(_) => None,
        })
    }) {
        put(
            "engram_num_embeddings",
            FactValue::Integers(engram.tables.iter().map(|rows| *rows as i64).collect()),
        );
        put(
            "engram_max_ngram_size",
            FactValue::Integer(engram.max_ngram as i64),
        );
        put(
            "engram_pad_id",
            FactValue::Integer(i64::from(engram.pad_id)),
        );
        put(
            "engram_compressed_vocab_size",
            FactValue::Integer(engram.compressed_vocab as i64),
        );
        put("engram_n_heads", FactValue::Integer(engram.heads as i64));
        put(
            "engram_head_dim",
            FactValue::Integer(engram.head_dim as i64),
        );
        put(
            "engram_vocab_size",
            FactValue::Integer(engram.vocab_size as i64),
        );
    }
    put(
        "dspark_markov_rank",
        FactValue::Integer(
            architecture
                .layout_vars
                .get("markov_rank")
                .copied()
                .unwrap_or(0),
        ),
    );
    put(
        "dspark_noise_token_id",
        FactValue::Integer(
            architecture
                .layout_vars
                .get("noise_token")
                .copied()
                .unwrap_or(0),
        ),
    );
    // The layout states the producer's ModelArgs default (model.py:67-68); the release stores
    // no key for it, so the fallback is the one the reference constructor uses.
    put(
        "gate_temp",
        FactValue::Float(
            architecture
                .layout_vars
                .get("gate_temp")
                .copied()
                .unwrap_or(1) as f64,
        ),
    );
    if let Some(vision) = &architecture.vision {
        put("vision_n_layers", FactValue::Integer(vision.depth as i64));
        put("vision_dim", FactValue::Integer(vision.hidden_size as i64));
        put("vision_n_heads", FactValue::Integer(vision.heads as i64));
        put(
            "vision_inter_dim",
            FactValue::Integer(vision.intermediate as i64),
        );
        put(
            "vision_patch_size",
            FactValue::Integer(vision.patch_size as i64),
        );
        put(
            "vision_downsample_ratio",
            FactValue::Integer(vision.spatial_merge as i64),
        );
        put(
            "vision_max_n_token",
            FactValue::Integer(
                architecture
                    .layout_vars
                    .get("vision_max_tokens")
                    .copied()
                    .unwrap_or(0),
            ),
        );
        put(
            "vision_min_pixels",
            FactValue::Integer(vision.min_pixels as i64),
        );
        // The released document stores `null`; the plan says so rather than inventing a cap.
        put("vision_max_wh_ratio", FactValue::Null);
        put("vision_rope_theta", FactValue::Float(vision.rope_theta));
    }
    facts
}

/// The description's stacks, preferring the layout's expanded claims when a layout matches.
fn plan_stacks(architecture: &Architecture) -> Vec<Stack> {
    if let Ok(Some(layout)) = sparkflash_model::layout::matching(architecture)
        && let Ok(stacks) = layout.stacks(architecture)
    {
        return stacks;
    }
    architecture.stacks()
}

/// Whether the four-rank device bind loads this tensor (the baseline manifest's `resident`
/// placement). The predicate mirrors the arm's bind: attention projections, expert and shared
/// matrices with their scale grids, the router, the tail, the norms and the hyper-connections.
fn device_bound(name: &str, part: Part) -> bool {
    if name.starts_with("mtp.") {
        return false;
    }
    match part {
        Part::Csa2Compression | Part::Engram | Part::Ple => false,
        Part::SparseIndexer => name.ends_with(".wq_b.weight"),
        Part::MlaAttention => {
            name.ends_with(".wq_a.weight")
                || name.ends_with(".wq_b.weight")
                || name.ends_with(".wkv.weight")
                || name.ends_with(".wo_a.weight")
                || name.ends_with(".wo_b.weight")
                || name.ends_with(".q_norm.weight")
                || name.ends_with(".kv_norm.weight")
        }
        Part::Moe => true,
        Part::RmsNorm | Part::LayerNorm | Part::HyperConnection | Part::ManifoldHyperConnection => {
            true
        }
        Part::Embedding | Part::LmHead => true,
        _ => false,
    }
}

/// Every tensor the description claims at this rank, with the layout's split and the device
/// placement the baseline names.
fn plan_tensors(architecture: &Architecture, tp: &TpPlan, stacks: &[Stack]) -> Vec<PlanTensor> {
    use crate::tensor_plan::ArchitectureTensorPlan;
    let Ok(plan) = ArchitectureTensorPlan::served_and_mtp(architecture, *tp) else {
        return Vec::new();
    };
    let mut claims = BTreeMap::new();
    for stack in stacks {
        for block in &stack.blocks {
            for instance in &block.parts {
                for claim in &instance.claims {
                    claims.insert(
                        claim.name.clone(),
                        (instance.part, Some(block.index), claim.split.clone()),
                    );
                }
            }
        }
    }
    let mut out = Vec::with_capacity(plan.len());
    for index in 0..plan.len() {
        let spec = plan.spec_at(index);
        let rank = plan.rank_spec_at(index);
        let global = match spec.name.as_str() {
            "embed.weight" => Some(Part::Embedding),
            "head.weight" => Some(Part::LmHead),
            "norm.weight" => Some(Part::RmsNorm),
            _ => None,
        };
        let (part, block, split) = claims
            .get(&spec.name)
            .cloned()
            .unwrap_or_else(|| (global.unwrap_or(Part::Collective), None, "whole".to_owned()));
        let elements = rank
            .shape
            .iter()
            .fold(1usize, |product, dim| product.saturating_mul(*dim));
        // The checkpoint stores a 4-bit expert packed two values per byte: the manifest and
        // the loader see I8 with the last axis halved, while the plan's arithmetic sees the
        // logical E2M1 shape. Carry the stored form and keep the byte count the same.
        let (format, shape, rank_shape) = if spec.dtype == sparkflash_weights::DType::F4E2M1 {
            let mut shape = spec.shape.clone();
            let mut rank_shape = rank.shape.clone();
            if let Some(last) = shape.last_mut() {
                *last /= 2;
            }
            if let Some(last) = rank_shape.last_mut() {
                *last /= 2;
            }
            ("I8".to_owned(), shape, rank_shape)
        } else {
            (
                spec.dtype.name().to_owned(),
                spec.shape.clone(),
                rank.shape.clone(),
            )
        };
        out.push(PlanTensor {
            name: spec.name.clone(),
            shape,
            format,
            rank_shape,
            shard_bytes: spec.dtype.storage_bytes(elements),
            part,
            block,
            split,
            placement: if device_bound(&spec.name, part) {
                "resident".to_owned()
            } else {
                "host".to_owned()
            },
        });
    }
    out
}

/// The DeepSeek image's symbols for each part instance the plan executes, grouped by part.
fn part_symbols(architecture: &Architecture) -> Vec<PartSymbols> {
    if !architecture.family.starts_with("deepseek") {
        return Vec::new();
    }
    crate::parts::CATALOGUE
        .iter()
        .filter(|entry| {
            entry.image == sparkflash_kernels::images::Image::Deepseek
                && !entry.symbols.is_empty()
                && !matches!(
                    entry.part,
                    Part::Vision
                        | Part::VisionPatchEmbed
                        | Part::VisionPosition
                        | Part::VisionNorm
                        | Part::VisionAttention
                        | Part::VisionMlp
                        | Part::VisionDownsample
                )
        })
        .map(|entry| PartSymbols {
            part: entry.part,
            block: None,
            image: "deepseek",
            symbols: entry.symbols.iter().map(|symbol| symbol.name).collect(),
        })
        .collect()
}
fn layout_format(dtype: LayoutDtype, architecture: &Architecture) -> Option<WeightFormat> {
    let quantization = &architecture.quantization;
    match dtype {
        LayoutDtype::Bf16 => Some(WeightFormat::Bf16),
        LayoutDtype::F32 => Some(WeightFormat::F32),
        LayoutDtype::Dense => Some(quantization.dense_format),
        LayoutDtype::Expert => Some(quantization.expert_format),
        LayoutDtype::F8E4M3 => {
            // An explicit E4M3 tensor keeps the tile or MX block the checkpoint's own formats
            // name. A release that states no FP8 class still stores 128x128 bf16 scales here.
            for format in [
                quantization.dense_format,
                quantization.expert_format,
                quantization.mtp_expert_format.unwrap_or(WeightFormat::Bf16),
            ] {
                if matches!(
                    format,
                    WeightFormat::Fp8Tile { .. } | WeightFormat::MxFp8 { .. }
                ) {
                    return Some(format);
                }
            }
            Some(WeightFormat::Fp8Tile {
                block_m: 128,
                block_n: 128,
                scale_dtype: ScaleDType::Bf16,
                dynamic_activations: false,
            })
        }
        LayoutDtype::F4E2M1
            if matches!(
                quantization.expert_format,
                WeightFormat::NvFp4 { .. } | WeightFormat::MxFp4 { .. }
            ) =>
        {
            Some(quantization.expert_format)
        }
        LayoutDtype::F8E8M0 | LayoutDtype::F4E2M1 | LayoutDtype::U8 | LayoutDtype::I64 => None,
    }
}

fn format_for(
    architecture: &Architecture,
    formats: &PartFormats,
    part: Part,
    block: Option<usize>,
) -> Option<StoredFormat> {
    formats.get(&(part, block)).copied().or_else(|| {
        architecture
            .stored_format(part)
            .map(WeightFormat::stored_format)
    })
}

fn global_requirements(
    architecture: &Architecture,
    tp: &TpPlan,
    precision: PrecisionPolicy,
    formats: &PartFormats,
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
            format: format_for(architecture, formats, Part::Embedding, None)
                .unwrap_or_else(|| format_of_precision(precision.dense)),
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
            format: format_for(architecture, formats, Part::RmsNorm, None)
                .unwrap_or_else(|| format_of_precision(precision.dense)),
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
            format: format_for(architecture, formats, Part::LmHead, None)
                .unwrap_or_else(|| format_of_precision(precision.dense)),
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
            format: format_for(architecture, formats, Part::Mtp, None)
                .unwrap_or_else(|| format_of_precision(precision.mtp)),
        });
    }
    if let Some(vision) = architecture.vision_spec() {
        let geometry = &vision.geometry;
        let head_dim = geometry
            .hidden_size
            .checked_div(geometry.num_heads)
            .unwrap_or(0);
        let base = PartGeometry {
            hidden: geometry.hidden_size,
            rows: geometry.depth,
            intermediate: geometry.intermediate_size,
            heads: geometry.num_heads,
            head_dim,
            patch_size: geometry.patch_size,
            spatial_merge: vision.downsample.factor(),
            out_hidden: geometry.out_hidden_size,
            in_channels: geometry.in_channels,
            temporal_patch: vision.patch_temporal_size(),
            norm_epsilon_bits: vision.norm_epsilon.to_bits(),
            vision_norm: vision.norm.code(),
            vision_activation: vision.activation.code(),
            attention_bias: vision.attention_bias,
            qk_norm: Some(vision.qk_norm),
            vision_position: vision.position.code(),
            vision_downsample: vision.downsample.code(),
            merger_intermediate: vision.merger_intermediate,
            positions: geometry.num_position_embeddings,
            world: tp.world(),
            rank: tp.rank(),
            ..PartGeometry::new()
        };
        let mut push = |part: Part, geometry: PartGeometry| {
            globals.push(PartRequirement {
                part,
                geometry,
                // A vision part with no tensor of its own (a rotary-only position scheme) has
                // no stored dtype to read; the tower's arithmetic is bf16.
                format: format_for(architecture, formats, part, None).unwrap_or(StoredFormat::BF16),
            });
        };
        push(Part::VisionPatchEmbed, base);
        push(Part::VisionPosition, base);
        push(Part::VisionNorm, base);
        push(Part::VisionAttention, base);
        push(Part::VisionMlp, base);
        let merger = PartGeometry {
            vision_activation: vision.merger_activation.code(),
            ..base
        };
        if !matches!(
            vision.downsample,
            sparkflash_model::VisionDownsample::Merge(_) | sparkflash_model::VisionDownsample::None
        ) {
            push(Part::VisionDownsample, base);
        }
        push(Part::VisionMerger, merger);
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
                // The gate, the q/k norms and the sinks change the part's arithmetic, not its
                // geometry; an arm that does not run them refuses them by name
                // ([Architecture::refuse_switches]).
                output_gate: _,
                qk_norm: _,
                sinks: _,
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
            AttentionKind::Gqa { kv_heads, .. } | AttentionKind::Inkling { kv_heads } => {
                let heads = split(tp, attention.heads, TpAxis::Heads, 1)?;
                let kv = split(tp, *kv_heads, TpAxis::KvHeads, 1)?;
                geometry.heads = heads.padded;
                geometry.head_offset = heads.offset;
                geometry.kv_heads = kv.padded;
                geometry.kv_head_offset = kv.offset;
                geometry.head_dim = attention.head_dim;
                geometry.window = attention.window.unwrap_or(0);
                // A layer whose indexer selects its keys attends at most that many a query, which
                // decides the grouped body ([crate::attention::GqaBody::choose]).
                geometry.top_k = attention
                    .indexer
                    .as_ref()
                    .map_or(0, |indexer| indexer.max_selected());
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
        Part::RelativeAttention => {
            let inkling = architecture.inkling.as_ref().ok_or_else(|| {
                PlanError::Invalid("a relative-attention part on a non-Inkling layer".to_owned())
            })?;
            geometry.heads = attention.heads;
            geometry.head_dim = attention.head_dim;
            geometry.window = inkling.rel_extent;
        }
        Part::ShortConv => {
            let inkling = architecture.inkling.as_ref().ok_or_else(|| {
                PlanError::Invalid("a short-convolution part on a non-Inkling layer".to_owned())
            })?;
            geometry.window = inkling.short_conv;
        }
        Part::BlockAttentionResidual => {
            let block = architecture
                .inkling
                .as_ref()
                .and_then(|inkling| inkling.block_residual)
                .ok_or_else(|| {
                    PlanError::Invalid(
                        "a block-attention residual part on a plain residual".to_owned(),
                    )
                })?;
            geometry.streams = block.layers_per_block;
        }
        Part::SoftcappedActivation => {}
        Part::Csa2Compression => {
            // The compressor pools and appends the MLA head's latent rows, so its compiled body
            // carries the attention head width beside the compression ratio. The generated
            // catalogue rule names both; leaving the width at zero refused the released config.
            geometry.head_dim = attention.head_dim;
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
                geometry.output = geometry.intermediate;
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
            Residual::Plain | Residual::BlockAttention => {
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
        | Part::VisionPatchEmbed
        | Part::VisionPosition
        | Part::VisionNorm
        | Part::VisionAttention
        | Part::VisionMlp
        | Part::VisionDownsample
        | Part::VisionMerger
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

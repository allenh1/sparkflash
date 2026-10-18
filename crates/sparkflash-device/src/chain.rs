//! The M3 hybrid 40-layer chain skeleton: the CED schedule, the per-layer stage order, the planned
//! host-seam set and the ledger self-check, with no device execution and no token.
//!
//! **Scope.** This module is M3-a plumbing (design docs/build/device-forward-chain.md section 7 M3
//! and docs/build/serving-readiness.md 3.3 M3-a). It constructs the 130 + 8 schedule, names the 40
//! layer executions and every stage in the per-layer graph, and fixes the host-seam plan before a
//! run: the plan constructs the expected seam set, the run records seams, and the self-check fails
//! on a planned seam with no record **and** on a record with no plan entry. It does not launch a
//! kernel, stage weights, allocate device memory or compare a token: chain_execution stays false
//! and the report says so. The device stages are named here so the later wiring cannot invent a
//! different stage map.
//!
//! **Labels.** The first runnable hybrid pass must carry the design 4.3 labels verbatim:
//! "hybrid: token produced by host math over device logits", "first chain execution, unverified"
//! and the certified-head label. [ChainLabels] is the one place they live.
//!
//! **Seam counts are plan accounting, not measurements.** They are derived from the CED schedule
//! (extent per layer) and the layer predicates, never copied from a report. The count totals are
//! 5,482 token-layer executions for 130 + 8 (see [ChainSchedule::token_layer_executions]).

use std::collections::BTreeMap;

use sparkflash_kernels::sequence_chain::{SeamKind, SeamLedger, SeamRecord, SequencePlan};
use sparkflash_kv::config::{
    CANDIDATE_SOURCE_LAYER, Csa2Mode, ENCODER_LAST_LAYER, NUM_HIDDEN_LAYERS, SWA_WINDOW_SIZE,
    compress_ratio,
};
use sparkflash_kv::schedule::ced_schedule;

/// The prompt length the Phase-2 milestone runs (design 2.1).
pub const MILESTONE_PROMPT_TOKENS: usize = 130;

/// The decode steps the Phase-2 milestone runs.
pub const MILESTONE_DECODE_STEPS: usize = 8;

/// The first design 4.3 label, verbatim.
pub const HYBRID_TOKEN_LABEL: &str = "hybrid: token produced by host math over device logits";

/// The design 4.3 status label for the first chain run, verbatim.
pub const FIRST_CHAIN_LABEL: &str = "first chain execution, unverified";

/// The design 4.2/4.3 certified-head label, verbatim.
pub const HEAD_LABEL: &str = "NOT YET DEVICE-EXECUTED AS A PRUNING PATH: one stage-5 launch per rank, 0 certified, never compared against the chain reference";

/// The M3-a plan-only label: the exact words that keep a plan from reading as an execution.
pub const PLAN_ONLY_LABEL: &str =
    "plan only: no chain execution, no device execution, no token and no logits produced";

/// Everything a chain run can reject before it reaches a comparison.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ChainError {
    /// The CED schedule could not be built.
    #[error("CED schedule: {0}")]
    Schedule(String),
    /// The CED schedule and the sequence plan disagree about a layer extent.
    #[error(
        "CED schedule and sequence plan disagree at layer {layer}: schedule {schedule_first}+{schedule_tokens}, plan {plan_first}+{plan_tokens}"
    )]
    PlanDisagreement {
        /// The layer.
        layer: usize,
        /// First position in the CED schedule.
        schedule_first: usize,
        /// Tokens in the CED schedule.
        schedule_tokens: usize,
        /// First position in the sequence plan.
        plan_first: usize,
        /// Tokens in the sequence plan.
        plan_tokens: usize,
    },
    /// A planned seam has no record (or fewer than planned).
    #[error("planned seam {name} ({kind:?}): expected {expected} records, found {found}")]
    SeamCount {
        /// The seam name.
        name: &'static str,
        /// The planned kind.
        kind: SeamKind,
        /// Expected records.
        expected: usize,
        /// Found records.
        found: usize,
    },
    /// A recorded seam has no plan entry.
    #[error("unplanned seam {name}: {found} record(s) with no plan entry")]
    UnplannedSeam {
        /// The seam name.
        name: &'static str,
        /// Records found.
        found: usize,
    },
    /// A record's kind does not match the plan.
    #[error("seam {name}: plan kind {expected:?}, record kind {found:?}")]
    SeamKind {
        /// The seam name.
        name: &'static str,
        /// Planned kind.
        expected: SeamKind,
        /// Recorded kind.
        found: SeamKind,
    },
    /// A stage returned a seam record under a different seam name.
    #[error("stage {stage:?} (layer {layer}) returned seam {found}, expected {expected}")]
    SeamName {
        /// The stage that ran.
        stage: ChainStage,
        /// The layer.
        layer: usize,
        /// The seam the stage must record.
        expected: &'static str,
        /// The seam the backend returned.
        found: &'static str,
    },
    /// A record does not name a phase-3 removal plan.
    #[error("seam {name}: phase3_followup is false; every seam must name its removal")]
    SeamWithoutRemoval {
        /// The seam name.
        name: &'static str,
    },
    /// The backend returned a seam record for a stage the plan does not mark as a host seam.
    #[error("stage {stage:?} (layer {layer}) returned a seam record but is not a host seam")]
    NonSeamRecord {
        /// The stage.
        stage: ChainStage,
        /// The layer.
        layer: usize,
    },
    /// The backend omitted a seam record for a host-seam stage.
    #[error("stage {stage:?} (layer {layer}) is a host seam but the backend recorded none")]
    MissingStageSeam {
        /// The stage.
        stage: ChainStage,
        /// The layer.
        layer: usize,
    },
    /// The schedule has no layers.
    #[error("the chain schedule needs at least one layer")]
    EmptyLayers,
    /// The decode step count is zero.
    #[error("the chain schedule needs at least one decode step")]
    NoDecodeSteps,
}

/// One layer's CED extent and static mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainLayer {
    /// Backbone layer id, 0..40.
    pub layer: usize,
    /// First absolute position this layer runs.
    pub first_position: usize,
    /// Tokens this layer runs in the prefill.
    pub tokens: usize,
    /// The layer's static CSA2 mode.
    pub mode: Csa2Mode,
    /// The layer's compress ratio.
    pub ratio: usize,
}

impl ChainLayer {
    /// The positions this layer runs in the prefill, ascending.
    #[must_use]
    pub fn positions(&self) -> std::ops::Range<usize> {
        self.first_position..self.first_position + self.tokens
    }

    /// Whether this layer has an Engram path (layers 1 and 14, design 2.2).
    #[must_use]
    pub fn has_engram(&self) -> bool {
        matches!(self.layer, 1 | 14)
    }

    /// Whether this layer owns an indexer (Full and Reindex layers).
    #[must_use]
    pub fn owns_index(&self) -> bool {
        self.mode.scores_indices()
    }

    /// Whether this layer builds or masks the HSI candidate pool.
    #[must_use]
    pub fn has_hsi(&self) -> bool {
        CANDIDATE_SOURCE_LAYER == self.layer || self.mode == Csa2Mode::Reindex
    }

    /// Whether this layer writes compressed main KV (ratio > 0).
    #[must_use]
    pub fn compresses(&self) -> bool {
        self.ratio > 0
    }
}

/// The 130 + 8 CED schedule the chain runs (one prefill, then greedy decode steps).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainSchedule {
    /// Prompt length.
    pub prompt_tokens: usize,
    /// Greedy decode steps after the prefill.
    pub decode_steps: usize,
    /// The sliding window the CED decoder extent uses.
    pub window: usize,
    /// One row per backbone layer, in order.
    pub layers: Vec<ChainLayer>,
}

impl ChainSchedule {
    /// The Phase-2 milestone schedule: 130 prompt tokens and 8 decode steps.
    ///
    /// # Errors
    /// [ChainError] when the schedule or the cross-checked sequence plan cannot be built.
    pub fn milestone() -> Result<Self, ChainError> {
        Self::new(MILESTONE_PROMPT_TOKENS, MILESTONE_DECODE_STEPS)
    }

    /// Builds a schedule from the CED plan and cross-checks it against
    /// [SequencePlan::ced] (the chain's other statement of the same extents).
    ///
    /// # Errors
    /// [ChainError] for an empty prompt (via the CED schedule), zero decode steps, or a disagreement
    /// between the CED schedule and the sequence plan.
    pub fn new(prompt_tokens: usize, decode_steps: usize) -> Result<Self, ChainError> {
        if decode_steps == 0 {
            return Err(ChainError::NoDecodeSteps);
        }
        let schedule =
            ced_schedule(prompt_tokens).map_err(|error| ChainError::Schedule(error.to_string()))?;
        let plan = SequencePlan::ced(
            prompt_tokens,
            NUM_HIDDEN_LAYERS,
            ENCODER_LAST_LAYER + 1,
            SWA_WINDOW_SIZE.min(prompt_tokens),
        )
        .map_err(ChainError::Schedule)?;
        if plan.extents.len() != NUM_HIDDEN_LAYERS {
            return Err(ChainError::EmptyLayers);
        }
        let decoder_first = prompt_tokens.saturating_sub(schedule.decoder_tokens);
        let mut layers = Vec::with_capacity(NUM_HIDDEN_LAYERS);
        for layer in 0..NUM_HIDDEN_LAYERS {
            let tokens = if layer <= ENCODER_LAST_LAYER {
                schedule.encoder_tokens
            } else {
                schedule.decoder_tokens
            };
            let first_position = if layer <= ENCODER_LAST_LAYER {
                0
            } else {
                decoder_first
            };
            let extent = plan.extents[layer];
            if extent.first_position != first_position || extent.tokens != tokens {
                return Err(ChainError::PlanDisagreement {
                    layer,
                    schedule_first: first_position,
                    schedule_tokens: tokens,
                    plan_first: extent.first_position,
                    plan_tokens: extent.tokens,
                });
            }
            let row = schedule.layer(layer).ok_or(ChainError::EmptyLayers)?;
            layers.push(ChainLayer {
                layer,
                first_position,
                tokens,
                mode: row.mode,
                ratio: compress_ratio(layer),
            });
        }
        Ok(Self {
            prompt_tokens,
            decode_steps,
            window: schedule.window,
            layers,
        })
    }

    /// Prefill token-layer executions: 21 x 130 + 19 x 128 = 5,162 for the milestone prompt.
    #[must_use]
    pub fn prefill_token_layer_executions(&self) -> usize {
        self.layers.iter().map(|layer| layer.tokens).sum()
    }

    /// Decode token-layer executions: one token per step per layer (8 x 40 = 320).
    #[must_use]
    pub fn decode_token_layer_executions(&self) -> usize {
        self.decode_steps.saturating_mul(self.layers.len())
    }

    /// Every token-layer execution in the run (prefill + decode; 5,482 for 130 + 8).
    #[must_use]
    pub fn token_layer_executions(&self) -> usize {
        self.prefill_token_layer_executions() + self.decode_token_layer_executions()
    }

    /// The one-line plan description.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "chain plan: {} prompt tokens + {} decode steps, {} layers, {} prefill token-layer executions, {} decode token-layer executions, {} total (CED; plan accounting, not measured)",
            self.prompt_tokens,
            self.decode_steps,
            self.layers.len(),
            self.prefill_token_layer_executions(),
            self.decode_token_layer_executions(),
            self.token_layer_executions()
        )
    }
}

/// What a stage is in the chain today: a device kernel (named, not launched in M3-a), a kernel that
/// has been measured in isolation but is not wired into the chain, or a labelled host seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageStatus {
    /// A device kernel the chain will launch once the first pass is wired.
    DeviceKernel(&'static str),
    /// A merged and measured device kernel that is not wired into the chain yet.
    DeviceMeasuredNotWired {
        /// Kernel name.
        kernel: &'static str,
        /// Why it is not a chain stage yet.
        note: &'static str,
    },
    /// Host work standing in for a device stage, with its seam name.
    HostSeam(&'static str),
}

/// One stage in the per-layer graph, in the design 2.2 order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainStage {
    /// Prompt/decode token embedding gather (host seam).
    Embedding,
    /// Engram read/write at layers 1 and 14 (host seam).
    Engram,
    /// Dispatch/launch scheduling (host seam; no arithmetic).
    Dispatch,
    /// Q/KV RMSNorm before attention (host seam).
    AttnNorm,
    /// RoPE on q and the output side (host seam).
    AttnRope,
    /// mHC entry mix (device kernel).
    MhcEntry,
    /// wq/wkv input projections (device kernel).
    AttentionProjections,
    /// Window-KV encode and ring rotation (host seam).
    WindowKv,
    /// Compressor output and compressed-cache round-trip (host seam).
    CompressorCache,
    /// Index-key wk/knorm/RoPE/store (host seam).
    IndexKeyPath,
    /// Indexer weights projection (host seam).
    IndexerWeights,
    /// HSI pool population and block scoring (host seam).
    HsiScoring,
    /// Top-512 merge over the pool (host seam).
    Top512Merge,
    /// Expanded/compressed store view (host seam).
    ExpandedView,
    /// CSA-union or window attention (device kernel).
    Attention,
    /// wo_a output projection (device kernel), after attention and before the post-mix.
    AttentionOutputProjection,
    /// mHC post-mix (device kernel).
    MhcPost,
    /// MoE router (measured one launch, not wired).
    MoeRouter,
    /// Routing weight application (host seam).
    MoeRoutingWeight,
    /// Routed expert gate/up (measured in isolation, not wired).
    MoeGateUp,
    /// SwiGLU and the activation quantiser (host seam).
    MoeSwiGLUQuant,
    /// Routed down/combine (device kernel).
    MoeDownCombine,
    /// Shared expert dense gate/up (device kernel).
    SharedExpertGate,
    /// Shared expert SwiGLU/quantiser (host seam).
    SharedExpertSwiGLUQuant,
    /// FFN epilogue: routed-weighted sum and shared add (host seam; design 2.11).
    FfnEpilogue,
    /// Final RMS norm (host seam; blocks a device token path).
    FinalNorm,
    /// Certified LM-head screening (device kernel).
    LmHeadScreen,
    /// Block reduction over per-block maxima and greedy argmax (host seam).
    LmHeadArgmax,
}

impl ChainStage {
    /// The stable stage name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Engram => "engram",
            Self::Dispatch => "dispatch",
            Self::AttnNorm => "attn_norm",
            Self::AttnRope => "attn_rope",
            Self::MhcEntry => "mhc_entry",
            Self::AttentionProjections => "attention_projections",
            Self::WindowKv => "window_kv",
            Self::CompressorCache => "compressor_cache",
            Self::IndexKeyPath => "index_key_path",
            Self::IndexerWeights => "indexer_weights",
            Self::HsiScoring => "hsi_scoring",
            Self::Top512Merge => "top512_merge",
            Self::ExpandedView => "expanded_view",
            Self::Attention => "attention",
            Self::AttentionOutputProjection => "attention_output_projection",
            Self::MhcPost => "mhc_post",
            Self::MoeRouter => "moe_router",
            Self::MoeRoutingWeight => "moe_routing_weight",
            Self::MoeGateUp => "moe_gate_up",
            Self::MoeSwiGLUQuant => "moe_swiglu_quant",
            Self::MoeDownCombine => "moe_down_combine",
            Self::SharedExpertGate => "shared_expert_gate",
            Self::SharedExpertSwiGLUQuant => "shared_expert_swiglu_quant",
            Self::FfnEpilogue => "ffn_epilogue",
            Self::FinalNorm => "final_norm",
            Self::LmHeadScreen => "lm_head_screen",
            Self::LmHeadArgmax => "lm_head_argmax",
        }
    }

    /// The design section/item this stage comes from. The ordering test and the plan's stage map
    /// both cite these, so a reordering has to say which design item it is changing.
    #[must_use]
    pub fn design_item(self) -> &'static str {
        match self {
            Self::Embedding => "2.1 input",
            Self::Engram | Self::Dispatch => "2.2 item 1",
            Self::MhcEntry => "2.2 item 2",
            Self::AttnNorm => "2.2 items 2-3",
            Self::AttentionProjections | Self::AttentionOutputProjection | Self::AttnRope => {
                "2.2 item 3"
            }
            Self::WindowKv | Self::CompressorCache | Self::ExpandedView => "2.2 item 4",
            Self::IndexKeyPath
            | Self::IndexerWeights
            | Self::HsiScoring
            | Self::Top512Merge
            | Self::Attention => "2.2 item 5",
            Self::MhcPost => "2.2 item 6",
            Self::MoeRouter
            | Self::MoeRoutingWeight
            | Self::MoeGateUp
            | Self::MoeSwiGLUQuant
            | Self::MoeDownCombine
            | Self::SharedExpertGate
            | Self::SharedExpertSwiGLUQuant
            | Self::FfnEpilogue => "2.2 item 7",
            Self::FinalNorm | Self::LmHeadScreen | Self::LmHeadArgmax => "2.2 item 8",
        }
    }

    /// This stage's status on main 5ee8599.
    #[must_use]
    pub fn status(self) -> StageStatus {
        match self {
            Self::Embedding => StageStatus::HostSeam("embedding_gather"),
            Self::Engram => StageStatus::HostSeam("engram_rw"),
            Self::Dispatch => StageStatus::HostSeam("dispatch"),
            Self::AttnNorm => StageStatus::HostSeam("attn_norm"),
            Self::AttnRope => StageStatus::HostSeam("attn_rope"),
            Self::MhcEntry | Self::MhcPost => StageStatus::DeviceKernel("mega_mhc_sm121"),
            Self::AttentionProjections => StageStatus::DeviceKernel("b12x_mxfp8_dense_sm121"),
            Self::WindowKv => StageStatus::HostSeam("window_kv_roundtrip"),
            Self::CompressorCache => StageStatus::HostSeam("compressor_cache"),
            Self::IndexKeyPath => StageStatus::HostSeam("index_key_path"),
            Self::IndexerWeights => StageStatus::HostSeam("indexer_weights"),
            Self::HsiScoring => StageStatus::HostSeam("hsi_scoring"),
            Self::Top512Merge => StageStatus::HostSeam("top512_merge"),
            Self::ExpandedView => StageStatus::HostSeam("expanded_view"),
            Self::Attention => StageStatus::DeviceKernel("flash_mla_csa2_sm121"),
            Self::AttentionOutputProjection => StageStatus::DeviceKernel("b12x_mxfp8_wo_a_sm121"),
            Self::MoeRouter => StageStatus::DeviceMeasuredNotWired {
                kernel: "moe_router_sm121",
                note: "device-proven one launch (M2 window b5), not wired into the chain yet",
            },
            Self::MoeRoutingWeight => StageStatus::HostSeam("moe_routing_weight"),
            Self::MoeGateUp => StageStatus::DeviceMeasuredNotWired {
                kernel: "routed_gate_up_sm121",
                note: "device kernel merged and measured in isolation (m10), not wired into the chain yet",
            },
            Self::MoeSwiGLUQuant => StageStatus::HostSeam("moe_swiglu_quant"),
            Self::MoeDownCombine => StageStatus::DeviceKernel("b12x_mxfp4_moe_sm121"),
            Self::SharedExpertGate => StageStatus::DeviceKernel("b12x_mxfp8_dense_sm121"),
            Self::SharedExpertSwiGLUQuant => StageStatus::HostSeam("shared_expert"),
            Self::FfnEpilogue => StageStatus::HostSeam("ffn_epilogue"),
            Self::FinalNorm => StageStatus::HostSeam("final_norm"),
            Self::LmHeadScreen => StageStatus::DeviceKernel("cert_lm_head_sm121"),
            Self::LmHeadArgmax => StageStatus::HostSeam("lm_head_argmax"),
        }
    }

    /// The host seam this stage records, if it is host-side.
    #[must_use]
    pub fn seam_name(self) -> Option<&'static str> {
        match self.status() {
            StageStatus::HostSeam(name) => Some(name),
            StageStatus::DeviceKernel(_) | StageStatus::DeviceMeasuredNotWired { .. } => None,
        }
    }

    /// The device kernel this stage names, if it has one.
    #[must_use]
    pub fn kernel(self) -> Option<&'static str> {
        match self.status() {
            StageStatus::DeviceKernel(kernel)
            | StageStatus::DeviceMeasuredNotWired { kernel, .. } => Some(kernel),
            StageStatus::HostSeam(_) => None,
        }
    }

    /// Whether this stage runs at all for a given layer: Engram only at layers 1 and 14, the
    /// compressor only where the ratio is non-zero, the indexer only on layers that own an indexer,
    /// HSI only at its candidate layer and the Reindex layers, and the expanded compressed-store
    /// view only where the layer reads compressed KV.
    #[must_use]
    pub fn runs_at(self, layer: &ChainLayer) -> bool {
        match self {
            Self::Engram => layer.has_engram(),
            Self::CompressorCache => layer.compresses(),
            Self::IndexKeyPath | Self::IndexerWeights | Self::Top512Merge => layer.owns_index(),
            Self::HsiScoring => layer.has_hsi(),
            Self::ExpandedView => layer.compresses(),
            _ => true,
        }
    }

    /// This stage's one-line map entry, for the plan report.
    #[must_use]
    pub fn map_line(self) -> String {
        match self.status() {
            StageStatus::HostSeam(seam) => {
                format!(
                    "{}: host seam {} (phase-3 removal planned) [design {}]",
                    self.name(),
                    seam,
                    self.design_item()
                )
            }
            StageStatus::DeviceKernel(kernel) => {
                format!(
                    "{}: device kernel {} (named; not launched in M3-a) [design {}]",
                    self.name(),
                    kernel,
                    self.design_item()
                )
            }
            StageStatus::DeviceMeasuredNotWired { kernel, note } => {
                format!(
                    "{}: device kernel {} ({note}) [design {}]",
                    self.name(),
                    kernel,
                    self.design_item()
                )
            }
        }
    }
}

/// The per-token-layer stage order of design 2.2 (the layer body, before the tail).
///
/// The order is the engine's: engram (2.2 item 1), dispatch (item 1), mHC entry (item 2),
/// rms_norm (items 2-3), q/kv projections (item 3), RoPE (item 3), window KV (item 4), the
/// compressed/indexer views (items 4-5), attention (item 5), the wo_a output projection (item 3),
/// post-mix mHC (item 6), MoE and the shared expert (item 7), then the tail (item 8).
pub const LAYER_STAGE_ORDER: &[ChainStage] = &[
    ChainStage::Engram,
    ChainStage::Dispatch,
    ChainStage::MhcEntry,
    ChainStage::AttnNorm,
    ChainStage::AttentionProjections,
    ChainStage::AttnRope,
    ChainStage::WindowKv,
    ChainStage::CompressorCache,
    ChainStage::IndexKeyPath,
    ChainStage::IndexerWeights,
    ChainStage::HsiScoring,
    ChainStage::Top512Merge,
    ChainStage::ExpandedView,
    ChainStage::Attention,
    ChainStage::AttentionOutputProjection,
    ChainStage::MhcPost,
    ChainStage::MoeRouter,
    ChainStage::MoeRoutingWeight,
    ChainStage::MoeGateUp,
    ChainStage::MoeSwiGLUQuant,
    ChainStage::MoeDownCombine,
    ChainStage::SharedExpertGate,
    ChainStage::SharedExpertSwiGLUQuant,
    ChainStage::FfnEpilogue,
];

/// The per-token-step prefix stages, in order.
pub const STEP_STAGE_ORDER: &[ChainStage] = &[ChainStage::Embedding];

/// The per-token-step tail after the 40 layer bodies.
pub const TAIL_STAGE_ORDER: &[ChainStage] = &[
    ChainStage::FinalNorm,
    ChainStage::LmHeadScreen,
    ChainStage::LmHeadArgmax,
];

/// One planned host seam: the name, kind, expected records and its removal plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeamPlanEntry {
    /// The seam name the ledger records.
    pub name: &'static str,
    /// The seam kind.
    pub kind: SeamKind,
    /// The number of records the run must produce.
    pub expected: usize,
    /// The phase-3 device work that removes the seam.
    pub removal: &'static str,
}

/// The planned seam set for one schedule, built before the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainSeamPlan {
    /// One entry per seam name, in stage order.
    pub entries: Vec<SeamPlanEntry>,
}

impl ChainSeamPlan {
    /// Builds the plan for a schedule: every host seam's expected record count is derived from the
    /// CED extents and the layer predicates, never copied from a report.
    #[must_use]
    pub fn for_schedule(schedule: &ChainSchedule) -> Self {
        let all = schedule.token_layer_executions();
        let predicate_tokens = |predicate: fn(&ChainLayer) -> bool| {
            let selected: Vec<&ChainLayer> = schedule
                .layers
                .iter()
                .filter(|layer| predicate(layer))
                .collect();
            let prefill: usize = selected.iter().map(|layer| layer.tokens).sum();
            prefill + schedule.decode_steps * selected.len()
        };
        let engram = predicate_tokens(ChainLayer::has_engram);
        let compressing = predicate_tokens(ChainLayer::compresses);
        let index = predicate_tokens(ChainLayer::owns_index);
        let hsi = predicate_tokens(ChainLayer::has_hsi);
        let step_tokens = 1 + schedule.decode_steps;
        let entries = vec![
            SeamPlanEntry {
                name: "embedding_gather",
                kind: SeamKind::Other,
                expected: schedule.prompt_tokens + schedule.decode_steps,
                removal: "embedding gather kernel (Phase 3)",
            },
            SeamPlanEntry {
                name: "engram_rw",
                kind: SeamKind::Engram,
                expected: engram,
                removal: "device Engram read/write (Phase 3)",
            },
            SeamPlanEntry {
                name: "dispatch",
                kind: SeamKind::Other,
                expected: all,
                removal: "Phase 3 CUDA-graph runner",
            },
            SeamPlanEntry {
                name: "attn_norm",
                kind: SeamKind::Other,
                expected: all,
                removal: "fused norm+RoPE kernel",
            },
            SeamPlanEntry {
                name: "attn_rope",
                kind: SeamKind::Other,
                expected: all,
                removal: "fused norm+RoPE kernel",
            },
            SeamPlanEntry {
                name: "window_kv_roundtrip",
                kind: SeamKind::Other,
                expected: all,
                removal: "device KV store S1",
            },
            SeamPlanEntry {
                name: "compressor_cache",
                kind: SeamKind::Compressor,
                expected: compressing,
                removal: "compressor kernel",
            },
            SeamPlanEntry {
                name: "index_key_path",
                kind: SeamKind::Other,
                expected: index,
                removal: "index-key kernel",
            },
            SeamPlanEntry {
                name: "indexer_weights",
                kind: SeamKind::Other,
                expected: index,
                removal: "indexer kernel",
            },
            SeamPlanEntry {
                name: "hsi_scoring",
                kind: SeamKind::Other,
                expected: hsi,
                removal: "HSI/indexer kernel",
            },
            SeamPlanEntry {
                name: "top512_merge",
                kind: SeamKind::IndexerSelection,
                expected: index,
                removal: "indexer/top-k kernel",
            },
            SeamPlanEntry {
                name: "expanded_view",
                kind: SeamKind::Other,
                expected: compressing,
                removal: "expanded-view kernel (Phase 3)",
            },
            SeamPlanEntry {
                name: "moe_routing_weight",
                kind: SeamKind::Other,
                expected: all,
                removal: "routed weighted-sum kernel (Phase 3)",
            },
            SeamPlanEntry {
                name: "moe_swiglu_quant",
                kind: SeamKind::Other,
                expected: all,
                removal: "fused SwiGLU + activation-quantiser kernel (Phase 3)",
            },
            SeamPlanEntry {
                name: "shared_expert",
                kind: SeamKind::Other,
                expected: all,
                removal: "fused shared-expert kernel (Phase 3)",
            },
            SeamPlanEntry {
                name: "ffn_epilogue",
                kind: SeamKind::Other,
                expected: all,
                removal: "FFN epilogue kernel (Phase 3)",
            },
            SeamPlanEntry {
                name: "final_norm",
                kind: SeamKind::Other,
                expected: step_tokens,
                removal: "final-norm kernel",
            },
            SeamPlanEntry {
                name: "lm_head_argmax",
                kind: SeamKind::Other,
                expected: step_tokens,
                removal: "head/sampling kernel",
            },
        ];
        Self { entries }
    }

    /// The entry for one seam name.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<&SeamPlanEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// The plan lines the report prints before any comparison.
    #[must_use]
    pub fn report_lines(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.entries.len() + 1);
        for entry in &self.entries {
            lines.push(format!(
                "planned seam {} ({:?}): {} records, removal: {}",
                entry.name, entry.kind, entry.expected, entry.removal
            ));
        }
        let total: usize = self.entries.iter().map(|entry| entry.expected).sum();
        lines.push(format!(
            "planned seam total: {total} records across {} seams (host work, NOT device work; plan accounting, not measured)",
            self.entries.len()
        ));
        lines
    }

    /// Fails on a planned seam with no record, a planned seam whose count differs, a record with no
    /// plan entry, a kind mismatch or a seam with no phase-3 removal.
    ///
    /// This is the structural guard the design requires: the ledger alone cannot see a host buffer
    /// that never got a record, so the plan and the ledger are checked against each other.
    ///
    /// # Errors
    /// [ChainError] naming the first missing, unplanned or mismatched seam.
    pub fn validate(&self, ledger: &SeamLedger) -> Result<(), ChainError> {
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for record in ledger.records() {
            if !record.phase3_followup {
                return Err(ChainError::SeamWithoutRemoval { name: record.name });
            }
            let entry = self.entry(record.name).ok_or(ChainError::UnplannedSeam {
                name: record.name,
                found: counts.get(record.name).copied().unwrap_or(0) + 1,
            })?;
            if record.kind != entry.kind {
                return Err(ChainError::SeamKind {
                    name: record.name,
                    expected: entry.kind,
                    found: record.kind,
                });
            }
            *counts.entry(record.name).or_default() += 1;
        }
        for entry in &self.entries {
            let found = counts.get(entry.name).copied().unwrap_or(0);
            if found != entry.expected {
                return Err(ChainError::SeamCount {
                    name: entry.name,
                    kind: entry.kind,
                    expected: entry.expected,
                    found,
                });
            }
        }
        Ok(())
    }
}

/// The prefix for a design 4.3 label that a later stage will use. M3-a prints these three only
/// with this prefix, so a plan cannot be read as an execution.
pub const FUTURE_LABEL_PREFIX: &str = "label reserved for a later stage:";

/// The labels a run report carries; one place so a report cannot invent a softer claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainLabels;

impl ChainLabels {
    /// The exact labels a real run must carry, in report order. The plan-only label is not one
    /// of them: a run that carries it would contradict itself.
    #[must_use]
    pub fn run_lines() -> [&'static str; 3] {
        [HYBRID_TOKEN_LABEL, FIRST_CHAIN_LABEL, HEAD_LABEL]
    }

    /// The labels an M3-a plan prints: the plan-only label as a state, and the three run labels
    /// explicitly marked future-only. No line states that a token, chain execution or head
    /// execution happened.
    #[must_use]
    pub fn plan_lines() -> Vec<String> {
        let mut lines = Vec::with_capacity(4);
        lines.push(format!("label: {PLAN_ONLY_LABEL}"));
        for label in [HYBRID_TOKEN_LABEL, FIRST_CHAIN_LABEL, HEAD_LABEL] {
            lines.push(format!("{FUTURE_LABEL_PREFIX} {label}"));
        }
        lines
    }
}

/// One stage invocation handed to a [ChainOps] backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageCall {
    /// Layer id; the embedding/tail stages use layer 0.
    pub layer: usize,
    /// Absolute token position the stage runs at.
    pub position: usize,
    /// The stage.
    pub stage: ChainStage,
}

/// A chain backend. M3-a has no device implementation; the skeleton executor and its tests use this
/// trait, and the first runnable pass will implement it over the staged weights and the kernels.
pub trait ChainOps {
    /// Runs one stage. A host-seam stage returns its [SeamRecord]; a device stage returns None.
    ///
    /// # Errors
    /// The backend's own error, surfaced as a [ChainError] by the backend.
    fn stage(&mut self, call: StageCall) -> Result<Option<SeamRecord>, ChainError>;
}

/// The skeleton report: a plan, the ledger the backend produced, and explicit no-execution flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainRunReport {
    /// The schedule the plan was built for.
    pub schedule: ChainSchedule,
    /// The planned seam set.
    pub seam_plan: ChainSeamPlan,
    /// The seams the backend actually recorded.
    pub ledger: SeamLedger,
    /// Whether the device chain executed (false in M3-a).
    pub chain_execution: bool,
    /// Whether any device kernel executed (false in M3-a).
    pub device_execution: bool,
    /// Whether a token was produced (false in M3-a).
    pub token_produced: bool,
}

impl ChainRunReport {
    /// The report lines, seam ledger summary **before** any comparison/label line.
    #[must_use]
    pub fn report_lines(&self) -> Vec<String> {
        let mut lines = vec![
            self.schedule.describe(),
            format!(
                "execution flags: chain_execution={}; device_execution={}; token_produced={}",
                self.chain_execution, self.device_execution, self.token_produced
            ),
        ];
        lines.extend(self.seam_plan.report_lines());
        lines.extend(ledger_summary_lines(&self.ledger, &self.seam_plan));
        lines.extend(ChainLabels::plan_lines());
        lines.push("no comparison was made: M3-a is a plan/ledger skeleton, not a run".to_string());
        lines
    }
}

/// The ledger summary: one line per seam and a total, so a plan-only report stays readable while
/// still showing every planned seam fired its expected number of times.
#[must_use]
pub fn ledger_summary_lines(ledger: &SeamLedger, plan: &ChainSeamPlan) -> Vec<String> {
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut total_bytes = 0u64;
    for record in ledger.records() {
        *counts.entry(record.name).or_default() += 1;
        total_bytes = total_bytes.saturating_add(record.total_bytes());
    }
    let mut lines = Vec::with_capacity(counts.len() + 1);
    for entry in &plan.entries {
        let found = counts.get(entry.name).copied().unwrap_or(0);
        lines.push(format!(
            "seam ledger {}: {} records (planned {}), host work",
            entry.name, found, entry.expected
        ));
    }
    lines.push(format!(
        "seam ledger total: {} records, {} host bytes (NOT device bytes); phase-3 device-kernel follow-up for every seam",
        ledger.records().len(),
        total_bytes
    ));
    lines
}

/// Runs the stage graph against a backend and validates the ledger against the plan.
///
/// It executes no kernel: the backend decides what a stage call means. The order and the seam
/// accounting are the skeleton's contract, and the tests' recording backend pins both.
///
/// # Errors
/// [ChainError] when a backend returns a seam for a non-seam stage or omits one for a seam, or when
/// the ledger does not match the plan.
pub fn run_skeleton(
    schedule: &ChainSchedule,
    ops: &mut impl ChainOps,
) -> Result<ChainRunReport, ChainError> {
    let seam_plan = ChainSeamPlan::for_schedule(schedule);
    let mut ledger = SeamLedger::default();
    // Prefill: each token is embedded, then every layer whose CED extent covers it runs its body.
    for position in 0..schedule.prompt_tokens {
        run_stage_order(ops, &mut ledger, 0, position, STEP_STAGE_ORDER)?;
        for layer in &schedule.layers {
            if layer.positions().contains(&position) {
                for stage in LAYER_STAGE_ORDER
                    .iter()
                    .filter(|stage| stage.runs_at(layer))
                {
                    run_stage(ops, &mut ledger, layer.layer, position, *stage)?;
                }
            }
        }
        if position + 1 == schedule.prompt_tokens {
            run_stage_order(ops, &mut ledger, 0, position, TAIL_STAGE_ORDER)?;
        }
    }
    // Decode: each step's token is embedded, then all 40 layers run at that new position.
    for step in 0..schedule.decode_steps {
        let position = schedule.prompt_tokens + step;
        run_stage_order(ops, &mut ledger, 0, position, STEP_STAGE_ORDER)?;
        for layer in &schedule.layers {
            for stage in LAYER_STAGE_ORDER
                .iter()
                .filter(|stage| stage.runs_at(layer))
            {
                run_stage(ops, &mut ledger, layer.layer, position, *stage)?;
            }
        }
        run_stage_order(ops, &mut ledger, 0, position, TAIL_STAGE_ORDER)?;
    }
    seam_plan.validate(&ledger)?;
    Ok(ChainRunReport {
        schedule: schedule.clone(),
        seam_plan,
        ledger,
        chain_execution: false,
        device_execution: false,
        token_produced: false,
    })
}

fn run_stage_order(
    ops: &mut impl ChainOps,
    ledger: &mut SeamLedger,
    layer: usize,
    position: usize,
    stages: &[ChainStage],
) -> Result<(), ChainError> {
    for &stage in stages {
        run_stage(ops, ledger, layer, position, stage)?;
    }
    Ok(())
}

fn run_stage(
    ops: &mut impl ChainOps,
    ledger: &mut SeamLedger,
    layer: usize,
    position: usize,
    stage: ChainStage,
) -> Result<(), ChainError> {
    let call = StageCall {
        layer,
        position,
        stage,
    };
    let record = ops.stage(call)?;
    match (stage.seam_name(), record) {
        (Some(expected), Some(record)) => {
            // Equal counts are not enough: a backend that swaps two seam names would otherwise
            // pass the plan-vs-ledger check. The record must name the stage that produced it.
            if record.name != expected {
                return Err(ChainError::SeamName {
                    stage,
                    layer,
                    expected,
                    found: record.name,
                });
            }
            ledger.record(record);
        }
        (Some(_), None) => {
            return Err(ChainError::MissingStageSeam { stage, layer });
        }
        (None, Some(_)) => {
            return Err(ChainError::NonSeamRecord { stage, layer });
        }
        (None, None) => {}
    }
    Ok(())
}

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

//! Model identity and selection contracts.
//!
//! This crate is deliberately small: it is the boundary between the protocol/runtime crates and
//! the model-specific packs. It owns the **descriptor** (what a model is called, how much context
//! it serves, what it can do) and the **registry** (which descriptors this build knows about).
//! It owns no weights, no kernels, no HTTP types, and no CUDA.
//!
//! It also holds what several packs share at that boundary: the error their configuration parsers
//! return ([config]), the DFlash 2 drafter's configuration ([dflash2]), which more than one
//! target drafts with, and the host [sampler] a served arm applies to a logits row it already
//! has. The sampler is request bookkeeping over that row, not a kernel.
//!
//! The first pack is DeepSeek-V4.1-Flash. The registry is compile-time and static for now; a
//! second pack adds a descriptor here (or in a pack crate that registers at startup) without
//! changing the server, the web control plane, or the CLI dispatcher.

pub mod architecture;
pub mod arm;
pub mod arms;
pub mod backend;
pub mod capabilities;
pub mod compose;
pub mod config;
pub mod coverage;
pub mod defaults;
pub mod descriptor;
pub mod dflash2;
pub mod engine;
pub mod gumbel;
pub mod layout;
pub mod ledger;
pub mod numerics;
pub mod numerics_coverage;
pub mod numerics_monitors;
pub mod parts;
pub mod prof;
pub mod profile;
pub mod prompt;
pub mod prompt_ranks;
pub mod sampler;
pub mod settings;
pub mod spec;
pub mod speculation;
pub mod stack;
pub mod telemetry;
pub mod tp_plan;
pub mod vision;
pub mod vision_geometry;

pub use architecture::{Architecture, LoopSchedule, names_the_producer};
pub use arm::{
    Admission, AdmissionRefusal, KvPlan, MEMORY_FLOOR_BYTES, NODE_MEMORY_BUDGET_BYTES,
    PackIdentity, gib, kv_budget_bytes,
};
pub use arms::{
    ARMS, Arm, Conversion, Converter, DeepSeek, Dense, Deployment, Dialect, Dory, GROUP_LANES, Glm,
    GroupDefaults, Inkling, Qwen, Qwen36, QwenFp8, Route, Source, Tokenizer, arm_for_family,
    arm_for_weights, crate_for, numerics_status, weights_numerics_status,
};
pub use backend::{BackendError, ModelBackend, ModelSession, Sampling, Speculation, StepOutput};
pub use capabilities::Capabilities;
pub use compose::{LayerStep, StepRole};
pub use config::ConfigError;
pub use coverage::{ConfigCoverage, IgnoredKey, KeyCoverageState, UnknownKey};
pub use defaults::{DEFAULTS, DefaultValue, Defaults, check_profile};
pub use descriptor::{
    DEFAULT_MODEL_ID, DEFAULT_WEIGHTS_ID, DORY1_0_4B_BF16_ID, DORY1_0_4B_BF16_WEIGHTS_ID,
    DORY1_0_4B_CONTEXT_TOKENS, DORY1_0_4B_MAX_OUTPUT_TOKENS, DORY1_0_4B_VOCAB_SIZE,
    INKLING2_CONTEXT_TOKENS, INKLING2_MAX_OUTPUT_TOKENS, INKLING2_SMALL_ID,
    INKLING2_SMALL_WEIGHTS_ID, INKLING2_UNPADDED_VOCAB_SIZE, INKLING2_VOCAB_SIZE, MODELS,
    ModelDescriptor, QWEN36_CONTEXT_TOKENS, QWEN36_FLASH_35B_A3B_FP8_ID,
    QWEN36_FLASH_35B_A3B_FP8_WEIGHTS_ID, QWEN36_MAX_OUTPUT_TOKENS, QWEN36_VOCAB_SIZE,
    V41_CONTEXT_TOKENS, V41_MAX_OUTPUT_TOKENS, V41_VOCAB_SIZE, WEIGHT_IDS, canonical_model_type,
    default_model, find_model, find_model_by_type, find_model_by_weights, model_context_tokens,
    model_max_output_tokens,
};
pub use engine::{DEFAULT_ENGINE_KIND, ENGINE_NAMES, EngineKind, parse_engine_kind};
pub use layout::{
    Layout, LayoutDescription, LayoutDtype, LayoutError, LayoutSet, LayoutTensor, LayoutValue,
    TensorSplit,
};
pub use ledger::{ARM_LEDGERS, ArmLedger, State, Strategy, ledger_for_arm, ledger_for_weights};
pub use numerics::{ConfigKey, NumericsIdentity, NumericsProfile, NumericsStatus};
pub use parts::{Part, PartSet, Target, built, is_built};
pub use prof::StageProfiler;
pub use profile::{
    DepthMode, DraftLogitsKernel, DrafterProfile, ModelProfile, PlanProfile, ProfileError,
    active as active_profile, install_active,
};
pub use prompt::{PromptCodec, PromptError};
pub use prompt_ranks::{MergedAlternative, MergedPosition};
pub use sampler::{Sampler, SamplerMark, argmax};
pub use spec::{Pending, Queue, Speculator, kept_to_end, lookup_draft, verdict};
pub use speculation::{DepthPolicy, StepCost};
pub use stack::{Block, PartInstance, Shape, Stack, StackKind};
pub use tp_plan::{Split, TpAxis, TpPlan, TpPlanError};
pub use vision::{
    ImageGrid, ImageSpan, PreparedImage, SpanRow, VisionCodec, VisionError, VisionOptions,
};
pub use vision_geometry::{
    VisionActivation, VisionDownsample, VisionGeometry, VisionNorm, VisionPosition, VisionTowerSpec,
};

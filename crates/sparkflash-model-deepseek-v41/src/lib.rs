//! DeepSeek-V4.1-Flash model pack.
//!
//! This crate owns the V4.1 definition end to end: the flat runtime hyperparameters and the
//! released-HF mapping, the storage formats, the tensor and layer plans, the host reference
//! forward, the checkpoint loader, the sequence fixture, and the prompt encoding/tokenizer seam.
//! `sparkflash-engine` is a facade that re-exports these modules under their old paths.
//!
//! The arithmetic is unchanged from the engine extraction: this is a move, not a rewrite. The
//! device path and the CLI still reach the same types and functions through
//! `sparkflash_engine`.

#![forbid(unsafe_code)]

pub mod config;
pub mod dtype;
pub mod encoding;
pub mod forward;
pub mod layer_plan;
pub mod linalg;
pub mod loader;
pub mod parallel;
pub mod parallel_tp;
pub mod prompt_ranks;
pub mod quant;
pub mod report;
pub mod safetensors;
pub mod sequence_fixture;
pub mod synthetic;
pub mod tensor;
pub mod vision;
pub mod weights;

pub use config::ModelConfig;
pub use dtype::DType;
pub use layer_plan::{
    AttentionGeometry, CompressionKind, LayerMode, LayerPlan, LayerSpec, PlanError,
};
pub use sequence_fixture::{
    CertifiedMask, CompareError, DeviceStep, FixtureError, FixtureProvenance, FixtureSource,
    FixtureStep, FixtureWeightKind, FixtureWeights, KvRowDigest, KvRowKind, LogitsCapture,
    MaskedCompareError, MaskedLogitsComparison, ScreenedLogits, SequenceFixture, compare_logits,
    compare_logits_exact, compare_steps, compared_positions, deterministic_prompt,
    exclusion_boundary, generate_fixture,
};
pub use tensor::{TensorPlan, TensorSpec};
pub use weights::{TensorView, WeightError, WeightSource};

//! The SparkFlash engine facade.
//!
//! The V4.1 model pack now owns the configuration, plans, host reference forward, loader,
//! sequence fixture, and prompt encoding. This crate re-exports those modules so the existing
//! `sparkflash_engine::...` paths used by the CLI, device crates, kernels and tests keep working
//! while the rest of the workspace migrates to the model pack directly.

#![forbid(unsafe_code)]

pub use sparkflash_model_deepseek_v41::{
    AttentionGeometry, CompressionKind, DType, LayerMode, LayerPlan, LayerSpec, ModelConfig,
    PlanError, TensorPlan, TensorSpec, TensorView, WeightError, WeightSource, config, dtype,
    encoding, forward, layer_plan, linalg, loader, parallel, parallel_tp, prompt_ranks, quant,
    report, safetensors, sequence_fixture, synthetic, tensor, weights,
};

pub use sparkflash_model_deepseek_v41::sequence_fixture::{
    CertifiedMask, CompareError, DeviceStep, FixtureError, FixtureProvenance, FixtureSource,
    FixtureStep, FixtureWeightKind, FixtureWeights, KvRowDigest, KvRowKind, LogitsCapture,
    MaskedCompareError, MaskedLogitsComparison, ScreenedLogits, SequenceFixture, compare_logits,
    compare_logits_exact, compare_steps, compared_positions, deterministic_prompt,
    exclusion_boundary, generate_fixture,
};

//! The model description an engine needs before it has any weights.
//!
//! This crate is the first slice of the engine: it holds the architecture constants of
//! DeepSeek-V4.1-Flash, the exact tensor plan a forward pass consumes, and a weight source that
//! produces every one of those tensors deterministically **without the checkpoint**. That last part
//! is the point - the engine has to be testable on a machine that has no 510 GB of weights, no GPU
//! and no cluster, so the plan and the source are separated from the arithmetic that will consume
//! them.
//!
//! Modules:
//!
//! - [`config`]: [`ModelConfig`], the flat runtime hyperparameters, with a checked constructor for
//!   the released model and a tiny one for tests.
//! - [`dtype`]: the storage formats in the checkpoint - FP8 E4M3, FP4 E2M1 packed two-per-byte,
//!   the UE8M0 scale bytes, and the block rules that decide how many scale bytes a tensor carries.
//! - [`quant`]: the checkpoint's storage formats decoded to f32 - FP8 E4M3, E8M0 scales, FP4
//! - [`report`]: the Phase 1 one-token report - rows, seams, audit and determinism as one
//!   runnable contract.
//!   E2M1, bf16 and f32 - and the reference's per-32 activation quantisers.
//! - [`tensor`]: [`TensorPlan`], the name, shape and dtype of every tensor the forward reads.
//! - [`layer_plan`]: [`LayerPlan`], the attention variant, KV source, Engram and DSpark role of
//!   every backbone layer and MTP stage, plus the execution order its KV DAG requires.
//! - [`weights`]: the [`WeightSource`] trait - the only thing the arithmetic code will see.
//! - [`synthetic`]: a [`WeightSource`] backed by a seeded generator instead of a file.
//! - [`safetensors`]: the checkpoint's container and index, read without reading tensor data, so a
//!   reader that has the index can describe 96,085 tensors without opening a 510 GB checkpoint.
//! - [`loader`]: the checkpoint opened against the plan - verified shard by shard, then read one
//!   tensor at a time, with the two conversions convert.py performs and the Engram tables refused.
//!
//! Nothing here allocates a model-sized buffer or touches a device. The one module that opens a
//! checkpoint, [`loader`], reads headers first and then a single tensor at a time, under a limit.

#![forbid(unsafe_code)]

pub mod config;
pub mod dtype;
pub mod forward;
pub mod layer_plan;
pub mod linalg;
pub mod loader;
pub mod quant;
pub mod report;
pub mod safetensors;
pub mod synthetic;
pub mod tensor;
pub mod weights;

pub use config::ModelConfig;
pub use dtype::DType;
pub use layer_plan::{
    AttentionGeometry, CompressionKind, LayerMode, LayerPlan, LayerSpec, PlanError,
};
pub use tensor::{TensorPlan, TensorSpec};
pub use weights::{TensorView, WeightError, WeightSource};

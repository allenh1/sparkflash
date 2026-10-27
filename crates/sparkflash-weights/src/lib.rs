//! Generic weight storage and host arithmetic.
//!
//! This crate is model-agnostic. It owns the storage formats ([module:dtype]), the scalar and
//! block decoders ([module:quant]), the safetensors container ([module:safetensors]), the weight
//! source contract ([module:weights]), and the CPU kernels the host reference is built from
//! ([module:linalg]). A model pack supplies tensor names, shapes and plans; none of those appear
//! here.
//!
//! The arithmetic is unchanged from the V4.1 engine extraction: this is a move, not a rewrite.

#![forbid(unsafe_code)]

pub mod config;
pub mod dtype;
pub mod linalg;
pub mod parallel;
pub mod quant;
pub mod safetensors;
pub mod spec;
pub mod weights;

pub use config::ConfigError;
pub use dtype::DType;
pub use linalg::EngineError;
pub use spec::TensorSpec;
pub use weights::{Conformance, TensorView, WeightError, WeightSource, check_source_against_specs};

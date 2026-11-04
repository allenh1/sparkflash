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

//! Generic weight storage and host arithmetic.
//!
//! This crate is model-agnostic. It owns the storage formats ([module:dtype]), the scalar and
//! block decoders ([module:quant]), the ModelOpt NVFP4 format ([module:nvfp4]), the safetensors
//! container and its fixture writer ([module:safetensors]), the per-tensor stream synthetic
//! checkpoints draw from ([module:rng]), the weight source contract ([module:weights]), and the CPU
//! kernels the host reference is built from ([module:linalg], the hyper-connections of
//! [module:mhc], the delta-rule recurrence of [module:linear_attention], and the block and expert
//! selections of [module:topk]), and the tensor-parallel seam every pack shards through
//! ([module@tp]). A model pack supplies tensor names and shapes; none of those appear here, only
//! the role-tagged container they are planned in ([module:plan]), the per-rank converted file
//! a rank's slices are written to and read back from ([module:rank_file]), and the Hugging Face
//! hub cache a model id resolves to its checkpoint in ([module:hub]).
//!
//! The arithmetic is unchanged from the V4.1 engine extraction: this is a move, not a rewrite.

#![forbid(unsafe_code)]

pub mod config;
pub mod dtype;
pub mod hub;
pub mod linalg;
pub mod linear_attention;
pub mod mhc;
pub mod nvfp4;
pub mod parallel;
pub mod plan;
pub mod quant;
pub mod rank_file;
pub mod rng;
pub mod safetensors;
pub mod spec;
pub mod topk;
pub mod tp;
pub mod weights;

pub use config::ConfigError;
pub use dtype::DType;
pub use linalg::EngineError;
pub use plan::TensorPlan;
pub use spec::TensorSpec;
pub use weights::{
    Conformance, TensorView, WeightError, WeightSource, check_length, check_source_against_specs,
};

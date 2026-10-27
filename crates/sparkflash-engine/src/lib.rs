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

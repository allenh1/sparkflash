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
//! It owns no weights, no arithmetic, no HTTP types, and no CUDA.
//!
//! The first pack is DeepSeek-V4.1-Flash. The registry is compile-time and static for now; a
//! second pack adds a descriptor here (or in a pack crate that registers at startup) without
//! changing the server, the web control plane, or the CLI dispatcher.

pub mod backend;
pub mod capabilities;
pub mod descriptor;
pub mod engine;
pub mod prompt;
pub mod telemetry;
pub mod vision;

pub use backend::{BackendError, ModelBackend, ModelSession, Sampling, StepOutput};
pub use capabilities::Capabilities;
pub use descriptor::{
    DEFAULT_MODEL_ID, DEFAULT_WEIGHTS_ID, MODELS, ModelDescriptor, V41_CONTEXT_TOKENS,
    V41_MAX_OUTPUT_TOKENS, V41_VOCAB_SIZE, WEIGHT_IDS, default_model, find_model,
    find_model_by_weights, model_context_tokens, model_max_output_tokens,
};
pub use engine::{DEFAULT_ENGINE_KIND, ENGINE_NAMES, EngineKind, parse_engine_kind};
pub use prompt::{PromptCodec, PromptError};
pub use vision::{
    ImageGrid, ImageSpan, PreparedImage, SpanRow, VisionCodec, VisionError, VisionOptions,
};

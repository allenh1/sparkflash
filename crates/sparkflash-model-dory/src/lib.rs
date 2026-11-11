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

//! The NVIDIA Nemotron Labs Dory1.0-4B pack.
//!
//! Dory is a looped normalized-GPT: 72 stored sublayers form 36 physical blocks, the middle 40
//! sublayers are a shared recurrent block applied n_recurrent_loops times, and every residual
//! is a normalized interpolation rather than an addition. The checkpoint is BF16 throughout.
//!
//! This pack owns the typed configuration and its derived block schedule, the 507-name tensor
//! plan, the f32 reference arithmetic (including the loop and its KV caches), the host backend,
//! and the TP=1 device arm that runs the shared Qwen GEMV/rope/cache kernels beside the
//! nGPT-specific kernels in kernels/dory_sm121.cu.

pub mod architecture;
pub mod backend;
pub mod config;
pub mod device;
#[doc(hidden)]
pub mod fixture;
pub mod forward;
pub mod prompt;
pub mod recurrence;
pub mod tensor;

pub use architecture::{architecture_from_carried, architecture_from_json};
pub use backend::DoryHostBackend;
pub use config::{
    AttentionType, Block, DoryConfig, KvCacheMode, MODEL_ID, MODEL_TYPE, Partition, WEIGHTS_ID,
};
pub use device::{DoryDeviceBackend, DoryStage};
pub use forward::{
    AttentionWeights, BlockWeights, DoryState, DoryWeights, KvCache, Mat, MlpWeights, justnorm,
    slerp,
};
pub use recurrence::{
    DoryRecurrenceBackend, DoryRecurrenceStage, LinkStreams, RecurrenceLink, RecurrenceLinks,
    recurrence_draft_lanes,
};
pub use tensor::{DoryTensorPlan, Role};

use sparkflash_model::ModelDescriptor;

/// The descriptor this pack serves.
#[must_use]
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor::dory1_0_4b_bf16()
}

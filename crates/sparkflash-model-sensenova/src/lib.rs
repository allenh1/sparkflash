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

//! The SenseNova-Loop-9B pack.
//!
//! SenseNova-Loop is a Qwen3.5 hybrid decoder with a shared-weight loop over physical layers
//! [4, 28) applied twice (56 visits from 32 stored layers) and two-stream
//! manifold-constrained hyper-connections over the same range. The release has 512K context via
//! YaRN and a 27-layer vision tower.
//!
//! This pack owns the typed configuration and execution plan, the 772-name text tensor plan,
//! the f32 reference forward with per-visit caches and the exact mHC router, and the host
//! backend. The device arm is the next layer on the same plan.

pub mod architecture;
pub mod backend;
pub mod config;
pub mod device;
pub mod forward;
pub mod tensor;

pub use architecture::{architecture_from_carried, architecture_from_json};
pub use backend::SenseNovaHostBackend;
pub use config::{
    LayerType, LinearSettings, LoopSettings, MODEL_ID, MODEL_TYPE, MhcSettings, RopeSettings,
    SenseNovaConfig, Storage, WEIGHTS_ID,
};
pub use device::SenseNovaDeviceBackend;
pub use forward::{
    AttentionWeights, GdnState, KvCache, LayerWeights, Mat, MhcWeights, SenseNovaState,
    SenseNovaWeights, VisitState, yarn_attention_scale, yarn_inv_freq,
};
pub use tensor::{Role, SenseNovaTensorPlan};

/// The context the FP8 device arm admits when the operator does not name one: 160K with the
/// 8-bit K/V cache and the embedding/head on the host. It leaves about 2 GiB of device headroom
/// on a 16 GB card; 256K needs 4-bit weights, because the 8-bit cache alone is 7.5 GB there.
pub const DEFAULT_CONTEXT: usize = 163_840;

/// The carried runtime profile for the FP8 release.
pub const PROFILE_TOML: &str = include_str!("../config/sensenova_loop_9b_fp8.toml");

use sparkflash_model::ModelDescriptor;

/// The released configuration carried in this crate.
#[must_use]
pub fn carried_config() -> &'static str {
    include_str!("../config/sensenova_loop_9b.hf.json")
}

/// The descriptor this pack serves.
#[must_use]
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor::sensenova_loop_9b()
}

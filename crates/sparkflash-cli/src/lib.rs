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

//! `sparkflash` command line interface.
//!
//! The binary is a thin dispatcher: every subcommand lives in its own module here so that it
//! can be exercised as a library, with injected probes, without spawning a process.

pub mod allocator;
#[cfg(feature = "cuda")]
pub use sparkflash_deepseek_forward::attention_device;
pub mod attn_seam_weights;
pub mod bootlog;
pub use sparkflash_deepseek_forward::chain_timing;
pub mod convert;
pub mod device_chain;
pub mod dynamic;
pub use sparkflash_deepseek_forward::device_counters;
pub use sparkflash_deepseek_forward::device_link;
pub use sparkflash_deepseek_forward::doctor;
pub use sparkflash_deepseek_forward::engram_device_state;
pub mod glm_tp4;
pub use sparkflash_deepseek_forward::layer_attention_stages;
pub use sparkflash_deepseek_forward::layer_device;
pub use sparkflash_deepseek_forward::logits_dump;
pub mod models;
#[cfg(feature = "cuda")]
pub use sparkflash_deepseek_forward::moe_device;
pub use sparkflash_deepseek_forward::resident_weights;
pub use sparkflash_deepseek_forward::run_header;
pub mod serve;
pub mod tp4_chain;
pub use sparkflash_deepseek_forward::tp4_combine;
pub mod tp4_device_runner;
pub use sparkflash_deepseek_forward::tp4_forward;
#[cfg(feature = "cuda")]
pub mod tp4_host;
pub use sparkflash_deepseek_forward::tp4_operands;
pub use sparkflash_deepseek_forward::tp4_prefill;
pub use sparkflash_deepseek_forward::tp4_reduce;
pub use sparkflash_deepseek_forward::tp4_resident;
pub use sparkflash_deepseek_forward::tp4_smoke;
pub use sparkflash_deepseek_forward::tp4_tail;
pub use sparkflash_deepseek_forward::tp4_token;

pub use sparkflash_deepseek_forward::comm_rdma_shim;

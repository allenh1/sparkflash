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
pub mod attention_device;
pub mod attn_seam_weights;
pub mod bootlog;
pub mod chain_timing;
pub mod convert;
pub mod device_chain;
pub mod device_counters;
pub mod device_link;
pub mod doctor;
pub mod engram_device_state;
pub mod layer_attention_stages;
pub mod layer_device;
pub mod logits_dump;
pub mod models;
#[cfg(feature = "cuda")]
pub mod moe_device;
pub mod resident_weights;
pub mod run_header;
pub mod serve;
pub mod tp4_chain;
pub mod tp4_combine;
pub mod tp4_device_runner;
pub mod tp4_forward;
#[cfg(feature = "cuda")]
pub(crate) mod tp4_graph;
pub mod tp4_host;
pub mod tp4_operands;
pub mod tp4_prefill;
pub mod tp4_reduce;
pub mod tp4_resident;
pub mod tp4_smoke;
pub mod tp4_tail;
pub mod tp4_token;

/// The RDMA inline-shim build mode, embedded unconditionally in every evidence schema.
///
/// "No silent fallback" applies to the build configuration too: a window must be able to tell a
/// real-verbs binary from an ENOSYS-stub one from the JSON alone, so every stage that writes
/// evidence reports this (see [`tp4_smoke`] and [`tp4_resident`]).
pub fn comm_rdma_shim() -> &'static str {
    #[cfg(feature = "rdma")]
    {
        match sparkflash_comm::rdma::shim_mode() {
            sparkflash_comm::rdma::ShimMode::System => "system",
            sparkflash_comm::rdma::ShimMode::Stub => "stub",
        }
    }
    #[cfg(not(feature = "rdma"))]
    {
        "not_built"
    }
}

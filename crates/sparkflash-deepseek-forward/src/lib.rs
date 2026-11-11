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

//! The DeepSeek-V4.1-Flash device forward, out of `sparkflash-cli`.
//!
//! The forward is model code: it composes the V4.1 layer plan, the resident weights and the
//! shared kernel stack. It lived in the CLI, so every serving change and every forward change
//! touched the same 90k-line crate. It is here now, and the CLI is a dispatcher over it.
//!
//! The modules keep the names and `crate::` paths they had, which is why the move is mechanical
//! and numerically inert. The typed op plans in `sparkflash-ops` are the next layer: an op that
//! moves into a plan here loses its hand-rolled launch arguments without changing its bits.

#[cfg(feature = "cuda")]
pub mod attention_device;
pub mod chain_timing;
#[cfg(feature = "cuda")]
pub mod device_counters;
pub mod device_link;
pub mod doctor;
pub mod engram_device_state;
pub mod layer_attention_stages;
pub mod layer_device;
pub mod logits_dump;
#[cfg(feature = "cuda")]
pub mod moe_device;
// The exchange needs the CUDA driver (a pinned, mapped slab and the staging kernel's cubin) as
// well as the verbs backend, so it is built only when both are on. Gating it on `rdma` alone broke
// `--features rdma`, which is a supported build for the host-side collective work.
#[cfg(all(feature = "cuda", feature = "rdma"))]
pub mod rdma_exchange;
pub mod resident_weights;
pub mod run_header;
pub mod tp4_combine;
pub mod tp4_forward;
pub mod tp4_graph;
pub mod tp4_operands;
pub mod tp4_prefill;
pub mod tp4_reduce;
pub mod tp4_resident;
pub mod tp4_smoke;
pub mod tp4_tail;
pub mod tp4_token;

/// The RDMA inline-shim build mode, embedded in every evidence schema.
#[must_use]
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

//! `sparkflash` command line interface.
//!
//! The binary is a thin dispatcher: every subcommand lives in its own module here so that it
//! can be exercised as a library, with injected probes, without spawning a process.

pub mod allocator;
#[cfg(feature = "cuda")]
pub mod attention_device;
pub mod chain_timing;
pub mod convert;
pub mod device_chain;
pub mod doctor;
pub mod logits_dump;
#[cfg(feature = "cuda")]
pub mod moe_device;
pub mod resident_weights;
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

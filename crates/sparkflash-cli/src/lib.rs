//! `sparkflash` command line interface.
//!
//! The binary is a thin dispatcher: every subcommand lives in its own module here so that it
//! can be exercised as a library, with injected probes, without spawning a process.

#[cfg(feature = "cuda")]
pub mod attention_device;
pub mod device_chain;
pub mod doctor;
#[cfg(feature = "cuda")]
pub mod moe_device;
pub mod serve;
pub mod tp4_smoke;

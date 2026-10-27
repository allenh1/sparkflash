//! Model-agnostic serving lifecycle.
//!
//! This crate sits between the HTTP surface in sparkflash-api and a model backend. It owns the
//! request queue, the engine thread, streaming replies, stop-string handling, sample logprobs and
//! the request telemetry envelope. A model pack supplies the arithmetic through
//! sparkflash_model::ModelBackend, and a prompt codec supplies tokenization.
//!
//! The first adapter is the generic host driver. The V4.1 reference driver still lives in the CLI
//! until the next move; it can migrate onto this crate without changing the HTTP contract.

pub mod host;

pub use host::{HostDriver, HostOptions};

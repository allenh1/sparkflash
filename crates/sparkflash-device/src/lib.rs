//! `sparkflash-device`: the device forward chain crate. **M1 is a fake-session skeleton.**
//!
//! Milestone M1 of `docs/build/device-forward-chain.md` section 7, not a chain. It exists so the
//! serving seam's capability boundary becomes real without a device claim:
//!
//! - It implements the api engine seam, [`sparkflash_api::engine::InferenceEngine`] (design 1.2:
//!   `sparkflash-api` supplies "the `InferenceEngine` trait and `EngineRequest`/`Capabilities`
//!   surface"), over a [`session::DeviceSession`].
//! - The only session M1 has is [`session::FakeSession`]: it records calls and produces no device
//!   result. It opens no driver, loads no cubin, allocates no device buffer, and reads no
//!   checkpoint or cluster (design 7 M1: "a fake device session for CI (no CUDA, no checkpoint, no
//!   cluster)").
//! - `generate` delegates to a reference engine supplied by the cli wiring. `ReferenceDriver` is
//!   bin-only, so this library cannot name it; the cli passes it as `Arc<dyn InferenceEngine>`
//!   (design 1.3).
//! - It refuses an image part through the typed [`sparkflash_api::MissingCapability::vision`]
//!   path, and it refuses a non-greedy sampling configuration before any delegation (design 1.3:
//!   "refusing a non-greedy configuration").
//! - The capability report says `vision: false` and `fake_session: true`; the cli flag
//!   `--engine reference|device` defaults to `reference`, so nothing on the existing serve path
//!   changes unless an operator selects it (design 1.3, 7 M1).
//!
//! **What M1 does not do.** No kernel launch, no chain, no boundary capture, no window ring, no
//! weight staging and no criterion-1 claim (design 3.1, 3.2, 4.3, 7 M2/M3). The token path in a
//! first real chain is hybrid (design 3.2b); M1 does not even have a device token path. The first
//! real session is M2, after M1 merges (design 7).
//!
//! # Dependency direction
//!
//! `sparkflash-device` depends on the api crate (the engine seam) and futures only. It does not
//! depend on `sparkflash-kernels` or libcuda yet; M2/M3 add the kernel and fixture dependencies
//! named in design 1.2. Nothing here needs a device, a checkpoint, a cluster or a network, so it
//! builds and tests on a CPU-only host and adds no unconditional device dependency to any build.

#![forbid(unsafe_code)]

pub mod engine;
pub mod session;

pub use engine::{CapabilityReport, DeviceEngine, DeviceRefusal, NON_GREEDY_CODE, is_greedy};
pub use session::{DeviceSession, FakeSession, SessionError};

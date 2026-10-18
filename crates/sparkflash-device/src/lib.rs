//! `sparkflash-device`: the device forward chain crate. **M2 is one real kernel, still not a chain.**
//!
//! Milestones M1 and M2 of `docs/build/device-forward-chain.md` section 7. The crate exists so the
//! serving seam's capability boundary becomes real and a device session is proved through it:
//!
//! - It implements the api engine seam, [`sparkflash_api::engine::InferenceEngine`] (design 1.2:
//!   `sparkflash-api` supplies "the `InferenceEngine` trait and `EngineRequest`/`Capabilities`
//!   surface"), over a [`session::DeviceSession`].
//! - M1's session is [`session::FakeSession`]: it records calls and produces no device result, so
//!   the default build, the CI host and every test here stay GPU-free (design 7 M1).
//! - M2's session is `session::cuda::CudaSession`, behind the `cuda` feature: it opens the
//!   driver on a one-shot worker thread, verifies and loads the pinned device MoE router cubin,
//!   allocates the router buffers, executes exactly one real `moe_router_sm121` launch, reads the
//!   outputs back and reports that evidence (design 7 M2). It makes no token, logit, correctness or
//!   chain claim; the chain is still absent.
//! - `generate` delegates to a reference engine supplied by the cli wiring. `ReferenceDriver` is
//!   bin-only, so this library cannot name it; the cli passes it as `Arc<dyn InferenceEngine>`
//!   (design 1.3).
//! - It refuses an image part through the typed [`sparkflash_api::MissingCapability::vision`]
//!   path, and it refuses a non-greedy sampling configuration before any delegation (design 1.3:
//!   "refusing a non-greedy configuration").
//! - The capability report states `vision: false`, whether the session is fake, and exactly what a
//!   real session executed (kernel id, cubin sha256 and size, launches). The cli flag
//!   `--engine reference|device` defaults to `reference`, so nothing on the existing serve path
//!   changes unless an operator selects it (design 1.3, 7 M1, 7 M2).
//!
//! **What the crate does not do.** No chain, no boundary capture, no window ring, no window-ring
//! rotation, no weight staging for the chain, no token/logit comparison and no criterion-1 claim
//! (design 3.1, 3.2, 4.3, 7 M3). The token path in a first real chain is hybrid (design 3.2b); this
//! crate has no device token path. M3 is the first point at which a chain claim may be made.
//!
//! # Dependency direction and the cuda feature
//!
//! Default dependencies are the api crate (the engine seam) and futures only, so a CPU-only build
//! and CI see no driver path. The `cuda` feature adds `sparkflash-kernels` (the dlopen driver, the
//! pinned router ABI/launch and the cubin pin) and `sha2`; nothing links libcuda, because the
//! driver is loaded through `Cuda::load`. The cli forwards its own `cuda` feature here, and a
//! real session is only constructed with that feature, an explicit `--engine device` and an
//! explicit cubin path.

#![forbid(unsafe_code)]

pub mod engine;
pub mod session;

pub use engine::{CapabilityReport, DeviceEngine, DeviceRefusal, NON_GREEDY_CODE, is_greedy};
#[cfg(feature = "cuda")]
pub use session::cuda::CudaSession;
pub use session::{DeviceSession, ExecutionEvidence, FakeSession, SessionError};

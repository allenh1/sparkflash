//! `sparkflash-device`: the device forward chain crate. **M2 is one real kernel; M3-a is the chain
//! skeleton, still not an execution.**
//!
//! Milestones M1, M2 and M3-a of `docs/build/device-forward-chain.md` section 7. The crate exists so
//! the serving seam's capability boundary becomes real and a device session is proved through it:
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
//! - M3-a adds the chain skeleton: [`chain`] derives the 130 + 8 CED schedule (5,482 token-layer
//!   executions), the 40-layer stage map, the planned host-seam set and the plan-vs-ledger
//!   self-check; [`window`] accounts the 128-slot window rings and round-trips the oldest-first
//!   view; [`cubins`] parses the baseline pins and verifies size and SHA-256 fail-closed. The
//!   skeleton runs through a [`chain::ChainOps`] backend and reports
//!   `chain_execution = false`, `device_execution = false`, `token_produced = false`.
//!
//! **What the crate does not do.** No chain execution, no device weight staging, no device token or
//! logits path, no boundary capture and no criterion-1 claim (design 3.1, 3.2, 4.3, 7 M3). The token
//! path in a first real chain is hybrid (design 3.2b); this crate has no device token path. M3's
//! first hybrid pass, with a Lead window, is the first point at which a chain claim may be made.
//!
//! # Dependency direction and the cuda feature
//!
//! The chain plan, window arithmetic, seam ledger and masked comparison are host code, so default
//! dependencies include `sparkflash-engine`, `sparkflash-kernels` and `sparkflash-kv`; those crates
//! are CPU-only-buildable (kernels resolves the driver with dlopen and nothing links libcuda), and
//! the default build still opens no driver: the device path is behind `cuda`. `sparkflash-engine`
//! must never depend on this crate (the dependency direction is device -> engine); the MR's
//! verification runs `cargo tree -p sparkflash-device` and
//! `cargo tree -p sparkflash-engine` to show the tree stays acyclic. The cli forwards its own
//! `cuda` feature here, and a real session is
//! only constructed with that feature, an explicit `--engine device` and an explicit cubin path.

#![forbid(unsafe_code)]

pub mod chain;
pub mod cubins;
pub mod engine;
pub mod session;
pub mod window;

pub use chain::{
    ChainError, ChainLabels, ChainLayer, ChainOps, ChainRunReport, ChainSchedule, ChainSeamPlan,
    ChainStage, FIRST_CHAIN_LABEL, FUTURE_LABEL_PREFIX, HEAD_LABEL, HYBRID_TOKEN_LABEL,
    LAYER_STAGE_ORDER, MILESTONE_DECODE_STEPS, MILESTONE_PROMPT_TOKENS, PLAN_ONLY_LABEL,
    STEP_STAGE_ORDER, SeamPlanEntry, StageCall, StageStatus, TAIL_STAGE_ORDER, run_skeleton,
};
pub use cubins::{
    CHAIN_KERNELS, GATE_UP_KERNEL, REQUIRED_KERNELS, ROUTER_KERNEL, required_pins,
    verify_required_dir,
};
pub use engine::{CapabilityReport, DeviceEngine, DeviceRefusal, NON_GREEDY_CODE, is_greedy};
#[cfg(feature = "cuda")]
pub use session::cuda::CudaSession;
pub use session::{DeviceSession, ExecutionEvidence, FakeSession, SessionError};
pub use window::{WindowAccounting, layer_view, layer_view_bytes};

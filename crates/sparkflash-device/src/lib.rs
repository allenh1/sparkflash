//! `sparkflash-device`: the device forward chain crate. **M2 is one real kernel; M3-a is the chain
//! skeleton, M3-b is layer-0 weight staging and a budget-gated first-stage path; none is a chain
//! execution.**
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
//! - M3 wiring adds [`chain_device`]: the one `ChainOps` implementation a device run uses. Host
//!   seams go to an injected host-seam delegate; device-status stages go to an injected launcher,
//!   never to host math. The fake launcher drives the full 130 + 8 plan through
//!   `device-chain --dry-run` (108,017 calls, plan-vs-ledger validated); the real `CudaLauncher`
//!   is behind `cuda`, verifies the pinned cubin directory and wires the mHC typed builder.
//! - M3-b adds layer-0 staging and admission: [`staging`] reads only `layers.0.hc_attn_fn`,
//!   `layers.0.hc_attn_base`, `layers.0.hc_attn_scale`, `layers.0.attn_norm.weight` and paged embed
//!   rows through [`sparkflash_engine::weights::WeightSource`], checks each dtype/shape before the
//!   read, writes no whole embedding table, records a SHA-256 per accepted tensor/row and exposes the
//!   exact sixteen-buffer [`staging::MhcBufferPlan`]; [`budget`] admits a stage only when the
//!   projected host and device bytes leave at least 1 GiB of MemAvailable, before any allocation.
//!   `CudaLauncher::stage_layer_zero` allocates the sixteen buffers and copies the staged tensors
//!   only after that admission; the cuda-only `device-chain --run-stage mhc_entry` arm drives one
//!   layer-0 mHC entry stage through `ChainDevice`. It is a stage path, not a chain.
//!
//! **What the crate does not do.** No chain execution, no token or logits path, no boundary capture
//! and no criterion-1 claim (design 3.1, 3.2, 4.3, 7 M3). Weight staging is layer-0-only; no layer 1
//! to 39, shared-expert, head or routed-expert path is staged, and no device execution was performed
//! by this crate's tests. The token path in a first real chain is hybrid (design 3.2b); M3's first
//! hybrid pass, in a Lead window, is the first point at which a chain claim may be made.
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

pub mod budget;
pub mod chain;
pub mod chain_device;
pub mod cubins;
pub mod engine;
pub mod layer_zero;
pub mod session;
pub mod staging;
pub mod window;

pub use budget::{
    ADMISSION_FLOOR_BYTES, BudgetAdmission, BudgetError, BudgetRequest, read_mem_available_bytes,
};
pub use chain::{
    ChainError, ChainLabels, ChainLayer, ChainOps, ChainRunReport, ChainSchedule, ChainSeamPlan,
    ChainStage, FIRST_CHAIN_LABEL, FUTURE_LABEL_PREFIX, HEAD_LABEL, HYBRID_TOKEN_LABEL,
    LAYER_STAGE_ORDER, MILESTONE_DECODE_STEPS, MILESTONE_PROMPT_TOKENS, PLAN_ONLY_LABEL,
    STEP_STAGE_ORDER, SeamPlanEntry, StageCall, StageStatus, TAIL_STAGE_ORDER, run_skeleton,
};
pub use chain_device::{
    ChainDevice, ChainDeviceStats, DeviceLaunchError, DeviceLauncher, DeviceStageCall,
    DryRunReport, FakeLauncher, HostSeamDelegate, NoCudaLauncher, SeatHostSeams, dry_run,
    host_seat_counts, run_chain,
};
#[cfg(feature = "cuda")]
pub use chain_device::{CudaLauncher, MhcBuffers, MhcStageError};
pub use cubins::{
    CHAIN_KERNELS, GATE_UP_KERNEL, REQUIRED_KERNELS, ROUTER_KERNEL, required_pins,
    verify_required_dir,
};
pub use engine::{CapabilityReport, DeviceEngine, DeviceRefusal, NON_GREEDY_CODE, is_greedy};
#[cfg(feature = "cuda")]
pub use session::cuda::CudaSession;
pub use session::{DeviceSession, ExecutionEvidence, FakeSession, SessionError};
pub use staging::{
    EMBED_WEIGHT, EmbedRow, LAYER0_ATTN_NORM, LAYER0_MHC_BASE, LAYER0_MHC_FN, LAYER0_MHC_SCALE,
    LAYER0_PREFIX, LayerStaging, LayerZeroStaging, MHC_BUFFER_COUNT, MhcBufferPlan, StagedTensor,
    StagingError,
};
pub use window::{WindowAccounting, layer_view, layer_view_bytes};

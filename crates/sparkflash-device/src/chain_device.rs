//! M3 device-stage wiring: the ChainOps implementation the device run will use.
//!
//! What this module is. The chain skeleton in crate::chain fixes the schedule, the stage map, the
//! planned host-seam set and the plan-vs-ledger self-check. This module implements ChainOps for that
//! skeleton: host-seam stages go to an injected HostSeamDelegate (the M3-a seat recorder by default),
//! and every device-status stage goes to an injected DeviceLauncher. There is no fallback between
//! the two: a device stage can never quietly run host arithmetic.
//!
//! What this module is not. It does not run a chain, compare a token or logit, or make a device
//! claim. M3-b adds the layer-0 staging path in [crate::staging]; only that path allocates
//! buffers. The fake
//! launcher is the default test/dry-run launcher. The wiring is nevertheless the single source of
//! truth: the dry-run drives the full 130 + 8 plan through this code and validates the same ledger a
//! device run would validate.
//!
//! Call accounting. The full plan makes 108,017 ChainOps::stage calls: 58,670 host-seam records and
//! 49,347 device-launcher calls. Tests and device-chain --dry-run pin those totals.
//!
//! Cubins. The real launcher reads every required cubin exactly once and verifies that same slice
//! with crate::cubins::verify_bytes before it can load anything (no verify-then-re-read window),
//! then resolves the typed builder for the stage. The first wired builder is
//! sparkflash_kernels::mega_mhc_launch; its params are built only when a buffer set is supplied,
//! which M3-b's stage_layer_zero provides. A missing or mis-pinned cubin is an error, never a host
//! fallback.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use sparkflash_kernels::sequence_chain::SeamRecord;

use crate::chain::{
    ChainError, ChainOps, ChainRunReport, ChainSchedule, ChainSeamPlan, ChainStage, SeamPlanEntry,
    StageCall, StageStatus, run_skeleton,
};

/// One device stage call: the stage, its position in the schedule and the kernel it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceStageCall {
    /// The stage the chain wanted to run.
    pub stage: ChainStage,
    /// The layer id.
    pub layer: usize,
    /// The absolute token position.
    pub position: usize,
    /// The kernel named by the stage map.
    pub kernel: &'static str,
}

/// A launcher failure. Every variant is typed so the wiring says what failed instead of treating a
/// refusal as a successful stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceLaunchError {
    /// The launcher has no verified cubin for this kernel.
    MissingCubin {
        /// The kernel the stage named.
        kernel: &'static str,
    },
    /// The launcher's own failure (driver, load, launch or an injected test failure).
    Launcher {
        /// Human-readable detail.
        detail: String,
    },
    /// The cubin is present but the stage's buffers/weights are not staged yet.
    NotStaged {
        /// The kernel the stage named.
        kernel: &'static str,
    },
    /// The stage is measured but not wired into the chain.
    NotWired {
        /// The kernel the stage named.
        kernel: &'static str,
    },
    /// This binary was built without the cuda feature.
    CudaUnavailable,
}

impl fmt::Display for DeviceLaunchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCubin { kernel } => {
                write!(formatter, "no verified cubin for kernel {kernel}")
            }
            Self::Launcher { detail } => write!(formatter, "launcher failed: {detail}"),
            Self::NotStaged { kernel } => write!(
                formatter,
                "kernel {kernel} is verified but its device buffers/weights are not staged (M3-b)"
            ),
            Self::NotWired { kernel } => write!(
                formatter,
                "kernel {kernel} is measured but not wired into the chain"
            ),
            Self::CudaUnavailable => write!(
                formatter,
                "the device chain was built without the cuda feature; no device stage can run"
            ),
        }
    }
}

impl std::error::Error for DeviceLaunchError {}

/// The injected device side of the wiring.
///
/// The fake implementation records calls and can inject failures; the real implementation is
/// CudaLauncher behind the cuda feature. The trait deliberately has no host-math method.
pub trait DeviceLauncher {
    /// Runs one device stage.
    ///
    /// # Errors
    /// A DeviceLaunchError when the cubin, buffer set, driver or launch cannot proceed.
    fn launch(&mut self, call: DeviceStageCall) -> Result<(), DeviceLaunchError>;
}

/// The default host-seam delegate: records a seat, not host arithmetic.
///
/// M3-a prints plan accounting; M3-c supplies the real host arithmetic. A seat has zero host bytes
/// because no host computation or transfer happened, and its note says so. The ledger still checks
/// the name, kind, count and phase-3 removal plan, which is what the plan can prove without weights.
pub trait HostSeamDelegate {
    /// Records one host-seam stage.
    fn record(&mut self, call: StageCall, entry: &SeamPlanEntry) -> SeamRecord;
}

/// The M3-a seat recorder used by the dry-run and by tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct SeatHostSeams;

impl HostSeamDelegate for SeatHostSeams {
    fn record(&mut self, _call: StageCall, entry: &SeamPlanEntry) -> SeamRecord {
        SeamRecord {
            name: entry.name,
            kind: entry.kind,
            host_bytes_in: 0,
            host_bytes_out: 0,
            transfer_bytes: 0,
            phase3_followup: !entry.removal.is_empty(),
            note: "M3 wiring host-seam seat: zero host bytes accounted; host arithmetic is M3-c",
        }
    }
}

/// A launcher used by tests and device-chain --dry-run.
///
/// It records every call, returns success by default, and can inject either a missing cubin or a
/// launcher error. It intentionally has no cubin directory: the dry-run proves dispatch and ledger
/// plumbing, not the pinned bytes.
#[derive(Debug, Default, Clone)]
pub struct FakeLauncher {
    calls: Vec<DeviceStageCall>,
    missing_cubins: BTreeSet<&'static str>,
    fail_at: Option<(ChainStage, usize, usize)>,
    fail_detail: String,
}

impl FakeLauncher {
    /// The calls the chain made, in order.
    #[must_use]
    pub fn calls(&self) -> &[DeviceStageCall] {
        &self.calls
    }

    /// Makes every call for kernel fail as a missing cubin.
    pub fn inject_missing_cubin(&mut self, kernel: &'static str) {
        self.missing_cubins.insert(kernel);
    }

    /// Makes the call at (stage, layer, position) fail with a launcher error.
    pub fn inject_launcher_error(
        &mut self,
        stage: ChainStage,
        layer: usize,
        position: usize,
        detail: impl Into<String>,
    ) {
        self.fail_at = Some((stage, layer, position));
        self.fail_detail = detail.into();
    }
}

impl DeviceLauncher for FakeLauncher {
    fn launch(&mut self, call: DeviceStageCall) -> Result<(), DeviceLaunchError> {
        self.calls.push(call);
        if self.missing_cubins.contains(call.kernel) {
            return Err(DeviceLaunchError::MissingCubin {
                kernel: call.kernel,
            });
        }
        if self.fail_at == Some((call.stage, call.layer, call.position)) {
            return Err(DeviceLaunchError::Launcher {
                detail: self.fail_detail.clone(),
            });
        }
        Ok(())
    }
}

/// A launcher for a build without cuda: every device stage refuses, with a reason.
///
/// This is the explicit refusal path required by the wiring contract. It is not a host fallback and
/// it is not selected by the dry-run, which uses FakeLauncher and says so.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoCudaLauncher;

impl DeviceLauncher for NoCudaLauncher {
    fn launch(&mut self, _call: DeviceStageCall) -> Result<(), DeviceLaunchError> {
        Err(DeviceLaunchError::CudaUnavailable)
    }
}

/// One stage's call count in the wiring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainDeviceStats {
    /// All ChainOps::stage calls, host and device.
    pub total_calls: usize,
    /// Host-seam stages delegated to HostSeamDelegate.
    pub host_calls: usize,
    /// Device stages delegated to DeviceLauncher.
    pub device_calls: usize,
    /// Per-stage call counts, in first-call order.
    pub per_stage: Vec<(&'static str, usize)>,
}

/// The wiring: the one ChainOps implementation a device run uses.
pub struct ChainDevice<L: DeviceLauncher, H: HostSeamDelegate> {
    plan: ChainSeamPlan,
    launcher: L,
    host: H,
    stage_counts: Vec<(&'static str, usize)>,
    host_calls: usize,
    device_calls: usize,
}

impl<L: DeviceLauncher, H: HostSeamDelegate> ChainDevice<L, H> {
    /// Builds the wiring for a schedule. It touches no device and stages nothing.
    #[must_use]
    pub fn new(schedule: &ChainSchedule, launcher: L, host: H) -> Self {
        Self {
            plan: ChainSeamPlan::for_schedule(schedule),
            launcher,
            host,
            stage_counts: Vec::new(),
            host_calls: 0,
            device_calls: 0,
        }
    }

    /// The call counts after a run (or during one).
    #[must_use]
    pub fn stats(&self) -> ChainDeviceStats {
        ChainDeviceStats {
            total_calls: self.host_calls + self.device_calls,
            host_calls: self.host_calls,
            device_calls: self.device_calls,
            per_stage: self.stage_counts.clone(),
        }
    }

    /// The launcher, for a dry-run's recorded calls.
    #[must_use]
    pub fn launcher(&self) -> &L {
        &self.launcher
    }

    /// The launcher, for the M3-b stage path that writes one position before each launch.
    pub fn launcher_mut(&mut self) -> &mut L {
        &mut self.launcher
    }

    fn count(&mut self, stage: ChainStage) {
        let name = stage.name();
        if let Some((_, count)) = self
            .stage_counts
            .iter_mut()
            .find(|(existing, _)| *existing == name)
        {
            *count += 1;
        } else {
            self.stage_counts.push((name, 1));
        }
    }

    fn device_call(call: StageCall, kernel: &'static str) -> DeviceStageCall {
        DeviceStageCall {
            stage: call.stage,
            layer: call.layer,
            position: call.position,
            kernel,
        }
    }
}

impl<L: DeviceLauncher, H: HostSeamDelegate> ChainOps for ChainDevice<L, H> {
    fn stage(&mut self, call: StageCall) -> Result<Option<SeamRecord>, ChainError> {
        self.count(call.stage);
        match call.stage.status() {
            StageStatus::HostSeam(seam) => {
                self.host_calls += 1;
                let entry = self.plan.entry(seam).ok_or(ChainError::UnplannedSeam {
                    name: seam,
                    found: 1,
                })?;
                Ok(Some(self.host.record(call, entry)))
            }
            StageStatus::DeviceKernel(kernel)
            | StageStatus::DeviceMeasuredNotWired { kernel, .. } => {
                self.device_calls += 1;
                self.launcher
                    .launch(Self::device_call(call, kernel))
                    .map_err(|error| ChainError::DeviceStage {
                        stage: call.stage,
                        layer: call.layer,
                        position: call.position,
                        detail: error.to_string(),
                    })?;
                Ok(None)
            }
        }
    }
}

/// Runs the full plan through the wiring and returns the report plus the live wiring.
///
/// # Errors
/// Any ChainError the wiring or the ledger self-check returns.
pub fn run_chain<L: DeviceLauncher, H: HostSeamDelegate>(
    schedule: &ChainSchedule,
    launcher: L,
    host: H,
) -> Result<(ChainRunReport, ChainDevice<L, H>), ChainError> {
    let mut device = ChainDevice::new(schedule, launcher, host);
    let report = run_skeleton(schedule, &mut device)?;
    Ok((report, device))
}

/// The dry-run result: the validated report, the wiring statistics and the fake launcher's calls.
#[derive(Debug, Clone)]
pub struct DryRunReport {
    /// The report whose ledger the plan-vs-ledger self-check validated.
    pub report: ChainRunReport,
    /// Total and per-stage call counts.
    pub stats: ChainDeviceStats,
    /// Every device-stage call the fake launcher received, in order.
    pub device_calls: Vec<DeviceStageCall>,
}

/// Runs the full 130 + 8 plan through ChainDevice with FakeLauncher and SeatHostSeams.
///
/// This is the dry-run / test path: no driver, no cubin, no allocation and no launch. It still runs
/// every stage through the real dispatch and the real plan-vs-ledger self-check.
///
/// # Errors
/// A wiring error or a failed ledger self-check. The fake launcher is configured to succeed.
pub fn dry_run(schedule: &ChainSchedule) -> Result<DryRunReport, ChainError> {
    let (report, device) = run_chain(schedule, FakeLauncher::default(), SeatHostSeams)?;
    Ok(DryRunReport {
        report,
        stats: device.stats(),
        device_calls: device.launcher().calls().to_vec(),
    })
}

/// The host-seam seat counts, by seam name, from a validated ledger.
#[must_use]
pub fn host_seat_counts(report: &ChainRunReport) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for record in report.ledger.records() {
        *counts.entry(record.name).or_insert(0) += 1;
    }
    counts
}

// The real launcher is compiled only with the feature. M3-b stages the layer-0 buffers and exposes
// one cuda-only CLI stage arm; keeping it behind the feature means the default binary has no device
// path at all.
#[cfg(feature = "cuda")]
mod cuda_launcher {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
    use sparkflash_kernels::mega_mhc_launch::{self, MegaMhcParams};

    use super::{ChainStage, DeviceLaunchError, DeviceLauncher, DeviceStageCall};
    use crate::cubins::{cubin_path, required_pins, verify_bytes};
    use sparkflash_kernels::baseline::KernelPin;

    use sparkflash_kernels::m4_quant::bf16_bits;
    use sparkflash_kernels::mega_mhc_ref::{HC_MULT, HIDDEN};

    use crate::budget::{BudgetAdmission, BudgetError, BudgetRequest};
    use crate::staging::{
        EmbedRow, LayerPostStaging, LayerStaging, LayerZeroStaging, MhcBufferPlan,
    };
    use sparkflash_engine::forward::layer_zero::LayerZeroEntry;

    /// Builds the layer-zero host handoff from the mHC kernel's f32 readback buffers.
    ///
    /// The kernel writes `hc_coeffs` as B (n*n, destination-major), A (n, the input/pre mixes) and
    /// C (n, the output/post mixes). `Mixes` (and therefore [LayerZeroEntry]) stores comb as
    /// source-major, so the 4x4 block is transposed here; the A and C slices copy in order.
    fn entry_from_readback(xhat_normed: &[u8], hc_coeffs: &[u8]) -> Result<LayerZeroEntry, String> {
        let expected_capture = HIDDEN * 4;
        let expected_coeffs = 24 * 4;
        if xhat_normed.len() != expected_capture {
            return Err(format!(
                "the mHC xhat_normed readback is {} bytes, expected {expected_capture}",
                xhat_normed.len()
            ));
        }
        if hc_coeffs.len() != expected_coeffs {
            return Err(format!(
                "the mHC hc_coeffs readback is {} bytes, expected {expected_coeffs}",
                hc_coeffs.len()
            ));
        }
        let f32s = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()
        };
        let capture = f32s(xhat_normed);
        let coeffs = f32s(hc_coeffs);
        let hc = HC_MULT;
        let mut comb = vec![0.0f32; hc * hc];
        for destination in 0..hc {
            for source in 0..hc {
                comb[source * hc + destination] = coeffs[destination * hc + source];
            }
        }
        Ok(LayerZeroEntry {
            pre_attention_hidden: capture,
            pre: coeffs[hc * hc..hc * hc + hc].to_vec(),
            post: coeffs[hc * hc + hc..hc * hc + 2 * hc].to_vec(),
            comb,
            source: sparkflash_engine::forward::layer_zero::LayerZeroEntrySource::Device,
        })
    }

    fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn f32_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| bf16_bits(*value).to_le_bytes())
            .collect()
    }

    /// The per-launch zeroes the mHC grid rendezvous needs.
    ///
    /// The kernel's tiles meet through atomicAdd(hc_sync[0]) and wait on hc_sync[1]; the statistics
    /// scratch is also per-launch. Stale counters from a previous launch never publish the next
    /// coefficients, so both buffers are re-zeroed before every launch, exactly as m11's run_once
    /// does.
    fn mhc_rendezvous_zeroes(buffers: &MhcBuffers) -> [(Deviceptr, Vec<u8>); 2] {
        let plan = MhcBufferPlan::LAYER_ZERO;
        [
            (buffers.hc_stats, vec![0u8; plan.hc_stats]),
            (buffers.hc_sync, vec![0u8; plan.hc_sync]),
        ]
    }

    fn reset_mhc_rendezvous_with(
        buffers: &MhcBuffers,
        mut copy: impl FnMut(&Deviceptr, &[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        for (destination, zeroes) in mhc_rendezvous_zeroes(buffers) {
            copy(&destination, &zeroes)?;
        }
        Ok(())
    }

    /// The single entry launch sequence: reset the grid rendezvous, then launch the kernel.
    ///
    /// This is the function [DeviceLauncher::launch] calls, so the reset and its byte sizes are
    /// tested through the same seam the real launch uses.
    fn launch_mhc_entry_with(
        buffers: &MhcBuffers,
        mut copy: impl FnMut(&Deviceptr, &[u8]) -> Result<(), String>,
        mut launch: impl FnMut() -> Result<(), String>,
    ) -> Result<(), String> {
        reset_mhc_rendezvous_with(buffers, |destination, zeroes| copy(destination, zeroes))?;
        launch()
    }

    /// A typed failure from staging the layer-0 mHC buffers or from its admission.
    #[derive(Debug, thiserror::Error)]
    pub enum MhcStageError {
        /// The budget admission refused before any device allocation.
        #[error(transparent)]
        Budget(#[from] BudgetError),
        /// The driver or a staging copy failed.
        #[error("device staging failed: {detail}")]
        Launcher {
            /// The driver or copy detail.
            detail: String,
        },
    }

    /// The sixteen device pointers one mHC launch needs, in the kernel declaration order.
    ///
    /// Only [CudaLauncher::stage_layer_zero]'s own allocation set constructs this type: the fields
    /// are private and there is no public setter, so a caller cannot inject a null, partial or
    /// wrong-order pointer set into a launch.
    #[derive(Debug, Clone, Copy)]
    pub struct MhcBuffers {
        x: Deviceptr,
        y_prev: Deviceptr,
        b_prev: Deviceptr,
        c_prev: Deviceptr,
        a_prev: Deviceptr,
        predictor: Deviceptr,
        hc_scale: Deviceptr,
        hc_base: Deviceptr,
        rms_weight: Deviceptr,
        xhat_fp8: Deviceptr,
        xhat_scales: Deviceptr,
        hc_stats: Deviceptr,
        hc_sync: Deviceptr,
        hc_coeffs: Deviceptr,
        hc_rms_out: Deviceptr,
        xhat_normed: Deviceptr,
    }

    impl MhcBuffers {
        fn from_allocations(allocated: &[Deviceptr]) -> Self {
            Self {
                x: allocated[0],
                y_prev: allocated[1],
                b_prev: allocated[2],
                c_prev: allocated[3],
                a_prev: allocated[4],
                predictor: allocated[5],
                hc_scale: allocated[6],
                hc_base: allocated[7],
                rms_weight: allocated[8],
                xhat_fp8: allocated[9],
                xhat_scales: allocated[10],
                hc_stats: allocated[11],
                hc_sync: allocated[12],
                hc_coeffs: allocated[13],
                hc_rms_out: allocated[14],
                xhat_normed: allocated[15],
            }
        }
    }

    /// Copies one layer's predictor/base/scale/norm and resets the boundary seed and the grid
    /// rendezvous. The allocation itself is the caller's; this is the one copy sequence both
    /// [CudaLauncher::stage_layer] and [CudaLauncher::restage_layer] use.
    fn copy_layer_tensors(
        cuda: &Cuda,
        buffers: &MhcBuffers,
        staging: &LayerStaging,
    ) -> Result<(), String> {
        let plan = MhcBufferPlan::LAYER_ZERO;
        let (b, c, a, y) = LayerZeroStaging::layer_zero_seed();
        let y_bytes = f32_to_bf16_bytes(&y);
        cuda.copy_htod(&buffers.x, &vec![0u8; plan.x])?;
        cuda.copy_htod(&buffers.y_prev, &y_bytes)?;
        cuda.copy_htod(&buffers.b_prev, &f32_to_bytes(&b))?;
        cuda.copy_htod(&buffers.c_prev, &f32_to_bytes(&c))?;
        cuda.copy_htod(&buffers.a_prev, &f32_to_bytes(&a))?;
        cuda.copy_htod(&buffers.predictor, &f32_to_bytes(&staging.predictor))?;
        cuda.copy_htod(&buffers.hc_scale, &f32_to_bytes(&staging.hc_scale))?;
        cuda.copy_htod(&buffers.hc_base, &f32_to_bytes(&staging.hc_base_kernel))?;
        cuda.copy_htod(&buffers.rms_weight, &f32_to_bytes(&staging.attn_norm))?;
        reset_mhc_rendezvous_with(buffers, |destination, zeroes| {
            cuda.copy_htod(destination, zeroes)
        })?;
        Ok(())
    }
    /// The real launcher: verifies the required cubins, then dispatches the first typed builder
    /// (mHC) when its buffers are staged.
    ///
    /// Construction reads each pinned cubin exactly once and verifies that same slice with
    /// [crate::cubins::verify_bytes], so the loaded bytes are the verified bytes (there is no
    /// verify-then-re-read window). A missing or mis-pinned cubin is refused before any driver is
    /// opened. It opens no driver until a launch is actually attempted; M3-b's
    /// [CudaLauncher::stage_layer_zero] is the path that allocates.
    pub struct CudaLauncher {
        cuda: Option<Cuda>,
        cubins: BTreeMap<String, Vec<u8>>,
        mhc_buffers: Option<MhcBuffers>,
        mhc_allocations: Vec<Deviceptr>,
    }

    /// Reads each pinned cubin once, verifies that exact slice, and returns it as the loaded catalog.
    ///
    /// The read result is both the verified object and the object that is stored, so a file changed
    /// between a verify pass and a load pass cannot be launched.
    fn load_verified_cubins(
        dir: &Path,
        pins: &[KernelPin],
    ) -> Result<BTreeMap<String, Vec<u8>>, String> {
        let mut cubins = BTreeMap::new();
        for pin in pins {
            let path: PathBuf = cubin_path(dir, &pin.name);
            let bytes =
                std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
            verify_bytes(&bytes, pin)?;
            cubins.insert(pin.name.clone(), bytes);
        }
        Ok(cubins)
    }

    impl CudaLauncher {
        /// Verifies a cubin directory against every required pin and loads its bytes.
        ///
        /// # Errors
        /// A required-pin catalog error, a cubin read error, or a size/digest mismatch.
        pub fn open(dir: impl AsRef<Path>) -> Result<Self, String> {
            let pins = required_pins()?;
            Self::from_pins(dir.as_ref(), &pins)
        }

        /// Verifies and loads only one stage's pinned cubin.
        ///
        /// This is the M3-b layer-0 path: the mHC entry does not need the attention, MoE or head
        /// cubins to be present, and a missing or mis-pinned mHC cubin is still refused before any
        /// driver or allocation.
        ///
        /// # Errors
        /// The kernel is not in the baseline catalog, or its read/size/digest check fails.
        pub fn open_for_stage(dir: impl AsRef<Path>, kernel: &str) -> Result<Self, String> {
            let pin = sparkflash_kernels::baseline::pin(kernel)?.ok_or_else(|| {
                format!("kernel {kernel} is required by the stage but the baseline does not pin it")
            })?;
            Self::from_pins(dir.as_ref(), &[pin])
        }

        fn from_pins(dir: &Path, pins: &[KernelPin]) -> Result<Self, String> {
            let cubins = load_verified_cubins(dir, pins)?;
            Ok(Self {
                cuda: None,
                cubins,
                mhc_buffers: None,
                mhc_allocations: Vec::new(),
            })
        }

        /// A launcher with no cubins or driver, used only to prove the admission refusal path.
        #[cfg(test)]
        fn for_test() -> Self {
            Self {
                cuda: None,
                cubins: BTreeMap::new(),
                mhc_buffers: None,
                mhc_allocations: Vec::new(),
            }
        }

        /// Admits the budget first, then validates the staging and its projection and allocates the
        /// layer-0 mHC buffers.
        ///
        /// The admission is the first statement: a refused budget returns before every validation
        /// that can cost device work, before Cuda::load, before the first alloc and before any copy.
        /// The projected device bytes must equal [MhcBufferPlan::LAYER_ZERO]'s exact byte total, and
        /// the projected host bytes must cover the retained staging plus the verified cubins; a caller
        /// cannot understate the stage and still allocate. A driver or copy failure frees every
        /// allocation made by this call and leaves mhc_buffers unset.
        ///
        /// # Errors
        /// [MhcStageError::Budget] on refusal, [MhcStageError::Launcher] on a projection, driver or
        /// copy failure.
        pub fn stage_layer_zero(
            &mut self,
            staging: &LayerZeroStaging,
            request: BudgetRequest,
        ) -> Result<BudgetAdmission, MhcStageError> {
            let admission = BudgetAdmission::admit(request)?;
            let plan = MhcBufferPlan::LAYER_ZERO;
            let expected_device = plan.device_bytes() as u64;
            if request.projected_device_bytes != expected_device {
                return Err(MhcStageError::Launcher {
                    detail: format!(
                        "projected device bytes {} do not match the mHC buffer plan's {} bytes",
                        request.projected_device_bytes, expected_device
                    ),
                });
            }
            let expected_host = staging.host_bytes().saturating_add(self.cubin_bytes());
            if request.projected_host_bytes < expected_host {
                return Err(MhcStageError::Launcher {
                    detail: format!(
                        "projected host bytes {} do not cover the staged {} staging bytes plus the {} verified cubin bytes",
                        request.projected_host_bytes,
                        staging.host_bytes(),
                        self.cubin_bytes()
                    ),
                });
            }
            if staging.predictor.len() * 4 != plan.predictor
                || staging.hc_scale.len() * 4 != plan.hc_scale
                || staging.hc_base_kernel.len() * 4 != plan.hc_base
                || staging.attn_norm.len() * 4 != plan.rms_weight
            {
                return Err(MhcStageError::Launcher {
                    detail: format!(
                        "staged tensor sizes do not match the mHC buffer plan: predictor {}B/{}B, scale {}B/{}B, base {}B/{}B, norm {}B/{}B",
                        staging.predictor.len() * 4,
                        plan.predictor,
                        staging.hc_scale.len() * 4,
                        plan.hc_scale,
                        staging.hc_base_kernel.len() * 4,
                        plan.hc_base,
                        staging.attn_norm.len() * 4,
                        plan.rms_weight
                    ),
                });
            }
            if self.mhc_buffers.is_some() {
                return Err(MhcStageError::Launcher {
                    detail: "the layer-0 mHC buffers are already staged".to_string(),
                });
            }
            if self.cuda.is_none() {
                self.cuda =
                    Some(Cuda::load().map_err(|error| MhcStageError::Launcher { detail: error })?);
            }
            let cuda = self.cuda.as_ref().ok_or(MhcStageError::Launcher {
                detail: "the CUDA driver is unavailable".to_string(),
            })?;
            let mut allocated: Vec<Deviceptr> = Vec::with_capacity(plan.sizes().len());
            let mut total = 0usize;
            let result = (|| -> Result<(), String> {
                for (_name, bytes) in plan.sizes() {
                    let pointer = cuda.alloc(bytes, &mut total)?;
                    allocated.push(pointer);
                }
                let buffers = MhcBuffers::from_allocations(&allocated);
                let (b, c, a, y) = LayerZeroStaging::layer_zero_seed();
                let y_bytes = f32_to_bf16_bytes(&y);
                cuda.copy_htod(&buffers.x, &vec![0u8; plan.x])?;
                cuda.copy_htod(&buffers.y_prev, &y_bytes)?;
                cuda.copy_htod(&buffers.b_prev, &f32_to_bytes(&b))?;
                cuda.copy_htod(&buffers.c_prev, &f32_to_bytes(&c))?;
                cuda.copy_htod(&buffers.a_prev, &f32_to_bytes(&a))?;
                cuda.copy_htod(&buffers.predictor, &f32_to_bytes(&staging.predictor))?;
                cuda.copy_htod(&buffers.hc_scale, &f32_to_bytes(&staging.hc_scale))?;
                cuda.copy_htod(&buffers.hc_base, &f32_to_bytes(&staging.hc_base_kernel))?;
                cuda.copy_htod(&buffers.rms_weight, &f32_to_bytes(&staging.attn_norm))?;
                reset_mhc_rendezvous_with(&buffers, |destination, zeroes| {
                    cuda.copy_htod(destination, zeroes)
                })?;
                Ok(())
            })();
            if let Err(detail) = result {
                let refs: Vec<&Deviceptr> = allocated.iter().collect();
                cuda.free_buffers(&refs);
                return Err(MhcStageError::Launcher { detail });
            }
            self.mhc_buffers = Some(MhcBuffers::from_allocations(&allocated));
            self.mhc_allocations = allocated;
            Ok(admission)
        }

        /// Allocates the sixteen mHC buffers for one backbone layer, then copies that layer's
        /// released predictor/base/scale/norm into them. The first layer of a chain calls this;
        /// later layers call [Self::restage_layer] so the allocation is reused.
        ///
        /// # Errors
        /// [MhcStageError::Budget] on refusal, [MhcStageError::Launcher] on a projection, driver or
        /// copy failure.
        pub fn stage_layer(
            &mut self,
            staging: &LayerStaging,
            request: BudgetRequest,
        ) -> Result<BudgetAdmission, MhcStageError> {
            let admission = BudgetAdmission::admit(request)?;
            let plan = MhcBufferPlan::LAYER_ZERO;
            let expected_device = plan.device_bytes() as u64;
            if request.projected_device_bytes != expected_device {
                return Err(MhcStageError::Launcher {
                    detail: format!(
                        "projected device bytes {} do not match the mHC buffer plan's {} bytes",
                        request.projected_device_bytes, expected_device
                    ),
                });
            }
            let expected_host = staging.host_bytes().saturating_add(self.cubin_bytes());
            if request.projected_host_bytes < expected_host {
                return Err(MhcStageError::Launcher {
                    detail: format!(
                        "projected host bytes {} do not cover the staged {} staging bytes plus the {} verified cubin bytes",
                        request.projected_host_bytes,
                        staging.host_bytes(),
                        self.cubin_bytes()
                    ),
                });
            }
            if staging.predictor.len() * 4 != plan.predictor
                || staging.hc_scale.len() * 4 != plan.hc_scale
                || staging.hc_base_kernel.len() * 4 != plan.hc_base
                || staging.attn_norm.len() * 4 != plan.rms_weight
            {
                return Err(MhcStageError::Launcher {
                    detail: "staged tensor sizes do not match the mHC buffer plan".to_string(),
                });
            }
            if self.mhc_buffers.is_some() {
                return Err(MhcStageError::Launcher {
                    detail: "the mHC buffers are already staged; use restage_layer".to_string(),
                });
            }
            if self.cuda.is_none() {
                self.cuda =
                    Some(Cuda::load().map_err(|error| MhcStageError::Launcher { detail: error })?);
            }
            let cuda = self.cuda.as_ref().ok_or(MhcStageError::Launcher {
                detail: "the CUDA driver is unavailable".to_string(),
            })?;
            let mut allocated: Vec<Deviceptr> = Vec::with_capacity(plan.sizes().len());
            let mut total = 0usize;
            let result = (|| -> Result<(), String> {
                for (_name, bytes) in plan.sizes() {
                    let pointer = cuda.alloc(bytes, &mut total)?;
                    allocated.push(pointer);
                }
                let buffers = MhcBuffers::from_allocations(&allocated);
                copy_layer_tensors(cuda, &buffers, staging)
            })();
            if let Err(detail) = result {
                let refs: Vec<&Deviceptr> = allocated.iter().collect();
                cuda.free_buffers(&refs);
                return Err(MhcStageError::Launcher { detail });
            }
            self.mhc_buffers = Some(MhcBuffers::from_allocations(&allocated));
            self.mhc_allocations = allocated;
            Ok(admission)
        }

        /// Copies a new layer's released mHC tensors into the already-allocated buffer set.
        ///
        /// # Errors
        /// No staged buffers, a driver that is unavailable, or a copy failure.
        pub fn restage_layer(&mut self, staging: &LayerStaging) -> Result<(), String> {
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the mHC buffers are not staged; call stage_layer first".to_string()
            })?;
            let cuda = self.cuda.as_ref().ok_or_else(|| {
                "the CUDA driver is unavailable while restaging a layer".to_string()
            })?;
            copy_layer_tensors(cuda, &buffers, staging)
        }

        /// Copies one layer's released FFN mHC tensors into the already-allocated buffer set,
        /// leaving X, Y, B, C and A in place. The c3b post call follows an entry call that already
        /// wrote X and the attention output, so resetting the boundary seed here would destroy the
        /// post operand; only the predictor/scale/base/norm are replaced.
        ///
        /// # Errors
        /// No staged buffers, driver unavailable, or a copy failure.
        pub fn restage_layer_post(&mut self, staging: &LayerPostStaging) -> Result<(), String> {
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the mHC buffers are not staged; call stage_layer first".to_string()
            })?;
            let cuda = self.cuda.as_ref().ok_or_else(|| {
                "the CUDA driver is unavailable while restaging a post layer".to_string()
            })?;
            cuda.copy_htod(&buffers.predictor, &f32_to_bytes(&staging.predictor))?;
            cuda.copy_htod(&buffers.hc_scale, &f32_to_bytes(&staging.hc_scale))?;
            cuda.copy_htod(&buffers.hc_base, &f32_to_bytes(&staging.hc_base_kernel))?;
            cuda.copy_htod(&buffers.rms_weight, &f32_to_bytes(&staging.ffn_norm))?;
            Ok(())
        }

        /// Writes the previous sublayer's B (comb, destination-major in the kernel) and C (post)
        /// coefficients for the post call. `comb` is the host source-major orientation; it is
        /// transposed here to the kernel's destination-major layout.
        ///
        /// # Errors
        /// No staged buffers, driver unavailable, or the wrong coefficient lengths.
        pub fn write_layer_bc(&mut self, post: &[f32], comb: &[f32]) -> Result<(), String> {
            if post.len() != HC_MULT || comb.len() != HC_MULT * HC_MULT {
                return Err(format!(
                    "the B/C rows have {} C and {} B values, expected {HC_MULT} and {}",
                    post.len(),
                    comb.len(),
                    HC_MULT * HC_MULT
                ));
            }
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the mHC buffers are not staged, so B/C cannot be written".to_string()
            })?;
            let cuda = self
                .cuda
                .as_ref()
                .ok_or_else(|| "the CUDA driver is unavailable while copying B/C".to_string())?;
            let mut b = vec![0.0f32; HC_MULT * HC_MULT];
            for destination in 0..HC_MULT {
                for source in 0..HC_MULT {
                    b[destination * HC_MULT + source] = comb[source * HC_MULT + destination];
                }
            }
            cuda.copy_htod(&buffers.b_prev, &f32_to_bytes(&b))?;
            cuda.copy_htod(&buffers.c_prev, &f32_to_bytes(post))
        }

        /// Writes the previous sublayer's block output Y as BF16, the format the kernel's y_prev
        /// buffer holds.
        ///
        /// # Errors
        /// No staged buffers, driver unavailable, or a length other than `HIDDEN * 2`.
        pub fn write_layer_y_bf16(&mut self, y: &[u8]) -> Result<(), String> {
            if y.len() != HIDDEN * 2 {
                return Err(format!(
                    "the Y row is {} bytes, the staged buffer expects {}",
                    y.len(),
                    HIDDEN * 2
                ));
            }
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the mHC buffers are not staged, so Y cannot be written".to_string()
            })?;
            let cuda = self
                .cuda
                .as_ref()
                .ok_or_else(|| "the CUDA driver is unavailable while copying Y".to_string())?;
            cuda.copy_htod(&buffers.y_prev, y)
                .map_err(|error| format!("copy the Y row: {error}"))
        }

        /// Writes the BF16 X stream for one layer into the staged X buffer.
        ///
        /// c3 builds the bytes from the previous layer's host residual, so the caller owns the
        /// BF16 staging seam and its digest; this method only enforces the exact buffer length.
        ///
        /// # Errors
        /// No staged buffers, driver unavailable, or a length other than `HC_MULT * HIDDEN * 2`.
        pub fn write_layer_x_bf16(&mut self, x: &[u8]) -> Result<(), String> {
            if x.len() != HC_MULT * HIDDEN * 2 {
                return Err(format!(
                    "the X stream is {} bytes, the staged buffer expects {}",
                    x.len(),
                    HC_MULT * HIDDEN * 2
                ));
            }
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the mHC buffers are not staged, so X cannot be written".to_string()
            })?;
            let cuda = self
                .cuda
                .as_ref()
                .ok_or_else(|| "the CUDA driver is unavailable while copying X".to_string())?;
            cuda.copy_htod(&buffers.x, x)
                .map_err(|error| format!("copy the X stream: {error}"))
        }

        /// Writes the previous sublayer's pre-mix coefficients as the kernel's A row for this
        /// layer. c3's step 1 is identity, but step 2 is still H(A X_l) with the previous FFN
        /// pre-mix, so A must be restaged per layer rather than left at the layer-0 seed.
        ///
        /// # Errors
        /// No staged buffers, driver unavailable, or a length other than `HC_MULT`.
        pub fn write_layer_a(&mut self, a: &[f32]) -> Result<(), String> {
            if a.len() != HC_MULT {
                return Err(format!(
                    "the A row has {} values, the staged buffer expects {HC_MULT}",
                    a.len()
                ));
            }
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the mHC buffers are not staged, so A cannot be written".to_string()
            })?;
            let cuda = self
                .cuda
                .as_ref()
                .ok_or_else(|| "the CUDA driver is unavailable while copying A".to_string())?;
            cuda.copy_htod(&buffers.a_prev, &f32_to_bytes(a))
                .map_err(|error| format!("copy the A row: {error}"))
        }

        /// The layer-generic aliases for the c3 readbacks.
        ///
        /// # Errors
        /// As the layer-0 readback each one delegates to.
        pub fn read_layer_entry(&self) -> Result<LayerZeroEntry, String> {
            self.read_layer_zero_entry()
        }

        /// # Errors
        /// As [Self::read_layer_zero_x].
        pub fn read_layer_x(&self) -> Result<Vec<u8>, String> {
            self.read_layer_zero_x()
        }

        /// # Errors
        /// As [Self::read_layer_zero_inv_rms].
        pub fn read_layer_inv_rms(&self) -> Result<f32, String> {
            self.read_layer_zero_inv_rms()
        }

        /// # Errors
        /// As [Self::read_layer_zero_codes].
        pub fn read_layer_codes(&self) -> Result<Vec<u8>, String> {
            self.read_layer_zero_codes()
        }

        /// # Errors
        /// As [Self::read_layer_zero_scales].
        pub fn read_layer_scales(&self) -> Result<Vec<u8>, String> {
            self.read_layer_zero_scales()
        }

        /// Copies the staged embedding row's four BF16 copies into the staged X buffer.
        ///
        /// The bytes are built here from an admitted [EmbedRow]. Its fields are private and its
        /// only constructor is crate-private, so an external caller cannot supply arbitrary X
        /// bytes whose digest the stage report did not record.
        ///
        /// # Errors
        /// No staged buffers, a driver that is unavailable, or a row that is not HIDDEN BF16
        /// values (the four-copy X build is always HC_MULT * HIDDEN BF16 values).
        pub fn write_layer_zero_x(&mut self, row: &EmbedRow) -> Result<(), String> {
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the layer-0 mHC buffers are not staged; call stage_layer_zero first".to_string()
            })?;
            let x = row.x_bf16();
            if x.len() != HC_MULT * HIDDEN * 2 {
                return Err(format!(
                    "the X row is {} bytes, the staged buffer expects {}",
                    x.len(),
                    HC_MULT * HIDDEN * 2
                ));
            }
            let cuda = self.cuda.as_ref().ok_or_else(|| {
                "the CUDA driver is unavailable while copying the X row".to_string()
            })?;
            cuda.copy_htod(&buffers.x, &x)
                .map_err(|error| format!("copy the X row: {error}"))
        }

        /// Reads the layer-0 mHC entry handoff back to the host after a completed launch.
        ///
        /// The host trace continues from this capture and its coefficients; the values are the
        /// kernel's f32 `xhat_normed` and `hc_coeffs` buffers. The readback does not by itself make
        /// a chain or token claim.
        ///
        /// # Errors
        /// No staged buffers, no driver, or a readback length that does not match the plan.
        pub fn read_layer_zero_entry(&self) -> Result<LayerZeroEntry, String> {
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the layer-0 mHC buffers are not staged, so there is no entry handoff".to_string()
            })?;
            let cuda = self.cuda.as_ref().ok_or_else(|| {
                "the CUDA driver is unavailable, so the mHC entry cannot be read back".to_string()
            })?;
            let plan = MhcBufferPlan::LAYER_ZERO;
            let mut capture = vec![0u8; plan.xhat_normed];
            let mut coefficients = vec![0u8; plan.hc_coeffs];
            cuda.copy_dtoh(&mut capture, &buffers.xhat_normed)?;
            cuda.copy_dtoh(&mut coefficients, &buffers.hc_coeffs)?;
            entry_from_readback(&capture, &coefficients)
        }

        /// Reads back the staged X buffer's bytes after [Self::write_layer_zero_x].
        ///
        /// A window diagnostic: the caller can compare these bytes with the admitted row's
        /// `x_bf16` seed and prove the device row is the row the host trace used. It does not make
        /// a chain or token claim.
        ///
        /// # Errors
        /// No staged buffers, no driver, or a copy error.
        pub fn read_layer_zero_x(&self) -> Result<Vec<u8>, String> {
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the layer-0 mHC buffers are not staged, so X cannot be read back".to_string()
            })?;
            let cuda = self.cuda.as_ref().ok_or_else(|| {
                "the CUDA driver is unavailable, so X cannot be read back".to_string()
            })?;
            let mut x = vec![0u8; MhcBufferPlan::LAYER_ZERO.x];
            cuda.copy_dtoh(&mut x, &buffers.x)?;
            Ok(x)
        }

        /// Reads the kernel's `1/rms(X_hat)` f32 output after a completed launch.
        ///
        /// m11 b7 measured this scalar against the CPU twin with a `3e-4` bar; the window uses it
        /// as a diagnostic next to the normalised capture. It does not make a chain or token claim.
        ///
        /// # Errors
        /// No staged buffers, no driver, a short read, or a non-finite value.
        pub fn read_layer_zero_inv_rms(&self) -> Result<f32, String> {
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the layer-0 mHC buffers are not staged, so 1/rms cannot be read back".to_string()
            })?;
            let cuda = self.cuda.as_ref().ok_or_else(|| {
                "the CUDA driver is unavailable, so 1/rms cannot be read back".to_string()
            })?;
            let mut bytes = [0u8; 4];
            cuda.copy_dtoh(&mut bytes, &buffers.hc_rms_out)?;
            let value = f32::from_le_bytes(bytes);
            if !value.is_finite() {
                return Err(format!(
                    "the mHC 1/rms readback is {value}; the kernel writes a finite scalar"
                ));
            }
            Ok(value)
        }

        /// Reads the kernel's E4M3 codes for the block input, after a completed launch.
        ///
        /// # Errors
        /// No staged buffers, no driver, or a copy error.
        pub fn read_layer_zero_codes(&self) -> Result<Vec<u8>, String> {
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the layer-0 mHC buffers are not staged, so the FP8 codes cannot be read back"
                    .to_string()
            })?;
            let cuda = self.cuda.as_ref().ok_or_else(|| {
                "the CUDA driver is unavailable, so the FP8 codes cannot be read back".to_string()
            })?;
            let mut codes = vec![0u8; MhcBufferPlan::LAYER_ZERO.xhat_fp8];
            cuda.copy_dtoh(&mut codes, &buffers.xhat_fp8)?;
            Ok(codes)
        }

        /// Reads the kernel's E8M0 block scales for the block input, after a completed launch.
        ///
        /// # Errors
        /// No staged buffers, no driver, or a copy error.
        pub fn read_layer_zero_scales(&self) -> Result<Vec<u8>, String> {
            let buffers = self.mhc_buffers.ok_or_else(|| {
                "the layer-0 mHC buffers are not staged, so the FP8 scales cannot be read back"
                    .to_string()
            })?;
            let cuda = self.cuda.as_ref().ok_or_else(|| {
                "the CUDA driver is unavailable, so the FP8 scales cannot be read back".to_string()
            })?;
            let mut scales = vec![0u8; MhcBufferPlan::LAYER_ZERO.xhat_scales];
            cuda.copy_dtoh(&mut scales, &buffers.xhat_scales)?;
            Ok(scales)
        }

        /// The raw bytes of every verified cubin this launcher loaded.
        #[must_use]
        pub fn cubin_bytes(&self) -> u64 {
            self.cubins.values().map(|bytes| bytes.len() as u64).sum()
        }

        /// Whether this launcher verified a cubin with kernel's name.
        #[must_use]
        pub fn has_cubin(&self, kernel: &str) -> bool {
            self.cubins.contains_key(kernel)
        }

        fn mhc_params(&self, buffers: &MhcBuffers, call: DeviceStageCall) -> MegaMhcParams {
            // step_seq is the absolute token position (0..prompt+decode-1) from ChainDevice, not a
            // decoder-step index and not a per-layer counter. The kernel ignores it while
            // trace_ring=0; if a later arm enables tracing it must be read as this absolute position.
            let step_seq = call.position as u64;
            MegaMhcParams::new(
                buffers.x,
                buffers.y_prev,
                buffers.b_prev,
                buffers.c_prev,
                buffers.a_prev,
                buffers.predictor,
                buffers.hc_scale,
                buffers.hc_base,
                buffers.rms_weight,
                buffers.xhat_fp8,
                buffers.xhat_scales,
                buffers.hc_stats,
                buffers.hc_sync,
                buffers.hc_coeffs,
                buffers.hc_rms_out,
                buffers.xhat_normed,
                0,
                u32::try_from(call.layer).unwrap_or(u32::MAX),
                step_seq,
            )
        }
    }

    impl Drop for CudaLauncher {
        fn drop(&mut self) {
            if let Some(cuda) = self.cuda.as_ref() {
                let refs: Vec<&Deviceptr> = self.mhc_allocations.iter().collect();
                cuda.free_buffers(&refs);
            }
        }
    }

    impl DeviceLauncher for CudaLauncher {
        fn launch(&mut self, call: DeviceStageCall) -> Result<(), DeviceLaunchError> {
            // The post call reuses the mHC buffer set, but only after an entry call has allocated
            // and staged it. Without that allocation it is refused before any driver work.
            if call.stage == ChainStage::MhcPost && self.mhc_buffers.is_none() {
                return Err(DeviceLaunchError::NotStaged {
                    kernel: call.kernel,
                });
            }
            let bytes =
                self.cubins
                    .get(call.kernel)
                    .cloned()
                    .ok_or(DeviceLaunchError::MissingCubin {
                        kernel: call.kernel,
                    })?;
            match call.stage {
                ChainStage::MhcEntry | ChainStage::MhcPost => {
                    let buffers = self.mhc_buffers.ok_or(DeviceLaunchError::NotStaged {
                        kernel: call.kernel,
                    })?;
                    if self.cuda.is_none() {
                        self.cuda = Some(
                            Cuda::load()
                                .map_err(|error| DeviceLaunchError::Launcher { detail: error })?,
                        );
                    }
                    let cuda = self.cuda.as_ref().ok_or(DeviceLaunchError::Launcher {
                        detail: "the CUDA driver is unavailable".to_owned(),
                    })?;
                    let (module, kernel) = mega_mhc_launch::load(cuda, &bytes)
                        .map_err(|error| DeviceLaunchError::Launcher { detail: error })?;
                    let params = self.mhc_params(&buffers, call);
                    let result = launch_mhc_entry_with(
                        &buffers,
                        |destination, zeroes| cuda.copy_htod(destination, zeroes),
                        || mega_mhc_launch::launch(cuda, kernel, &params),
                    )
                    .map_err(|error| DeviceLaunchError::Launcher { detail: error });
                    cuda.free(module, &[]);
                    result
                }
                ChainStage::MoeRouter | ChainStage::MoeGateUp => Err(DeviceLaunchError::NotWired {
                    kernel: call.kernel,
                }),
                _ => Err(DeviceLaunchError::NotStaged {
                    kernel: call.kernel,
                }),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use sha2::{Digest, Sha256};
        use sparkflash_kernels::mega_mhc_ref::PREDICTOR_ROWS;

        use crate::cubins::{cubin_path, hex};

        fn empty_staging() -> LayerZeroStaging {
            LayerZeroStaging {
                predictor: Vec::new(),
                hc_base: [0.0; PREDICTOR_ROWS],
                hc_scale: [0.0; 3],
                attn_norm: Vec::new(),
                hc_base_kernel: [0.0; PREDICTOR_ROWS],
                embed_rows: Vec::new(),
                digests: Vec::new(),
                buffer_digests: Vec::new(),
            }
        }

        #[test]
        fn a_refused_budget_never_loads_the_driver_or_allocates() {
            let mut launcher = CudaLauncher::for_test();
            let error = launcher
                .stage_layer_zero(
                    &empty_staging(),
                    BudgetRequest {
                        mem_available_bytes: 0,
                        projected_host_bytes: 0,
                        projected_device_bytes: MhcBufferPlan::LAYER_ZERO.device_bytes() as u64,
                    },
                )
                .expect_err("the budget refuses");
            assert!(matches!(
                error,
                MhcStageError::Budget(BudgetError::BelowFloor { .. })
            ));
            assert!(launcher.cuda.is_none(), "no driver was opened");
            assert!(launcher.mhc_buffers.is_none(), "no buffers were staged");
            assert!(
                launcher.mhc_allocations.is_empty(),
                "no allocation was made"
            );
        }

        #[test]
        fn a_stage_projection_must_match_the_exact_buffer_plan() {
            let mut launcher = CudaLauncher::for_test();
            let error = launcher
                .stage_layer_zero(
                    &empty_staging(),
                    BudgetRequest {
                        mem_available_bytes: 1 << 40,
                        projected_host_bytes: 0,
                        projected_device_bytes: 0,
                    },
                )
                .expect_err("a zero device projection must not allocate the plan");
            match error {
                MhcStageError::Launcher { detail } => {
                    assert!(
                        detail.contains("projected device bytes"),
                        "the error names the projection: {detail}"
                    );
                }
                other => panic!("expected Launcher, got {other:?}"),
            }
            assert!(launcher.cuda.is_none(), "no driver was opened");
            assert!(
                launcher.mhc_allocations.is_empty(),
                "no allocation was made"
            );
        }

        #[test]
        fn an_unstaged_post_is_refused_before_the_driver() {
            let mut launcher = CudaLauncher::for_test();
            let error = launcher
                .launch(DeviceStageCall {
                    stage: ChainStage::MhcPost,
                    layer: 0,
                    position: 0,
                    kernel: "mega_mhc_sm121",
                })
                .expect_err("the launch must be refused");
            assert!(
                matches!(error, DeviceLaunchError::NotStaged { .. }),
                "unexpected refusal: {error:?}"
            );
            assert!(launcher.cuda.is_none(), "no driver was opened");
        }

        #[test]
        fn a_nonzero_layer_entry_without_cubins_is_refused_before_the_driver() {
            let mut launcher = CudaLauncher::for_test();
            let error = launcher
                .launch(DeviceStageCall {
                    stage: ChainStage::MhcEntry,
                    layer: 1,
                    position: 0,
                    kernel: "mega_mhc_sm121",
                })
                .expect_err("the launch must be refused");
            assert!(
                matches!(error, DeviceLaunchError::MissingCubin { .. }),
                "unexpected refusal: {error:?}"
            );
            assert!(launcher.cuda.is_none(), "no driver was opened");
            assert!(launcher.mhc_buffers.is_none(), "no buffers were staged");
        }

        #[test]
        fn a_second_launch_starts_with_zeroed_rendezvous_counters() {
            use std::collections::BTreeMap;

            let pointers: [Deviceptr; 16] =
                core::array::from_fn(|index| Deviceptr(0x2000 * (index as u64 + 1)));
            let buffers = MhcBuffers::from_allocations(&pointers);
            let plan = MhcBufferPlan::LAYER_ZERO;
            let zeroes = mhc_rendezvous_zeroes(&buffers);
            assert_eq!(zeroes[0].0, buffers.hc_stats);
            assert_eq!(zeroes[0].1.len(), 520, "hc_stats is 130 words");
            assert_eq!(zeroes[1].0, buffers.hc_sync);
            assert_eq!(zeroes[1].1.len(), 8, "hc_sync is two words");
            let mut device = BTreeMap::new();
            device.insert(buffers.hc_stats.0, vec![0xAAu8; plan.hc_stats]);
            device.insert(buffers.hc_sync.0, vec![0xBBu8; plan.hc_sync]);
            let mut kernels = 0;
            for launch in 0..2 {
                if launch > 0 {
                    // This is the stale state a previous launch leaves behind.
                    device.insert(buffers.hc_stats.0, vec![0xAAu8; plan.hc_stats]);
                    device.insert(buffers.hc_sync.0, vec![0xBBu8; plan.hc_sync]);
                }
                launch_mhc_entry_with(
                    &buffers,
                    |destination, zeroes| {
                        device.insert(destination.0, zeroes.to_vec());
                        Ok(())
                    },
                    || {
                        kernels += 1;
                        Ok(())
                    },
                )
                .expect("the fake launch accepts the reset and kernel");
                assert!(
                    device[&buffers.hc_stats.0].iter().all(|byte| *byte == 0),
                    "launch {launch} starts with stale stats"
                );
                assert!(
                    device[&buffers.hc_sync.0].iter().all(|byte| *byte == 0),
                    "launch {launch} starts with stale sync counters"
                );
                // Simulate the grid's tile rendezvous before the next iteration.
                device.insert(buffers.hc_stats.0, vec![4u8; plan.hc_stats]);
                device.insert(buffers.hc_sync.0, vec![1u8; plan.hc_sync]);
            }
            assert_eq!(kernels, 2, "every launch reached the kernel seam");
        }

        #[test]
        fn write_layer_zero_x_checks_the_four_copy_length_before_the_driver() {
            use crate::staging::EmbedRow;

            let pointers: [Deviceptr; 16] =
                core::array::from_fn(|index| Deviceptr(0x4000 * (index as u64 + 1)));
            let mut launcher = CudaLauncher::for_test();
            launcher.mhc_buffers = Some(MhcBuffers::from_allocations(&pointers));

            let valid = EmbedRow::new(1, vec![0x3cu8; HIDDEN * 2], 8)
                .expect("a valid admitted row is accepted");
            let error = launcher
                .write_layer_zero_x(&valid)
                .expect_err("no driver is available in this test");
            assert!(
                error.contains("CUDA driver is unavailable"),
                "a valid four-copy row must pass the length check and reach the driver check: {error}"
            );

            let short = EmbedRow::new_unchecked(1, vec![0x3cu8; HIDDEN * 2 - 2]);
            let error = launcher
                .write_layer_zero_x(&short)
                .expect_err("a short row must be refused");
            let expected = HC_MULT * (HIDDEN * 2 - 2);
            assert!(
                error.contains(&format!("the X row is {expected} bytes")),
                "the length check names the short X size: {error}"
            );
            assert!(
                !error.contains("CUDA driver is unavailable"),
                "the length check must run before the driver/copy path: {error}"
            );
        }

        #[test]
        fn the_mhc_entry_readback_is_transposed_into_the_host_coefficient_orientation() {
            let capture: Vec<f32> = (0..HIDDEN).map(|index| index as f32 * 0.25).collect();
            let coefficients: Vec<f32> = (0..24).map(|index| index as f32 + 1.0).collect();
            let entry = entry_from_readback(&f32_to_bytes(&capture), &f32_to_bytes(&coefficients))
                .expect("a full readback is accepted");
            assert_eq!(entry.pre_attention_hidden, capture);
            assert_eq!(
                entry.source,
                sparkflash_engine::forward::layer_zero::LayerZeroEntrySource::Device,
                "the readback must carry device provenance explicitly"
            );
            assert_eq!(entry.pre, coefficients[16..20].to_vec());
            assert_eq!(entry.post, coefficients[20..24].to_vec());
            for source in 0..HC_MULT {
                for destination in 0..HC_MULT {
                    assert_eq!(
                        entry.comb[source * HC_MULT + destination],
                        coefficients[destination * HC_MULT + source],
                        "comb[{source}][{destination}] is the kernel's destination-major value"
                    );
                }
            }
            let error = entry_from_readback(&[0u8; 4], &f32_to_bytes(&coefficients))
                .expect_err("a short capture is refused");
            assert!(error.contains("xhat_normed"));
            let error = entry_from_readback(&f32_to_bytes(&capture), &[0u8; 4])
                .expect_err("short coefficients are refused");
            assert!(error.contains("hc_coeffs"));
        }

        #[test]
        fn a_launch_resets_the_sized_counters_before_the_kernel() {
            use std::cell::RefCell;

            let pointers: [Deviceptr; 16] =
                core::array::from_fn(|index| Deviceptr(0x3000 * (index as u64 + 1)));
            let buffers = MhcBuffers::from_allocations(&pointers);
            let events = RefCell::new(Vec::new());
            launch_mhc_entry_with(
                &buffers,
                |destination, zeroes| {
                    if *destination == buffers.hc_stats {
                        assert_eq!(zeroes.len(), 520);
                        events.borrow_mut().push("stats");
                    } else if *destination == buffers.hc_sync {
                        assert_eq!(zeroes.len(), 8);
                        events.borrow_mut().push("sync");
                    } else {
                        panic!("the reset covers a buffer outside the rendezvous");
                    }
                    assert!(zeroes.iter().all(|byte| *byte == 0));
                    Ok(())
                },
                || {
                    events.borrow_mut().push("kernel");
                    Ok(())
                },
            )
            .expect("the fake launch accepts the reset sequence");
            assert_eq!(&*events.borrow(), &["stats", "sync", "kernel"]);
        }

        #[test]
        fn mhc_params_are_all_sixteen_buffers_in_kernel_order() {
            let launcher = CudaLauncher::for_test();
            let pointers: [Deviceptr; 16] =
                core::array::from_fn(|index| Deviceptr(0x1000 * (index as u64 + 1)));
            let buffers = MhcBuffers {
                x: pointers[0],
                y_prev: pointers[1],
                b_prev: pointers[2],
                c_prev: pointers[3],
                a_prev: pointers[4],
                predictor: pointers[5],
                hc_scale: pointers[6],
                hc_base: pointers[7],
                rms_weight: pointers[8],
                xhat_fp8: pointers[9],
                xhat_scales: pointers[10],
                hc_stats: pointers[11],
                hc_sync: pointers[12],
                hc_coeffs: pointers[13],
                hc_rms_out: pointers[14],
                xhat_normed: pointers[15],
            };
            let params = launcher.mhc_params(
                &buffers,
                DeviceStageCall {
                    stage: ChainStage::MhcEntry,
                    layer: 7,
                    position: 129,
                    kernel: "mega_mhc_sm121",
                },
            );
            let words = params.as_slice();
            assert_eq!(words.len(), 19);
            assert_eq!(&words[..16], &pointers.map(|pointer| pointer.0));
            assert_eq!(words[16], 0, "trace ring is the trace-disabled zero");
            assert_eq!(words[17], 7);
            assert_eq!(words[18], 129);
        }

        #[test]
        fn a_loaded_cubin_is_the_exact_slice_that_was_verified() {
            let dir = tempfile::tempdir().expect("temp dir");
            let contents = b"mega mhc cubin bytes";
            let path = cubin_path(dir.path(), "mega_mhc_sm121");
            std::fs::write(&path, contents).expect("write cubin");
            let pin = KernelPin {
                name: "mega_mhc_sm121".to_string(),
                cubin_bytes: contents.len(),
                cubin_sha256: hex(&Sha256::digest(contents)),
                ptx_membar_lines: 0,
            };
            let loaded = load_verified_cubins(dir.path(), std::slice::from_ref(&pin))
                .expect("the loaded slice verifies");
            assert_eq!(
                loaded.get("mega_mhc_sm121").map(Vec::as_slice),
                Some(contents.as_slice()),
                "the catalog stores the slice that was verified, not a second read"
            );

            // A changed file cannot be loaded under the old pin: the read/verify step fails
            // closed rather than keeping stale verified bytes.
            std::fs::write(&path, b"tampered").expect("tamper cubin");
            let error = load_verified_cubins(dir.path(), std::slice::from_ref(&pin))
                .expect_err("the tampered file must not verify");
            assert!(error.contains("bytes") || error.contains("sha256"));
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_launcher::{CudaLauncher, MhcBuffers, MhcStageError};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{ChainError, ChainStage, run_skeleton};

    fn schedule() -> ChainSchedule {
        ChainSchedule::milestone().expect("schedule")
    }

    fn count(stats: &ChainDeviceStats, name: &str) -> usize {
        stats
            .per_stage
            .iter()
            .find(|(stage, _)| *stage == name)
            .map(|(_, count)| *count)
            .unwrap_or(0)
    }

    #[test]
    fn the_dry_run_drives_the_full_plan_and_the_ledger_self_check_passes() {
        let dry = dry_run(&schedule()).expect("dry run");
        assert_eq!(dry.stats.total_calls, 108_017);
        assert_eq!(dry.stats.host_calls, 58_670);
        assert_eq!(dry.stats.device_calls, 49_347);
        assert_eq!(dry.device_calls.len(), 49_347);
        assert_eq!(dry.report.ledger.records().len(), 58_670);
        assert_eq!(
            dry.stats.total_calls,
            dry.stats.host_calls + dry.stats.device_calls
        );
        assert_eq!(
            dry.stats.per_stage.iter().map(|(_, n)| n).sum::<usize>(),
            dry.stats.total_calls
        );
        assert!(!dry.report.chain_execution);
        assert!(!dry.report.device_execution);
        assert!(!dry.report.token_produced);
        // The per-stage counts are the schedule's, including the non-layer stages and both
        // measured-but-unwired kernels.
        assert_eq!(count(&dry.stats, "embedding"), 138);
        assert_eq!(count(&dry.stats, "engram"), 276);
        assert_eq!(count(&dry.stats, "dispatch"), 5_482);
        assert_eq!(count(&dry.stats, "mhc_entry"), 5_482);
        assert_eq!(count(&dry.stats, "moe_router"), 5_482);
        assert_eq!(count(&dry.stats, "moe_gate_up"), 5_482);
        assert_eq!(count(&dry.stats, "final_norm"), 9);
        assert_eq!(count(&dry.stats, "lm_head_argmax"), 9);
        // Every device stage call names a stage-map kernel; none is a host seam.
        for call in &dry.device_calls {
            assert!(!call.kernel.is_empty());
            assert!(call.stage.kernel().is_some());
        }
    }

    #[test]
    fn a_missing_cubin_fails_the_wiring_and_names_the_stage() {
        let mut launcher = FakeLauncher::default();
        launcher.inject_missing_cubin("flash_mla_csa2_sm121");
        let mut device = ChainDevice::new(&schedule(), launcher, SeatHostSeams);
        let error = run_skeleton(&schedule(), &mut device).expect_err("missing cubin");
        assert!(matches!(
            error,
            ChainError::DeviceStage {
                stage: ChainStage::Attention,
                ..
            }
        ));
        assert!(error.to_string().contains("no verified cubin"));
        assert!(error.to_string().contains("flash_mla_csa2_sm121"));
    }

    #[test]
    fn a_launcher_error_fails_the_wiring_and_names_the_stage() {
        let mut launcher = FakeLauncher::default();
        launcher.inject_launcher_error(ChainStage::MhcEntry, 0, 0, "injected failure");
        let mut device = ChainDevice::new(&schedule(), launcher, SeatHostSeams);
        let error = run_skeleton(&schedule(), &mut device).expect_err("launcher failure");
        assert!(matches!(
            error,
            ChainError::DeviceStage {
                stage: ChainStage::MhcEntry,
                layer: 0,
                position: 0,
                ..
            }
        ));
        assert!(error.to_string().contains("injected failure"));
    }

    #[test]
    fn a_default_build_refuses_every_device_stage_with_a_reason() {
        let mut device = ChainDevice::new(&schedule(), NoCudaLauncher, SeatHostSeams);
        let error = run_skeleton(&schedule(), &mut device).expect_err("no cuda feature");
        assert!(matches!(
            error,
            ChainError::DeviceStage {
                stage: ChainStage::MhcEntry,
                ..
            }
        ));
        assert!(error.to_string().contains("without the cuda feature"));
        // Host seams before the first device stage still went to the host delegate: the refusal
        // happens at the device boundary, not by silently executing host math for it.
        assert_eq!(device.stats().host_calls, 2);
        assert_eq!(device.stats().device_calls, 1);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_cuda_launcher_refuses_a_missing_cubin_directory_before_the_driver() {
        let error = match CudaLauncher::open("/definitely/not/a/cubin/dir") {
            Ok(_) => panic!("there is no cubin directory there"),
            Err(error) => error,
        };
        assert!(!error.is_empty());
        let error =
            match CudaLauncher::open_for_stage("/definitely/not/a/cubin/dir", "mega_mhc_sm121") {
                Ok(_) => panic!("there is no cubin directory there"),
                Err(error) => error,
            };
        assert!(!error.is_empty());
    }
}

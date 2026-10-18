//! M3 device-stage wiring: the ChainOps implementation the device run will use.
//!
//! What this module is. The chain skeleton in crate::chain fixes the schedule, the stage map, the
//! planned host-seam set and the plan-vs-ledger self-check. This module implements ChainOps for that
//! skeleton: host-seam stages go to an injected HostSeamDelegate (the M3-a seat recorder by default),
//! and every device-status stage goes to an injected DeviceLauncher. There is no fallback between
//! the two: a device stage can never quietly run host arithmetic.
//!
//! What this module is not. It does not stage weights, allocate device buffers, launch a kernel on
//! the dev host, compare a token or logit, or make a device claim. The fake launcher is the default
//! test/dry-run launcher; the real one is behind the cuda feature and still refuses a launch until
//! M3-b stages weights and buffers. The wiring is nevertheless the single source of truth: the
//! dry-run drives the full 130 + 8 plan through this code and validates the same ledger a device run
//! would validate.
//!
//! Call accounting. The full plan makes 108,017 ChainOps::stage calls: 58,670 host-seam records and
//! 49,347 device-launcher calls. Tests and device-chain --dry-run pin those totals.
//!
//! Cubins. The real launcher reads every required cubin exactly once and verifies that same slice
//! with crate::cubins::verify_bytes before it can load anything (no verify-then-re-read window),
//! then resolves the typed builder for the stage. The first wired builder is
//! sparkflash_kernels::mega_mhc_launch; its params are built only when a buffer set is supplied,
//! which M3-b owns. A missing or mis-pinned cubin is an error, never a host fallback.

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

// The real launcher is compiled only with the feature. It still has no CLI path in this MR: M3-b
// stages the buffers, and only then can a launch be attempted. Keeping it behind the feature means
// the default binary has no device path at all.
#[cfg(feature = "cuda")]
mod cuda_launcher {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
    use sparkflash_kernels::mega_mhc_launch::{self, MegaMhcParams};

    use super::{ChainStage, DeviceLaunchError, DeviceLauncher, DeviceStageCall};
    use crate::cubins::{cubin_path, required_pins, verify_bytes};
    use sparkflash_kernels::baseline::KernelPin;

    /// The sixteen device pointers one mHC launch needs, in the kernel declaration order.
    ///
    /// M3-b constructs this from its staged buffers; this MR only defines and passes it through.
    /// A null/placeholder Deviceptr must never reach a launch, so construction is explicit.
    #[derive(Debug, Clone, Copy)]
    pub struct MhcBuffers {
        /// The x buffer.
        pub x: Deviceptr,
        /// The y_prev buffer.
        pub y_prev: Deviceptr,
        /// The b_prev buffer.
        pub b_prev: Deviceptr,
        /// The c_prev buffer.
        pub c_prev: Deviceptr,
        /// The a_prev buffer.
        pub a_prev: Deviceptr,
        /// The predictor buffer.
        pub predictor: Deviceptr,
        /// The hc_scale buffer.
        pub hc_scale: Deviceptr,
        /// The hc_base buffer.
        pub hc_base: Deviceptr,
        /// The rms_weight buffer.
        pub rms_weight: Deviceptr,
        /// The xhat_fp8 buffer.
        pub xhat_fp8: Deviceptr,
        /// The xhat_scales buffer.
        pub xhat_scales: Deviceptr,
        /// The hc_stats buffer.
        pub hc_stats: Deviceptr,
        /// The hc_sync buffer.
        pub hc_sync: Deviceptr,
        /// The hc_coeffs buffer.
        pub hc_coeffs: Deviceptr,
        /// The hc_rms_out buffer.
        pub hc_rms_out: Deviceptr,
        /// The xhat_normed buffer.
        pub xhat_normed: Deviceptr,
    }

    /// The real launcher: verifies the required cubins, then dispatches the first typed builder
    /// (mHC) when its buffers are staged.
    ///
    /// Construction reads each pinned cubin exactly once and verifies that same slice with
    /// [crate::cubins::verify_bytes], so the loaded bytes are the verified bytes (there is no
    /// verify-then-re-read window). A missing or mis-pinned cubin is refused before any driver is
    /// opened. It opens no driver until a launch is actually attempted, and no caller in this MR
    /// supplies MhcBuffers: that is M3-b.
    pub struct CudaLauncher {
        cuda: Option<Cuda>,
        cubins: BTreeMap<String, Vec<u8>>,
        mhc_buffers: Option<MhcBuffers>,
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
        /// Verifies the cubin directory and loads its bytes into the catalog.
        ///
        /// # Errors
        /// A required-pin catalog error, a cubin read error, or a size/digest mismatch.
        pub fn open(dir: impl AsRef<Path>) -> Result<Self, String> {
            let dir = dir.as_ref();
            let pins = required_pins()?;
            let cubins = load_verified_cubins(dir, &pins)?;
            Ok(Self {
                cuda: None,
                cubins,
                mhc_buffers: None,
            })
        }

        /// Whether this launcher verified a cubin with kernel's name.
        #[must_use]
        pub fn has_cubin(&self, kernel: &str) -> bool {
            self.cubins.contains_key(kernel)
        }

        /// Supplies the mHC buffer set. M3-b owns the staging that creates it.
        pub fn set_mhc_buffers(&mut self, buffers: MhcBuffers) {
            self.mhc_buffers = Some(buffers);
        }

        fn mhc_params(&self, buffers: &MhcBuffers, call: DeviceStageCall) -> MegaMhcParams {
            // step_seq is the absolute token position (0..prompt+decode-1) from ChainDevice, not a
            // decoder-step index and not a per-layer counter. The kernel ignores it while
            // trace_ring=0; when M3-b enables tracing it must be read as this absolute position.
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

    impl DeviceLauncher for CudaLauncher {
        fn launch(&mut self, call: DeviceStageCall) -> Result<(), DeviceLaunchError> {
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
                    let result = mega_mhc_launch::launch(cuda, kernel, &params)
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

        use crate::cubins::{cubin_path, hex};

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
pub use cuda_launcher::{CudaLauncher, MhcBuffers};

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
    }
}

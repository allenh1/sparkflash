//! The shared CUDA driver path for the kernel-launching examples.
//!
//! The driver is resolved with dlopen at run time, so nothing links libcuda and the workspace still
//! builds on hosts without a device. This module exists so the examples do not each carry their own
//! copy of the driver: the m7 launcher and the forthcoming m8 sequence launcher import one Cuda,
//! one module loader and one launch path.
//!
//! The raw driver function pointers stay private to this module. A kernel handle is an opaque
//! crate-owned value (`Kernel`) that only the non-flash_mla kernel ids can produce; the
//! flash_mla kernel is reachable only through the typed launches in crate::cuda_harness, which is
//! the compile-time enforcement of the one-launch-path rule. The unsafe blocks in this module are
//! the dlopen boundary: everything above it is safe code.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

type CuResult = i32;

/// An opaque CUDA event handle.
type CuEvent = *mut c_void;

/// The host's MemAvailable in kB, parsed from /proc/meminfo.
///
/// Interpolating the whole /proc line into a message produced "...the node has ~MemAvailable:
/// 3768676 kB GB of system memory available...", which is exactly the kind of unreadable failure
/// text this launcher exists to avoid; the launcher admission checks read the number instead.
pub fn mem_available_kb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

/// The cuLaunchKernel signature. The parameter list is long by nature, so it is named once here
/// rather than spelled out at the one call site.
type LaunchFn = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut c_void,
) -> CuResult;

/// The driver's CUresult names, so a report says OUT_OF_MEMORY rather than 2.
fn name(code: CuResult) -> String {
    let s = match code {
        0 => "SUCCESS",
        1 => "INVALID_VALUE",
        2 => "OUT_OF_MEMORY",
        3 => "NOT_INITIALIZED",
        4 => "DEINITIALIZED",
        100 => "NO_DEVICE",
        101 => "INVALID_DEVICE",
        201 => "INVALID_CONTEXT",
        400 => "INVALID_HANDLE",
        700 => "ILLEGAL_ADDRESS",
        801 => "NOT_SUPPORTED",
        _ => "UNKNOWN",
    };
    format!("{code} ({s})")
}
type CuDeviceptr = u64;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

/// A device allocation, in the driver's flat address space.
///
/// The address is transparent on purpose, not opaque: the launcher marshals device addresses into
/// the u64 kernel-argument vector, so the harness reads `.0` when it builds an argument list and
/// tests can construct one. A device address is not a capability - it cannot name or launch a
/// kernel; launches only accept an opaque Kernel (or the typed flash_mla handle).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deviceptr(
    /// The driver's flat device address, as the driver returns it from cuMemAlloc_v2.
    pub CuDeviceptr,
);

impl Deviceptr {
    /// Advances this address by a byte offset.
    ///
    /// The rotation copy moves one row group at a time (512-byte code rows, 16-byte scale rows),
    /// so it needs an address inside an allocation, not only the allocation base. The arithmetic is
    /// the driver's flat address space and is not checked against an allocation bound: a caller
    /// that offsets past its allocation gets the driver's own ILLEGAL_ADDRESS, exactly as it would
    /// from a hand-built address. Only the device path calls this; a host-only build never does.
    #[must_use]
    pub fn offset(self, bytes: usize) -> Self {
        Self(self.0 + bytes as u64)
    }
}

/// The opaque handle of a kernel that is not flash_mla.
///
/// The raw CUfunction stays private: examples cannot construct a Kernel and cannot recover the
/// pointer, so a launch always goes through this module. The flash_mla kernel deliberately has no
/// KernelId variant; see crate::cuda_harness::launch_flash_mla.
///
/// Constructing a Kernel by hand is a compile error, so there is no symbol-name bypass:
///
/// ```compile_fail
/// use sparkflash_kernels::cuda_driver::{Cuda, Kernel};
///
/// fn bypass(cuda: &Cuda) -> Result<(), String> {
///     // The raw field is private and there is no public constructor or conversion.
///     let kernel = Kernel { function: std::ptr::null_mut() };
///     cuda.launch(kernel, (1, 1, 1), (1, 1, 1), &[])
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Kernel {
    function: CuFunction,
    /// The canonical cubin pin name, so a timing bracket can key its sample by kernel.
    name: &'static str,
}

/// A page-locked host allocation with its UVA device mapping.
///
/// The host pointer is the allocation's owner; the device pointer is only a mapping into it and
/// must not be freed separately. [Cuda::free_host_mapped] releases the host allocation.
#[derive(Clone, Copy, Debug)]
pub struct HostMapped {
    host: *mut c_void,
    device: Deviceptr,
}

impl HostMapped {
    /// The device-visible pointer for kernel arguments.
    #[must_use]
    pub fn device(&self) -> Deviceptr {
        self.device
    }

    /// The host-visible pointer the caller fills before a launch.
    #[must_use]
    pub fn host_ptr(&self) -> *mut c_void {
        self.host
    }
}

/// The opaque handle of a CUDA module.
///
/// The module is what keeps a Kernel's code loaded, so callers hold it for as long as the kernel
/// runs; freeing it goes through Cuda::free. The raw CUmodule stays private.
#[derive(Clone, Copy, Debug)]
pub struct Module(CuModule);

/// One of the non-flash_mla kernels the shared driver resolves by symbol.
///
/// The enum deliberately has no flash_mla variant: the flash_mla kernel is loaded and launched by
/// crate::cuda_harness's typed entry points, so an example cannot obtain a raw handle for it.
/// Adding a variant here would reopen the raw-launch path that the typed launch closes.
///
/// A symbol name cannot stand in for a KernelId, so there is no symbol-name loader to reach the
/// flash_mla kernel with:
///
/// ```compile_fail
/// use sparkflash_kernels::cuda_driver::Cuda;
///
/// fn bypass(cuda: &Cuda, cubin: &[u8]) -> Result<(), String> {
///     // load_kernel takes a KernelId; a string does not typecheck.
///     let (_module, _kernel) = cuda.load_kernel(cubin, "flash_mla_csa2_sm121")?;
///     Ok(())
/// }
/// ```
///
/// And the flash_mla variant does not exist to be named:
///
/// ```compile_fail
/// use sparkflash_kernels::cuda_driver::KernelId;
///
/// fn bypass() -> KernelId {
///     KernelId::FlashMla
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelId {
    /// mega_mhc_fused_bf16_sm121: the Mega-mHC fused kernel.
    MegaMhcBf16,
    /// b12x_mxfp8_dense_sm121: the MXFP8 dense twin.
    Mxfp8Dense,
    /// b12x_mxfp8_wo_a_sm121: the output-projection MXFP8 kernel.
    Mxfp8WoA,
    /// b12x_mxfp4_moe_sm121: the MXFP4 MoE kernel.
    Mxfp4Moe,
    /// swiglu_mxfp8_quant_sm121: the SwiGLU plus MXFP8 quantiser kernel.
    SwigluMxfp8Quant,
    /// swiglu_mxfp8_quant_weighted_sm121: the same epilogue with the routing weight multiplied in
    /// before the quantiser (the F6 seam's device closure).
    SwigluMxfp8QuantWeighted,
    /// cert_lm_head_sm121: the certified LM-head kernel.
    CertLmHead,
    /// moe_router_sm121: the device MoE router (task-37, roadmap 6.3 item 4).
    MoeRouter,
    /// routed_gate_up_sm121: the routed-expert W4A8 gate/up projection (task-8 iteration B).
    RoutedGateUp,
    /// rmsnorm_sm121: the standalone final RMSNorm of the serving tail (BF16 in, BF16 out).
    RmsNorm,
    /// lm_head_reduce_sm121: the device block reduction and argmax over the four certified-head
    /// shard outputs.
    LmHeadReduce,
    /// flash_mla_csa2_seq_sm121: the deferred sequence's position-causal CSA2 attention entry, the
    /// start_pos > 0 path the c4a-2 call-site guard currently denies.
    FlashMlaCsa2Seq,
    /// flash_mla_csa2_ring_sm121: D49's ring-capacity CSA2 entry. Capacity is an explicit argument
    /// separate from the 128-token arithmetic window; the frozen sequence entry is not mutated.
    FlashMlaCsa2Ring,
    /// b12x_bf16a_mxfp8w_dense_sm121: the BF16-activation variant of the dense MXFP8 projection
    /// (D37's precision-first path); weights stay B12x FP8 codes and ue8m0 scales.
    Bf16aMxfp8wDense,
    /// b12x_bf16a_mxfp8w_wo_a_sm121: the BF16-activation variant of the grouped wo_a projection.
    Bf16aMxfp8wWoA,
    /// engram_embed_sm121: the Engram table row reader and activation quantiser (task-9, audit F4).
    EngramEmbed,
    /// engram_wkv_sm121: the Engram wkv projection, fp8 W8A8 (task-9, audit F4).
    EngramWkv,
    /// engram_gate_write_sm121: the Engram gate and residual write (task-9, audit F4).
    EngramGateWrite,
}

impl KernelId {
    /// Every kernel the shared driver owns, in the order of this enum.
    pub const ALL: [KernelId; 18] = [
        KernelId::MegaMhcBf16,
        KernelId::Mxfp8Dense,
        KernelId::Mxfp8WoA,
        KernelId::Mxfp4Moe,
        KernelId::SwigluMxfp8Quant,
        KernelId::SwigluMxfp8QuantWeighted,
        KernelId::CertLmHead,
        KernelId::MoeRouter,
        KernelId::RoutedGateUp,
        KernelId::RmsNorm,
        KernelId::LmHeadReduce,
        KernelId::FlashMlaCsa2Seq,
        KernelId::FlashMlaCsa2Ring,
        KernelId::Bf16aMxfp8wDense,
        KernelId::Bf16aMxfp8wWoA,
        KernelId::EngramEmbed,
        KernelId::EngramWkv,
        KernelId::EngramGateWrite,
    ];

    /// The cubin symbol of this kernel. Private: callers name the enum, never the symbol.
    fn symbol(self) -> &'static str {
        match self {
            KernelId::MegaMhcBf16 => "mega_mhc_fused_bf16_sm121",
            KernelId::Mxfp8Dense => "b12x_mxfp8_dense_sm121",
            KernelId::Mxfp8WoA => "b12x_mxfp8_wo_a_sm121",
            KernelId::Mxfp4Moe => "b12x_mxfp4_moe_sm121",
            KernelId::SwigluMxfp8Quant => "swiglu_mxfp8_quant_sm121",
            KernelId::SwigluMxfp8QuantWeighted => "swiglu_mxfp8_quant_weighted_sm121",
            KernelId::CertLmHead => "cert_lm_head_sm121",
            KernelId::MoeRouter => "moe_router_sm121",
            KernelId::RoutedGateUp => "routed_gate_up_sm121",
            KernelId::RmsNorm => "rmsnorm_sm121",
            KernelId::LmHeadReduce => "lm_head_reduce_sm121",
            KernelId::FlashMlaCsa2Seq => "flash_mla_csa2_seq_sm121",
            KernelId::FlashMlaCsa2Ring => "flash_mla_csa2_ring_sm121",
            KernelId::Bf16aMxfp8wDense => "b12x_bf16a_mxfp8w_dense_sm121",
            KernelId::Bf16aMxfp8wWoA => "b12x_bf16a_mxfp8w_wo_a_sm121",
            KernelId::EngramEmbed => "engram_embed_sm121",
            KernelId::EngramWkv => "engram_wkv_sm121",
            KernelId::EngramGateWrite => "engram_gate_write_sm121",
        }
    }

    /// The cubin file names this kernel may be stored under, symbol first.
    ///
    /// The symbol is the name the harness documents and must open. Mega-MHC carries a second,
    /// documented alias because the pinned baseline manifest keys that kernel by its source-file
    /// stem, `mega_mhc_sm121` (`baseline/kernel_baseline.txt`). A cubin's file name is not part of
    /// its ABI - the symbol inside it is - so both names are the same artifact, and accepting the
    /// alias removes the run-D failure of 2026-10-18 (the pinned artifact was copied under the
    /// manifest's name and the harness opened the symbol's name instead).
    pub fn cubin_names(self) -> &'static [&'static str] {
        match self {
            KernelId::MegaMhcBf16 => &["mega_mhc_fused_bf16_sm121", "mega_mhc_sm121"],
            KernelId::Mxfp8Dense => &["b12x_mxfp8_dense_sm121"],
            KernelId::Mxfp8WoA => &["b12x_mxfp8_wo_a_sm121"],
            KernelId::Mxfp4Moe => &["b12x_mxfp4_moe_sm121"],
            KernelId::SwigluMxfp8Quant => &["swiglu_mxfp8_quant_sm121"],
            KernelId::SwigluMxfp8QuantWeighted => &["swiglu_mxfp8_quant_weighted_sm121"],
            KernelId::CertLmHead => &["cert_lm_head_sm121"],
            KernelId::MoeRouter => &["moe_router_sm121"],
            KernelId::RoutedGateUp => &["routed_gate_up_sm121"],
            KernelId::RmsNorm => &["rmsnorm_sm121"],
            KernelId::LmHeadReduce => &["lm_head_reduce_sm121"],
            KernelId::FlashMlaCsa2Seq => &["flash_mla_csa2_seq_sm121"],
            KernelId::FlashMlaCsa2Ring => &["flash_mla_csa2_ring_sm121"],
            KernelId::Bf16aMxfp8wDense => &["b12x_bf16a_mxfp8w_dense_sm121"],
            KernelId::Bf16aMxfp8wWoA => &["b12x_bf16a_mxfp8w_wo_a_sm121"],
            KernelId::EngramEmbed => &["engram_embed_sm121"],
            KernelId::EngramWkv => &["engram_wkv_sm121"],
            KernelId::EngramGateWrite => &["engram_gate_write_sm121"],
        }
    }

    /// Maps a cubin file name (the symbol or a documented alias) to the kernel.
    ///
    /// `flash_mla_csa2_sm121` is deliberately absent: that kernel is loaded through
    /// `cuda_harness::load_flash_mla`, which owns its symbol, so it has no `KernelId`. The
    /// deferred sequence entry IS here: it is a separate cubin with its own symbol, and the call
    /// site resolves it like any other shared-driver kernel.
    #[must_use]
    pub fn from_cubin_name(name: &str) -> Option<KernelId> {
        KernelId::ALL
            .into_iter()
            .find(|id| id.cubin_names().contains(&name))
    }
}

/// The cubin name of the flash_mla kernel, which has no [`KernelId`]: `crate::cuda_harness`
/// owns its symbol and its typed launches, so it sits outside the driver's kernel-id vocabulary.
pub const FLASH_MLA_CUBIN: &str = "flash_mla_csa2_sm121";

/// The built cubin for a kernel name, under `dir`, for a harness that launches it.
///
/// The kernel's [`KernelId::cubin_names`] are tried in order - the symbol first, then any documented
/// alias such as the pinned baseline manifest's Mega-MHC source stem - so a cubin copied out of the
/// pinned artifact directory under either name is found. [`FLASH_MLA_CUBIN`] is resolved by its own
/// name because it has no `KernelId`. Any other name is refused with the driver-kernel message
/// rather than a file error, and a known kernel with no built file names every candidate tried.
///
/// # Errors
///
/// The name is not a shared-driver kernel or the flash_mla kernel, or no candidate file exists under
/// `dir`.
pub fn cubin_path(dir: &Path, name: &str) -> Result<PathBuf, String> {
    let candidates: &[&str] = if name == FLASH_MLA_CUBIN {
        &[FLASH_MLA_CUBIN]
    } else {
        KernelId::from_cubin_name(name)
            .ok_or_else(|| {
                format!(
                    "{name} is not one of the shared driver kernels; the flash_mla kernel is loaded through cuda_harness::load_flash_mla and has no KernelId"
                )
            })?
            .cubin_names()
    };
    let mut tried = Vec::new();
    for candidate in candidates {
        let path = dir.join(format!("{candidate}.cubin"));
        if path.is_file() {
            return Ok(path);
        }
        tried.push(path.display().to_string());
    }
    Err(format!(
        "no cubin for {name} in {}; tried {}",
        dir.display(),
        tried.join(", ")
    ))
}

/// The chain scope a launch is attributed to: phase, step, layer and position.
///
/// The harness sets it at its position/step loop; every launch until the next set is attributed to
/// that scope. All fields default to 0 (and the phase to "stage") so an unwired arm still produces
/// a well-formed ledger row rather than a wrong one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StageScope {
    /// The phase name: prefill, decode or a stage-arm name.
    pub phase: &'static str,
    /// The step index inside the phase.
    pub step: usize,
    /// The backbone layer, when the caller knows it.
    pub layer: usize,
    /// The absolute token position.
    pub position: usize,
}

/// One kernel's aggregated device time in one phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelTimingRow {
    /// The phase the launches were attributed to.
    pub phase: &'static str,
    /// The canonical cubin pin name.
    pub kernel: &'static str,
    /// How many launches were bracketed.
    pub calls: u64,
    /// The summed event time, nanoseconds.
    pub total_ns: u64,
    /// The mean event time, nanoseconds.
    pub mean_ns: u64,
    /// The shortest launch, nanoseconds.
    pub min_ns: u64,
    /// The longest launch, nanoseconds.
    pub max_ns: u64,
    /// The ordinal of the first launch of this kernel in this phase.
    pub first_launch_order: usize,
}

/// One kernel's launches inside one step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepLaunchRow {
    /// The canonical cubin pin name.
    pub kernel: &'static str,
    /// How many launches were bracketed.
    pub calls: u64,
    /// The summed event time, nanoseconds.
    pub total_ns: u64,
}

/// One step's launches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepTimingRow {
    /// The step index the scope reported.
    pub step: usize,
    /// True when the caller marked this step as warmup (cold-vs-steady separation).
    pub warm: bool,
    /// How many launches were bracketed.
    pub launches: u64,
    /// The summed event time, nanoseconds.
    pub kernel_ns: u64,
    /// The per-kernel rows, in kernel-name order.
    pub kernels: Vec<StepLaunchRow>,
}

/// The per-kernel ledger one run collected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageTimingReport {
    /// The per-phase, per-kernel rows, in first-launch order.
    pub kernels: Vec<KernelTimingRow>,
    /// The per-step rows, in step order.
    pub steps: Vec<StepTimingRow>,
    /// How many distinct kernel names beyond the 128-name cap were dropped.
    pub dropped_names: usize,
}

/// The distinct-kernel cap, so a run cannot grow the ledger without bound.
const STAGE_TIMING_KERNEL_CAP: usize = 128;

#[derive(Default)]
struct KernelRow {
    calls: u64,
    total_ns: u64,
    min_ns: u64,
    max_ns: u64,
    first_order: usize,
}

#[derive(Default)]
struct StepRow {
    launches: u64,
    kernel_ns: u64,
    kernels: BTreeMap<&'static str, (u64, u64)>,
    warm: bool,
}

struct StageRecorder {
    scope: StageScope,
    order: usize,
    dropped: usize,
    kernels: BTreeMap<(&'static str, &'static str), KernelRow>,
    steps: BTreeMap<usize, StepRow>,
}

impl Default for StageRecorder {
    fn default() -> Self {
        Self {
            scope: StageScope {
                phase: "stage",
                step: 0,
                layer: 0,
                position: 0,
            },
            order: 0,
            dropped: 0,
            kernels: BTreeMap::new(),
            steps: BTreeMap::new(),
        }
    }
}

/// The process-wide recorder. The ledger is per process, not per [Cuda], because one chain opens
/// several driver handles (attention, MoE and the mHC launcher) and all of their launches belong in
/// one table.
static STAGE_RECORDER: Mutex<Option<StageRecorder>> = Mutex::new(None);

/// Set by the device arm before its dummy steps; read when a step row is first touched.
static WARM_STEP: AtomicBool = AtomicBool::new(false);

/// Whether SPARKFLASH_STAGE_TIMING=1 armed the bracket. Off by default: no events, no allocations
/// and a byte-identical run when it is unset.
pub fn stage_timing_enabled() -> bool {
    std::env::var("SPARKFLASH_STAGE_TIMING").ok().as_deref() == Some("1")
}

/// Sets the scope every following launch is attributed to. A no-op when the recorder is off.
pub fn stage_timing_scope(phase: &'static str, step: usize, layer: usize, position: usize) {
    if !stage_timing_enabled() {
        return;
    }
    let Ok(mut guard) = STAGE_RECORDER.lock() else {
        return;
    };
    let Some(recorder) = guard.as_mut() else {
        return;
    };
    recorder.scope = StageScope {
        phase,
        step,
        layer,
        position,
    };
}

/// Marks subsequent steps as warmup steps until cleared. The device arm sets this for its dummy
/// pre-serving steps and clears it before the first real token, so the ledger can split first-step
/// (cold) time from steady-state time instead of averaging them together.
pub fn stage_timing_set_warm(warm: bool) {
    WARM_STEP.store(warm, Ordering::Relaxed);
}

/// Updates only the layer of the current scope (the chain's per-layer loop calls this).
pub fn stage_timing_set_layer(layer: usize) {
    if !stage_timing_enabled() {
        return;
    }
    let Ok(mut guard) = STAGE_RECORDER.lock() else {
        return;
    };
    if let Some(recorder) = guard.as_mut() {
        recorder.scope.layer = layer;
    }
}

/// Records one bracketed launch. Called by [Cuda::launch] only.
fn record_launch(kernel: &'static str, elapsed_ns: u64) {
    let Ok(mut guard) = STAGE_RECORDER.lock() else {
        return;
    };
    let Some(recorder) = guard.as_mut() else {
        return;
    };
    let phase = recorder.scope.phase;
    let step = recorder.scope.step;
    let order = recorder.order;
    recorder.order += 1;
    let known = recorder.kernels.contains_key(&(phase, kernel));
    if !known && recorder.kernels.len() >= STAGE_TIMING_KERNEL_CAP {
        recorder.dropped += 1;
        return;
    }
    let row = recorder
        .kernels
        .entry((phase, kernel))
        .or_insert(KernelRow {
            first_order: order,
            min_ns: elapsed_ns,
            max_ns: elapsed_ns,
            ..KernelRow::default()
        });
    row.calls += 1;
    row.total_ns = row.total_ns.saturating_add(elapsed_ns);
    row.min_ns = row.min_ns.min(elapsed_ns);
    row.max_ns = row.max_ns.max(elapsed_ns);
    let step_row = recorder.steps.entry(step).or_default();
    step_row.warm = WARM_STEP.load(Ordering::Relaxed);
    step_row.launches += 1;
    step_row.kernel_ns = step_row.kernel_ns.saturating_add(elapsed_ns);
    let entry = step_row.kernels.entry(kernel).or_insert((0, 0));
    entry.0 += 1;
    entry.1 = entry.1.saturating_add(elapsed_ns);
}

/// The ledger one run collected, or None when the recorder was never armed.
pub fn stage_timing_report() -> Option<StageTimingReport> {
    let guard = STAGE_RECORDER.lock().ok()?;
    let recorder = guard.as_ref()?;
    let mut kernels: Vec<KernelTimingRow> = recorder
        .kernels
        .iter()
        .map(|((phase, kernel), row)| KernelTimingRow {
            phase,
            kernel,
            calls: row.calls,
            total_ns: row.total_ns,
            mean_ns: row.total_ns.checked_div(row.calls).unwrap_or(0),
            min_ns: row.min_ns,
            max_ns: row.max_ns,
            first_launch_order: row.first_order,
        })
        .collect();
    kernels.sort_by_key(|row| row.first_launch_order);
    let steps: Vec<StepTimingRow> = recorder
        .steps
        .iter()
        .map(|(step, row)| StepTimingRow {
            step: *step,
            warm: row.warm,
            launches: row.launches,
            kernel_ns: row.kernel_ns,
            kernels: row
                .kernels
                .iter()
                .map(|(kernel, (calls, total_ns))| StepLaunchRow {
                    kernel,
                    calls: *calls,
                    total_ns: *total_ns,
                })
                .collect(),
        })
        .collect();
    Some(StageTimingReport {
        kernels,
        steps,
        dropped_names: recorder.dropped,
    })
}

/// Counts one launch bracket site, so a test can prove the bracket is armed.
#[must_use]
pub fn stage_timing_launches() -> u64 {
    STAGE_RECORDER
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|recorder| recorder.order as u64))
        .unwrap_or(0)
}
/// The CUDA context and the dlopen'd driver functions this process uses.
///
/// The context is the retained primary context, made current for the thread that created it, and
/// is released by Drop on every exit path.
pub struct Cuda {
    lib: libloading::Library,
    ctx: *mut c_void,
    /// The device ordinal the retained primary context belongs to.
    ///
    /// `cuDevicePrimaryCtxRelease` takes a `CUdevice` (an `int`), not the `CUcontext`: storing the
    /// ordinal is what lets `Drop` make the matching release call with the right argument instead of
    /// passing a pointer through a mis-typed symbol.
    dev: i32,
}

impl Cuda {
    /// Opens libcuda.so.1, initialises the driver and makes the primary context current.
    ///
    /// cuInit alone is not enough: cuModuleLoadData and cuMemAlloc need a CURRENT context, and
    /// without one the driver answers CUDA_ERROR_INVALID_CONTEXT (201), which is what the first
    /// run on the node returned. The primary context is retained (refcounted, shared with anything
    /// else driving the device) and made current for this thread.
    pub fn load() -> Result<Self, String> {
        unsafe {
            let lib = libloading::Library::new("libcuda.so.1")
                .map_err(|e| format!("libcuda.so.1: {e}"))?;
            {
                let init: libloading::Symbol<unsafe extern "C" fn(u32) -> CuResult> =
                    lib.get(b"cuInit").map_err(|e| e.to_string())?;
                let r = init(0);
                if r != 0 {
                    return Err(format!("cuInit failed: {r}"));
                }
            }
            let mut count: i32 = 0;
            {
                let f: libloading::Symbol<unsafe extern "C" fn(*mut i32) -> CuResult> =
                    lib.get(b"cuDeviceGetCount").map_err(|e| e.to_string())?;
                let r = f(&mut count);
                if r != 0 {
                    return Err(format!("cuDeviceGetCount failed: {}", name(r)));
                }
            }
            let mut dev: i32 = 0;
            {
                let get: libloading::Symbol<unsafe extern "C" fn(*mut i32, i32) -> CuResult> =
                    lib.get(b"cuDeviceGet").map_err(|e| e.to_string())?;
                let r = get(&mut dev, 0);
                if r != 0 {
                    return Err(format!("cuDeviceGet(0) failed: {}", name(r)));
                }
            }
            let mut device_name = String::from("unknown device");
            {
                let mut nm = [0i8; 128];
                let f: libloading::Symbol<unsafe extern "C" fn(*mut i8, i32, i32) -> CuResult> =
                    lib.get(b"cuDeviceGetName").map_err(|e| e.to_string())?;
                let r = f(nm.as_mut_ptr(), 128, dev);
                if r == 0 {
                    // c_char is u8 on aarch64 and i8 on x86_64; cast rather than assume either.
                    let s = std::ffi::CStr::from_ptr(nm.as_ptr().cast());
                    device_name = s.to_string_lossy().into_owned();
                    let mut total: u64 = 0;
                    let t: libloading::Symbol<unsafe extern "C" fn(*mut u64, i32) -> CuResult> =
                        lib.get(b"cuDeviceTotalMem_v2").map_err(|e| e.to_string())?;
                    let _ = t(&mut total, dev);
                    println!(
                        "device 0: {device_name} of {count} (total {} MiB, MemAvailable {} MiB)",
                        total / (1024 * 1024),
                        mem_available_kb().map(|kb| kb / 1024).unwrap_or(0)
                    );
                }
            }
            let mut ctx: *mut c_void = std::ptr::null_mut();
            {
                // CU_CTX_LMEM_RESIZE_TO_MAX (0x10) asks the driver NOT to pre-reserve the maximum
                // local memory for this context. On a unified-memory GB10 where the device total is
                // 124 GB but only ~3.7 GB of system memory is free, that pre-reservation is the
                // difference between a context and CUDA_ERROR_OUT_OF_MEMORY.
                let set_flags: libloading::Symbol<unsafe extern "C" fn(i32, u32) -> CuResult> = lib
                    .get(b"cuDevicePrimaryCtxSetFlags")
                    .map_err(|e| e.to_string())?;
                let _ = set_flags(dev, 0x10);
                let retain: libloading::Symbol<
                    unsafe extern "C" fn(*mut *mut c_void, i32) -> CuResult,
                > = lib
                    .get(b"cuDevicePrimaryCtxRetain")
                    .map_err(|e| e.to_string())?;
                let r = retain(&mut ctx, dev);
                if r != 0 {
                    // One alternative route, tried exactly once - not a retry loop. Retaining the
                    // primary context is the normal route for a process that does not own the
                    // device; cuCtxCreate_v2 is what a launcher would use if it did. Both results
                    // are reported so the failure cannot be read as one API's quirk.
                    let mut alt: *mut c_void = std::ptr::null_mut();
                    let create: libloading::Symbol<
                        unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> CuResult,
                    > = lib.get(b"cuCtxCreate_v2").map_err(|e| e.to_string())?;
                    let r2 = create(&mut alt, 0x10, dev);
                    // A context created by the alternative route is not the one this type stores,
                    // so destroy it now: it would otherwise leak on the error return below.
                    if r2 == 0
                        && let Ok(destroy) = lib
                            .get::<unsafe extern "C" fn(*mut c_void) -> CuResult>(
                                b"cuCtxDestroy_v2",
                            )
                    {
                        let _ = destroy(alt);
                    }
                    // Nothing is left half-initialised: no context was retained, the alternative
                    // context (if any) was destroyed, no device memory was allocated, and this
                    // condition is not retried.
                    let avail = mem_available_kb()
                        .map(|kb| format!("{} MiB", kb / 1024))
                        .unwrap_or_else(|| "unknown".to_string());
                    return Err(format!(
                        "no CUDA context could be created on device 0 ({}): \
                         cuDevicePrimaryCtxRetain -> {}, cuCtxCreate_v2 -> {} (one attempt each, not \
                         retried). cuInit, cuDeviceGet and cuDeviceGetName all succeeded, so the \
                         driver is reachable and the device is present: on this node a second CUDA \
                         context does not fit beside the resident model. MemAvailable now: {avail}. \
                         No device memory was allocated and no context is left retained. This is a \
                         capacity fact about the node, not a defect in this launcher - do not work \
                         around it here.",
                        device_name,
                        name(r),
                        name(r2),
                    ));
                }
            }
            {
                // This failure path cannot rely on Drop, because the Cuda it would release has not
                // been constructed yet; release the retained primary context explicitly.
                let set: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> CuResult> =
                    match lib.get(b"cuCtxSetCurrent") {
                        Ok(set) => set,
                        Err(error) => {
                            if let Ok(release_v2) = lib
                                .get::<unsafe extern "C" fn(i32) -> CuResult>(
                                    b"cuDevicePrimaryCtxRelease_v2",
                                )
                            {
                                let _ = release_v2(dev);
                            }
                            return Err(error.to_string());
                        }
                    };
                let r = set(ctx);
                if r != 0 {
                    if let Ok(release_v2) = lib.get::<unsafe extern "C" fn(i32) -> CuResult>(
                        b"cuDevicePrimaryCtxRelease_v2",
                    ) {
                        let _ = release_v2(dev);
                    }
                    return Err(format!("cuCtxSetCurrent failed: {r}"));
                }
            }
            Ok(Cuda { lib, ctx, dev })
        }
    }

    fn sym<T>(&self, name: &[u8]) -> Result<libloading::Symbol<'_, T>, String> {
        unsafe {
            self.lib
                .get(name)
                .map_err(|e| format!("{}: {e}", String::from_utf8_lossy(name)))
        }
    }

    /// Allocates device memory and adds the byte count to the running total.
    pub fn alloc(&self, bytes: usize, total: &mut usize) -> Result<Deviceptr, String> {
        unsafe {
            let f: libloading::Symbol<unsafe extern "C" fn(*mut CuDeviceptr, usize) -> CuResult> =
                self.sym(b"cuMemAlloc_v2")?;
            let mut p: CuDeviceptr = 0;
            let r = f(&mut p, bytes);
            if r != 0 {
                return Err(format!("cuMemAlloc({bytes}) failed: {r}"));
            }
            *total += bytes;
            Ok(Deviceptr(p))
        }
    }

    /// Allocates managed (UVA/unified) memory with CU_MEM_ATTACH_GLOBAL.
    ///
    /// The returned pointer is valid on both the host and the device on a UMA system, so a caller
    /// can fill an arena once and launch kernels against it without a separate copy. Device code
    /// may migrate pages; the allocation is freed with [Cuda::free_managed] (or cuMemFree).
    pub fn alloc_managed(&self, bytes: usize, total: &mut usize) -> Result<Deviceptr, String> {
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(*mut CuDeviceptr, usize, u32) -> CuResult,
            > = self.sym(b"cuMemAllocManaged")?;
            let mut p: CuDeviceptr = 0;
            const CU_MEM_ATTACH_GLOBAL: u32 = 1;
            let r = f(&mut p, bytes, CU_MEM_ATTACH_GLOBAL);
            if r != 0 {
                return Err(format!("cuMemAllocManaged({bytes}) failed: {r}"));
            }
            *total += bytes;
            Ok(Deviceptr(p))
        }
    }

    /// Allocates a page-locked host slab and resolves its device mapping.
    ///
    /// The slab is portable and device-mapped, so the device can read it through UVA while the
    /// host keeps the original pointer for filling and inspection. The returned [HostMapped] owns
    /// both pointers; free it with [Cuda::free_host_mapped], not cuMemFree.
    pub fn alloc_host_mapped(&self, bytes: usize, total: &mut usize) -> Result<HostMapped, String> {
        unsafe {
            let alloc: libloading::Symbol<
                unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> CuResult,
            > = self.sym(b"cuMemHostAlloc")?;
            const CU_MEMHOSTALLOC_PORTABLE: u32 = 1;
            const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 2;
            let mut host: *mut c_void = std::ptr::null_mut();
            let r = alloc(
                &mut host,
                bytes,
                CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP,
            );
            if r != 0 {
                return Err(format!("cuMemHostAlloc({bytes}) failed: {r}"));
            }
            let get: libloading::Symbol<
                unsafe extern "C" fn(*mut CuDeviceptr, *mut c_void, u32) -> CuResult,
            > = match self.sym(b"cuMemHostGetDevicePointer_v2") {
                Ok(symbol) => symbol,
                Err(error) => {
                    if let Ok(free) =
                        self.sym::<unsafe extern "C" fn(*mut c_void) -> CuResult>(b"cuMemFreeHost")
                    {
                        let _ = free(host);
                    }
                    return Err(error);
                }
            };
            let mut device: CuDeviceptr = 0;
            let r = get(&mut device, host, 0);
            if r != 0 {
                if let Ok(free) =
                    self.sym::<unsafe extern "C" fn(*mut c_void) -> CuResult>(b"cuMemFreeHost")
                {
                    let _ = free(host);
                }
                return Err(format!("cuMemHostGetDevicePointer failed: {r}"));
            }
            *total += bytes;
            Ok(HostMapped {
                host,
                device: Deviceptr(device),
            })
        }
    }

    /// Frees an allocation returned by [Cuda::alloc_managed].
    ///
    /// Managed memory is released through cuMemFree, the same call as an ordinary device
    /// allocation. The byte count is not tracked here; the caller's arena keeps its own total.
    pub fn free_managed(&self, ptr: &Deviceptr) {
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                let _ = f(ptr.0);
            }
        }
    }

    /// Frees an allocation returned by [Cuda::alloc_host_mapped].
    pub fn free_host_mapped(&self, mapped: &HostMapped) {
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(*mut c_void) -> CuResult>(b"cuMemFreeHost")
            {
                let _ = f(mapped.host);
            }
        }
    }

    /// Copies host bytes to a device allocation.
    pub fn copy_htod(&self, dst: &Deviceptr, src: &[u8]) -> Result<(), String> {
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(CuDeviceptr, *const c_void, usize) -> CuResult,
            > = self.sym(b"cuMemcpyHtoD_v2")?;
            let r = f(dst.0, src.as_ptr().cast(), src.len());
            if r != 0 {
                return Err(format!("cuMemcpyHtoD failed: {r}"));
            }
            Ok(())
        }
    }

    /// Copies a device allocation back to host bytes.
    pub fn copy_dtoh(&self, dst: &mut [u8], src: &Deviceptr) -> Result<(), String> {
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(*mut c_void, CuDeviceptr, usize) -> CuResult,
            > = self.sym(b"cuMemcpyDtoH_v2")?;
            let r = f(dst.as_mut_ptr().cast(), src.0, dst.len());
            if r != 0 {
                return Err(format!("cuMemcpyDtoH failed: {r}"));
            }
            Ok(())
        }
    }

    /// Copies between two device allocations.
    ///
    /// The byte count is explicit because the callers copy fixed-size rows (512 bytes of window
    /// codes, 16 bytes of window scales) into contiguous buffers, not whole slices. The real
    /// cuMemcpyDtoD_v2 symbol and copy are exercised by the m8 device run; this doctest is no_run
    /// so the call's arity and signature are compiled in CI without loading libcuda.
    ///
    /// ```no_run
    /// use sparkflash_kernels::cuda_driver::Cuda;
    ///
    /// # fn main() -> Result<(), String> {
    /// let cuda = Cuda::load()?;
    /// let mut allocated = 0usize;
    /// let src = cuda.alloc(512, &mut allocated)?;
    /// let dst = cuda.alloc(512, &mut allocated)?;
    /// cuda.copy_dtod(&dst, &src, 512)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn copy_dtod(&self, dst: &Deviceptr, src: &Deviceptr, bytes: usize) -> Result<(), String> {
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(CuDeviceptr, CuDeviceptr, usize) -> CuResult,
            > = self.sym(b"cuMemcpyDtoD_v2")?;
            let r = f(dst.0, src.0, bytes);
            if r != 0 {
                return Err(format!("cuMemcpyDtoD failed: {r}"));
            }
            Ok(())
        }
    }

    /// Loads one non-flash_mla kernel from a cubin image.
    ///
    /// The cubin symbol is chosen by the KernelId, not by the caller, so the returned handle can
    /// never refer to the flash_mla kernel.
    pub fn load_kernel(&self, cubin: &[u8], id: KernelId) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, id.symbol())?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: id.cubin_names()[0],
            },
        ))
    }

    /// Loads the router's stage-two top-6 merge kernel from the router cubin.
    ///
    /// Stage two is a second symbol in the same cubin as [KernelId::MoeRouter]. It is not a new
    /// [KernelId] because it has no cubin of its own and no independent baseline entry: the
    /// router's cubin pin covers both symbols. The symbol is deliberately not resolvable through
    /// [KernelId::from_cubin_name], so a harness cannot accidentally launch the merge as a router.
    pub fn load_router_topk(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "moe_router_topk_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "moe_router_topk_sm121",
            },
        ))
    }
    /// Loads the K8 batched certified-head symbol by name from the cert cubin.
    pub fn load_cert_lm_head_batched(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "cert_lm_head_batched_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "cert_lm_head_batched_sm121",
            },
        ))
    }

    /// Loads the K1 warp-per-column dense symbol from the dense cubin.
    ///
    /// The scalar `b12x_mxfp8_dense_sm121` symbol remains the pinned reference and the incumbent
    /// harness target; this second symbol is the one the chain switches to after its own
    /// tolerance-qualified run. It shares the cubin pin with the scalar symbol.
    pub fn load_dense_warp(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_dense_warp_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_dense_warp_sm121",
            },
        ))
    }

    /// Loads the K1-b A-reuse tiled dense symbol by name.
    pub fn load_dense_tile(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_dense_tile_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_dense_tile_sm121",
            },
        ))
    }

    /// Loads the K1-c(a) parallel-K dense symbol by name.
    pub fn load_dense_par_k(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_dense_par_k_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_dense_par_k_sm121",
            },
        ))
    }

    /// Loads the K8 batched mHC main and init symbols by name.
    pub fn load_mega_mhc_batched(
        &self,
        cubin: &[u8],
    ) -> Result<(Module, Kernel, Module, Kernel), String> {
        let (mm, mf) = self.load_function(cubin, "mega_mhc_fused_bf16_batched_sm121")?;
        let (im, init) = self.load_function(cubin, "mega_mhc_init_sm121")?;
        Ok((
            Module(mm),
            Kernel {
                function: mf,
                name: "mega_mhc_fused_bf16_batched_sm121",
            },
            Module(im),
            Kernel {
                function: init,
                name: "mega_mhc_init_sm121",
            },
        ))
    }

    /// Loads the K8 batched LM-head reduction symbol by name.
    pub fn load_lm_head_reduce_batched(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "lm_head_reduce_batched_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "lm_head_reduce_batched_sm121",
            },
        ))
    }

    /// Loads both K1-c(b) split-K symbols by name.
    pub fn load_dense_split_k(
        &self,
        cubin: &[u8],
    ) -> Result<(Module, Kernel, Module, Kernel), String> {
        let (pm, pf) = self.load_function(cubin, "b12x_mxfp8_dense_split_k_sm121")?;
        let (cm, cf) = self.load_function(cubin, "b12x_mxfp8_dense_split_k_combine_sm121")?;
        Ok((
            Module(pm),
            Kernel {
                function: pf,
                name: "b12x_mxfp8_dense_split_k_sm121",
            },
            Module(cm),
            Kernel {
                function: cf,
                name: "b12x_mxfp8_dense_split_k_combine_sm121",
            },
        ))
    }

    /// Loads the D49 ring-capacity CSA2 symbol by name.
    pub fn load_flash_mla_csa2_ring(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "flash_mla_csa2_ring_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "flash_mla_csa2_ring_sm121",
            },
        ))
    }

    /// Loads the K8 batched CSA2 sequence symbol by name.
    pub fn load_flash_mla_csa2_seq_batched(
        &self,
        cubin: &[u8],
    ) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "flash_mla_csa2_seq_batched_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "flash_mla_csa2_seq_batched_sm121",
            },
        ))
    }

    /// Loads the K1 warp-per-column wo_a symbol from the wo_a cubin.
    pub fn load_wo_a_warp(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_wo_a_warp_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_wo_a_warp_sm121",
            },
        ))
    }

    /// Loads the K1-b A-reuse tiled wo_a symbol by name.
    pub fn load_wo_a_tile(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_wo_a_tile_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_wo_a_tile_sm121",
            },
        ))
    }

    /// Loads the K8 batched wo_a symbol by name.
    pub fn load_wo_a_batched(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_wo_a_batched_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_wo_a_batched_sm121",
            },
        ))
    }

    /// Loads the flash_mla kernel. Used only by crate::cuda_harness's typed loader.
    pub(crate) fn load_flash_mla(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "flash_mla_csa2_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: FLASH_MLA_CUBIN,
            },
        ))
    }

    fn load_function(&self, cubin: &[u8], name: &str) -> Result<(CuModule, CuFunction), String> {
        unsafe {
            let load: libloading::Symbol<
                unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
            > = self.sym(b"cuModuleLoadData")?;
            let mut module: CuModule = std::ptr::null_mut();
            let r = load(&mut module, cubin.as_ptr().cast());
            if r != 0 {
                return Err(format!("cuModuleLoadData failed: {r}"));
            }
            let get: libloading::Symbol<
                unsafe extern "C" fn(*mut CuFunction, CuModule, *const i8) -> CuResult,
            > = self.sym(b"cuModuleGetFunction")?;
            let cname = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
            let mut func: CuFunction = std::ptr::null_mut();
            let r = get(&mut func, module, cname.as_ptr().cast());
            if r != 0 {
                return Err(format!("cuModuleGetFunction({name}) failed: {r}"));
            }
            Ok((module, func))
        }
    }

    /// Launches one non-flash_mla kernel, then synchronises the context.
    ///
    /// With SPARKFLASH_STAGE_TIMING=1 the launch is bracketed by one cuEventCreate pair recorded
    /// immediately around cuLaunchKernel and resolved after the existing cuCtxSynchronize; the
    /// bracketed launches are the same launches in the same order and nothing else changes. Without
    /// the variable no event is created, no allocation happens and the launch path is unchanged.
    pub fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        unsafe {
            let launch: libloading::Symbol<LaunchFn> = self.sym(b"cuLaunchKernel")?;
            let mut storage: Vec<[u64; 1]> = args.iter().map(|a| [*a]).collect();
            let mut ptrs: Vec<*mut c_void> =
                storage.iter_mut().map(|s| s.as_mut_ptr().cast()).collect();
            let timing = stage_timing_enabled();
            let mut start_event: CuEvent = std::ptr::null_mut();
            let mut stop_event: CuEvent = std::ptr::null_mut();
            if timing {
                let create: libloading::Symbol<
                    unsafe extern "C" fn(*mut CuEvent, u32) -> CuResult,
                > = self.sym(b"cuEventCreate")?;
                let record: libloading::Symbol<
                    unsafe extern "C" fn(CuEvent, *mut c_void) -> CuResult,
                > = self.sym(b"cuEventRecord")?;
                let mut r = create(&mut start_event, 0);
                if r == 0 {
                    r = create(&mut stop_event, 0);
                }
                if r != 0 {
                    return Err(format!(
                        "cuEventCreate for the stage timing bracket failed: {r}"
                    ));
                }
                let r = record(start_event, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord(start) failed: {r}"));
                }
            }
            let r = launch(
                kernel.function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                std::ptr::null_mut(),
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if r != 0 {
                return Err(format!("cuLaunchKernel failed: {r}"));
            }
            if timing {
                let record: libloading::Symbol<
                    unsafe extern "C" fn(CuEvent, *mut c_void) -> CuResult,
                > = self.sym(b"cuEventRecord")?;
                let r = record(stop_event, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord(stop) failed: {r}"));
                }
            }
            let sync: libloading::Symbol<unsafe extern "C" fn() -> CuResult> =
                self.sym(b"cuCtxSynchronize")?;
            let r = sync();
            if r != 0 {
                return Err(format!("cuCtxSynchronize failed: {r}"));
            }
            if timing {
                let elapsed: libloading::Symbol<
                    unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult,
                > = self.sym(b"cuEventElapsedTime")?;
                let destroy: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
                    self.sym(b"cuEventDestroy")?;
                let mut ms: f32 = 0.0;
                let r = elapsed(&mut ms, start_event, stop_event);
                destroy(start_event);
                destroy(stop_event);
                if r != 0 {
                    return Err(format!("cuEventElapsedTime failed: {r}"));
                }
                record_launch(kernel.name, (f64::from(ms) * 1_000_000.0) as u64);
            }
            Ok(())
        }
    }

    /// Times one non-flash_mla kernel: warm-up, per-launch events, then one batched region.
    ///
    /// A throughput claim needs the warm-up outside the timed region and one event pair per launch
    /// for the distribution; the batched region has no per-launch synchronisation and is the figure
    /// to quote when the per-launch event overhead would dominate a kernel this small. No
    /// allocation, no upload and no checkpoint read is inside either timed region, so the numbers
    /// are the kernel's own and not the host's.
    pub fn launch_repeated(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
        repeat: usize,
    ) -> Result<(Vec<f64>, f64), String> {
        unsafe {
            let create: libloading::Symbol<unsafe extern "C" fn(*mut CuEvent, u32) -> CuResult> =
                self.sym(b"cuEventCreate")?;
            let record: libloading::Symbol<unsafe extern "C" fn(CuEvent, *mut c_void) -> CuResult> =
                self.sym(b"cuEventRecord")?;
            let wait: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
                self.sym(b"cuEventSynchronize")?;
            let elapsed: libloading::Symbol<
                unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult,
            > = self.sym(b"cuEventElapsedTime")?;
            let destroy: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
                self.sym(b"cuEventDestroy")?;
            let mut start: CuEvent = std::ptr::null_mut();
            let mut stop: CuEvent = std::ptr::null_mut();
            let r = create(&mut start, 0);
            if r != 0 {
                return Err(format!("cuEventCreate failed: {r}"));
            }
            let r = create(&mut stop, 0);
            if r != 0 {
                return Err(format!("cuEventCreate failed: {r}"));
            }
            // Warm-up: fills the module cache and the launch path, and is deliberately not timed.
            self.launch(kernel, grid, block, args)?;
            let sync: libloading::Symbol<unsafe extern "C" fn() -> CuResult> =
                self.sym(b"cuCtxSynchronize")?;
            let r = sync();
            if r != 0 {
                return Err(format!("cuCtxSynchronize failed: {r}"));
            }
            let mut per_launch = Vec::with_capacity(repeat);
            for _ in 0..repeat {
                let r = record(start, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord failed: {r}"));
                }
                self.launch(kernel, grid, block, args)?;
                let r = record(stop, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord failed: {r}"));
                }
                let r = wait(stop);
                if r != 0 {
                    return Err(format!("cuEventSynchronize failed: {r}"));
                }
                let mut ms: f32 = 0.0;
                let r = elapsed(&mut ms, start, stop);
                if r != 0 {
                    return Err(format!("cuEventElapsedTime failed: {r}"));
                }
                per_launch.push(f64::from(ms));
            }
            let r = record(start, std::ptr::null_mut());
            if r != 0 {
                return Err(format!("cuEventRecord failed: {r}"));
            }
            for _ in 0..repeat {
                self.launch(kernel, grid, block, args)?;
            }
            let r = record(stop, std::ptr::null_mut());
            if r != 0 {
                return Err(format!("cuEventRecord failed: {r}"));
            }
            let r = wait(stop);
            if r != 0 {
                return Err(format!("cuEventSynchronize failed: {r}"));
            }
            let mut total: f32 = 0.0;
            let r = elapsed(&mut total, start, stop);
            if r != 0 {
                return Err(format!("cuEventElapsedTime failed: {r}"));
            }
            destroy(start);
            destroy(stop);
            Ok((per_launch, f64::from(total) / repeat as f64))
        }
    }

    /// Frees device buffers and unloads the module that produced them.
    pub fn free(&self, module: Module, ptrs: &[Deviceptr]) {
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                for p in ptrs {
                    let _ = f(p.0);
                }
            }
            if let Ok(f) = self.sym::<unsafe extern "C" fn(CuModule) -> CuResult>(b"cuModuleUnload")
            {
                let _ = f(module.0);
            }
            // The primary context is released by Drop, on every exit path.
        }
    }

    /// Frees device buffers without unloading the module that produced them.
    ///
    /// Stage 7 runs several different-shaped launches per loaded module, so the module has to
    /// outlive them.
    pub fn free_buffers(&self, ptrs: &[&Deviceptr]) {
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                for p in ptrs {
                    let _ = f(p.0);
                }
            }
        }
    }
}

impl Drop for Cuda {
    /// Releases this process's reference to the primary context on every exit path, including a
    /// failure after the context was created, so a run can never leave a half-initialised Cuda.
    /// Releasing is refcounted and does not disturb anything else driving the device.
    fn drop(&mut self) {
        if self.ctx.is_null() {
            return;
        }
        unsafe {
            if let Ok(release_v2) =
                self.sym::<unsafe extern "C" fn(i32) -> CuResult>(b"cuDevicePrimaryCtxRelease_v2")
            {
                let _ = release_v2(self.dev);
            }
        }
    }
}

#[cfg(test)]
mod stage_timing_tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset() {
        *STAGE_RECORDER.lock().expect("lock") = Some(StageRecorder::default());
    }

    fn scope(phase: &'static str, step: usize, layer: usize, position: usize) {
        let mut guard = STAGE_RECORDER.lock().expect("lock");
        let recorder = guard.as_mut().expect("armed");
        recorder.scope = StageScope {
            phase,
            step,
            layer,
            position,
        };
    }

    #[test]
    fn the_ledger_aggregates_per_kernel_phase_and_step() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        reset();
        scope("decode", 2, 5, 7);
        record_launch("b12x_mxfp8_dense_sm121", 1_000);
        record_launch("b12x_mxfp8_dense_sm121", 3_000);
        record_launch("moe_router_sm121", 500);
        let report = stage_timing_report().expect("report");
        assert_eq!(report.kernels.len(), 2);
        let dense = &report.kernels[0];
        assert_eq!(dense.phase, "decode");
        assert_eq!(dense.kernel, "b12x_mxfp8_dense_sm121");
        assert_eq!(dense.calls, 2);
        assert_eq!(dense.total_ns, 4_000);
        assert_eq!(dense.mean_ns, 2_000);
        assert_eq!(dense.min_ns, 1_000);
        assert_eq!(dense.max_ns, 3_000);
        assert_eq!(dense.first_launch_order, 0);
        assert_eq!(report.steps.len(), 1);
        assert_eq!(report.steps[0].step, 2);
        assert_eq!(report.steps[0].launches, 3);
        assert_eq!(report.steps[0].kernel_ns, 4_500);
        *STAGE_RECORDER.lock().expect("lock") = None;
    }

    #[test]
    fn the_ledger_caps_the_distinct_kernel_names() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        reset();
        scope("decode", 0, 0, 0);
        let names: Vec<&'static str> = (0..STAGE_TIMING_KERNEL_CAP + 8)
            .map(|index| Box::leak(format!("kernel_{index}").into_boxed_str()) as &'static str)
            .collect();
        for name in &names {
            record_launch(name, 10);
        }
        let report = stage_timing_report().expect("report");
        assert_eq!(report.kernels.len(), STAGE_TIMING_KERNEL_CAP);
        assert_eq!(report.dropped_names, 8);
        *STAGE_RECORDER.lock().expect("lock") = None;
    }

    #[test]
    fn the_ledger_is_absent_until_armed_and_the_bracket_is_off_by_default() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        *STAGE_RECORDER.lock().expect("lock") = None;
        assert!(stage_timing_report().is_none());
        assert_eq!(stage_timing_launches(), 0);
    }

    #[test]
    fn warmup_steps_are_marked_so_cold_and_steady_state_can_be_reported_separately() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        reset();
        stage_timing_set_warm(true);
        scope("decode", 0, 0, 0);
        record_launch("b12x_mxfp8_dense_sm121", 9_000);
        stage_timing_set_warm(false);
        scope("decode", 1, 0, 0);
        record_launch("b12x_mxfp8_dense_sm121", 1_000);
        let report = stage_timing_report().expect("report");
        assert_eq!(report.steps.len(), 2);
        assert!(report.steps[0].warm, "step 0 is the warmup step");
        assert!(!report.steps[1].warm, "step 1 is steady state");
        *STAGE_RECORDER.lock().expect("lock") = None;
    }
}

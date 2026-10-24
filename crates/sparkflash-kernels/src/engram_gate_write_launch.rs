//! The typed launch capability of the Engram gate and residual write
//! (kernels/engram_gate_write_sm121.cu; task-9, audit F4).
//
// The argument array is the kernel declaration's parameter list in order - five device pointers,
// the head width, the copy count, the epsilon, the trace ring, the layer id and the step sequence.
//
// Geometry. One thread per hc copy owns that copy's whole ascending sum, so the kernel's grid is one
// block of 256 threads and `hc` must fit the kernel's shared gate array (SPARK_ENGRAM_MAX_HC,
// 8). The write walks `hc * dim` elements with the block's threads striding, so a head width
// larger than the block is expected, not an error. A zero head width or a copy count over the
// kernel's bound is refused here rather than launched.

use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};

/// The kernel id in the shared driver's enum, named here so a caller need not import the driver.
pub const KERNEL_ID: KernelId = KernelId::EngramGateWrite;

/// The number of argument words the kernel declaration takes.
pub const ARG_WORDS: usize = 11;

/// Threads per block. The kernel's grid is one block (see the module note).
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);

/// The kernel's shared gate array bound (SPARK_ENGRAM_MAX_HC).
pub const MAX_HC: u32 = 8;

/// The grid: one block, because the kernel's own threads cover the head.
#[must_use]
pub fn grid() -> (u32, u32, u32) {
    (1, 1, 1)
}

/// One launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngramGateWriteLaunchParams {
    /// `[hc * dim + dim]` bf16 wkv output: key then value (`model.py:355-357`).
    pub kv: Deviceptr,
    /// `[hc * dim]` bf16 residual stream (the released `x`, rounded by the caller).
    pub stream: Deviceptr,
    /// `[hc * dim]` f32 `q_weight * k_weight`.
    pub product: Deviceptr,
    /// `[hc * dim]` bf16 write result.
    pub out: Deviceptr,
    /// `[hc]` the gates, for the caller's record.
    pub gates: Deviceptr,
    /// Columns per copy (the released `dim` of the Engram head).
    pub dim: u32,
    /// Copies of the head (hc; 4 in the released configuration).
    pub hc: u32,
    /// The configured epsilon (the released Engram `eps`).
    pub eps: f32,
    /// The trace ring's device address, or 0 for the trace-disabled cubin.
    pub trace_ring: u64,
    /// The layer id a trace event would record.
    pub layer_id: u32,
    /// The step sequence a trace event would record.
    pub step_seq: u64,
}

impl EngramGateWriteLaunchParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; ARG_WORDS] {
        [
            self.kv.0,
            self.stream.0,
            self.product.0,
            self.out.0,
            self.gates.0,
            u64::from(self.dim),
            u64::from(self.hc),
            u64::from(self.eps.to_bits()),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// Loads the kernel from a cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain `engram_gate_write_sm121`.
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches the kernel for one parameter set.
///
/// # Errors
/// A zero head width, a zero copy count or one over the kernel's bound is refused; otherwise the
/// driver's launch error.
pub fn launch(
    cuda: &Cuda,
    kernel: Kernel,
    params: &EngramGateWriteLaunchParams,
) -> Result<(), String> {
    if params.dim == 0 {
        return Err("the Engram gate reads at least one column, so dim is not zero".to_owned());
    }
    if params.hc == 0 || params.hc > MAX_HC {
        return Err(format!(
            "the Engram gate holds the copies in a shared array of {MAX_HC}, so hc {} is out of range",
            params.hc
        ));
    }
    let word = OPT_FUNCTION.load(std::sync::atomic::Ordering::Acquire);
    if word != 0 && params.dim.is_multiple_of(2) {
        return cuda.launch(
            Kernel::from_raw(word, OPT_SYMBOL),
            (params.hc, 1, 1),
            BLOCK,
            &params.words(),
        );
    }
    cuda.launch(kernel, grid(), BLOCK, &params.words())
}

static OPT_FUNCTION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const OPT_SYMBOL: &str = "opt_engram_gate_write";

/// Loads the block-per-copy gate write (opt_misc_sm121.cubin) and routes [`launch`] through it.
///
/// # Errors
/// The driver's error when the cubin lacks the symbol.
pub fn install_opt(cuda: &Cuda, cubin: &[u8]) -> Result<Module, String> {
    let (module, kernel) = cuda.load_named(cubin, OPT_SYMBOL)?;
    OPT_FUNCTION.store(kernel.to_raw().0, std::sync::atomic::Ordering::Release);
    Ok(module)
}

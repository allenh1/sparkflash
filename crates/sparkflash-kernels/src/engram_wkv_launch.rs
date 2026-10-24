//! The typed launch capability of the Engram wkv projection (kernels/engram_wkv_sm121.cu; task-9,
//! audit F4).
//
// The argument array is the kernel declaration's parameter list in order - four device pointers,
// the output width and inner width, the trace ring, the layer id and the step sequence.
//
// Geometry. One warp owns one output element's dot product, one lane per column of a 32-column K
// block, so a 256-thread block is eight outputs and the grid is `ceil(out_width / 8)` blocks.
// The kernel reads the checkpoint's raw `[out_width][inner]` fp8 codes and its
// `[out_width / 32][inner / 32]` E8M0 grid directly: no repack, and a zero width or an inner
// width that is not whole 32-value blocks is refused here rather than launched.

use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};

/// The kernel id in the shared driver's enum, named here so a caller need not import the driver.
pub const KERNEL_ID: KernelId = KernelId::EngramWkv;

/// The cubin's pinned name (crates/sparkflash-kernels/baseline/kernel_baseline.txt), the key a
/// caller verifies the staged bytes by before it loads them.
pub const KERNEL: &str = "engram_wkv_sm121";

/// The number of argument words the kernel declaration takes.
pub const ARG_WORDS: usize = 10;

/// Warps per block (SPARK_ENGRAM_WKV_WARPS).
pub const WARPS: u32 = 8;

/// Threads per block: one warp per output element.
pub const BLOCK: (u32, u32, u32) = (WARPS * 32, 1, 1);

/// The K block of the released fp8 W8A8 rule (inference/kernel.py:41-112).
pub const K_BLOCK: u32 = 32;

/// The grid for `out_width` outputs: `ceil(out_width / 8)` blocks.
///
/// # Errors
/// `out_width == 0` is refused rather than launched as a zero-sized grid.
pub fn grid(out_width: u32) -> Result<(u32, u32, u32), String> {
    if out_width == 0 {
        return Err(
            "the Engram wkv projects at least one output, so the grid is not zero-sized".to_owned(),
        );
    }
    Ok((out_width.div_ceil(WARPS), 1, 1))
}

/// One launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngramWkvLaunchParams {
    /// `[inner]` E4M3 activation codes (the bf16 row quantised by `engram_embed_sm121`).
    pub act_codes: Deviceptr,
    /// `[inner / 32]` E8M0 activation scales.
    pub act_scales: Deviceptr,
    /// `[out_width][inner]` E4M3 weight codes, checkpoint order.
    pub weight: Deviceptr,
    /// `[out_width / 32][inner / 32]` E8M0 weight scales, the checkpoint's grid.
    pub weight_scales: Deviceptr,
    /// `[out_width]` bf16 output: the Linear's bf16 result.
    pub out: Deviceptr,
    /// Outputs (2 * n_embed + 2 * n_embed_proj in the released configuration).
    pub out_width: u32,
    /// The reduction width (n_embed; 6,144 in the released configuration).
    pub inner: u32,
    /// The trace ring's device address, or 0 for the trace-disabled cubin.
    pub trace_ring: u64,
    /// The layer id a trace event would record.
    pub layer_id: u32,
    /// The step sequence a trace event would record.
    pub step_seq: u64,
}

impl EngramWkvLaunchParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; ARG_WORDS] {
        [
            self.act_codes.0,
            self.act_scales.0,
            self.weight.0,
            self.weight_scales.0,
            self.out.0,
            u64::from(self.out_width),
            u64::from(self.inner),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// Loads the kernel from a cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain `engram_wkv_sm121`.
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches the kernel for one parameter set.
///
/// # Errors
/// A zero output width, a zero inner width or an inner width that is not a whole number of 32-value
/// blocks is refused; otherwise the driver's launch error.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &EngramWkvLaunchParams) -> Result<(), String> {
    if params.inner == 0 {
        return Err("the Engram wkv reduces at least one column, so inner is not zero".to_owned());
    }
    if !params.inner.is_multiple_of(K_BLOCK) {
        return Err(format!(
            "the released fp8 GEMM reduces whole blocks of {K_BLOCK}, so inner {} is not a released shape",
            params.inner
        ));
    }
    let grid = grid(params.out_width)?;
    let word = OPT_FUNCTION.load(std::sync::atomic::Ordering::Acquire);
    if word != 0 {
        return cuda.launch(
            Kernel::from_raw(word, OPT_SYMBOL),
            grid,
            BLOCK,
            &params.words(),
        );
    }
    cuda.launch(kernel, grid, BLOCK, &params.words())
}

static OPT_FUNCTION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const OPT_SYMBOL: &str = "opt_engram_wkv";

/// Loads the sixteen-byte-load wkv (opt_misc_sm121.cubin) and routes [`launch`] through it.
///
/// # Errors
/// The driver's error when the cubin lacks the symbol.
pub fn install_opt(cuda: &Cuda, cubin: &[u8]) -> Result<Module, String> {
    let (module, kernel) = cuda.load_named(cubin, OPT_SYMBOL)?;
    OPT_FUNCTION.store(kernel.to_raw().0, std::sync::atomic::Ordering::Release);
    Ok(module)
}

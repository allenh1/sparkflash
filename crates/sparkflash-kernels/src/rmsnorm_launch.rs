//! The typed launch capability of the standalone final RMSNorm (kernels/rmsnorm_sm121.cu).
//!
//! The argument array is the kernel declaration's parameter list in order - three device pointers,
//! the row and column counts, the epsilon bits, the trace ring, the layer id and the step sequence -
//! so the only way to fill a word wrongly is to fill a field wrongly, and the fields are named.
//!
//! Geometry. One thread owns one whole row (the engine's row-serial order; see the kernel header),
//! so the grid is \`ceil(rows / 256)\` blocks of 256 threads. A zero row or column count is refused
//! here rather than launched as a zero grid.
//!
//! Tracing. The kernel carries the trace words for ABI uniformity but emits no event yet: the
//! protocol's sixteen scope bits are all assigned and none describes a tail norm (kernel header,
//! TRACE). This module passes a null ring and a caller can pass an address without changing the
//! argument order.

use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};

/// The kernel id in the shared driver's enum, named here so a caller need not import the driver.
pub const KERNEL_ID: KernelId = KernelId::RmsNorm;

/// The number of argument words the kernel declaration takes.
pub const ARG_WORDS: usize = 9;

/// Threads per block. One thread owns one row, so the block size is only occupancy geometry.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);

/// The grid for \`rows\` rows: \`ceil(rows / 256)\` blocks.
///
/// # Errors
/// \`rows == 0\` is refused rather than launched as a zero-sized grid.
pub fn grid(rows: u32) -> Result<(u32, u32, u32), String> {
    if rows == 0 {
        return Err(
            "the RMSNorm normalises at least one row, so the grid is not zero-sized".to_owned(),
        );
    }
    Ok((rows.div_ceil(BLOCK.0), 1, 1))
}

/// One launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNormLaunchParams {
    /// \`[rows][cols]\` BF16 input.
    pub x: Deviceptr,
    /// \`[cols]\` BF16 weight (the released norm.weight bytes).
    pub weight: Deviceptr,
    /// \`[rows][cols]\` BF16 output.
    pub out: Deviceptr,
    /// Rows in the launch (tokens in the serving tail).
    pub rows: u32,
    /// Columns per row (5120 in the released configuration).
    pub cols: u32,
    /// The configured epsilon (\`rms_norm_eps\` cast to f32; released value 1e-20).
    pub eps: f32,
    /// The trace ring's device address, or 0 for the trace-disabled cubin.
    pub trace_ring: u64,
    /// The layer id the trace event would record.
    pub layer_id: u32,
    /// The step sequence the trace event would record.
    pub step_seq: u64,
}

impl RmsNormLaunchParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; ARG_WORDS] {
        [
            self.x.0,
            self.weight.0,
            self.out.0,
            u64::from(self.rows),
            u64::from(self.cols),
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
/// Returns the driver's error when the cubin does not contain \`rmsnorm_sm121\`.
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

static OPT_FUNCTION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const OPT_SYMBOL: &str = "opt_rmsnorm_bf16";

/// Loads the block-per-row RMSNorm (opt_misc_sm121.cubin) and routes [`launch`] through it.
///
/// # Errors
/// The driver's error when the cubin lacks the symbol.
pub fn install_opt(cuda: &Cuda, cubin: &[u8]) -> Result<Module, String> {
    let (module, kernel) = cuda.load_named(cubin, OPT_SYMBOL)?;
    OPT_FUNCTION.store(kernel.to_raw().0, std::sync::atomic::Ordering::Release);
    Ok(module)
}

/// Launches the kernel for one parameter set.
///
/// # Errors
/// A zero row count, or the driver's launch error.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &RmsNormLaunchParams) -> Result<(), String> {
    if params.cols == 0 {
        return Err("the RMSNorm normalises at least one column, so cols is not zero".to_owned());
    }
    let word = OPT_FUNCTION.load(std::sync::atomic::Ordering::Acquire);
    if word != 0 {
        return cuda.launch(
            Kernel::from_raw(word, OPT_SYMBOL),
            (params.rows, 1, 1),
            (256, 1, 1),
            &params.words(),
        );
    }
    let grid = grid(params.rows)?;
    cuda.launch(kernel, grid, BLOCK, &params.words())
}

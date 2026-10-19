//! The typed launch capability of the Engram row reader and activation quantiser
//! (kernels/engram_embed_sm121.cu; task-9, audit F4).
//
// The argument array is the kernel declaration's parameter list in order - five device pointers,
// the row count and head width, the trace ring, the layer id and the step sequence - so a launch
// fills named fields rather than an anonymous word array.
//
// Geometry. One block of 256 threads, deliberately: the kernel's grid is one block and its threads
// walk the whole [rows_cols][head_dim] row set and then every 32-value quantisation block, so the
// quantiser's ascending-index order is the kernel's contract rather than a grid choice. A head
// width that is not a whole number of 32-value blocks is refused here instead of launched into a
// partial block the released act_quant rule does not define.

use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};

/// The kernel id in the shared driver's enum, named here so a caller need not import the driver.
pub const KERNEL_ID: KernelId = KernelId::EngramEmbed;

/// The number of argument words the kernel declaration takes.
pub const ARG_WORDS: usize = 10;

/// Threads per block. The kernel's grid is one block (see the module note).
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);

/// The activation quantisation block, from the released rule (inference/kernel.py:41-112).
pub const QUANT_BLOCK: u32 = 32;

/// The grid: one block, because the kernel's own loops cover the tensors.
#[must_use]
pub fn grid() -> (u32, u32, u32) {
    (1, 1, 1)
}

/// One launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngramEmbedLaunchParams {
    /// `[rows_cols][head_dim]` E4M3 codes, checkpoint order (the on-disk table row).
    pub row_codes: Deviceptr,
    /// `[rows_cols][head_dim / 32]` E8M0 block scales.
    pub row_scales: Deviceptr,
    /// `[rows_cols * head_dim]` bf16 fetched row (the kernel's `model.py:318` cast).
    pub fetched: Deviceptr,
    /// `[rows_cols * head_dim]` E4M3 activation codes.
    pub act_codes: Deviceptr,
    /// `[rows_cols * head_dim / 32]` E8M0 activation scales.
    pub act_scales: Deviceptr,
    /// `(max_ngram_size - 1) * heads` row groups (24 in the released configuration).
    pub rows_cols: u32,
    /// Columns per row (256 in the released configuration).
    pub head_dim: u32,
    /// The trace ring's device address, or 0 for the trace-disabled cubin.
    pub trace_ring: u64,
    /// The layer id a trace event would record.
    pub layer_id: u32,
    /// The step sequence a trace event would record.
    pub step_seq: u64,
}

impl EngramEmbedLaunchParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; ARG_WORDS] {
        [
            self.row_codes.0,
            self.row_scales.0,
            self.fetched.0,
            self.act_codes.0,
            self.act_scales.0,
            u64::from(self.rows_cols),
            u64::from(self.head_dim),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// Loads the kernel from a cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain `engram_embed_sm121`.
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches the kernel for one parameter set.
///
/// # Errors
/// A zero row count or a head width that is not a whole number of quantisation blocks is refused
/// rather than launched; otherwise the driver's launch error.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &EngramEmbedLaunchParams) -> Result<(), String> {
    if params.rows_cols == 0 {
        return Err(
            "the Engram reader reads at least one row group, so rows_cols is not zero".to_owned(),
        );
    }
    if params.head_dim == 0 {
        return Err("the Engram reader reads a non-empty head, so head_dim is not zero".to_owned());
    }
    if !params.head_dim.is_multiple_of(QUANT_BLOCK) {
        return Err(format!(
            "the released act_quant rule quantises whole blocks of {QUANT_BLOCK}, so head_dim {} is not a released shape",
            params.head_dim
        ));
    }
    cuda.launch(kernel, grid(), BLOCK, &params.words())
}

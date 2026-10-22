//! Typed launch capability of `kernels/ref_select_sm121.cu`: the reference candidate select.
//!
//! One CTA per query, 256 threads ([BLOCK]); a decode step launches it once per index-source layer
//! and it reads the query row, the index keys and the per-query weights, then writes the selected
//! ids in the shared index space. The kernel's rules, limits and the total order are stated in the
//! kernel's header and mirrored by [crate::ref_select_ref]; this module owns only the ABI.
//!
//! THE ARTIFACT IS PINNED: CUBIN_BYTES and CUBIN_SHA256 are what nvcc V13.0.88 compiles
//! kernels/ref_select_sm121.cu into for sm_121a with the baseline flags, recorded by
//! scripts/regen-kernel-baseline.sh and checked against baseline/kernel_baseline.txt by the
//! tests.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, Module};

/// The kernel symbol and the baseline entry's name.
pub const KERNEL: &str = "ref_select_sm121";

/// The kernel source, for the declaration test.
pub const SOURCE: &str = include_str!("../../../kernels/ref_select_sm121.cu");

/// The compiled artifact byte count, as nvcc V13.0.88 (CUDA 13.0) compiles
/// kernels/ref_select_sm121.cu for sm_121a with the baseline flags.
pub const CUBIN_BYTES: usize = 28264;
/// The compiled artifact SHA-256, from the same build as [CUBIN_BYTES].
pub const CUBIN_SHA256: &str = "b90baccac9937386c54c5fa69da88e41d759d3218ff2a1ac220dcfbba860f3e8";

/// The argument words of one launch.
pub const ARG_WORDS: usize = 18;

/// The thread block the kernel declares.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);

/// The most compressed positions one launch can score.
pub const MAX_POSITIONS: usize = 4096;

/// The mask is always this many 32-bit words; the kernel writes all of them.
pub const MASK_WORDS: usize = MAX_POSITIONS / 32;

/// `flags` bit 0: write every position's post-mask score to `out_scores`.
pub const FLAG_WRITE_SCORES: u32 = 1;

/// The declaration's parameters in order: `(name, base type, pointer, pointee const)`.
#[must_use]
pub fn params() -> [(&'static str, &'static str, bool, bool); ARG_WORDS] {
    [
        ("q_index", "float", true, true),
        ("keys", "float", true, true),
        ("weights", "float", true, true),
        ("out_ids", "uint32_t", true, false),
        ("out_scores", "float", true, false),
        ("mask_in", "uint32_t", true, true),
        ("mask_out", "uint32_t", true, false),
        ("n_pos", "uint32_t", false, false),
        ("heads", "uint32_t", false, false),
        ("head_dim", "uint32_t", false, false),
        ("reach", "uint32_t", false, false),
        ("topk", "uint32_t", false, false),
        ("offset", "uint32_t", false, false),
        ("mask_mode", "uint32_t", false, false),
        ("topk_blocks", "uint32_t", false, false),
        ("block_size", "uint32_t", false, false),
        ("no_index", "uint32_t", false, false),
        ("flags", "uint32_t", false, false),
    ]
}

/// Loads the kernel from a cubin, resolving the pinned symbol by name.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [KERNEL].
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_ref_select(cubin)
}

/// Parses the pinned declaration out of the kernel source.
#[must_use]
pub fn declaration_of() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, KERNEL)
}

/// Whether the parsed declaration is the pinned shape and order.
#[must_use]
pub fn declaration_matches() -> bool {
    let decl = declaration_of();
    let pinned = params();
    decl.name == KERNEL
        && decl.extern_c
        && decl.global
        && decl.params.len() == ARG_WORDS
        && decl.params.iter().zip(pinned).all(|(param, pin)| {
            param.name == pin.0
                && param.base == pin.1
                && param.pointer == pin.2
                && param.const_pointee == pin.3
                && (!param.pointer || param.restrict)
        })
}

/// The eighteen argument words of one launch, in the kernel's order.
#[derive(Debug, Clone, Copy)]
pub struct RefSelectParams([u64; ARG_WORDS]);

impl RefSelectParams {
    /// Builds the argument array with the kernel's own parameter names, in order.
    ///
    /// `out_scores` and `mask_in`/`mask_out` may be null device pointers when the stage does not
    /// write scores or does not use the mask; the kernel checks them for null in exactly those
    /// cases and never dereferences a null it was not told to use.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        q_index: Deviceptr,
        keys: Deviceptr,
        weights: Deviceptr,
        out_ids: Deviceptr,
        out_scores: Deviceptr,
        mask_in: Deviceptr,
        mask_out: Deviceptr,
        n_pos: u32,
        heads: u32,
        head_dim: u32,
        reach: u32,
        topk: u32,
        offset: u32,
        mask_mode: u32,
        topk_blocks: u32,
        block_size: u32,
        no_index: u32,
        flags: u32,
    ) -> Self {
        Self([
            q_index.0,
            keys.0,
            weights.0,
            out_ids.0,
            out_scores.0,
            mask_in.0,
            mask_out.0,
            u64::from(n_pos),
            u64::from(heads),
            u64::from(head_dim),
            u64::from(reach),
            u64::from(topk),
            u64::from(offset),
            u64::from(mask_mode),
            u64::from(topk_blocks),
            u64::from(block_size),
            u64::from(no_index),
            u64::from(flags),
        ])
    }

    /// The argument array, as the driver takes it.
    #[must_use]
    pub fn words(&self) -> [u64; ARG_WORDS] {
        self.0
    }
}

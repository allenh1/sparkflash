//! Typed launch capability of the grouped wo_a output projection
//! (`kernels/b12x_mxfp8_wo_a_sm121.cu`).
//!
//! The argument array is the `b12x_mxfp8_wo_a_sm121` declaration's parameter list in order - five
//! device pointers and the group count - so a launch cannot swap a pointer for the group count.
//! The declaration is pinned here against the frozen ABI's embedded copy
//! ([m4_abi::MXFP8_WO_A]); [PARAMS] is the pinned name and type of every parameter and
//! [declaration_matches] is the check, with a mutation test proving it is not vacuous.
//!
//! Geometry. The group is a grid dimension, not a GEMM shape: the grid is
//! `(groups, O_LORA_RANK / TILE, 1)` blocks of 256 threads, and each block computes one 256-wide
//! output-channel tile of one group. The kernel's `O_LORA_RANK` and tile constants are asserted
//! against [GRID_Y] in the tests, so a kernel change cannot silently leave this module with a
//! stale grid.
//!
//! No cubin pin. The device chain's catalog owns the cubin's bytes and digest.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use crate::m4_abi;

/// The kernel symbol in the cubin.
pub const KERNEL: &str = "b12x_mxfp8_wo_a_sm121";
/// The kernel source the declaration pin is read from (the frozen ABI's embedded copy).
pub const SOURCE: &str = m4_abi::MXFP8_WO_A;
/// The number of argument words a launch fills.
pub const ARG_WORDS: usize = 9;

/// The declaration's parameters in order: `(name, base type, pointer, pointee const)`.
pub const PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("attn_fp8", "uint8_t", true, true),
    ("attn_scales", "uint8_t", true, true),
    ("w_fp8", "uint8_t", true, true),
    ("w_scales", "uint8_t", true, true),
    ("lo_out", "__nv_bfloat16", true, false),
    ("groups", "uint32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// Parses the kernel declaration out of the pinned source.
#[must_use]
pub fn declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, KERNEL)
}

/// Whether a parsed declaration has [PARAMS]' shape and order.
#[must_use]
pub fn declaration_matches(decl: &KernelDecl) -> bool {
    decl.name == KERNEL
        && decl.extern_c
        && decl.global
        && decl.params.len() == ARG_WORDS
        && decl.params.iter().zip(PARAMS).all(|(param, pin)| {
            param.name == pin.0
                && param.base == pin.1
                && param.pointer == pin.2
                && param.const_pointee == pin.3
                && (!param.pointer || param.restrict)
        })
}

/// The kernel id in the shared driver's enum.
pub const KERNEL_ID: KernelId = KernelId::Mxfp8WoA;
/// 256 threads, one 256-wide output-channel tile per block.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);
/// The output-channel tile the kernel computes per block.
pub const TILE: u32 = 256;
/// The number of output-channel tiles per group: `O_LORA_RANK / TILE`.
pub const GRID_Y: u32 = 4;
/// The index of `groups` in [Mxfp8WoAParams::as_slice].
pub const GROUPS_INDEX: usize = 5;

/// The grid for `groups` groups: `(groups, GRID_Y, 1)`.
///
/// # Errors
/// A zero group count is refused rather than launched as a zero-sized grid.
pub fn grid(groups: u32) -> Result<(u32, u32, u32), String> {
    if groups == 0 {
        return Err("wo_a projects at least one group, so the grid is not zero-sized".to_owned());
    }
    Ok((groups, GRID_Y, 1))
}

/// One launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mxfp8WoAParams([u64; ARG_WORDS]);

impl Mxfp8WoAParams {
    /// Builds the argument array with the kernel's own parameter names, in order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        attn_fp8: Deviceptr,
        attn_scales: Deviceptr,
        w_fp8: Deviceptr,
        w_scales: Deviceptr,
        lo_out: Deviceptr,
        groups: u32,
        trace_ring: u64,
        layer_id: u32,
        step_seq: u64,
    ) -> Self {
        Self([
            attn_fp8.0,
            attn_scales.0,
            w_fp8.0,
            w_scales.0,
            lo_out.0,
            u64::from(groups),
            trace_ring,
            u64::from(layer_id),
            step_seq,
        ])
    }

    /// The group count.
    #[must_use]
    pub fn groups(&self) -> u32 {
        self.0[GROUPS_INDEX] as u32
    }

    /// The parameter words a launch passes, the shape `cuLaunchKernel` takes.
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.0
    }
}

/// Loads the kernel from a cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [KERNEL].
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches the projection once.
///
/// # Errors
/// Refuses a zero group count, and returns the driver's error otherwise.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &Mxfp8WoAParams) -> Result<(), String> {
    let grid = grid(params.groups())?;
    cuda.launch(kernel, grid, BLOCK, params.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_decl::Consts;

    fn params(groups: u32) -> Mxfp8WoAParams {
        Mxfp8WoAParams::new(
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            Deviceptr(0x3000),
            Deviceptr(0x4000),
            Deviceptr(0x5000),
            groups,
            0x6000,
            7,
            11,
        )
    }

    fn constants() -> Consts {
        let mut table = Consts::parse(SOURCE);
        table.extend(m4_abi::header_source());
        table
    }

    #[test]
    fn the_kernel_declaration_is_the_pinned_one() {
        let decl = declaration();
        assert!(
            declaration_matches(&decl),
            "the declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), ARG_WORDS);
    }

    #[test]
    fn the_declaration_pin_is_not_vacuous() {
        let mutated = SOURCE.replace(
            "const uint8_t *__restrict__ attn_fp8",
            "const uint8_t * attn_fp8",
        );
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        let decl = c_decl::kernel_declaration(&mutated, KERNEL);
        assert!(
            !declaration_matches(&decl),
            "the qualifier pin is vacuous: {decl:?}"
        );
    }

    #[test]
    fn the_argument_array_is_the_declaration_order() {
        let p = params(8);
        assert_eq!(
            p.as_slice(),
            [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 8, 0x6000, 7, 11]
        );
        assert_eq!(p.groups(), 8);
        assert_eq!(p.as_slice().len(), ARG_WORDS);
    }

    #[test]
    fn the_grid_is_the_kernel_tiling() {
        let table = constants();
        assert_eq!(table.value("SPARK_M4_WOA_THREADS"), i128::from(BLOCK.0));
        assert_eq!(table.value("SPARK_M4_WOA_TILE"), i128::from(TILE));
        assert_eq!(
            table.value("SPARK_M4_O_LORA_RANK") / i128::from(TILE),
            i128::from(GRID_Y)
        );
        assert_eq!(grid(8), Ok((8, GRID_Y, 1)));
        assert!(grid(0).is_err());
    }
}

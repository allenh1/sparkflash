//! Typed launch capability of the MXFP4 down/combine MoE kernel
//! (`kernels/b12x_mxfp4_moe_sm121.cu`).
//!
//! The argument array is the `b12x_mxfp4_moe_sm121` declaration's parameter list in order - six
//! device pointers and the token count - so a launch cannot swap the expert ids for the slot
//! accumulator or the token count for a pointer. The declaration is pinned here against the frozen ABI's
//! embedded copy ([m4_abi::MXFP4_MOE]); [PARAMS] is the pinned name and type of every parameter
//! and [declaration_matches] is the check, with a mutation test proving it is not vacuous.
//!
//! Geometry. The hidden state is the grid's second dimension: one block computes the 256-value tile
//! `blockIdx.y` of one routed row `blockIdx.x`, so the grid is `(tokens, 20, 1)` blocks of 256
//! threads and [grid] refuses a zero token count rather than launching an empty grid. The kernel's
//! tile and hidden constants are asserted against [GRID_Y] in the tests.
//!
//! No cubin pin. The device chain's catalog owns the cubin's bytes and digest.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use crate::m4_abi;

/// The kernel symbol in the cubin.
pub const KERNEL: &str = "b12x_mxfp4_moe_sm121";
/// The kernel source the declaration pin is read from (the frozen ABI's embedded copy).
pub const SOURCE: &str = m4_abi::MXFP4_MOE;
/// The number of argument words a launch fills.
pub const ARG_WORDS: usize = 10;

/// The declaration's parameters in order: `(name, base type, pointer, pointee const)`.
pub const PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("act_fp8", "uint8_t", true, true),
    ("act_scales", "uint8_t", true, true),
    ("w_fp4", "uint8_t", true, true),
    ("w_scales", "uint8_t", true, true),
    ("expert_ids", "uint32_t", true, true),
    ("slot_acc", "float", true, false),
    ("tokens", "uint32_t", false, false),
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
pub const KERNEL_ID: KernelId = KernelId::Mxfp4Moe;
/// 256 threads, one 256-value hidden tile per block.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);
/// The hidden values one block writes.
pub const TILE_D: u32 = 256;
/// The number of hidden tiles: `HIDDEN / TILE_D`.
pub const GRID_Y: u32 = 20;
/// The index of `tokens` in [Mxfp4MoeParams::as_slice].
pub const TOKENS_INDEX: usize = 6;

/// The grid for `tokens` routed rows: `(tokens, GRID_Y, 1)`.
///
/// # Errors
/// A zero token count is refused rather than launched as a zero-sized grid.
pub fn grid(tokens: u32) -> Result<(u32, u32, u32), String> {
    if tokens == 0 {
        return Err(
            "the combine runs on at least one token, so the grid is not zero-sized".to_owned(),
        );
    }
    Ok((tokens, GRID_Y, 1))
}

/// One launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mxfp4MoeParams([u64; ARG_WORDS]);

impl Mxfp4MoeParams {
    /// Builds the argument array with the kernel's own parameter names, in order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        act_fp8: Deviceptr,
        act_scales: Deviceptr,
        w_fp4: Deviceptr,
        w_scales: Deviceptr,
        expert_ids: Deviceptr,
        slot_acc: Deviceptr,
        tokens: u32,
        trace_ring: u64,
        layer_id: u32,
        step_seq: u64,
    ) -> Self {
        Self([
            act_fp8.0,
            act_scales.0,
            w_fp4.0,
            w_scales.0,
            expert_ids.0,
            slot_acc.0,
            u64::from(tokens),
            trace_ring,
            u64::from(layer_id),
            step_seq,
        ])
    }

    /// The token count.
    #[must_use]
    pub fn tokens(&self) -> u32 {
        self.0[TOKENS_INDEX] as u32
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

/// Launches the combine once.
///
/// # Errors
/// Refuses a zero token count, and returns the driver's error otherwise.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &Mxfp4MoeParams) -> Result<(), String> {
    let grid = grid(params.tokens())?;
    cuda.launch(kernel, grid, BLOCK, params.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_decl::Consts;

    fn params(tokens: u32) -> Mxfp4MoeParams {
        Mxfp4MoeParams::new(
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            Deviceptr(0x3000),
            Deviceptr(0x4000),
            Deviceptr(0x5000),
            Deviceptr(0x6000),
            tokens,
            0x7000,
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
        let mutated = SOURCE.replace("const uint8_t *__restrict__ w_fp4", "const uint8_t * w_fp4");
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        let decl = c_decl::kernel_declaration(&mutated, KERNEL);
        assert!(
            !declaration_matches(&decl),
            "the qualifier pin is vacuous: {decl:?}"
        );
    }

    #[test]
    fn the_argument_array_is_the_declaration_order() {
        let p = params(130);
        assert_eq!(
            p.as_slice(),
            [
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 130, 0x7000, 7, 11
            ]
        );
        assert_eq!(p.tokens(), 130);
        assert_eq!(p.as_slice().len(), ARG_WORDS);
    }

    #[test]
    fn the_grid_is_the_kernel_tiling() {
        let table = constants();
        assert_eq!(table.value("SPARK_M4_MOE_TILE_D"), i128::from(TILE_D));
        assert_eq!(table.value("SPARK_M4_MOE_THREADS"), i128::from(BLOCK.0));
        assert_eq!(
            table.value("SPARK_M4_HIDDEN") / i128::from(TILE_D),
            i128::from(GRID_Y)
        );
        assert_eq!(grid(8), Ok((8, GRID_Y, 1)));
        assert!(grid(0).is_err());
    }
}

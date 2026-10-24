//! Typed launch capability of the BF16-activation grouped wo_a output projection
//! (`kernels/b12x_bf16a_mxfp8w_wo_a_sm121.cu`), the D37 precision-first variant of the grouped
//! MXFP8 projection.
//!
//! The argument array is the kernel declaration's parameter list in order - four device pointers
//! and the group count, then the trace ring, the layer id and the step sequence - so a launch
//! cannot swap a pointer for the group count. The declaration is pinned against the kernel source
//! itself (SOURCE, `include_str!` of the translation unit); [PARAMS] is the pinned name and type of
//! every parameter and [declaration_matches] is the check, with a mutation test proving it is not
//! vacuous.
//!
//! Geometry. It is the MXFP8 wo_a kernel's geometry, deliberately: the group is a grid dimension,
//! the grid is `(groups, O_LORA_RANK / TILE, 1)` blocks of 256 threads, and each block computes one
//! 256-wide output-channel tile of one group. The kernel's `O_LORA_RANK` and tile constants are
//! asserted against [GRID_Y] in the tests, so a kernel change cannot silently leave this module
//! with a stale grid.
//!
//! No cubin pin. The device catalog owns the cubin's bytes and digest; `baseline::pin` is the
//! manifest reader a harness uses to verify the compiled artifact before allocation.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};

/// The kernel symbol in the cubin.
pub const KERNEL: &str = "b12x_bf16a_mxfp8w_wo_a_sm121";
/// The kernel source the declaration pin is read from.
pub const SOURCE: &str = include_str!("../../../kernels/b12x_bf16a_mxfp8w_wo_a_sm121.cu");
/// The driver identity of this kernel.
pub const KERNEL_ID: KernelId = KernelId::Bf16aMxfp8wWoA;
/// The number of argument words a launch fills.
pub const ARG_WORDS: usize = 8;

/// The declaration's parameters in order: `(name, base type, pointer, pointee const)`.
pub const PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("attn_bf16", "__nv_bfloat16", true, true),
    ("w_fp8", "uint8_t", true, true),
    ("w_scales", "uint8_t", true, true),
    ("lo_out", "__nv_bfloat16", true, false),
    ("groups", "uint32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// 256 threads, one 256-wide output-channel tile per block.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);
/// The output-channel tile the kernel computes per block.
pub const TILE: u32 = 256;
/// The number of output-channel tiles per group: `O_LORA_RANK / TILE`.
pub const GRID_Y: u32 = 4;
/// The index of `groups` in [Bf16aMxfp8wWoAParams::as_slice].
pub const GROUPS_INDEX: usize = 4;

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
pub struct Bf16aMxfp8wWoAParams([u64; ARG_WORDS]);

impl Bf16aMxfp8wWoAParams {
    /// Builds the argument array with the kernel's own parameter names, in order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        attn_bf16: Deviceptr,
        w_fp8: Deviceptr,
        w_scales: Deviceptr,
        lo_out: Deviceptr,
        groups: u32,
        trace_ring: u64,
        layer_id: u32,
        step_seq: u64,
    ) -> Self {
        Self([
            attn_bf16.0,
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

/// The split-K opt GEMV's bf16-activation symbol (kernels/opt_gemv_sm121.cu): when installed, the
/// grouped wo_a runs as 32-column tiles x groups with 8 warps each instead of 256-column tiles.
static OPT_FUNCTION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const OPT_SYMBOL: &str = "opt_gemv_bf16a_bf16o_t1";

/// Loads the opt wo_a symbol and routes [`launch`] through it.
pub fn install_opt(cuda: &Cuda, cubin: &[u8]) -> Result<Module, String> {
    let (module, kernel) = cuda.load_named(cubin, OPT_SYMBOL)?;
    OPT_FUNCTION.store(kernel.to_raw().0, std::sync::atomic::Ordering::Release);
    Ok(module)
}

/// Launches the projection once.
///
/// # Errors
/// Refuses a zero group count, and returns the driver's error otherwise.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &Bf16aMxfp8wWoAParams) -> Result<(), String> {
    let grid = grid(params.groups())?;
    let word = OPT_FUNCTION.load(std::sync::atomic::Ordering::Acquire);
    if word != 0 {
        // [attn, w, w_scales, out, groups, trace, layer, step]; one group is [1024][4096].
        let p = params.as_slice();
        let (cols, inner) = (1024u64, 4096u64);
        return cuda.launch(
            Kernel::from_raw(word, OPT_SYMBOL),
            (1024 / 32, grid.0, 1),
            (256, 1, 1),
            &[
                p[0],
                0,
                p[1],
                p[2],
                p[3],
                cols,
                inner,
                1,
                inner * p[4],
                cols * p[4],
                inner,
                cols * inner,
                (cols / 32) * (inner / 32),
                cols,
                0,
            ],
        );
    }
    cuda.launch(kernel, grid, BLOCK, params.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_decl::Consts;
    use crate::m4_abi;

    fn params(groups: u32) -> Bf16aMxfp8wWoAParams {
        Bf16aMxfp8wWoAParams::new(
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            Deviceptr(0x3000),
            Deviceptr(0x4000),
            groups,
            0x5000,
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
            "const __nv_bfloat16 *__restrict__ attn_bf16",
            "const __nv_bfloat16 * attn_bf16",
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
            [0x1000, 0x2000, 0x3000, 0x4000, 8, 0x5000, 7, 11]
        );
        assert_eq!(p.groups(), 8);
        assert_eq!(p.as_slice().len(), ARG_WORDS);
    }

    #[test]
    fn the_grid_is_the_mxfp8_wo_a_kernels_tiling() {
        let table = constants();
        assert_eq!(
            table.value("SPARK_M4_BF16A_WOA_THREADS"),
            i128::from(BLOCK.0)
        );
        assert_eq!(table.value("SPARK_M4_BF16A_WOA_TILE"), i128::from(TILE));
        assert_eq!(
            table.value("SPARK_M4_O_LORA_RANK") / i128::from(TILE),
            i128::from(GRID_Y)
        );
        assert_eq!(grid(8), Ok((8, GRID_Y, 1)));
        assert!(grid(0).is_err());
    }

    #[test]
    fn the_argument_count_is_one_fewer_than_the_mxfp8_wo_a_kernel() {
        // The precision path removes the activation scale buffer and nothing else: the MXFP8 wo_a
        // declaration has nine words (attn_fp8, attn_scales, w_fp8, w_scales, lo_out, groups,
        // trace_ring, layer_id, step_seq); this one has eight because the activation scale pointer
        // is gone and attn_fp8 became attn_bf16.
        assert_eq!(ARG_WORDS, mxfp8_wo_a_arg_words() - 1);
    }

    fn mxfp8_wo_a_arg_words() -> usize {
        crate::mxfp8_wo_a_launch::ARG_WORDS
    }
}

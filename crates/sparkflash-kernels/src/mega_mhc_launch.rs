//! Typed launch capability of the fused Mega-mHC kernel (`kernels/mega_mhc_sm121.cu`).
//!
//! The argument array is the `mega_mhc_fused_bf16_sm121` declaration's parameter list in order -
//! fifteen device pointers, the trace ring, the layer id and the step sequence - so a launch cannot
//! fill a word out of order or with the wrong pointer. The declaration is pinned in this module:
//! [PARAMS] carries the name and type of every parameter, [declaration_matches] is the check, and a
//! mutation test proves the check is not vacuous. The source text is `m4_abi::MEGA_MHC`, the same
//! embedded copy the frozen ABI ratchet uses, so there is no second copy of the kernel text.
//!
//! Geometry. The kernel sweeps the hidden state in 1280-value tiles and one block owns one tile, so
//! the grid is [GRID] blocks of [BLOCK] threads; the four tiles rendezvous through `hc_sync` and
//! `hc_stats`, which is why the grid is fixed by the 5120-wide hidden size and this module has no
//! grid function that could compute a different shape.
//!
//! No cubin pin. The device chain's catalog (`sparkflash-device::cubins`) owns the bytes and the
//! digest; this module resolves the kernel's symbol and cannot load an unpinned cubin by itself.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use crate::m4_abi;

/// The kernel symbol in the cubin.
pub const KERNEL: &str = "mega_mhc_fused_bf16_sm121";
/// The kernel source the declaration pin is read from (the frozen ABI's embedded copy).
pub const SOURCE: &str = m4_abi::MEGA_MHC;
/// The number of argument words a launch fills: one per declaration parameter.
pub const ARG_WORDS: usize = 19;

/// The declaration's parameters in order: `(name, base type, pointer, pointee const)`.
pub const PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("x", "__nv_bfloat16", true, false),
    ("y_prev", "__nv_bfloat16", true, true),
    ("b_prev", "float", true, true),
    ("c_prev", "float", true, true),
    ("a_prev", "float", true, true),
    ("predictor", "float", true, true),
    ("hc_scale", "float", true, true),
    ("hc_base", "float", true, true),
    ("rms_weight", "float", true, true),
    ("xhat_fp8", "uint8_t", true, false),
    ("xhat_scales", "uint8_t", true, false),
    ("hc_stats", "float", true, false),
    ("hc_sync", "uint32_t", true, false),
    ("hc_coeffs", "float", true, false),
    ("hc_rms_out", "float", true, false),
    ("xhat_normed", "float", true, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// Parses the kernel declaration out of the pinned source.
#[must_use]
pub fn declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, KERNEL)
}

/// Whether a parsed declaration has [PARAMS]' shape and order. Used by the mutation test, which
/// proves the comparison is not vacuous.
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
pub const KERNEL_ID: KernelId = KernelId::MegaMhcBf16;
/// The one piece of launch geometry the kernel fixes: 128 threads, one block per hidden tile.
pub const BLOCK: (u32, u32, u32) = (128, 1, 1);
/// The grid: one block per 1280-value tile of the 5120-wide hidden state.
pub const GRID: (u32, u32, u32) = (4, 1, 1);

/// One launch's arguments, in the kernel declaration's order.
///
/// Every device buffer is a [Deviceptr] the caller allocated with [Cuda::alloc]; `trace_ring` is 0
/// for the shipped, trace-disabled cubin and the ring's device address otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MegaMhcParams([u64; ARG_WORDS]);

impl MegaMhcParams {
    /// Builds the argument array with the kernel's own parameter names, in order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        x: Deviceptr,
        y_prev: Deviceptr,
        b_prev: Deviceptr,
        c_prev: Deviceptr,
        a_prev: Deviceptr,
        predictor: Deviceptr,
        hc_scale: Deviceptr,
        hc_base: Deviceptr,
        rms_weight: Deviceptr,
        xhat_fp8: Deviceptr,
        xhat_scales: Deviceptr,
        hc_stats: Deviceptr,
        hc_sync: Deviceptr,
        hc_coeffs: Deviceptr,
        hc_rms_out: Deviceptr,
        xhat_normed: Deviceptr,
        trace_ring: u64,
        layer_id: u32,
        step_seq: u64,
    ) -> Self {
        Self([
            x.0,
            y_prev.0,
            b_prev.0,
            c_prev.0,
            a_prev.0,
            predictor.0,
            hc_scale.0,
            hc_base.0,
            rms_weight.0,
            xhat_fp8.0,
            xhat_scales.0,
            hc_stats.0,
            hc_sync.0,
            hc_coeffs.0,
            hc_rms_out.0,
            xhat_normed.0,
            trace_ring,
            u64::from(layer_id),
            step_seq,
        ])
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

/// Launches the kernel once with its fixed grid.
///
/// # Errors
/// Returns the driver's error.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &MegaMhcParams) -> Result<(), String> {
    cuda.launch(kernel, GRID, BLOCK, params.as_slice())
}

/// Launches the kernel `repeat` times and returns `(per-launch ms, batched ms/launch)`.
///
/// # Errors
/// As [launch].
pub fn launch_repeated(
    cuda: &Cuda,
    kernel: Kernel,
    params: &MegaMhcParams,
    repeat: usize,
) -> Result<(Vec<f64>, f64), String> {
    cuda.launch_repeated(kernel, GRID, BLOCK, params.as_slice(), repeat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_decl::Consts;

    fn pointers() -> [Deviceptr; 16] {
        core::array::from_fn(|index| Deviceptr(0x1000 * (index as u64 + 1)))
    }

    fn params() -> MegaMhcParams {
        let p = pointers();
        MegaMhcParams::new(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13],
            p[14], p[15], 0x9000, 7, 11,
        )
    }

    /// The kernel's constants with the shared M4 header's names resolvable.
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
        assert_eq!(PARAMS.len(), ARG_WORDS);
    }

    #[test]
    fn the_declaration_pin_is_not_vacuous() {
        // Drop a qualifier from one pointer the way a careless ABI edit would; the pin must see it.
        let mutated = SOURCE.replace("uint8_t *__restrict__ xhat_scales", "uint8_t * xhat_scales");
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        let decl = c_decl::kernel_declaration(&mutated, KERNEL);
        assert!(
            !declaration_matches(&decl),
            "the qualifier pin is vacuous: {decl:?}"
        );
    }

    #[test]
    fn the_argument_array_is_the_declaration_order() {
        let p = params();
        assert_eq!(
            p.as_slice(),
            [
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 0x8000, 0x9000, 0xa000,
                0xb000, 0xc000, 0xd000, 0xe000, 0xf000, 0x10000, 0x9000, 7, 11,
            ]
        );
        assert_eq!(p.as_slice().len(), ARG_WORDS);
    }

    #[test]
    fn the_geometry_constants_are_the_kernels() {
        let table = constants();
        assert_eq!(table.value("SPARK_M4_MHC_TILES"), i128::from(GRID.0));
        assert_eq!(table.value("SPARK_M4_MHC_THREADS"), i128::from(BLOCK.0));
        assert_eq!(table.value("SPARK_M4_MHC_TILE_ELEMS"), 1280);
        // The grid is the hidden state's tiling, not a hand-picked number.
        assert_eq!(
            table.value("SPARK_M4_HIDDEN") / table.value("SPARK_M4_MHC_TILE_ELEMS"),
            i128::from(GRID.0)
        );
        assert_eq!(GRID.1, 1);
        assert_eq!(BLOCK.1, 1);
    }
}

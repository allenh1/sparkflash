//! Typed launch capability of the columns-only M=1 dense MXFP8 twin
//! (`kernels/b12x_mxfp8_dense_m1_sm121.cu`).
//!
//! The decode step (one token, one row) runs the dense projections at M=1. The pinned 64x64 dense
//! kernel `b12x_mxfp8_dense_sm121` is shaped for a prefill batch: it owns sixty-four rows of output
//! tile and gives each of its 256 threads a 4x4 sub-tile, so at M=1 only lanes 0-15 have a row inside
//! the tile and 16 of 256 thread slots do the arithmetic (measured 1.92 GMAC/s on the attention
//! projections). [KERNEL] keeps that kernel's arithmetic EXACTLY - the per-output pass is the same
//! loop in the same order, with the same decodes and the same single bf16 round - and changes only
//! which thread owns which output: one row, [COLUMNS] output columns per block, one output column
//! per thread, K still ascending in 32-value blocks and ascending inside each block. The tiling
//! decides only who writes an output, so no sum is reordered and bit identity is by construction,
//! not by tolerance.
//!
//! This module is an addition, in the form of the standing new-symbol rule
//! (`routed_gate_up_swiglu_splitk_abi`): the pinned kernel's source, cubin bytes, digest and
//! `KernelId` are untouched, and [KERNEL] is its own cubin, its own baseline entry and its own
//! symbol loaded by name through [crate::cuda_driver::Cuda::load_dense_m1]. The declaration is
//! pinned against [PARAMS] (the same eleven-word ABI as the pinned kernel, so the same
//! [crate::mxfp8_dense_launch::Mxfp8DenseParams] fills the argument array) and the grid is pinned
//! against the source's `#define`s.
//!
//! Geometry. One block per [COLUMNS]-wide output slice: the grid is
//! `(ceil(cols / COLUMNS), 1, 1)` blocks of 256 threads. The shape contract is one output row; both
//! [grid] and [launch] refuse any other row count rather than launching a kernel whose only column
//! guard would be wrong.
//!
//! No cubin pin in the launch type: the device chain's catalog owns the cubin's bytes and digest.
//! The values this module records ([CUBIN_BYTES], [CUBIN_SHA256]) are the compiled artifact's
//! identity, pinned against the baseline manifest by the tests.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Kernel, KernelId, Module};
use crate::mxfp8_dense_launch::Mxfp8DenseParams;

/// The kernel symbol in its own cubin.
pub const KERNEL: &str = "b12x_mxfp8_dense_m1_sm121";

/// The kernel source, as the declaration pin and the constant pin see it.
pub const SOURCE: &str = include_str!("../../../kernels/b12x_mxfp8_dense_m1_sm121.cu");

/// The number of argument words a launch fills: the pinned dense kernel's eleven-word declaration.
pub const ARG_WORDS: usize = 11;

/// The declaration's parameters in order: `(name, base type, pointer, pointee const)`.
pub const PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("a_fp8", "uint8_t", true, true),
    ("a_scales", "uint8_t", true, true),
    ("w_fp8", "uint8_t", true, true),
    ("w_scales", "uint8_t", true, true),
    ("out", "__nv_bfloat16", true, false),
    ("rows", "uint32_t", false, false),
    ("cols", "uint32_t", false, false),
    ("inner", "uint32_t", false, false),
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

/// 256 threads, one output column each.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);
/// Output columns one block owns: the block's thread count, one column per thread.
pub const COLUMNS: u32 = 256;
/// The one output row the kernel serves.
pub const ROWS: u32 = 1;

/// The compiled artifact's byte count, as nvcc V13.0.88 (CUDA 13.0.3) compiles
/// `kernels/b12x_mxfp8_dense_m1_sm121.cu` for sm_121a with the baseline's flags.
pub const CUBIN_BYTES: usize = 14520;
/// The compiled artifact's SHA-256, from the same build as [CUBIN_BYTES].
pub const CUBIN_SHA256: &str = "8fcd899dad03debc5292106e5f05e6c545f61fcc0d268789d6f4ce64509fb5ed";

/// The kernel id a caller would use if it resolved the symbol through the shared driver's enum.
///
/// The symbolic loader ([crate::cuda_driver::Cuda::load_dense_m1]) is what the call site uses, so
/// this value is documentation of the owning family rather than a dispatch key: the symbol is loaded
/// by name and the pinned kernel stays the frozen ABI id.
pub const FAMILY: KernelId = KernelId::Mxfp8Dense;

/// The grid for a `rows x cols` output: `(ceil(cols / COLUMNS), 1, 1)`.
///
/// # Errors
/// A row count other than [ROWS] is refused, because the kernel computes exactly one output row and
/// a larger count would silently drop rows; a zero column count is refused rather than launched as a
/// zero-sized grid.
pub fn grid(rows: u32, cols: u32) -> Result<(u32, u32, u32), String> {
    if rows != ROWS {
        return Err(format!(
            "the M=1 dense GEMM computes exactly {ROWS} output row, the launch asked for {rows}"
        ));
    }
    if cols == 0 {
        return Err(
            "the M=1 dense GEMM runs on at least one output column, so the grid is not zero-sized"
                .to_owned(),
        );
    }
    Ok((cols.div_ceil(COLUMNS), 1, 1))
}

/// Loads the kernel from its cubin, resolving [KERNEL] by name.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [KERNEL].
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_dense_m1(cubin)
}

/// Launches the M=1 dense GEMM once.
///
/// # Errors
/// Refuses a shape the kernel does not serve (rows other than [ROWS]), a zero reduction length, and
/// returns the driver's error otherwise.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &Mxfp8DenseParams) -> Result<(), String> {
    if params.inner() == 0 {
        return Err("the dense GEMM reduces at least one value, so inner is not zero".to_owned());
    }
    let grid = grid(params.rows(), params.cols())?;
    cuda.launch(kernel, grid, BLOCK, params.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_decl::Consts;
    use crate::cuda_driver::Deviceptr;
    use crate::m4_abi;

    fn params(rows: u32, cols: u32, inner: u32) -> Mxfp8DenseParams {
        Mxfp8DenseParams::new(
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            Deviceptr(0x3000),
            Deviceptr(0x4000),
            Deviceptr(0x5000),
            rows,
            cols,
            inner,
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
        let mutated = SOURCE.replace("const uint8_t *__restrict__ a_fp8", "const uint8_t * a_fp8");
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        let decl = c_decl::kernel_declaration(&mutated, KERNEL);
        assert!(
            !declaration_matches(&decl),
            "the qualifier pin is vacuous: {decl:?}"
        );
    }

    // The addition's whole claim: the argument row is the pinned dense kernel's row, word for word.
    // A kernel that took a different shape word or a different pointer order would not be the same
    // arithmetic, so the two PARAMS tables must be equal rather than merely both well-formed.
    #[test]
    fn the_argument_row_is_the_pinned_kernels_row() {
        assert_eq!(PARAMS, crate::mxfp8_dense_launch::PARAMS);
        assert_eq!(ARG_WORDS, crate::mxfp8_dense_launch::ARG_WORDS);
        let p = params(1, 1280, 5120);
        assert_eq!(
            p.as_slice(),
            [
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 1, 1280, 5120, 0x6000, 7, 11
            ]
        );
        assert_eq!(p.rows(), 1);
        assert_eq!(p.cols(), 1280);
        assert_eq!(p.inner(), 5120);
    }

    #[test]
    fn the_grid_is_the_kernel_tiling() {
        let table = constants();
        assert_eq!(
            table.value("SPARK_M4_DENSE_M1_COLUMNS"),
            i128::from(COLUMNS)
        );
        assert_eq!(
            table.value("SPARK_M4_DENSE_M1_THREADS"),
            i128::from(BLOCK.0)
        );
        assert_eq!(grid(1, 256), Ok((1, 1, 1)));
        assert_eq!(grid(1, 1280), Ok((5, 1, 1)));
        assert_eq!(grid(1, 512), Ok((2, 1, 1)));
        assert_eq!(grid(1, 5120), Ok((20, 1, 1)));
        assert_eq!(grid(1, 8192), Ok((32, 1, 1)));
        assert_eq!(grid(1, 1300), Ok((6, 1, 1)));
        assert!(grid(0, 1).is_err(), "a zero-row launch must be refused");
        assert!(grid(2, 1).is_err(), "rows other than one must be refused");
        assert!(grid(1, 0).is_err(), "a zero-column launch must be refused");
    }

    // The static_asserts in the kernel are re-evaluated here against the same constants, so a tile
    // edit that breaks one fails the Rust gate without a CUDA host.
    #[test]
    fn the_kernel_static_asserts_evaluate() {
        let table = constants();
        assert_eq!(
            table.value("SPARK_M4_DENSE_M1_COLUMNS"),
            table.value("SPARK_M4_DENSE_M1_THREADS"),
            "one output column per thread"
        );
        assert_eq!(table.value("SPARK_M4_DENSE_M1_THREADS") % 32, 0);
        assert_eq!(
            table.value("SPARK_M4_DENSE_M1_COLUMNS") % table.value("SPARK_B12X_NTILE"),
            0
        );
        for spec in c_decl::static_asserts(SOURCE) {
            assert_eq!(
                table.eval(&spec.condition),
                1,
                "the kernel's static_assert does not hold: {}",
                spec.condition
            );
        }
    }

    // The manifest half of the new-cubin job: the baseline names this symbol and pins the artifact
    // these constants describe. A regenerated cubin must move both in the same commit.
    #[test]
    fn the_baseline_pins_the_compiled_artifact() {
        let pin = crate::baseline::pin(KERNEL)
            .expect("the baseline manifest must parse")
            .expect("the baseline must pin the M=1 dense kernel");
        assert_eq!(pin.name, KERNEL);
        assert_eq!(pin.cubin_bytes, CUBIN_BYTES);
        assert_eq!(pin.cubin_sha256, CUBIN_SHA256);
    }

    // The family constant is documentation; the loader is symbolic. Pinning it here keeps the
    // family ownership visible without adding a dispatch variant for a symbol loaded by name.
    #[test]
    fn the_family_stays_the_pinned_dense_id() {
        assert_eq!(FAMILY, KernelId::Mxfp8Dense);
        assert_eq!(crate::mxfp8_dense_launch::KERNEL, "b12x_mxfp8_dense_sm121");
        assert_ne!(KERNEL, crate::mxfp8_dense_launch::KERNEL);
    }
}

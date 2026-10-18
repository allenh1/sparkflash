//! Typed launch capability of the MXFP8 dense twin (`kernels/b12x_mxfp8_dense_sm121.cu`).
//!
//! The argument array is the `b12x_mxfp8_dense_sm121` declaration's parameter list in order - five
//! device pointers and the `rows`, `cols`, `inner` shape - so a launch cannot fill a word out of
//! order or swap a shape value for a pointer. The declaration is pinned here against the frozen
//! ABI's embedded copy ([m4_abi::MXFP8_DENSE]); [PARAMS] is the pinned name and type of every
//! parameter and [declaration_matches] is the check, with a mutation test proving it is not vacuous.
//!
//! Geometry. One block computes a 64x64 output tile, so the grid is
//! `(ceil(rows / 64), ceil(cols / 64), 1)` blocks of 256 threads; [grid] refuses a zero-sized
//! shape rather than handing the driver an empty grid. The kernel derives the B12x scale grid from
//! `inner`, so a zero `inner` is refused by [launch] as well.
//!
//! No cubin pin. The device chain's catalog owns the cubin's bytes and digest.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use crate::m4_abi;

/// The kernel symbol in the cubin.
pub const KERNEL: &str = "b12x_mxfp8_dense_sm121";
/// The kernel source the declaration pin is read from (the frozen ABI's embedded copy).
pub const SOURCE: &str = m4_abi::MXFP8_DENSE;
/// The number of argument words a launch fills.
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

/// The kernel id in the shared driver's enum.
pub const KERNEL_ID: KernelId = KernelId::Mxfp8Dense;
/// 256 threads, one 64x64 output tile per block.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);
/// The kernel's output tile edge.
pub const TILE: u32 = 64;
/// The index of `rows` in [Mxfp8DenseParams::as_slice].
pub const ROWS_INDEX: usize = 5;
/// The index of `cols` in [Mxfp8DenseParams::as_slice].
pub const COLS_INDEX: usize = 6;
/// The index of `inner` in [Mxfp8DenseParams::as_slice].
pub const INNER_INDEX: usize = 7;

/// The grid for a `rows x cols` output tile: `(ceil(rows / 64), ceil(cols / 64), 1)`.
///
/// # Errors
/// A zero row or column count is refused rather than launched as a zero-sized grid.
pub fn grid(rows: u32, cols: u32) -> Result<(u32, u32, u32), String> {
    if rows == 0 || cols == 0 {
        return Err("the dense GEMM runs on at least one output row and column, so the grid is not zero-sized".to_owned());
    }
    Ok((rows.div_ceil(TILE), cols.div_ceil(TILE), 1))
}

/// One launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mxfp8DenseParams([u64; ARG_WORDS]);

impl Mxfp8DenseParams {
    /// Builds the argument array with the kernel's own parameter names, in order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        a_fp8: Deviceptr,
        a_scales: Deviceptr,
        w_fp8: Deviceptr,
        w_scales: Deviceptr,
        out: Deviceptr,
        rows: u32,
        cols: u32,
        inner: u32,
        trace_ring: u64,
        layer_id: u32,
        step_seq: u64,
    ) -> Self {
        Self([
            a_fp8.0,
            a_scales.0,
            w_fp8.0,
            w_scales.0,
            out.0,
            u64::from(rows),
            u64::from(cols),
            u64::from(inner),
            trace_ring,
            u64::from(layer_id),
            step_seq,
        ])
    }

    /// The rows of the output tile.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.0[ROWS_INDEX] as u32
    }

    /// The columns of the output tile.
    #[must_use]
    pub fn cols(&self) -> u32 {
        self.0[COLS_INDEX] as u32
    }

    /// The reduction length.
    #[must_use]
    pub fn inner(&self) -> u32 {
        self.0[INNER_INDEX] as u32
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

/// Launches the GEMM once.
///
/// # Errors
/// Refuses a zero row, column or reduction length, and returns the driver's error otherwise.
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

    #[test]
    fn the_argument_array_is_the_declaration_order() {
        let p = params(128, 64, 5120);
        assert_eq!(
            p.as_slice(),
            [
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 128, 64, 5120, 0x6000, 7, 11
            ]
        );
        assert_eq!(p.rows(), 128);
        assert_eq!(p.cols(), 64);
        assert_eq!(p.inner(), 5120);
        assert_eq!(p.as_slice().len(), ARG_WORDS);
    }

    #[test]
    fn the_grid_is_the_kernel_tiling() {
        let table = constants();
        assert_eq!(table.value("SPARK_M4_DENSE_TILE_M"), i128::from(TILE));
        assert_eq!(table.value("SPARK_M4_DENSE_TILE_N"), i128::from(TILE));
        assert_eq!(table.value("SPARK_M4_DENSE_THREADS"), i128::from(BLOCK.0));
        assert_eq!(grid(1, 1), Ok((1, 1, 1)));
        assert_eq!(grid(64, 64), Ok((1, 1, 1)));
        assert_eq!(grid(128, 129), Ok((2, 3, 1)));
        assert!(grid(0, 1).is_err());
        assert!(grid(1, 0).is_err());
    }
}

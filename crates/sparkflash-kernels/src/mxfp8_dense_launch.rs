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

/// The K1 warp-per-column symbol; it shares the cubin with [KERNEL] but is not the frozen ABI
/// id and is loaded by name.
pub const WARP_KERNEL: &str = "b12x_mxfp8_dense_warp_sm121";
/// Warp-per-column block: eight columns per block, 32 lanes per column.
pub const WARP_BLOCK: (u32, u32, u32) = (256, 1, 1);
/// Columns per warp-per-column block.
pub const WARP_COLUMNS: u32 = 8;
/// Rows one warp-per-column block serves from one read of the weight row.
pub const WARP_ROWS: u32 = 8;
/// The K1-b A-reuse tiled symbol.
pub const TILE_KERNEL: &str = "b12x_mxfp8_dense_tile_sm121";
/// A-reuse block: 128 columns x 8 rows, 256 threads.
pub const TILE_BLOCK: (u32, u32, u32) = (1024, 1, 1);
/// Columns one A-reuse block owns.
pub const TILE_COLUMNS: u32 = 128;
/// Rows one A-reuse block owns.
pub const TILE_ROWS: u32 = 8;
/// K1-c(a) parallel-K symbol: 8 rows x 32 columns x 4 K slices, a declared tolerance path.
pub const PK_KERNEL: &str = "b12x_mxfp8_dense_par_k_sm121";
/// Parallel-K block: 1024 threads.
pub const PK_BLOCK: (u32, u32, u32) = (1024, 1, 1);
/// Columns one parallel-K block owns.
pub const PK_COLUMNS: u32 = 32;
/// Rows one parallel-K block owns.
pub const PK_ROWS: u32 = 8;
/// K slices per output.
pub const PK_SPLIT: u32 = 4;
/// K1-c(b) split-K partial symbol: 8 rows x 32 columns x one K quarter.
pub const SK_KERNEL: &str = "b12x_mxfp8_dense_split_k_sm121";
/// K1-c(b) deterministic combine symbol.
pub const SK_COMBINE_KERNEL: &str = "b12x_mxfp8_dense_split_k_combine_sm121";
/// K1-c(b) block: 256 threads, one output per thread per K quarter.
pub const SK_BLOCK: (u32, u32, u32) = (256, 1, 1);
/// Columns one split-K partial block owns.
pub const SK_COLUMNS: u32 = 32;
/// Rows one split-K partial block owns.
pub const SK_ROWS: u32 = 8;
/// K quarters.
pub const SK_SPLIT: u32 = 4;
/// Combine block: 256 threads over a linear rows*cols range.
pub const SK_COMBINE_BLOCK: (u32, u32, u32) = (256, 1, 1);

/// One split-K launch's arguments: [Mxfp8DenseParams]' five pointers plus the f32 partial buffer and
/// the two shape words the combine needs. The split count is fixed at four.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitKParams([u64; 12]);

impl SplitKParams {
    /// Builds the split-K argument array in the partial kernel's order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        a_fp8: Deviceptr,
        a_scales: Deviceptr,
        w_fp8: Deviceptr,
        w_scales: Deviceptr,
        partials: Deviceptr,
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
            partials.0,
            out.0,
            u64::from(rows),
            u64::from(cols),
            u64::from(inner),
            trace_ring,
            u64::from(layer_id),
            step_seq,
        ])
    }

    /// The partial kernel's words: five pointers, rows/cols/inner, then the trace tail.
    /// The output pointer is the combine kernel's, not the partial kernel's.
    #[must_use]
    pub fn partial_slice(&self) -> [u64; 11] {
        [
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[6], self.0[7], self.0[8],
            self.0[9], self.0[10], self.0[11],
        ]
    }

    /// The combine kernel's words: partials, out, rows, cols.
    #[must_use]
    pub fn combine_slice(&self) -> [u64; 4] {
        [self.0[4], self.0[5], self.0[6], self.0[7]]
    }
}

/// The K1-b grid: `(ceil(cols / 128), ceil(rows / 8), 1)`.
///
/// # Errors
/// A zero row or column count is refused rather than launched as a zero-sized grid.
pub fn tile_grid(rows: u32, cols: u32) -> Result<(u32, u32, u32), String> {
    if rows == 0 || cols == 0 {
        return Err("the dense GEMM runs on at least one output row and column, so the grid is not zero-sized".to_owned());
    }
    Ok((cols.div_ceil(TILE_COLUMNS), rows.div_ceil(TILE_ROWS), 1))
}

/// The K1 grid: `(ceil(cols / 8), ceil(rows / 8), 1)`.
///
/// # Errors
/// A zero row or column count is refused rather than launched as a zero-sized grid.
pub fn warp_grid(rows: u32, cols: u32) -> Result<(u32, u32, u32), String> {
    if rows == 0 || cols == 0 {
        return Err("the dense GEMM runs on at least one output row and column, so the grid is not zero-sized".to_owned());
    }
    Ok((cols.div_ceil(WARP_COLUMNS), rows.div_ceil(WARP_ROWS), 1))
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

/// Loads the K1 warp-per-column symbol by name.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [WARP_KERNEL].
pub fn load_warp(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_dense_warp(cubin)
}

/// Loads the K1-b A-reuse tile symbol by name.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [TILE_KERNEL].
pub fn load_tile(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_dense_tile(cubin)
}

/// Loads the K1-c(a) parallel-K dense symbol by name.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [PK_KERNEL].
pub fn load_par_k(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_dense_par_k(cubin)
}

/// Loads both K1-c(b) split-K symbols by name.
///
/// # Errors
/// Returns the driver's error when either symbol is missing.
pub fn load_split_k(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel, Module, Kernel), String> {
    cuda.load_dense_split_k(cubin)
}

/// Loads the kernel from a cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [KERNEL].
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches the K1 warp-per-column GEMM once.
///
/// # Errors
/// Refuses a zero row, column or reduction length, and returns the driver's error otherwise.
pub fn launch_warp(cuda: &Cuda, kernel: Kernel, params: &Mxfp8DenseParams) -> Result<(), String> {
    if params.inner() == 0 {
        return Err("the dense GEMM reduces at least one value, so inner is not zero".to_owned());
    }
    let grid = warp_grid(params.rows(), params.cols())?;
    cuda.launch(kernel, grid, WARP_BLOCK, params.as_slice())
}

/// Launches the K1-b A-reuse tile GEMM once.
///
/// # Errors
/// Refuses a zero reduction length, and returns the driver's error otherwise.
pub fn launch_tile(cuda: &Cuda, kernel: Kernel, params: &Mxfp8DenseParams) -> Result<(), String> {
    if params.inner() == 0 {
        return Err("the dense GEMM reduces at least one value, so inner is not zero".to_owned());
    }
    let grid = tile_grid(params.rows(), params.cols())?;
    cuda.launch(kernel, grid, TILE_BLOCK, params.as_slice())
}

/// Launches the K1-c(a) parallel-K GEMM once. The kernel assumes inner is divisible by 4.
///
/// # Errors
/// Refuses a zero reduction length, and returns the driver's error otherwise.
pub fn launch_par_k(cuda: &Cuda, kernel: Kernel, params: &Mxfp8DenseParams) -> Result<(), String> {
    if params.inner() == 0 || !params.inner().is_multiple_of(PK_SPLIT) {
        return Err("the parallel-K GEMM needs inner divisible by 4".to_owned());
    }
    let grid = (
        params.cols().div_ceil(PK_COLUMNS),
        params.rows().div_ceil(PK_ROWS),
        1,
    );
    cuda.launch(kernel, grid, PK_BLOCK, params.as_slice())
}

/// Launches K1-c(b) split-K: four partial launches then the deterministic combine.
///
/// # Errors
/// Refuses inner not divisible by 4 and returns the driver's error otherwise.
pub fn launch_split_k(
    cuda: &Cuda,
    partial_kernel: Kernel,
    combine_kernel: Kernel,
    params: &SplitKParams,
) -> Result<(), String> {
    let inner = params.0[8] as u32;
    let rows = params.0[6] as u32;
    let cols = params.0[7] as u32;
    if inner == 0 || !inner.is_multiple_of(SK_SPLIT) || rows == 0 || cols == 0 {
        return Err("split-K needs inner divisible by 4 and a non-empty shape".to_owned());
    }
    let partial_grid = (cols.div_ceil(SK_COLUMNS), rows.div_ceil(SK_ROWS), SK_SPLIT);
    let partial_args = params.partial_slice();
    cuda.launch(partial_kernel, partial_grid, SK_BLOCK, &partial_args)?;
    let total = rows * cols;
    let combine_grid = (total.div_ceil(SK_COMBINE_BLOCK.0), 1, 1);
    let args = params.combine_slice();
    cuda.launch(combine_kernel, combine_grid, SK_COMBINE_BLOCK, &args)
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

    #[test]
    fn the_k1c_paths_keep_their_declared_shapes_and_word_orders() {
        assert_eq!((PK_COLUMNS, PK_ROWS, PK_SPLIT), (32, 8, 4));
        assert_eq!((SK_COLUMNS, SK_ROWS, SK_SPLIT), (32, 8, 4));
        assert_eq!(PK_BLOCK, (1024, 1, 1));
        assert_eq!(SK_BLOCK, (256, 1, 1));
        let split = SplitKParams::new(
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            Deviceptr(0x3000),
            Deviceptr(0x4000),
            Deviceptr(0x5000),
            Deviceptr(0x6000),
            6,
            128,
            5120,
            0x7000,
            9,
            13,
        );
        // The partial kernel has no output pointer; its order is a,as,w,ws,partials,rows,cols,inner,tail.
        assert_eq!(
            split.partial_slice(),
            [
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 6, 128, 5120, 0x7000, 9, 13
            ]
        );
        assert_eq!(split.combine_slice(), [0x5000, 0x6000, 6, 128]);
    }
}

//! Typed launch capability of the dedicated decode-step (M=1) MXFP8 GEMV
//! (`kernels/b12x_mxfp8_gemv_m1_sm121.cu`).
//!
//! The decode step runs every dense projection at M=1, where the pinned 64x64 dense kernel
//! `b12x_mxfp8_dense_sm121` does sixteen lanes' worth of arithmetic per block and reads its weights
//! as scalars: the launch is not memory-limited, it is shaped for a prefill tile. [KERNEL] is a
//! bandwidth-first GEMV whose tiling follows the B12x layout instead: one warp owns one 32x32
//! micro-tile (32 output columns), one lane per output column, and each lane's 32 K-bytes per
//! K-block are contiguous, so a warp reads 1024 coalesced bytes per K-block with two 128-bit loads
//! per lane. The per-output chain is still the pinned contract - ascending K in 32-value blocks,
//! ascending inside a block, one fused multiply-add per element and one bf16 round - because the
//! tiling decides which thread owns an output, never the order of its sum. There is no split-K and
//! no reassociation, so bit identity is by construction and the identity gate keeps its meaning.
//!
//! FOUR ENTRY POINTS, ONE BODY. The kernel file instantiates one templated body four times, and the
//! four differ only in the operations the seam they serve is pinned to - never in the tiling, the
//! prefetch schedule or which thread owns an output:
//!
//! * [Mode::Fused] / [KERNEL] - the dense projections. One contracted fma per element.
//! * [Mode::NonFused] / [KERNEL_NONFUSED] - the engine's coded-matmul twin, the arithmetic a
//!   shared-expert seam is written with on the host (native_linear to linalg::matmul_f8_scaled to
//!   coded_matmul, whose Rust expression acc += value * element(row, col) rounds the multiply and
//!   then rounds the add). It exists so the twin is available and pinned, not because the shared
//!   expert needs it: for E4M3-coded operands the product of two decoded values is EXACT in f32
//!   (4 + 4 significant bits, far inside 24), so no multiply rounds and the fused and non-fused
//!   modes round the same single value in the same order - tests/mxfp8_gemv_oracle.rs proves the
//!   exactness for all 256 codes and the agreement of the two modes on every served shape. The
//!   shared expert therefore launches [Mode::Fused] and stays byte-identical to its host row while
//!   paying one instruction per element instead of two.
//! * [Mode::NonFusedF32Activation] / [KERNEL_NONFUSED_F32_ACTIVATION] - the compressor seam
//!   (ref_seam). Its host twin multiplies an f32 activation row by f32 weights decoded once per
//!   layer and accumulates with the same rounded multiply, rounded add. This mode stages the f32
//!   row unchanged in shared memory and keeps the weights F8 coded on the device, decoding them
//!   in-kernel with the same table arithmetic the host's one-time decode used, so the value is
//!   identical and the traffic stays at the coded bytes.
//! * [Mode::NonFusedF32ActivationF32Stores] / [KERNEL_NONFUSED_F32_ACTIVATION_F32_OUT] - the
//!   shared expert's f32-reference arm on the device. Same f32-activation staging as the compressor
//!   mode, but the output is the f32 accumulator itself rather than its bf16 rounding: the host row
//!   it replaces (shared_expert_rows -> expert_body) keeps the f32 row end to end, with no
//!   activation quantisation and no bf16 boundary rounding, so a bf16 store would move the values.
//!
//! The activation pointer of the f32-activation modes arrives in the a_fp8 word as const float* and
//! a_scales is unused; in the f32-store mode the out word is a float* holding four bytes per output
//! column. The word count and the declaration shape are the same eleven words in all four modes, so
//! one parameter builder fills every launch.
//!
//! This module is an addition, in the form of the standing new-symbol rule
//! (`routed_gate_up_swiglu_splitk_abi`): the pinned kernel's source, cubin bytes, digest and
//! [crate::cuda_driver::KernelId] are untouched, and [KERNEL] is its own cubin, its own baseline
//! entry and its own symbol loaded by name through
//! [crate::cuda_driver::Cuda::load_dense_gemv]. The declaration is pinned against [PARAMS], the same
//! eleven-word ABI as the pinned kernel, so the same
//! [crate::mxfp8_dense_launch::Mxfp8DenseParams] fills the argument array.
//!
//! Geometry. Grid `(ceil(cols / OUT_COLUMNS), 1, 1)` blocks of one warp ([BLOCK]); the block's 32
//! lanes own 32 consecutive output columns of the single output row. The activation row is decoded
//! once per block into [MAX_INNER] f32 of static shared memory, which is the only shape limit this
//! kernel imposes: the wrapper refuses a reduction longer than that rather than truncating it.
//!
//! No cubin pin in the launch type: the device chain's catalog owns the cubin's bytes and digest.
//! The values this module records ([CUBIN_BYTES], [CUBIN_SHA256]) are the compiled artifact's
//! identity, pinned against the baseline manifest by the tests.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Kernel, KernelId, Module};
use crate::mxfp8_dense_launch::Mxfp8DenseParams;

/// The kernel symbol in its own cubin: the dense-projection instantiation.
pub const KERNEL: &str = "b12x_mxfp8_gemv_m1_sm121";

/// The shared-expert instantiation: rounded multiply then rounded add, coded activation.
pub const KERNEL_NONFUSED: &str = "b12x_mxfp8_gemv_m1_nf_sm121";

/// The compressor instantiation: rounded multiply then rounded add, f32 activation row.
pub const KERNEL_NONFUSED_F32_ACTIVATION: &str = "b12x_mxfp8_gemv_m1_nf_f32a_sm121";

/// The shared expert's f32-reference instantiation: f32 activation row and f32 accumulators stored
/// as f32 (no quantisation, no bf16 output rounding).
pub const KERNEL_NONFUSED_F32_ACTIVATION_F32_OUT: &str = "b12x_mxfp8_gemv_m1_f32a_f32o_sm121";

/// Which instantiation of the one body a launch runs.
///
/// The mode is a compile-time parameter in the kernel file, so this value only selects the symbol
/// to load and the name the evidence records; it never reaches the kernel as an argument word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The dense projections: one contracted fma per element.
    Fused,
    /// The shared expert: a rounded multiply then a rounded add, coded activation.
    NonFused,
    /// The compressor seam: a rounded multiply then a rounded add, f32 activation row.
    NonFusedF32Activation,
    /// The shared expert's f32 reference: f32 activation row and f32 accumulators stored as f32.
    NonFusedF32ActivationF32Stores,
}

impl Mode {
    /// Every mode, in the order the file declares its entry points.
    #[must_use]
    pub fn all() -> [Self; 4] {
        [
            Self::Fused,
            Self::NonFused,
            Self::NonFusedF32Activation,
            Self::NonFusedF32ActivationF32Stores,
        ]
    }

    /// The symbol this mode loads.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Fused => KERNEL,
            Self::NonFused => KERNEL_NONFUSED,
            Self::NonFusedF32Activation => KERNEL_NONFUSED_F32_ACTIVATION,
            Self::NonFusedF32ActivationF32Stores => KERNEL_NONFUSED_F32_ACTIVATION_F32_OUT,
        }
    }

    /// The name an evidence record carries for a launch of this mode.
    #[must_use]
    pub fn arithmetic_mode(self) -> &'static str {
        match self {
            Self::Fused => "fused-multiply-add",
            Self::NonFused => "rounded-multiply-then-rounded-add",
            Self::NonFusedF32Activation => "rounded-multiply-then-rounded-add-f32-activation",
            Self::NonFusedF32ActivationF32Stores => {
                "rounded-multiply-then-rounded-add-f32-activation-f32-stores"
            }
        }
    }
}

/// The kernel source, as the declaration pin and the constant pin see it.
pub const SOURCE: &str = include_str!("../../../kernels/b12x_mxfp8_gemv_m1_sm121.cu");

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

/// Parses the declaration of one mode's entry point.
#[must_use]
pub fn declaration_of(mode: Mode) -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, mode.symbol())
}

/// Whether a parsed declaration names the symbol and has [PARAMS] shape and order.
#[must_use]
pub fn declaration_matches_symbol(decl: &KernelDecl, symbol: &str) -> bool {
    decl.name == symbol
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

/// One warp per B12x micro-tile: 32 threads, one output column each.
pub const BLOCK: (u32, u32, u32) = (32, 1, 1);
/// Output columns one block owns: the block's thread count, one column per lane.
pub const OUT_COLUMNS: u32 = 32;
/// The one output row the kernel serves.
pub const ROWS: u32 = 1;
/// K-blocks per lane held in flight while the previous ones accumulate.
pub const PIPELINE: u32 = 6;
/// The longest reduction the shared activation row holds, in values.
pub const MAX_INNER: u32 = 5120;

/// The compiled artifact's byte count, as nvcc V13.0.88 (CUDA 13.0.3) compiles
/// `kernels/b12x_mxfp8_gemv_m1_sm121.cu` for sm_121a with the baseline's flags.
pub const CUBIN_BYTES: usize = 112512;
/// The compiled artifact's SHA-256, from the same build as [CUBIN_BYTES].
pub const CUBIN_SHA256: &str = "c33cd129f0003e1a4778bdced1fe788bf1f4a1be84fe8033c7989bc8d4bab469";

/// The kernel id a caller would use if it resolved the symbol through the shared driver's enum.
///
/// The symbolic loader ([crate::cuda_driver::Cuda::load_dense_gemv]) is what the call site uses, so
/// this value is documentation of the owning family rather than a dispatch key: the symbol is loaded
/// by name and the pinned kernel stays the frozen ABI id.
pub const FAMILY: KernelId = KernelId::Mxfp8Dense;

/// The grid for a `rows x cols` output: `(ceil(cols / OUT_COLUMNS), 1, 1)`.
///
/// # Errors
/// A row count other than [ROWS] is refused, because the kernel computes exactly one output row and
/// a larger count would silently drop rows; a zero column count is refused rather than launched as a
/// zero-sized grid.
pub fn grid(rows: u32, cols: u32) -> Result<(u32, u32, u32), String> {
    if rows != ROWS {
        return Err(format!(
            "the decode GEMV computes exactly {ROWS} output row, the launch asked for {rows}"
        ));
    }
    if cols == 0 {
        return Err(
            "the decode GEMV runs on at least one output column, so the grid is not zero-sized"
                .to_owned(),
        );
    }
    Ok((cols.div_ceil(OUT_COLUMNS), 1, 1))
}

/// Loads the kernel from its cubin, resolving [KERNEL] by name.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [KERNEL].
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    load_mode(cuda, cubin, Mode::Fused)
}

/// Loads one mode's symbol from the shared cubin.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain that mode's symbol.
pub fn load_mode(cuda: &Cuda, cubin: &[u8], mode: Mode) -> Result<(Module, Kernel), String> {
    cuda.load_dense_gemv_mode(cubin, mode.symbol())
}

/// The split-K opt GEMV (kernels/opt_gemv_sm121.cu): when installed, every decode GEMV launch is
/// served by it - same operands, 8 warps per 32-column tile instead of one.
static OPT_FUNCTIONS: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
const OPT_SYMBOLS: [&str; 3] = [
    "opt_gemv_fp8a_bf16o_t1",
    "opt_gemv_f32a_bf16o_t1",
    "opt_gemv_f32a_f32o_t1",
];

/// The up-to-8-token instantiations [`launch_opt`] uses for a multi-row (verify) launch, indexed
/// as [`OptKind`]. Per-token arithmetic is the t1 kernel's, so a row's output is bit-identical.
static OPT_ROWS_FUNCTIONS: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
const OPT_ROWS_SYMBOLS: [&str; 4] = [
    "opt_gemv_fp8a_bf16o_t8",
    "opt_gemv_f32a_bf16o_t8",
    "opt_gemv_f32a_f32o_t8",
    "opt_gemv_bf16a_bf16o_t8",
];

/// The 2- and 4-token instantiations, indexed as [`OptKind`]: a narrow or short multi-row launch
/// splits its tokens over more blocks with them. Bit-identical to the t8 form per token.
static OPT_T2_FUNCTIONS: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
const OPT_T2_SYMBOLS: [&str; 4] = [
    "opt_gemv_fp8a_bf16o_t2",
    "opt_gemv_f32a_bf16o_t2",
    "opt_gemv_f32a_f32o_t2",
    "opt_gemv_bf16a_bf16o_t2",
];
static OPT_T4_FUNCTIONS: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
const OPT_T4_SYMBOLS: [&str; 4] = [
    "opt_gemv_fp8a_bf16o_t4",
    "opt_gemv_f32a_bf16o_t4",
    "opt_gemv_f32a_f32o_t4",
    "opt_gemv_bf16a_bf16o_t4",
];

/// Loads the opt GEMV symbols and routes [`launch`] through them. The module is returned so the
/// caller keeps it loaded for the run.
pub fn install_opt(cuda: &Cuda, cubin: &[u8]) -> Result<Module, String> {
    let mut module = None;
    for (slot, symbol) in OPT_FUNCTIONS.iter().zip(OPT_SYMBOLS) {
        let (m, k) = cuda.load_named(cubin, symbol)?;
        slot.store(k.to_raw().0, std::sync::atomic::Ordering::Release);
        module = Some(m);
    }
    for (slot, symbol) in OPT_ROWS_FUNCTIONS
        .iter()
        .zip(OPT_ROWS_SYMBOLS)
        .chain(OPT_T2_FUNCTIONS.iter().zip(OPT_T2_SYMBOLS))
        .chain(OPT_T4_FUNCTIONS.iter().zip(OPT_T4_SYMBOLS))
    {
        let (m, k) = cuda.load_named(cubin, symbol)?;
        slot.store(k.to_raw().0, std::sync::atomic::Ordering::Release);
        module = Some(m);
    }
    module.ok_or_else(|| "opt gemv: no symbols".to_owned())
}

/// The tensor-core GEMMs (kernels/opt_mma_sm121.cu), indexed as [`OptKind`]: [`launch_opt`] runs
/// a launch of more than 8 tokens (the bulk prefill) through them once [`install_mma`] has run.
static OPT_MMA_FUNCTIONS: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
const OPT_MMA_SYMBOLS: [&str; 4] = [
    "opt_mma_fp8_bf16o",
    "opt_mma_f32a_bf16o",
    "opt_mma_f32a_f32o",
    "opt_mma_bf16a_bf16o",
];

/// Loads the tensor-core GEMM symbols, which [`launch_opt`] then uses above 8 tokens. The module is
/// returned so the caller keeps it loaded for the run.
pub fn install_mma(cuda: &Cuda, cubin: &[u8]) -> Result<Module, String> {
    let mut module = None;
    for (slot, symbol) in OPT_MMA_FUNCTIONS.iter().zip(OPT_MMA_SYMBOLS) {
        let (m, k) = cuda.load_named(cubin, symbol)?;
        slot.store(k.to_raw().0, std::sync::atomic::Ordering::Release);
        module = Some(m);
    }
    module.ok_or_else(|| "opt mma: no symbols".to_owned())
}

/// True once [`install_mma`] has run.
#[must_use]
pub fn mma_installed() -> bool {
    OPT_MMA_FUNCTIONS[0].load(std::sync::atomic::Ordering::Acquire) != 0
}

/// The tensor-core form of [`launch_opt`]: 128 x 128 output tiles. The FP8 kernel walks the token
/// tiles fastest (its weights are the wider operand), the f32/bf16 kernels the column tiles (their
/// activation rows are).
fn launch_mma(cuda: &Cuda, index: usize, args: &OptArgs) -> Result<(), String> {
    let word = OPT_MMA_FUNCTIONS[index].load(std::sync::atomic::Ordering::Acquire);
    if word == 0 {
        return Err(format!(
            "opt gemv: {} tokens need the tensor-core GEMMs, which are not installed",
            args.tokens
        ));
    }
    let (m_tiles, n_tiles) = (args.tokens.div_ceil(128), args.cols.div_ceil(128));
    let grid = if index == 0 {
        (m_tiles, n_tiles, args.groups)
    } else {
        (n_tiles, m_tiles, args.groups)
    };
    cuda.launch(
        Kernel::from_raw(word, OPT_MMA_SYMBOLS[index]),
        grid,
        (256, 1, 1),
        &[
            args.a,
            args.a_scales,
            args.w,
            args.ws,
            args.out,
            u64::from(args.cols),
            u64::from(args.inner),
            u64::from(args.tokens),
            args.a_tok_stride,
            args.out_tok_stride,
            args.a_grp_stride,
            args.w_grp_stride,
            args.s_grp_stride,
            args.out_grp_stride,
            u64::from(args.k_stride_blocks),
        ],
    )
}

fn opt_for(kernel: Kernel) -> Option<Kernel> {
    let index = match kernel.to_raw().1 {
        KERNEL | KERNEL_NONFUSED => 0,
        KERNEL_NONFUSED_F32_ACTIVATION => 1,
        KERNEL_NONFUSED_F32_ACTIVATION_F32_OUT => 2,
        _ => return None,
    };
    let word = OPT_FUNCTIONS[index].load(std::sync::atomic::Ordering::Acquire);
    (word != 0).then(|| Kernel::from_raw(word, OPT_SYMBOLS[index]))
}

/// Which opt GEMV instantiation [`launch_opt`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptKind {
    /// E4M3 + ue8m0 activation, bf16 output.
    Fp8aBf16o,
    /// f32 activation, bf16 output.
    F32aBf16o,
    /// f32 activation, f32 output.
    F32aF32o,
    /// bf16 activation, bf16 output (multi-row launches only; the t1 form lives with wo_a).
    Bf16aBf16o,
}

/// The opt GEMV's full argument set: `groups` independent [cols][inner] products over the grid's
/// y axis, each with its own activation, weight, scale and output offset; `k_stride_blocks` is the
/// tile-row stride of a K-slice of a wider matrix (0 = `inner / 32`). Strides are in elements of
/// their buffer (bytes for weights and scales).
#[derive(Debug, Clone, Copy)]
pub struct OptArgs {
    /// Activation rows (E4M3, f32 or bf16 by kind).
    pub a: u64,
    /// The E4M3 activation's MX scales (unused by the f32 and bf16 kinds).
    pub a_scales: u64,
    /// Swizzled E4M3 weight tiles.
    pub w: u64,
    /// The weights' UE8M0 block scales.
    pub ws: u64,
    /// Output rows.
    pub out: u64,
    /// Output columns per group.
    pub cols: u32,
    /// Reduction length per group.
    pub inner: u32,
    /// Activation rows: 1..=8 on the GEMV, more on the tensor-core GEMMs.
    pub tokens: u32,
    /// Elements between consecutive tokens' activations.
    pub a_tok_stride: u64,
    /// Elements between consecutive tokens' outputs.
    pub out_tok_stride: u64,
    /// Independent products on the grid's y axis.
    pub groups: u32,
    /// Activation elements between groups.
    pub a_grp_stride: u64,
    /// Weight bytes between groups.
    pub w_grp_stride: u64,
    /// Scale bytes between groups.
    pub s_grp_stride: u64,
    /// Output elements between groups.
    pub out_grp_stride: u64,
    /// Tile-row stride in MX blocks (0 = `inner / 32`).
    pub k_stride_blocks: u32,
}

/// True once [`install_opt`] has run.
#[must_use]
pub fn opt_installed() -> bool {
    OPT_FUNCTIONS[0].load(std::sync::atomic::Ordering::Acquire) != 0
}

/// Launches one opt GEMV over 1..=8 tokens, choosing how many tokens each block serves (see the
/// body). Every choice gives the same bits per token: a token's chain does not depend on it. More
/// tokens (a bulk prefill chunk) run on the tensor-core GEMMs, which round differently.
pub fn launch_opt(cuda: &Cuda, kind: OptKind, args: &OptArgs) -> Result<(), String> {
    let index = opt_index(kind, args)?;
    if args.tokens > 8 {
        return launch_mma(cuda, index, args);
    }
    launch_gemv(cuda, kind, index, args)
}

/// [launch_opt] that keeps more than 8 tokens on the GEMV (split over the grid's z axis, at most
/// 8 per block) rather than the tensor-core GEMMs: every token's bits are the GEMV's, at the price
/// of reading the weights once per 8 tokens (mostly from L2). For a few dozen rows, such as several
/// sequences' draft blocks.
///
/// # Errors
/// As [launch_opt].
pub fn launch_opt_split(cuda: &Cuda, kind: OptKind, args: &OptArgs) -> Result<(), String> {
    let index = opt_index(kind, args)?;
    launch_gemv(cuda, kind, index, args)
}

/// The kind's slot in the installed function tables, after the shape checks every launch shares.
fn opt_index(kind: OptKind, args: &OptArgs) -> Result<usize, String> {
    if args.tokens == 0
        || !args.cols.is_multiple_of(32)
        || !args.inner.is_multiple_of(32)
        || args.groups == 0
    {
        return Err(format!("opt gemv: unsupported shape {args:?}"));
    }
    Ok(match kind {
        OptKind::Fp8aBf16o => 0,
        OptKind::F32aBf16o => 1,
        OptKind::F32aF32o => 2,
        OptKind::Bf16aBf16o => 3,
    })
}

fn launch_gemv(cuda: &Cuda, kind: OptKind, index: usize, args: &OptArgs) -> Result<(), String> {
    use std::sync::atomic::Ordering::Acquire;
    // Tokens per block (every form gives a token the same bits): about four, so a block's
    // accumulators stay short while few splits re-read the weights (from L2: the kernel runs a
    // tile's splits side by side). A wide matrix with a long reduction has the blocks already and
    // takes eight from five tokens; a very narrow one splits further to get them; bf16 rows run a
    // ragged second block of four slowly. By sweep over the decode shapes at 1 to 16 tokens.
    let blocks = (args.cols / 32) * args.groups;
    let one = if index == 3 {
        0
    } else {
        OPT_FUNCTIONS[index].load(Acquire)
    };
    let mut per_block = if args.tokens == 1 && one != 0 {
        1
    } else if blocks >= WIDE_BLOCKS && args.inner >= WIDE_INNER {
        match args.tokens {
            0..=2 => 2,
            3..=4 => 4,
            _ => 8,
        }
    } else if blocks <= SPLIT_BLOCKS && one != 0 {
        match args.tokens {
            0..=3 => 1,
            4..=10 => 2,
            _ => 4,
        }
    } else if args.tokens <= 2 || (index == 3 && matches!(args.tokens, 5..=6)) {
        2
    } else {
        4
    };
    // A block whose rows do not fit the kernel's staging bytes reads them from global memory at
    // several microseconds per extra token, so fewer tokens per block that stage are faster.
    let element_bytes = match kind {
        OptKind::Fp8aBf16o => 1,
        OptKind::Bf16aBf16o => 2,
        OptKind::F32aBf16o | OptKind::F32aF32o => 4,
    };
    let staged = |per_block: u32| {
        let live = per_block.min(args.tokens);
        let scales = if element_bytes == 1 {
            live * (args.inner / 32)
        } else {
            0
        };
        let budget = if per_block == 1 {
            OPT_STAGE_T1
        } else {
            OPT_STAGE
        };
        live * args.inner * element_bytes + scales <= budget
    };
    while per_block > 2 && !staged(per_block) {
        per_block /= 2;
    }
    let (word, symbol) = match per_block {
        1 => (one, OPT_SYMBOLS[index]),
        2 => (OPT_T2_FUNCTIONS[index].load(Acquire), OPT_T2_SYMBOLS[index]),
        4 => (OPT_T4_FUNCTIONS[index].load(Acquire), OPT_T4_SYMBOLS[index]),
        _ => (
            OPT_ROWS_FUNCTIONS[index].load(Acquire),
            OPT_ROWS_SYMBOLS[index],
        ),
    };
    if word == 0 {
        return Err("opt gemv: not installed".to_owned());
    }
    cuda.launch(
        Kernel::from_raw(word, symbol),
        (args.cols / 32, args.groups, args.tokens.div_ceil(per_block)),
        (256, 1, 1),
        &[
            args.a,
            args.a_scales,
            args.w,
            args.ws,
            args.out,
            u64::from(args.cols),
            u64::from(args.inner),
            u64::from(args.tokens),
            args.a_tok_stride,
            args.out_tok_stride,
            args.a_grp_stride,
            args.w_grp_stride,
            args.s_grp_stride,
            args.out_grp_stride,
            u64::from(args.k_stride_blocks),
        ],
    )
}

/// The widest matrix (blocks per token) [launch_opt] splits to one or two tokens per block.
const SPLIT_BLOCKS: u32 = 16;

/// From this many blocks per token, and a reduction of at least [WIDE_INNER], [launch_opt] keeps
/// up to eight tokens per block.
const WIDE_BLOCKS: u32 = 128;
const WIDE_INNER: u32 = 1024;

/// The opt GEMV's activation staging bytes per block (OPT_G_STAGE, OPT_G_STAGE_T1).
const OPT_STAGE: u32 = 48128;
const OPT_STAGE_T1: u32 = 20480;

/// Launches the decode GEMV once.
///
/// # Errors
/// Refuses a shape the kernel does not serve: rows other than [ROWS], a zero column count, a zero
/// reduction length, a reduction that is not a whole number of MX blocks, or one longer than
/// [MAX_INNER].
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &Mxfp8DenseParams) -> Result<(), String> {
    check(params)?;
    let grid = grid(params.rows(), params.cols())?;
    if let Some(opt) = opt_for(kernel) {
        let p = params.as_slice();
        let (cols, inner) = (u64::from(params.cols()), u64::from(params.inner()));
        return cuda.launch(
            opt,
            grid,
            (256, 1, 1),
            &[
                p[0], p[1], p[2], p[3], p[4], cols, inner, 1, inner, cols, 0, 0, 0, 0, 0,
            ],
        );
    }
    cuda.launch(kernel, grid, BLOCK, params.as_slice())
}

/// The shape checks every mode shares.
///
/// # Errors
/// Refuses a shape the kernel does not serve: rows other than [ROWS], a zero column count, a zero
/// reduction length, a reduction that is not a whole number of MX blocks, or one longer than
/// [MAX_INNER].
pub fn check(params: &Mxfp8DenseParams) -> Result<(), String> {
    if params.rows() != ROWS {
        return Err(format!(
            "the decode GEMV computes exactly {ROWS} output row, the launch asked for {}",
            params.rows()
        ));
    }
    if params.inner() == 0 {
        return Err("the decode GEMV reduces at least one value, so inner is not zero".to_owned());
    }
    if !params.inner().is_multiple_of(32) {
        return Err(format!(
            "the decode GEMV reduces whole 32-value MX blocks, the launch asked for inner = {}",
            params.inner()
        ));
    }
    if params.inner() > MAX_INNER {
        return Err(format!(
            "the decode GEMV holds {MAX_INNER} activation values in shared memory, the launch asked for inner = {}",
            params.inner()
        ));
    }
    Ok(())
}

/// Launches one mode of the decode GEMV once.
///
/// # Errors
/// As [check] and [grid].
pub fn launch_mode(
    cuda: &Cuda,
    kernel: Kernel,
    mode: Mode,
    params: &Mxfp8DenseParams,
) -> Result<(), String> {
    let _ = mode;
    launch(cuda, kernel, params)
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
            "the parsed declaration must be the eleven-word dense ABI: {decl:?}"
        );
        assert_eq!(decl.params.len(), ARG_WORDS);
    }

    // A mutation that the pin must catch: the same declaration with one parameter's const-ness
    // flipped is not the pinned ABI.
    #[test]
    fn the_declaration_pin_is_not_vacuous() {
        let mut decl = declaration();
        decl.params[0].const_pointee = false;
        assert!(!declaration_matches(&decl));
        let mut decl = declaration();
        decl.name = "not_the_symbol".to_owned();
        assert!(!declaration_matches(&decl));
    }

    #[test]
    fn every_mode_is_one_body_with_its_own_pinned_declaration() {
        let mut symbols = Vec::new();
        let mut modes = Vec::new();
        for mode in Mode::all() {
            let decl = declaration_of(mode);
            assert!(
                declaration_matches_symbol(&decl, mode.symbol()),
                "{mode:?}: the parsed declaration must be the eleven-word GEMV ABI: {decl:?}"
            );
            assert!(
                SOURCE.contains(mode.symbol()),
                "{mode:?}: the symbol must be declared in the kernel source"
            );
            symbols.push(mode.symbol());
            modes.push(mode.arithmetic_mode());
        }
        // Four distinct symbols, four distinct arithmetic names: the evidence record is only
        // meaningful if a reader can tell which accumulate a launch ran.
        symbols.sort_unstable();
        symbols.dedup();
        assert_eq!(symbols.len(), 4, "one symbol per mode");
        modes.sort_unstable();
        modes.dedup();
        assert_eq!(modes.len(), 4, "one arithmetic name per mode");
        // The fused mode is the dense ABI pin: declaration_matches is KERNEL's name plus PARAMS.
        assert!(declaration_matches(&declaration_of(Mode::Fused)));
        assert!(!declaration_matches(&declaration_of(Mode::NonFused)));
    }

    // The cubin pin is one manifest entry for one file, and that file must declare exactly the
    // four entry points the modes name: a fifth symbol would be a kernel the baseline does not
    // pin, and a missing one would make a mode load a symbol the cubin does not carry.
    #[test]
    fn the_one_kernel_file_declares_exactly_the_four_mode_symbols() {
        let mut declared = Vec::new();
        let mut rest = SOURCE;
        while let Some(found) = rest.find("extern \"C\" __global__ void ") {
            rest = &rest[found + "extern \"C\" __global__ void ".len()..];
            let end = rest
                .find('(')
                .expect("a kernel declaration opens its parameter list");
            declared.push(rest[..end].trim().to_owned());
        }
        declared.sort_unstable();
        let mut expected: Vec<String> = Mode::all()
            .iter()
            .map(|mode| mode.symbol().to_owned())
            .collect();
        expected.sort_unstable();
        assert_eq!(
            declared, expected,
            "the one GEMV file declares exactly the four modes"
        );
        // And the f32-store mode is exactly the f32-activation staging plus the f32 store: its
        // arithmetic name must carry both, so evidence can tell it from the bf16-store f32a mode.
        assert_eq!(
            Mode::NonFusedF32ActivationF32Stores.symbol(),
            KERNEL_NONFUSED_F32_ACTIVATION_F32_OUT
        );
        assert!(
            Mode::NonFusedF32ActivationF32Stores
                .arithmetic_mode()
                .contains("f32-stores")
        );
        assert_ne!(
            Mode::NonFusedF32ActivationF32Stores.arithmetic_mode(),
            Mode::NonFusedF32Activation.arithmetic_mode()
        );
    }

    #[test]
    fn the_parameters_are_the_pinned_dense_words() {
        assert_eq!(PARAMS, crate::mxfp8_dense_launch::PARAMS);
        let p = params(1, 512, 5120);
        assert_eq!(
            p.as_slice(),
            [
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 1, 512, 5120, 0x6000, 7, 11
            ]
        );
        assert_eq!(p.rows(), 1);
        assert_eq!(p.cols(), 512);
        assert_eq!(p.inner(), 5120);
    }

    #[test]
    fn the_grid_is_the_kernel_tiling() {
        let table = constants();
        assert_eq!(
            table.value("SPARK_M4_GEMV_OUTPUTS_PER_TILE"),
            i128::from(OUT_COLUMNS)
        );
        assert_eq!(table.value("SPARK_M4_GEMV_THREADS"), i128::from(BLOCK.0));
        assert_eq!(table.value("SPARK_M4_GEMV_PIPELINE"), i128::from(PIPELINE));
        assert_eq!(
            table.value("SPARK_M4_GEMV_MAX_INNER"),
            i128::from(MAX_INNER)
        );
        assert_eq!(table.value("SPARK_M4_GEMV_LANE_BYTES"), 32);
        assert_eq!(grid(1, 32), Ok((1, 1, 1)));
        assert_eq!(grid(1, 512), Ok((16, 1, 1)));
        assert_eq!(grid(1, 1280), Ok((40, 1, 1)));
        assert_eq!(grid(1, 5120), Ok((160, 1, 1)));
        assert_eq!(grid(1, 8192), Ok((256, 1, 1)));
        assert_eq!(grid(1, 1300), Ok((41, 1, 1)));
        assert!(grid(0, 1).is_err(), "a zero-row launch must be refused");
        assert!(grid(2, 1).is_err(), "rows other than one must be refused");
        assert!(grid(1, 0).is_err(), "a zero-column launch must be refused");
    }

    // The static_asserts in the kernel are re-evaluated here against the same constants, so a tiling
    // edit that breaks one fails the Rust gate without a CUDA host.
    #[test]
    fn the_kernel_static_asserts_evaluate() {
        let table = constants();
        assert_eq!(
            table.value("SPARK_M4_GEMV_OUTPUTS_PER_TILE"),
            table.value("SPARK_M4_GEMV_THREADS"),
            "one lane per output column"
        );
        assert_eq!(table.value("SPARK_M4_GEMV_THREADS"), 32, "one warp");
        assert_eq!(
            table.value("SPARK_M4_GEMV_OUTPUTS_PER_TILE"),
            table.value("SPARK_B12X_NTILE"),
            "a warp covers exactly one B12x micro-tile's outputs"
        );
        assert_eq!(
            table.value("SPARK_M4_GEMV_LANE_BYTES"),
            table.value("SPARK_B12X_KTILE")
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
            .expect("the baseline must pin the decode GEMV kernel");
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
        assert_ne!(KERNEL, crate::mxfp8_dense_m1_launch::KERNEL);
    }
}

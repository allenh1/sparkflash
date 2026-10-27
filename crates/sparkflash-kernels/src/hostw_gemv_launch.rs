// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Typed launch capability of the decode-step (M=1) GEMV whose weight is the checkpoint's own
//! values (`kernels/b12x_hostw_gemv_m1_sm121.cu`).
//!
//! The ref_seam block's remaining host statements are the compressor's `wkv` and `wgate` Linears
//! (f32 weights at ratio > 1, bf16 at ratio == 1), the indexer's key projection `wk` (bf16 weight,
//! f32 row) and its per-query weights projection `weights_proj` (bf16 weight, bf16 row). Their host
//! rules are the engine's `matmul_checked` inner loop, `acc += xb[col] * wr[col]`: a rounded f32
//! multiply then a rounded f32 add, ascending `col`, one accumulator per output. The pinned
//! [crate::mxfp8_gemv_launch] GEMV cannot serve them because it decodes MXFP8 weights and these
//! weights carry no MX grid - quantising them to E4M3 would change their values - so this is a
//! separate kernel, cubin, symbol set and baseline entry, in the form of the standing new-symbol
//! rule: the pinned GEMV's source, bytes, digest and symbols are untouched.
//!
//! THREE ENTRY POINTS, ONE BODY. The body is templated on the weight's element type and on the
//! store, and the three symbols differ only in those two switches, never in the tiling, the
//! prefetch depth or which thread owns an output:
//!
//! * [Mode::Bf16Weight] / [KERNEL_BF16_WEIGHT] - bf16 weights widened exactly, bf16 store. The
//!   indexer's per-query weights projection: a bf16 Linear whose bf16-module output the host rounds
//!   once more before the index factor multiplies it.
//! * [Mode::Bf16WeightF32Stores] / [KERNEL_BF16_WEIGHT_F32_OUT] - bf16 weights, f32 store. The
//!   indexer's `wk` (its output feeds rms_norm in f32) and the ratio-1 compressor projection.
//! * [Mode::F32WeightF32Stores] / [KERNEL_F32_WEIGHT_F32_OUT] - f32 weights, f32 store. The
//!   compressor's `wkv` and `wgate` at ratio > 1, whose rows are pooled in f32 before the norm.
//!
//! Geometry. Grid `(cols / 32, 1, 1)` blocks of one warp ([BLOCK]); the block's 32 lanes own 32
//! consecutive output columns of the single output row. The activation row is staged once per block
//! into [MAX_INNER] f32 of static shared memory - the only shape limit this kernel imposes - and the
//! weight is the checkpoint's own B12x permutation of 32x32 micro-tiles, sized by the element type.
//!
//! No cubin path in the launch type: the device chain's catalog owns the cubin's bytes and digest.
//! The values this module records ([CUBIN_BYTES], [CUBIN_SHA256]) are the compiled artifact's
//! identity, pinned against the baseline manifest by the tests.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, Module};

/// The kernel symbol family: the cubin and the baseline entry are named after this stem.
pub const KERNEL: &str = "b12x_hostw_gemv_m1_sm121";

/// bf16 weights widened exactly, bf16 store: the indexer's per-query weights projection.
pub const KERNEL_BF16_WEIGHT: &str = "b12x_hostw_gemv_m1_bf16w_sm121";

/// bf16 weights, f32 store: the indexer's key projection and the ratio-1 compressor projection.
pub const KERNEL_BF16_WEIGHT_F32_OUT: &str = "b12x_hostw_gemv_m1_bf16w_f32o_sm121";

/// f32 weights, f32 store: the compressor's wkv and wgate at ratio > 1.
pub const KERNEL_F32_WEIGHT_F32_OUT: &str = "b12x_hostw_gemv_m1_f32w_f32o_sm121";

/// Which instantiation of the one body a launch runs.
///
/// The mode is a compile-time parameter in the kernel file, so this value only selects the symbol to
/// load and the name the evidence records; it never reaches the kernel as an argument word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// bf16 weights, bf16 output row.
    Bf16Weight,
    /// bf16 weights, f32 accumulator written as f32.
    Bf16WeightF32Stores,
    /// f32 weights, f32 accumulator written as f32.
    F32WeightF32Stores,
}

impl Mode {
    /// Every mode, in the order the file declares its entry points.
    #[must_use]
    pub fn all() -> [Self; 3] {
        [
            Self::Bf16Weight,
            Self::Bf16WeightF32Stores,
            Self::F32WeightF32Stores,
        ]
    }

    /// The symbol this mode loads.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Bf16Weight => KERNEL_BF16_WEIGHT,
            Self::Bf16WeightF32Stores => KERNEL_BF16_WEIGHT_F32_OUT,
            Self::F32WeightF32Stores => KERNEL_F32_WEIGHT_F32_OUT,
        }
    }

    /// The name an evidence record carries for a launch of this mode.
    #[must_use]
    pub fn arithmetic_mode(self) -> &'static str {
        match self {
            Self::Bf16Weight => "rounded-multiply-then-rounded-add-bf16-weight-bf16-store",
            Self::Bf16WeightF32Stores => "rounded-multiply-then-rounded-add-bf16-weight-f32-stores",
            Self::F32WeightF32Stores => "rounded-multiply-then-rounded-add-f32-weight-f32-stores",
        }
    }

    /// The weight's element size in bytes, which sizes the B12x tile the host stages.
    #[must_use]
    pub fn weight_element_bytes(self) -> usize {
        match self {
            Self::Bf16Weight | Self::Bf16WeightF32Stores => 2,
            Self::F32WeightF32Stores => 4,
        }
    }

    /// Whether the output row is four bytes per column instead of two.
    #[must_use]
    pub fn out_element_bytes(self) -> usize {
        match self {
            Self::Bf16Weight => 2,
            Self::Bf16WeightF32Stores | Self::F32WeightF32Stores => 4,
        }
    }
}

/// The kernel source, as the declaration pin sees it.
pub const SOURCE: &str = include_str!("../../../kernels/b12x_hostw_gemv_m1_sm121.cu");

/// The number of argument words a launch fills.
pub const ARG_WORDS: usize = 9;

/// One block owns one B12x micro-tile: 32 threads, one output column each.
pub const BLOCK: (u32, u32, u32) = (32, 1, 1);
/// Output columns one block owns: the block's thread count, one column per lane.
pub const OUT_COLUMNS: u32 = 32;
/// The one output row the kernel serves.
pub const ROWS: u32 = 1;
/// K blocks per lane held in flight while the previous ones accumulate.
pub const PIPELINE: u32 = 4;
/// The longest reduction the shared activation row holds, in values.
pub const MAX_INNER: u32 = 5120;

/// The declaration's parameters in order: `(name, base type, pointer, pointee const)`.
///
/// The activation row is always f32; the weight pointer's pointee type follows the mode and the
/// output word is `void` because it is bf16 in one mode and f32 in the other two, so the shape is
/// checked per mode by [declaration_matches].
#[must_use]
pub fn params(mode: Mode) -> [(&'static str, &'static str, bool, bool); ARG_WORDS] {
    let weight = match mode.weight_element_bytes() {
        2 => "__nv_bfloat16",
        _ => "float",
    };
    [
        ("a_row", "float", true, true),
        ("w", weight, true, true),
        ("out", "void", true, false),
        ("rows", "uint32_t", false, false),
        ("cols", "uint32_t", false, false),
        ("inner", "uint32_t", false, false),
        ("trace_ring", "GpuTraceRing", true, false),
        ("layer_id", "uint32_t", false, false),
        ("step_seq", "uint64_t", false, false),
    ]
}

/// Parses the declaration of one mode's entry point out of the pinned source.
#[must_use]
pub fn declaration_of(mode: Mode) -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, mode.symbol())
}

/// Whether a parsed declaration names the mode's symbol and has that mode's pinned shape and order.
#[must_use]
pub fn declaration_matches(mode: Mode) -> bool {
    let decl = declaration_of(mode);
    let pinned = params(mode);
    decl.name == mode.symbol()
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

/// The compiled artifact's byte count, as nvcc V13.0.88 (CUDA 13.0) compiles
/// `kernels/b12x_hostw_gemv_m1_sm121.cu` for sm_121a with the baseline's flags.
pub const CUBIN_BYTES: usize = 73848;
/// The compiled artifact's SHA-256, from the same build as [CUBIN_BYTES].
pub const CUBIN_SHA256: &str = "ca1d8498830cddfcdb9a3f1a72e777cb0360a80e4ce7be18af514695bda6bc72";

/// The nine argument words of one launch.
#[derive(Debug, Clone, Copy)]
pub struct HostwGemvParams([u64; ARG_WORDS]);

impl HostwGemvParams {
    /// Builds the argument array with the kernel's own parameter names, in order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        a_row: Deviceptr,
        w: Deviceptr,
        out: Deviceptr,
        rows: u32,
        cols: u32,
        inner: u32,
        trace_ring: u64,
        layer_id: u32,
        step_seq: u64,
    ) -> Self {
        Self([
            a_row.0,
            w.0,
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
        self.0[3] as u32
    }

    /// The columns of the output tile.
    #[must_use]
    pub fn cols(&self) -> u32 {
        self.0[4] as u32
    }

    /// The reduction length.
    #[must_use]
    pub fn inner(&self) -> u32 {
        self.0[5] as u32
    }

    /// The parameter words a launch passes, the shape `cuLaunchKernel` takes.
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.0
    }
}

/// The grid for a `rows x cols` output: `(cols / OUT_COLUMNS, 1, 1)`.
///
/// # Errors
/// A row count other than [ROWS] is refused, because the kernel computes exactly one output row; a
/// column count that is not whole 32-column tiles is refused, because the kernel writes a whole
/// micro-tile; and a zero column count is refused rather than launched as a zero-sized grid.
pub fn grid(rows: u32, cols: u32) -> Result<(u32, u32, u32), String> {
    if rows != ROWS {
        return Err(format!(
            "the host-weight GEMV computes exactly {ROWS} output row, the launch asked for {rows}"
        ));
    }
    if cols == 0 || !cols.is_multiple_of(OUT_COLUMNS) {
        return Err(format!(
            "the host-weight GEMV writes whole {OUT_COLUMNS}-column micro-tiles, the launch asked for cols = {cols}"
        ));
    }
    Ok((cols / OUT_COLUMNS, 1, 1))
}

/// Loads one mode's symbol from the cubin.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain that mode's symbol.
pub fn load_mode(cuda: &Cuda, cubin: &[u8], mode: Mode) -> Result<(Module, Kernel), String> {
    cuda.load_dense_gemv_mode(cubin, mode.symbol())
}

/// Launches one mode of the host-weight decode GEMV once.
///
/// # Errors
/// As [check] and [grid].
pub fn launch_mode(
    cuda: &Cuda,
    kernel: Kernel,
    mode: Mode,
    params: &HostwGemvParams,
) -> Result<(), String> {
    let _ = mode;
    check(params)?;
    let grid = grid(params.rows(), params.cols())?;
    cuda.launch(kernel, grid, BLOCK, params.as_slice())
}

/// The shape checks every mode shares.
///
/// # Errors
/// Refuses a shape the kernel does not serve: rows other than [ROWS], a zero or non-tile-aligned
/// column count, a zero reduction length, a reduction that is not whole 32-value blocks, or one
/// longer than [MAX_INNER].
pub fn check(params: &HostwGemvParams) -> Result<(), String> {
    if params.rows() != ROWS {
        return Err(format!(
            "the host-weight GEMV computes exactly {ROWS} output row, the launch asked for {}",
            params.rows()
        ));
    }
    if params.cols() == 0 || !params.cols().is_multiple_of(OUT_COLUMNS) {
        return Err(format!(
            "the host-weight GEMV writes whole {OUT_COLUMNS}-column micro-tiles, the launch asked for cols = {}",
            params.cols()
        ));
    }
    if params.inner() == 0 {
        return Err(
            "the host-weight GEMV reduces at least one value, so inner is not zero".to_owned(),
        );
    }
    if !params.inner().is_multiple_of(32) {
        return Err(format!(
            "the host-weight GEMV reduces whole 32-value blocks, the launch asked for inner = {}",
            params.inner()
        ));
    }
    if params.inner() > MAX_INNER {
        return Err(format!(
            "the host-weight GEMV holds {MAX_INNER} activation values in shared memory, the launch asked for inner = {}",
            params.inner()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mode_is_declared_with_its_own_weight_element_type() {
        let mut seen = Vec::new();
        for mode in Mode::all() {
            let decl = declaration_of(mode);
            assert_eq!(decl.name, mode.symbol());
            assert!(
                declaration_matches(mode),
                "{}: the pinned declaration moved",
                mode.symbol()
            );
            seen.push(mode.symbol());
        }
        assert_eq!(seen.len(), 3, "three entry points, one body");
        assert_eq!(Mode::Bf16Weight.weight_element_bytes(), 2);
        assert_eq!(Mode::F32WeightF32Stores.weight_element_bytes(), 4);
        assert_eq!(Mode::Bf16Weight.out_element_bytes(), 2);
        assert_eq!(Mode::Bf16WeightF32Stores.out_element_bytes(), 4);
    }

    #[test]
    fn the_geometry_is_one_row_of_whole_micro_tiles() {
        assert_eq!(grid(1, 4096), Ok((128, 1, 1)));
        assert_eq!(grid(1, 512), Ok((16, 1, 1)));
        assert_eq!(grid(1, 32), Ok((1, 1, 1)));
        assert!(grid(2, 32).is_err());
        assert!(grid(1, 0).is_err());
        assert!(grid(1, 40).is_err());
        assert_eq!(BLOCK, (32, 1, 1));
        assert_eq!(OUT_COLUMNS, 32);
        assert_eq!(ROWS, 1);
        assert_eq!(PIPELINE, 4);
        assert_eq!(MAX_INNER, 5120);
        assert_eq!(ARG_WORDS, 9);
    }

    #[test]
    fn the_params_carry_the_documented_words() {
        let params = HostwGemvParams::new(
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            Deviceptr(0x3000),
            1,
            512,
            5120,
            0x4000,
            7,
            42,
        );
        assert_eq!(params.rows(), 1);
        assert_eq!(params.cols(), 512);
        assert_eq!(params.inner(), 5120);
        assert_eq!(params.as_slice()[0], 0x1000);
        assert_eq!(params.as_slice()[1], 0x2000);
        assert_eq!(params.as_slice()[2], 0x3000);
        assert_eq!(params.as_slice()[8], 42);
        assert!(check(&params).is_ok());
        let wide = HostwGemvParams::new(
            Deviceptr(1),
            Deviceptr(2),
            Deviceptr(3),
            1,
            512,
            5152,
            0,
            0,
            0,
        );
        assert!(check(&wide).is_err(), "the shared row is 5120 values wide");
    }
}

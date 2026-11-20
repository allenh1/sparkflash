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

//! The shared tiled decode GEMV: operand descriptions, split rules and the E4M3 launch.
//!
//! Qwen3.8, GLM-5.3 and DeepSeek all read a BF16 or E4M3 matrix with the same tiled kernel at
//! decode. The types and the split arithmetic live here so a sharding or launch change lands in
//! every arm at once.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_model::BackendError;

/// Tokens the token GEMM's tile holds.
pub const GEMV_MAX_BATCH: usize = 8;

/// Blocks a tiled decode GEMV aims for: four on each SM of a 48-SM part.
pub const TILED_GEMV_BLOCKS: usize = 192;

/// Floats of split sums a tiled GEMV can need.
pub const TILED_GEMV_PARTIALS: usize = 2 * TILED_GEMV_BLOCKS * GEMV_MAX_BATCH * 32;

/// Columns a tiled decode GEMV block takes a round: eight BF16 weights.
pub const TILED_GEMV_ROUND: usize = 256;

/// Columns an E4M3 tiled decode GEMV block takes a round: sixteen codes.
pub const TILED_GEMV_E4M3_ROUND: usize = 512;

/// The narrowest activation row the plain decode GEMV goes tiled at.
pub const TILED_GEMV_MIN_COLUMNS: usize = 4096;

/// A CUDA context handle that the model contract can carry.
///
/// The driver context is a process-wide primary context. The raw pointer inside [`Cuda`] is what
/// makes the compiler refuse `Send`/`Sync`. Each session owns a [`sparkflash_kernels::cuda_driver::RunStream`]
/// and puts all of its copies and launches there; the weights and kernel handles shared by those
/// sessions are read-only. Buffer allocation and free go through the context's own pool mutex.
pub struct CudaHandle(Cuda);

impl CudaHandle {
    /// Takes ownership of an opened context.
    #[must_use]
    pub fn new(cuda: Cuda) -> Self {
        Self(cuda)
    }
}

impl std::ops::Deref for CudaHandle {
    type Target = Cuda;

    fn deref(&self) -> &Cuda {
        &self.0
    }
}

// SAFETY: the handle owns one retained driver context, and the driver's context is process-wide:
// every method this crate calls on it is a query or a launch on a stream the caller names, and the
// CUDA driver serialises context access. A session owns its own stream and buffers, so a `Send`
// handle cannot be used concurrently for the same session's state.
unsafe impl Send for CudaHandle {}
// SAFETY: as above: the handle itself carries no mutable per-thread state, and the run-scope
// caches behind it lock internally.
unsafe impl Sync for CudaHandle {}

/// One uploaded BF16 matrix, rows by columns, row-major.
#[derive(Debug, Clone, Copy)]
pub struct GpuMat {
    /// The matrix's first element on the device.
    pub pointer: Deviceptr,
    /// Output rows the matrix holds.
    pub rows: usize,
    /// Values in a row, the inner dimension a GEMV walks.
    pub columns: usize,
}

/// One matrix as E4M3 codes with one f32 scale a row.
#[derive(Debug, Clone, Copy)]
pub struct GpuMatE4m3 {
    /// The matrix's E4M3 codes on the device, a byte a value.
    pub weight: Deviceptr,
    /// One f32 scale a row.
    pub scales: Deviceptr,
    /// Output rows the matrix holds.
    pub rows: usize,
    /// Values in a row.
    pub columns: usize,
}

impl GpuMatE4m3 {
    /// An E4M3 copy of an uploaded BF16 matrix, quantized by the shared kernel.
    ///
    /// # Errors
    /// When the rows are not whole 16-byte runs of codes, or the device refuses an allocation or
    /// the launch.
    pub fn quantize(
        cuda: &Cuda,
        quantize: Kernel,
        matrix: &GpuMat,
        total: &mut usize,
    ) -> Result<Self, String> {
        if !matrix.columns.is_multiple_of(16) {
            return Err(format!(
                "a [{}, {}] matrix has no whole 16-byte E4M3 rows",
                matrix.rows, matrix.columns
            ));
        }
        let weight = cuda.alloc(matrix.rows * matrix.columns, total)?;
        let scales = cuda
            .alloc(matrix.rows * 4, total)
            .inspect_err(|_| cuda.free_buffers(&[&weight]))?;
        let quantized = Self {
            weight,
            scales,
            rows: matrix.rows,
            columns: matrix.columns,
        };
        const WARPS: usize = 8;
        let launched = cuda
            .launch(
                quantize,
                (matrix.rows.div_ceil(WARPS) as u32, 1, 1),
                ((WARPS * 32) as u32, 1, 1),
                &[
                    matrix.pointer.0,
                    weight.0,
                    scales.0,
                    matrix.rows as u64,
                    matrix.columns as u64,
                ],
            )
            .and_then(|()| {
                cuda.synchronize()
                    .map_err(|error| format!("the E4M3 quantization did not finish: {error}"))
            });
        match launched {
            Ok(()) => Ok(quantized),
            Err(error) => {
                quantized.free(cuda);
                Err(error)
            }
        }
    }

    /// Frees the codes and scales; nothing may run on them after.
    pub fn free(&self, cuda: &Cuda) {
        cuda.free_buffers(&[&self.weight, &self.scales]);
    }
}

/// One matrix as E4M3 codes with one BF16 inverse scale a 128 x 128 tile, as the Qwen3.5/3.6 FP8
/// release stores its dense projections (`weight` and `weight_scale_inv`).
#[derive(Debug, Clone, Copy)]
pub struct GpuMatE4m3Tile {
    /// The matrix's E4M3 codes on the device, a byte a value.
    pub weight: Deviceptr,
    /// The `[ceil(rows / 128)][ceil(columns / 128)]` BF16 inverse scales.
    pub scales: Deviceptr,
    /// Output rows the matrix holds.
    pub rows: usize,
    /// Values in a row.
    pub columns: usize,
}

/// Columns a block-scaled E4M3 tile spans.
pub const E4M3_TILE: usize = 128;

/// The launch and scratch a tiled GEMV needs from an arm.
pub trait GemvDevice {
    /// Launch a kernel on the arm's stream.
    ///
    /// # Errors
    /// When the device refuses the launch.
    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), BackendError>;

    /// Launch a typed call ([crate::calls::Call]) on the arm's stream.
    ///
    /// # Errors
    /// When the device refuses the launch.
    fn call<const N: usize>(
        &self,
        kernel: Kernel,
        call: crate::calls::Call<N>,
    ) -> Result<(), BackendError>
    where
        Self: Sized,
    {
        self.launch(kernel, call.grid, call.block, &call.args)
    }

    /// output = matrix input for rows rows of matrix.columns values.
    ///
    /// # Errors
    /// When the arm has no GEMV for the shape or the device refuses a launch.
    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError>;

    /// The split sums and per-tile counters the tiled GEMVs share.
    fn tiled_scratch(&self) -> (Deviceptr, Deviceptr);
}

/// How many blocks a tiled decode GEMV over tiles of 32 rows splits its columns into.
#[must_use]
pub fn tiled_gemv_splits(tiles: usize, columns: usize) -> usize {
    split_rounds(tiles, columns.div_ceil(TILED_GEMV_ROUND))
}

/// tiled_gemv_splits for an E4M3 matrix, whose rounds are 512 columns.
#[must_use]
pub fn tiled_gemv_e4m3_splits(tiles: usize, columns: usize) -> usize {
    split_rounds(tiles, columns.div_ceil(TILED_GEMV_E4M3_ROUND))
}

fn split_rounds(tiles: usize, rounds: usize) -> usize {
    if tiles >= TILED_GEMV_BLOCKS || rounds <= 1 {
        return 1;
    }
    let wanted = TILED_GEMV_BLOCKS.div_ceil(tiles).min(rounds);
    rounds.div_ceil(rounds.div_ceil(wanted))
}

mod sealed {
    /// Keeps [super::TiledOperand] closed: a matrix kind outside this crate cannot claim a
    /// tiled kernel reads it.
    pub trait Tiled {}
}

/// A matrix a tiled GEMV reads, typed by the plane its kernel takes: BF16 rows, or E4M3 codes
/// with a row scale or a block-scale tile. [tiled_operands] takes one, so the predicate names
/// the operand kind where the launch is written, and a packed BF16 copy ([crate::packed::Packed])
/// or a matrix of a format no tiled kernel reads cannot be presented to it.
///
/// A packed copy is not a tiled operand:
///
/// ```compile_fail,E0277
/// use sparkflash_kernels::cuda_driver::Deviceptr;
/// use sparkflash_ops::gemv::tiled_operands;
/// use sparkflash_ops::packed::{Packed, Tiled};
/// let copy: Packed<Tiled> = unsafe { std::mem::zeroed() };
/// let _ = tiled_operands(&copy, Deviceptr(0));
/// ```
///
/// And an FP8 matrix is not the BF16 tiled GEMV's operand, which has no codes path:
///
/// ```compile_fail,E0277
/// use sparkflash_kernels::cuda_driver::Deviceptr;
/// use sparkflash_ops::formats::{Fp8E4m3, Mat};
/// use sparkflash_ops::gemv::tiled_operands;
/// let fp8: Mat<Fp8E4m3> = todo!("the uploaded FP8 matrix");
/// let _ = tiled_operands(&fp8, Deviceptr(0));
/// ```
pub trait TiledOperand: sealed::Tiled {
    /// The address of the plane the kernel reads.
    fn plane(&self) -> Deviceptr;
    /// The values in a row.
    fn columns(&self) -> usize;
}

impl sealed::Tiled for GpuMatE4m3 {}
impl TiledOperand for GpuMatE4m3 {
    fn plane(&self) -> Deviceptr {
        self.weight
    }

    fn columns(&self) -> usize {
        self.columns
    }
}

impl sealed::Tiled for GpuMatE4m3Tile {}
impl TiledOperand for GpuMatE4m3Tile {
    fn plane(&self) -> Deviceptr {
        self.weight
    }

    fn columns(&self) -> usize {
        self.columns
    }
}

impl sealed::Tiled for crate::formats::Mat<crate::formats::Bf16> {}
impl TiledOperand for crate::formats::Mat<crate::formats::Bf16> {
    fn plane(&self) -> Deviceptr {
        self.data()
    }

    fn columns(&self) -> usize {
        self.columns()
    }
}

/// Whether the tiled GEMV can read this matrix's plane and the activation rows it meets: rows of
/// whole 16-byte chunks, and a K run of whole eight-value steps. The matrix is the typed operand,
/// so the check is written on the kind its launch reads.
#[must_use]
pub fn tiled_operand(matrix: &impl TiledOperand, columns: usize) -> bool {
    columns.is_multiple_of(8) && matrix.plane().0.is_multiple_of(16)
}

/// Whether the tiled GEMV can read a launch's matrix and its activations: whole 16-byte chunks in
/// both, and a K run of whole eight-value steps.
#[must_use]
pub fn tiled_operands(matrix: &impl TiledOperand, input: Deviceptr) -> bool {
    input.0.is_multiple_of(16) && tiled_operand(matrix, matrix.columns())
}

/// output = matrix input over an E4M3 matrix for up to GEMV_MAX_BATCH rows.
///
/// # Errors
/// When there are more rows than the kernel takes, the operands are not 16-byte rows, or the
/// device refuses the launch.
pub fn gemv_e4m3(
    device: &impl GemvDevice,
    kernel: Kernel,
    matrix: &GpuMatE4m3,
    input: Deviceptr,
    output: Deviceptr,
    rows: usize,
) -> Result<(), BackendError> {
    if rows > GEMV_MAX_BATCH || !tiled_operands(matrix, input) {
        return Err(BackendError::new(
            "gemv_e4m3",
            format!(
                "no E4M3 GEMV of {rows} rows over a [{}, {}] matrix",
                matrix.rows, matrix.columns
            ),
        ));
    }
    let tiles = matrix.rows.div_ceil(32);
    let (partials, counters) = device.tiled_scratch();
    device.launch(
        kernel,
        (
            tiles as u32,
            tiled_gemv_e4m3_splits(tiles, matrix.columns) as u32,
            1,
        ),
        (256, 1, 1),
        &[
            matrix.weight.0,
            matrix.scales.0,
            input.0,
            output.0,
            matrix.rows as u64,
            matrix.columns as u64,
            rows as u64,
            partials.0,
            counters.0,
        ],
    )
}

/// output = matrix input over a block-scaled E4M3 matrix for up to GEMV_MAX_BATCH rows, the
/// activations kept f32 (q4e_gemv_tiled_e4m3_tile).
///
/// # Errors
/// When there are more rows than the kernel takes, the columns are not whole 128-column tiles,
/// the operands are not 16-byte rows, or the device refuses the launch.
pub fn gemv_e4m3_tile(
    device: &impl GemvDevice,
    kernel: Kernel,
    matrix: &GpuMatE4m3Tile,
    input: Deviceptr,
    output: Deviceptr,
    rows: usize,
) -> Result<(), BackendError> {
    if rows > GEMV_MAX_BATCH
        || !matrix.columns.is_multiple_of(E4M3_TILE)
        || !tiled_operands(matrix, input)
    {
        return Err(BackendError::new(
            "gemv_e4m3_tile",
            format!(
                "no block-scaled E4M3 GEMV of {rows} rows over a [{}, {}] matrix",
                matrix.rows, matrix.columns
            ),
        ));
    }
    let tiles = matrix.rows.div_ceil(32);
    let (partials, counters) = device.tiled_scratch();
    device.launch(
        kernel,
        (
            tiles as u32,
            tiled_gemv_e4m3_splits(tiles, matrix.columns) as u32,
            1,
        ),
        (256, 1, 1),
        &[
            matrix.weight.0,
            matrix.scales.0,
            input.0,
            output.0,
            matrix.rows as u64,
            matrix.columns as u64,
            rows as u64,
            partials.0,
            counters.0,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_rules_never_leave_an_empty_round() {
        for tiles in 1..=512 {
            for columns in [256, 512, 2560, 4096, 16384] {
                let splits = tiled_gemv_splits(tiles, columns);
                assert!(splits >= 1);
                assert!(columns.div_ceil(TILED_GEMV_ROUND) >= splits);
            }
        }
    }

    #[test]
    fn the_scratch_is_inside_the_reserved_partial_region() {
        const { assert!(TILED_GEMV_PARTIALS <= 4 << 20) };
    }

    /// The tiled admission is written on the typed operand: the matrix's own plane and the
    /// activations are 16-byte loads and the K run whole eight-value steps. A matrix of another
    /// kind, or a packed copy, is refused at the call (the [TiledOperand] doctests); this pins
    /// the shape rule over one kind.
    #[test]
    fn the_tiled_admission_is_the_typed_operand() {
        let matrix = GpuMatE4m3 {
            weight: Deviceptr(0x1000),
            scales: Deviceptr(0x9000),
            rows: 64,
            columns: 512,
        };
        assert!(tiled_operand(&matrix, 512));
        assert!(tiled_operands(&matrix, Deviceptr(0x2000)));
        // An activation off a 16-byte load, and a K run off whole eight-value steps.
        assert!(!tiled_operands(&matrix, Deviceptr(0x2004)));
        assert!(!tiled_operand(&matrix, 500));
        // A plane off a 16-byte load is not a tiled operand's.
        let misaligned = GpuMatE4m3 {
            weight: Deviceptr(0x1004),
            ..matrix
        };
        assert!(!tiled_operand(&misaligned, 512));
        assert!(!tiled_operands(&misaligned, Deviceptr(0x2000)));
    }
}

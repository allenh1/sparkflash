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

//! The typed launch capability of the two Engram stream conversions
//! (kernels/engram_stream_convert_sm121.cu): `engram_narrow_bf16` (f32 -> bf16) and
//! `engram_widen_f32` (bf16 -> f32).
//!
//! WHY THIS MODULE EXISTS. The S1 layer arm carries the residual stream as `hc_mult * dim` f32 values
//! (`LayerResidual::stream`, crates/sparkflash-cli/src/layer_device.rs:372-382), while the released
//! Engram head reads and writes it as bf16 (`model.py:311-365`) - so the landed
//! `engram_gate_write_sm121` (bf16 stream in, bf16 out) and the declared stage write
//! `engram_write_bf16` (f32 stream in and out) meet at a dtype boundary. These launches are that
//! boundary and nothing else: no arithmetic, no reduction, no reordering, and no allocation, readback
//! or synchronisation. The exactness argument is in the kernel header; the short form is that the
//! narrow is the released rounding (`__float2bfloat16_rn`, the same rule as
//! [crate::engram_native_ref::bf16_round]) and the widen is a bit insertion with no rounding.
//!
//! NOT A STAGE. Neither kernel carries the trace ring, layer id or step sequence the seven layer-stage
//! seam kernels carry for ABI uniformity: the repo's rule is the one `layer_seams_launch.rs:20-23,41-43`
//! states for `shared_swiglu_f32` - a non-stage kernel carries no ring, and the stage that launches it
//! records the trace. So both declarations are three argument words: source, destination, element
//! count.
//!
//! GEOMETRY. One launch converts a contiguous element run of `count` elements with
//! `ceil(count / 256)` blocks of 256 threads, capped at [MAX_BLOCKS] because both kernels are
//! grid-strided. The released sizes are [RELEASED_STREAM_VALUES] (hc_mult * dim = 4 * 5,120) and
//! [RELEASED_VALUE_VALUES] (dim = 5,120).
//!
//! SPAN GUARDS. [EngramConvertParams::check_spans] refuses a span shorter than the direction reads or
//! writes, naming the direction and the requirement, and both [launch_narrow] and [launch_widen] call
//! it before the driver sees the call: a device overrun is a silently corrupted residual stream, and a
//! refusal is a stage error the loop can print. The caller passes the spans' real byte lengths (the
//! arm's `DeviceSpan.len`), never a computed pointer.
//!
//! PIN. No cubin bytes or digest live here: the artifact is pinned in baseline/kernel_baseline.txt
//! (the Lead's registration step for a new cubin) and [NARROW_KERNEL]/[WIDEN_KERNEL] are the symbols
//! the by-name loaders resolve. This module resolves symbols; it does not own a pin.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, Module};

/// The cubin stem: the baseline pin's name and the staged file's stem.
pub const CUBIN: &str = "engram_stream_convert_sm121";

/// The f32 -> bf16 symbol.
pub const NARROW_KERNEL: &str = "engram_narrow_bf16";

/// The bf16 -> f32 symbol.
pub const WIDEN_KERNEL: &str = "engram_widen_f32";

/// The kernel source, for the two declaration pins.
pub const SOURCE: &str = include_str!("../../../kernels/engram_stream_convert_sm121.cu");

/// The number of argument words one launch fills: source, destination, element count.
pub const ARG_WORDS: usize = 3;

/// Threads per block; the kernels' grid-stride step.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);

/// Threads per block as the kernel source states it, pinned against [BLOCK] by a test.
pub const THREADS: u32 = 256;

/// The largest grid.x a launch forms; past it the kernels' grid-stride loop covers the remainder.
pub const MAX_BLOCKS: u32 = 65_535;

/// The released stream's element count: hc_mult * dim = 4 * 5,120 f32 values.
pub const RELEASED_STREAM_VALUES: u32 = 20_480;

/// The released value row's element count: dim = 5,120 f32 values.
pub const RELEASED_VALUE_VALUES: u32 = 5_120;

/// Bytes per f32 element.
const F32_BYTES: usize = 4;

/// Bytes per bf16 element.
const BF16_BYTES: usize = 2;

/// The narrow kernel's parameters in order: `(name, base type, pointer, pointee const)`.
pub const NARROW_PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("src", "float", true, true),
    ("dst", "__nv_bfloat16", true, false),
    ("count", "uint32_t", false, false),
];

/// The widen kernel's parameters in order: `(name, base type, pointer, pointee const)`.
pub const WIDEN_PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("src", "__nv_bfloat16", true, true),
    ("dst", "float", true, false),
    ("count", "uint32_t", false, false),
];

/// One of the two conversions, and the element sizes its spans are checked with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// f32 in, bf16 out: the arm's stream as the released bf16 stream.
    Narrow,
    /// bf16 in, f32 out: the released stream as the arm's f32 stream.
    Widen,
}

impl Direction {
    /// Bytes per source element.
    #[must_use]
    pub fn src_bytes(self) -> usize {
        match self {
            Self::Narrow => F32_BYTES,
            Self::Widen => BF16_BYTES,
        }
    }

    /// Bytes per destination element.
    #[must_use]
    pub fn dst_bytes(self) -> usize {
        match self {
            Self::Narrow => BF16_BYTES,
            Self::Widen => F32_BYTES,
        }
    }

    /// The symbol this direction launches.
    #[must_use]
    pub fn kernel(self) -> &'static str {
        match self {
            Self::Narrow => NARROW_KERNEL,
            Self::Widen => WIDEN_KERNEL,
        }
    }

    /// The direction's name, for a refusal message.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Narrow => "narrow",
            Self::Widen => "widen",
        }
    }
}

/// Parses the narrow kernel's declaration out of the pinned source.
#[must_use]
pub fn narrow_declaration_of() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, NARROW_KERNEL)
}

/// Parses the widen kernel's declaration out of the pinned source.
#[must_use]
pub fn widen_declaration_of() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, WIDEN_KERNEL)
}

/// Whether a parsed declaration has a pin's shape and order.
fn matches_pin(decl: &KernelDecl, pins: &[(&str, &str, bool, bool)]) -> bool {
    decl.extern_c
        && decl.global
        && decl.params.len() == pins.len()
        && decl.params.iter().zip(pins).all(|(param, pin)| {
            param.name == pin.0
                && param.base == pin.1
                && param.pointer == pin.2
                && param.const_pointee == pin.3
                && (!param.pointer || param.restrict)
        })
}

/// Whether a parsed narrow declaration has [NARROW_PARAMS]' shape and order.
#[must_use]
pub fn narrow_declaration_matches(decl: &KernelDecl) -> bool {
    decl.name == NARROW_KERNEL && matches_pin(decl, &NARROW_PARAMS)
}

/// Whether a parsed widen declaration has [WIDEN_PARAMS]' shape and order.
#[must_use]
pub fn widen_declaration_matches(decl: &KernelDecl) -> bool {
    decl.name == WIDEN_KERNEL && matches_pin(decl, &WIDEN_PARAMS)
}

/// The two spans one launch reads and writes, in bytes: the caller's real lengths, not computed
/// pointer arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngramConvertSpans {
    /// The source span's length in bytes (f32 for narrow, bf16 for widen).
    pub src_bytes: usize,
    /// The destination span's length in bytes (bf16 for narrow, f32 for widen).
    pub dst_bytes: usize,
}

/// One conversion launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngramConvertParams {
    /// The source span: f32 values for [Direction::Narrow], bf16 values for [Direction::Widen].
    pub src: Deviceptr,
    /// The destination span: bf16 values for [Direction::Narrow], f32 values for [Direction::Widen].
    pub dst: Deviceptr,
    /// Elements to convert, not bytes.
    pub count: u32,
}

impl EngramConvertParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; ARG_WORDS] {
        [self.src.0, self.dst.0, u64::from(self.count)]
    }

    /// The element count as a `usize`.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.count as usize
    }

    /// Refuses a geometry the kernel would read or write out of bounds on.
    ///
    /// # Errors
    /// A zero element count (a zero-sized grid is not a launch), or an element count whose byte size
    /// does not fit a `usize`.
    pub fn validate(&self) -> Result<(), String> {
        if self.count == 0 {
            return Err(
                "Engram conversion: count is zero, so there is no element to convert".to_owned(),
            );
        }
        self.elements().checked_mul(F32_BYTES).ok_or_else(|| {
            format!(
                "Engram conversion: {} elements do not fit a usize",
                self.count
            )
        })?;
        Ok(())
    }

    /// The source bytes the launch reads.
    ///
    /// # Errors
    /// The element count does not fit a `usize` product.
    pub fn need_src_bytes(&self, direction: Direction) -> Result<usize, String> {
        self.elements()
            .checked_mul(direction.src_bytes())
            .ok_or_else(|| {
                format!(
                    "Engram conversion ({}): the source bytes overflow a usize",
                    direction.name()
                )
            })
    }

    /// The destination bytes the launch writes.
    ///
    /// # Errors
    /// The element count does not fit a `usize` product.
    pub fn need_dst_bytes(&self, direction: Direction) -> Result<usize, String> {
        self.elements()
            .checked_mul(direction.dst_bytes())
            .ok_or_else(|| {
                format!(
                    "Engram conversion ({}): the destination bytes overflow a usize",
                    direction.name()
                )
            })
    }

    /// Refuses spans shorter than the launch reads or writes.
    ///
    /// # Errors
    /// Names the direction and the first span that is too short, with the length the launch needs.
    pub fn check_spans(
        &self,
        direction: Direction,
        spans: &EngramConvertSpans,
    ) -> Result<(), String> {
        let need_src = self.need_src_bytes(direction)?;
        let need_dst = self.need_dst_bytes(direction)?;
        if spans.src_bytes < need_src {
            return Err(format!(
                "Engram conversion ({name}): the source span is {} B, the {count} x {width}-byte elements need {need_src} B",
                spans.src_bytes,
                name = direction.name(),
                count = self.count,
                width = direction.src_bytes()
            ));
        }
        if spans.dst_bytes < need_dst {
            return Err(format!(
                "Engram conversion ({name}): the destination span is {} B, the {count} x {width}-byte elements need {need_dst} B",
                spans.dst_bytes,
                name = direction.name(),
                count = self.count,
                width = direction.dst_bytes()
            ));
        }
        Ok(())
    }

    /// The launch grid: one block per 256 elements, capped at [MAX_BLOCKS].
    ///
    /// # Errors
    /// [EngramConvertParams::validate]'s geometry refusals.
    pub fn grid(&self) -> Result<(u32, u32, u32), String> {
        self.validate()?;
        let blocks = u32::try_from(self.elements().div_ceil(THREADS as usize)).map_err(|_| {
            format!(
                "Engram conversion: {} blocks do not fit the grid",
                self.count
            )
        })?;
        Ok((blocks.min(MAX_BLOCKS), 1, 1))
    }
}

/// Loads the narrow kernel from a cubin, resolving the pinned symbol by name.
///
/// Neither conversion symbol has a `KernelId`: they share one cubin and the by-name loader is the
/// repo's existing route for a kernel with its own cubin and no id.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [NARROW_KERNEL].
pub fn load_narrow(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_engram_narrow(cubin)
}

/// Loads the widen kernel from a cubin, resolving the pinned symbol by name.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [WIDEN_KERNEL].
pub fn load_widen(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_engram_widen(cubin)
}

/// Launches the narrow once over one contiguous f32 run.
///
/// # Errors
/// The geometry refusal, the span guard's refusal, or the driver's launch error.
pub fn launch_narrow(
    cuda: &Cuda,
    kernel: Kernel,
    params: &EngramConvertParams,
    spans: &EngramConvertSpans,
) -> Result<(), String> {
    launch(cuda, kernel, Direction::Narrow, params, spans)
}

/// Launches the widen once over one contiguous bf16 run.
///
/// # Errors
/// The geometry refusal, the span guard's refusal, or the driver's launch error.
pub fn launch_widen(
    cuda: &Cuda,
    kernel: Kernel,
    params: &EngramConvertParams,
    spans: &EngramConvertSpans,
) -> Result<(), String> {
    launch(cuda, kernel, Direction::Widen, params, spans)
}

/// Launches one conversion: geometry and span guards, then the driver call.
fn launch(
    cuda: &Cuda,
    kernel: Kernel,
    direction: Direction,
    params: &EngramConvertParams,
    spans: &EngramConvertSpans,
) -> Result<(), String> {
    params.validate()?;
    params.check_spans(direction, spans)?;
    let grid = params.grid()?;
    cuda.launch(kernel, grid, BLOCK, &params.words())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram_native_ref;
    use crate::m4_quant;

    fn params() -> EngramConvertParams {
        EngramConvertParams {
            src: Deviceptr(0x1000),
            dst: Deviceptr(0x2000),
            count: RELEASED_STREAM_VALUES,
        }
    }

    fn spans_for(params: &EngramConvertParams, direction: Direction) -> EngramConvertSpans {
        EngramConvertSpans {
            src_bytes: params
                .need_src_bytes(direction)
                .expect("released geometry fits"),
            dst_bytes: params
                .need_dst_bytes(direction)
                .expect("released geometry fits"),
        }
    }

    #[test]
    fn both_declarations_are_the_pinned_ones() {
        let narrow = narrow_declaration_of();
        let widen = widen_declaration_of();
        assert!(
            narrow_declaration_matches(&narrow),
            "the Engram narrow declaration drifted: {narrow:?}"
        );
        assert!(
            widen_declaration_matches(&widen),
            "the Engram widen declaration drifted: {widen:?}"
        );
        assert_eq!(narrow.params.len(), ARG_WORDS);
        assert_eq!(widen.params.len(), ARG_WORDS);
        assert_eq!(NARROW_PARAMS.len(), ARG_WORDS);
        assert_eq!(WIDEN_PARAMS.len(), ARG_WORDS);
    }

    #[test]
    fn the_declaration_pins_are_not_vacuous() {
        // A dropped pointee const on the narrow source: the pin must refuse it.
        let source = SOURCE.replace("const float *__restrict__ src", "float *__restrict__ src");
        assert_ne!(
            source, SOURCE,
            "the narrow source's const moved; update the mutation"
        );
        let decl = c_decl::kernel_declaration(&source, NARROW_KERNEL);
        assert!(
            !narrow_declaration_matches(&decl),
            "a non-const narrow source must fail the pin: {decl:?}"
        );
        // A dropped `__restrict__` on the widen source.
        let source = SOURCE.replace(
            "const __nv_bfloat16 *__restrict__ src",
            "const __nv_bfloat16 *src",
        );
        assert_ne!(
            source, SOURCE,
            "the widen source moved; update the mutation"
        );
        let decl = c_decl::kernel_declaration(&source, WIDEN_KERNEL);
        assert!(
            !widen_declaration_matches(&decl),
            "an unrestricted widen source must fail the pin: {decl:?}"
        );
        // A wider element count word.
        let source = SOURCE.replace(
            "const uint32_t count) {\n  const uint32_t stride",
            "const int count) {\n  const uint32_t stride",
        );
        assert_ne!(source, SOURCE, "the count word moved; update the mutation");
        let decl = c_decl::kernel_declaration(&source, NARROW_KERNEL);
        assert!(
            !narrow_declaration_matches(&decl),
            "an int count must fail the pin: {decl:?}"
        );
    }

    #[test]
    fn the_argument_array_is_the_declaration_order() {
        let words = params().words();
        assert_eq!(words, [0x1000, 0x2000, u64::from(RELEASED_STREAM_VALUES)]);
        assert_eq!(words.len(), ARG_WORDS);
    }

    #[test]
    fn the_direction_sizes_are_the_declarations() {
        assert_eq!(Direction::Narrow.src_bytes(), 4);
        assert_eq!(Direction::Narrow.dst_bytes(), 2);
        assert_eq!(Direction::Widen.src_bytes(), 2);
        assert_eq!(Direction::Widen.dst_bytes(), 4);
        assert_eq!(Direction::Narrow.kernel(), NARROW_KERNEL);
        assert_eq!(Direction::Widen.kernel(), WIDEN_KERNEL);
    }

    #[test]
    fn the_grid_is_one_block_per_256_elements() {
        let mut p = params();
        p.count = 1;
        assert_eq!(p.grid().expect("one element is a launch"), (1, 1, 1));
        p.count = 256;
        assert_eq!(p.grid().expect("a full block"), (1, 1, 1));
        p.count = 257;
        assert_eq!(p.grid().expect("one element past a block"), (2, 1, 1));
        p.count = RELEASED_STREAM_VALUES;
        assert_eq!(p.grid().expect("the released stream"), (80, 1, 1));
        p.count = RELEASED_VALUE_VALUES;
        assert_eq!(p.grid().expect("the released value row"), (20, 1, 1));
        p.count = u32::MAX;
        assert_eq!(
            p.grid()
                .expect("the cap covers a count the grid cannot index"),
            (MAX_BLOCKS, 1, 1)
        );
        p.count = 0;
        assert!(p.grid().is_err(), "a zero count is not a launch");
    }

    #[test]
    fn a_span_shorter_than_the_launch_needs_is_refused() {
        let p = params();
        let spans = spans_for(&p, Direction::Narrow);
        assert!(p.check_spans(Direction::Narrow, &spans).is_ok());
        let short_src = EngramConvertSpans {
            src_bytes: spans.src_bytes - 1,
            ..spans
        };
        let error = p
            .check_spans(Direction::Narrow, &short_src)
            .expect_err("a short f32 source must be refused");
        assert!(
            error.contains("narrow"),
            "the refusal names the direction: {error}"
        );
        assert!(
            error.contains("need"),
            "the refusal names the requirement: {error}"
        );
        let short_dst = EngramConvertSpans {
            dst_bytes: spans.dst_bytes - 1,
            ..spans
        };
        assert!(p.check_spans(Direction::Narrow, &short_dst).is_err());
        // The widen's spans are the other way round; a narrow-sized source must be refused there.
        let widen_spans = spans_for(&p, Direction::Widen);
        assert!(p.check_spans(Direction::Widen, &widen_spans).is_ok());
        assert!(
            p.check_spans(Direction::Widen, &spans).is_err(),
            "an f32-sized source is short for the bf16->f32 widen"
        );
    }

    #[test]
    fn the_conversions_are_the_twins_roundings() {
        // The reference: the narrow is engram_native_ref::bf16_bits_of and the widen is the inverse
        // read of those bits, which is what engram_native_ref::bf16_round composes.
        let values = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            1.0 / 3.0,
            -7.5e-8,
            1.0e-45,
            3.0e38,
            -6.0e-39,
            5.5,
            12_345.678,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0 + 1.0 / 256.0, // exactly halfway between 1.0 and 1.0 + 2^-7: ties to even
        ];
        for &value in &values {
            let bits = engram_native_ref::bf16_bits_of(value);
            let widened = m4_quant::bf16_bits_to_f32(bits);
            assert_eq!(
                widened,
                engram_native_ref::bf16_round(value),
                "the widen of the narrow must be the twin's bf16_round for {value}"
            );
            // Idempotence: the narrow of an already-narrowed value is the identity, and the widen
            // of those bits returns the same f32.
            let again = engram_native_ref::bf16_bits_of(widened);
            assert_eq!(again, bits, "the narrow is idempotent for {value}");
        }
        // NaN keeps a NaN through both directions (the pattern is not required to be preserved).
        let nan_bits = engram_native_ref::bf16_bits_of(f32::NAN);
        assert!(
            m4_quant::bf16_bits_to_f32(nan_bits).is_nan(),
            "NaN stays NaN through narrow then widen"
        );
    }

    #[test]
    fn the_source_pins_the_conversion_intrinsics() {
        assert!(
            SOURCE.contains(NARROW_KERNEL),
            "the narrow symbol is declared"
        );
        assert!(
            SOURCE.contains(WIDEN_KERNEL),
            "the widen symbol is declared"
        );
        assert!(
            SOURCE.contains("__float2bfloat16_rn"),
            "the narrow is the CUDA round-to-nearest-even intrinsic"
        );
        assert!(
            SOURCE.contains("__bfloat162float"),
            "the widen is the CUDA widening intrinsic"
        );
        assert!(
            !SOURCE.contains("fma"),
            "no fused multiply-add in a conversion"
        );
        assert!(
            !SOURCE.contains("trace_ring"),
            "the conversions carry no trace ring (see the module doc)"
        );
        assert!(
            SOURCE.contains(&format!("SPARK_ENGRAM_CONVERT_THREADS {THREADS}u")),
            "the kernel's thread constant is the launcher's BLOCK.x"
        );
        assert_eq!(
            BLOCK.0, THREADS,
            "the launcher's block and the pinned threads agree"
        );
        assert_eq!(BLOCK.1, 1);
        assert_eq!(BLOCK.2, 1);
    }
}

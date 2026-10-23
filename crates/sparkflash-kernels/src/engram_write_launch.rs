//! The typed launch capability of the Engram residual write
//! (kernels/engram_write_sm121.cu; the `engram_write` stage of
//! crates/sparkflash-cli/src/layer_device.rs:764-768, kernel `engram_write_bf16`).
//!
//! The argument array is the kernel declaration's parameter list in order - four device pointers,
//! the row width `dim`, and the trace ring/layer/step words the other seam kernels carry for ABI
//! uniformity. The kernel emits no trace event (see the kernel header), so the launcher's
//! trace-disabled steady state passes 0 for the ring.
//!
//! WHAT THE LAUNCH DOES. `stream_out[copy * dim + d] = bf16_rne(stream[copy * dim + d] +
//! gate[copy] * value[d])` for every copy, exactly the released Engram write
//! (inference/model.py:363-365) and the write half of
//! [crate::engram_native_ref::gate_and_write]. The gate pass and the wkv projection are NOT here:
//! they are the stage's preceding launches (the landed `engram_gate_write_sm121` /
//! `engram_wkv_sm121`), and this module's only arithmetic authority is the one rounded multiply, the
//! one rounded add and the one bf16 cast. Nothing is allocated, nothing is read back, and the launch
//! returns as soon as it is enqueued on the caller's stream.
//!
//! GEOMETRY. One block per 1024 columns of one copy, 256 threads per block: grid =
//! `(ceil(dim / 1024), copies, 1)`. The contract's declaration has no `hc` parameter, so the copy
//! count is the grid's y dimension and not an argument word; it is a field of
//! [EngramWriteParams] so the launcher can refuse a grid the device cannot form (grid.y is 16-bit)
//! and so the span guards can compute the bytes the launch touches. The copy count is also the
//! released `hc_mult` (4).
//!
//! SPAN GUARDS. [EngramWriteParams::check_spans] refuses a span shorter than the launch reads or
//! writes, naming the span and the requirement, and [launch] calls it before the driver sees the
//! call: a device overrun is a silently corrupted residual stream, and a refusal is a stage error
//! the loop can print. The caller passes the spans' real byte lengths (the arm's `DeviceSpan.len`),
//! never a computed pointer.
//!
//! PIN. No cubin bytes or digest live here: the artifact is pinned in baseline/kernel_baseline.txt
//! (the Lead's registration step for a new cubin) and [KERNEL] is the symbol [load] resolves by
//! name. This module resolves a symbol; it does not own a pin.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, Module};

/// The kernel symbol and the baseline entry's name.
pub const KERNEL: &str = "engram_write_bf16";

/// The kernel source, for the declaration pin.
pub const SOURCE: &str = include_str!("../../../kernels/engram_write_sm121.cu");

/// The number of argument words one launch fills.
pub const ARG_WORDS: usize = 8;

/// Threads per block; the block strides the 1024-column window.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);

/// The contract's block granularity: one block per 1024 columns of one copy.
pub const COLUMNS_PER_BLOCK: u32 = 1024;

/// The released hidden width (the Engram head's `dim`, and the row stride here).
pub const RELEASED_DIM: u32 = 5120;

/// The released Engram copy count (`hc_mult`), which is the launch's grid.y.
pub const RELEASED_COPIES: u32 = 4;

/// The largest copy count a grid can carry in y (CUDA's 16-bit grid.y bound).
pub const MAX_COPIES: u32 = 65_535;

/// The declaration's parameters in order: `(name, base type, pointer, pointee const)`.
pub const PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("stream", "float", true, true),
    ("gate", "float", true, true),
    ("value", "float", true, true),
    ("stream_out", "float", true, false),
    ("dim", "int", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// Parses the kernel declaration out of the pinned source.
#[must_use]
pub fn declaration_of() -> KernelDecl {
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

/// The four spans one launch reads and writes, in bytes: the caller's real lengths, not a
/// computed pointer arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngramWriteSpans {
    /// `[copies][dim]` f32 residual stream (the arm's `residual.stream`).
    pub stream_bytes: usize,
    /// `[copies]` f32 gate coefficients.
    pub gate_bytes: usize,
    /// `[dim]` f32 value row.
    pub value_bytes: usize,
    /// `[copies][dim]` f32 output (may be the stream span itself: an in-place call is exact).
    pub stream_out_bytes: usize,
}

/// One launch's arguments, in the kernel declaration's order, plus the grid's y word.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngramWriteParams {
    /// `[copies][dim]` f32: the residual stream the write reads.
    pub stream: Deviceptr,
    /// `[copies]` f32: the gate coefficients the preceding gate pass produced.
    pub gate: Deviceptr,
    /// `[dim]` f32: the Engram head's shared value row.
    pub value: Deviceptr,
    /// `[copies][dim]` f32 out: the stream with `gate * value` added and rounded once.
    pub stream_out: Deviceptr,
    /// Columns per copy (the released `dim`, 5,120).
    pub dim: u32,
    /// Copies of the head (hc). NOT an argument word: it is the launch's grid.y, because the
    /// contract's declaration has no `hc` parameter.
    pub copies: u32,
    /// The trace ring's device address, or 0 for the trace-disabled steady state.
    pub trace_ring: u64,
    /// The layer id a trace event would record (the kernel records none).
    pub layer_id: u32,
    /// The step sequence a trace event would record (the kernel records none).
    pub step_seq: u64,
}

impl EngramWriteParams {
    /// The argument array a launch passes to the driver, in declaration order.
    ///
    /// `copies` is deliberately absent: it is the grid's y dimension, not a declaration parameter.
    #[must_use]
    pub fn words(&self) -> [u64; ARG_WORDS] {
        [
            self.stream.0,
            self.gate.0,
            self.value.0,
            self.stream_out.0,
            u64::from(self.dim),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }

    /// Refuses a geometry the kernel would read or write out of bounds on.
    ///
    /// # Errors
    /// A zero row width or copy count, a row width past the declaration's `int`, a copy count past
    /// the grid's y bound, or a copy x row product that does not fit the address space.
    pub fn validate(&self) -> Result<(), String> {
        if self.dim == 0 {
            return Err("Engram write: dim is zero, so there is no column to write".to_owned());
        }
        if self.dim > i32::MAX as u32 {
            return Err(format!(
                "Engram write: dim {} does not fit the declaration's int",
                self.dim
            ));
        }
        if self.copies == 0 {
            return Err("Engram write: copies is zero, so there is no copy to write".to_owned());
        }
        if self.copies > MAX_COPIES {
            return Err(format!(
                "Engram write: {copies} copies do not fit the grid's y bound of {MAX_COPIES}; the copy is grid.y because the declaration has no hc word",
                copies = self.copies
            ));
        }
        self.values()
            .map_err(|values| format!("Engram write: {values} x {} overflows", self.dim))?;
        Ok(())
    }

    /// The f32 values one launch reads from (and writes to) the stream: `copies * dim`.
    ///
    /// # Errors
    /// The product does not fit a `usize`.
    pub fn values(&self) -> Result<usize, &'static str> {
        (self.copies as usize)
            .checked_mul(self.dim as usize)
            .ok_or("the copy x row product")
    }

    /// Refuses spans shorter than the launch reads or writes.
    ///
    /// # Errors
    /// Names the first span that is too short, with the length the launch needs.
    pub fn check_spans(&self, spans: &EngramWriteSpans) -> Result<(), String> {
        let values = self.values()?;
        let need_stream = values
            .checked_mul(4)
            .ok_or("Engram write: the stream bytes overflow a usize")?;
        let need_gate = (self.copies as usize)
            .checked_mul(4)
            .ok_or("Engram write: the gate bytes overflow a usize")?;
        let need_value = (self.dim as usize)
            .checked_mul(4)
            .ok_or("Engram write: the value bytes overflow a usize")?;
        if spans.stream_bytes < need_stream {
            return Err(format!(
                "Engram write: the stream span is {} B, the {copies} x {dim} f32 stream needs {need_stream} B",
                spans.stream_bytes,
                copies = self.copies,
                dim = self.dim
            ));
        }
        if spans.gate_bytes < need_gate {
            return Err(format!(
                "Engram write: the gate span is {} B, the {copies} f32 gates need {need_gate} B",
                spans.gate_bytes,
                copies = self.copies
            ));
        }
        if spans.value_bytes < need_value {
            return Err(format!(
                "Engram write: the value span is {} B, the {dim}-value f32 row needs {need_value} B",
                spans.value_bytes,
                dim = self.dim
            ));
        }
        if spans.stream_out_bytes < need_stream {
            return Err(format!(
                "Engram write: the output span is {} B, the {copies} x {dim} f32 write needs {need_stream} B",
                spans.stream_out_bytes,
                copies = self.copies,
                dim = self.dim
            ));
        }
        Ok(())
    }

    /// The launch grid: one block per 1024 columns of one copy.
    ///
    /// # Errors
    /// [EngramWriteParams::validate]'s geometry refusals.
    pub fn grid(&self) -> Result<(u32, u32, u32), String> {
        self.validate()?;
        Ok((self.dim.div_ceil(COLUMNS_PER_BLOCK), self.copies, 1))
    }
}

/// Loads the kernel from a cubin, resolving the pinned symbol by name.
///
/// The symbol has no `KernelId`: the Engram write has its own cubin and the by-name loader is the
/// repo's existing route for a kernel with its own cubin and no id (the kv-path symbols use the
/// same one).
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [KERNEL].
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_engram_write(cubin)
}

/// Launches the write once for one token.
///
/// # Errors
/// The geometry refusal, the span guard's refusal, or the driver's launch error.
pub fn launch(
    cuda: &Cuda,
    kernel: Kernel,
    params: &EngramWriteParams,
    spans: &EngramWriteSpans,
) -> Result<(), String> {
    params.validate()?;
    params.check_spans(spans)?;
    let grid = params.grid()?;
    cuda.launch(kernel, grid, BLOCK, &params.words())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram_native_ref;
    use sparkflash_engine::quant::bf16_round;

    fn pointers() -> [Deviceptr; 4] {
        core::array::from_fn(|index| Deviceptr(0x1000 * (index as u64 + 1)))
    }

    fn params() -> EngramWriteParams {
        let p = pointers();
        EngramWriteParams {
            stream: p[0],
            gate: p[1],
            value: p[2],
            stream_out: p[3],
            dim: RELEASED_DIM,
            copies: RELEASED_COPIES,
            trace_ring: 0,
            layer_id: 7,
            step_seq: 13,
        }
    }

    fn spans_for(params: &EngramWriteParams) -> EngramWriteSpans {
        let values = params.values().expect("released geometry fits a usize");
        EngramWriteSpans {
            stream_bytes: values * 4,
            gate_bytes: params.copies as usize * 4,
            value_bytes: params.dim as usize * 4,
            stream_out_bytes: values * 4,
        }
    }

    #[test]
    fn the_write_declaration_is_the_pinned_one() {
        let decl = declaration_of();
        assert!(
            declaration_matches(&decl),
            "the Engram write declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), ARG_WORDS);
        assert_eq!(PARAMS.len(), ARG_WORDS);
    }

    #[test]
    fn the_argument_array_is_the_declaration_order() {
        let p = params();
        let words = p.words();
        assert_eq!(words.len(), ARG_WORDS);
        assert_eq!(&words[..4], &[0x1000, 0x2000, 0x3000, 0x4000]);
        assert_eq!(words[4], u64::from(RELEASED_DIM));
        assert_eq!(words[5], 0, "the trace ring word");
        assert_eq!(words[6], 7);
        assert_eq!(words[7], 13);
    }

    #[test]
    fn the_grid_is_one_block_per_1024_columns_of_one_copy() {
        assert_eq!(
            params()
                .grid()
                .expect("the released geometry is launchable"),
            (5, 4, 1)
        );
        // A row narrower than a window is still one block, and a row past it gains a block.
        let mut narrow = params();
        narrow.dim = 1;
        assert_eq!(narrow.grid().expect("one column"), (1, 4, 1));
        let mut past = params();
        past.dim = COLUMNS_PER_BLOCK + 1;
        assert_eq!(past.grid().expect("one block and a bit"), (2, 4, 1));
        // Exactly one window is one block, not two.
        let mut exact = params();
        exact.dim = COLUMNS_PER_BLOCK;
        assert_eq!(exact.grid().expect("one exact window"), (1, 4, 1));
    }

    #[test]
    fn a_zero_geometry_is_refused_by_name() {
        let mut zero_dim = params();
        zero_dim.dim = 0;
        let error = zero_dim.validate().expect_err("a zero row is refused");
        assert!(error.contains("dim is zero"), "{error}");
        let mut zero_copies = params();
        zero_copies.copies = 0;
        let error = zero_copies.validate().expect_err("no copy is refused");
        assert!(error.contains("copies is zero"), "{error}");
        let mut many = params();
        many.copies = MAX_COPIES + 1;
        let error = many.validate().expect_err("a grid.y overflow is refused");
        assert!(error.contains("grid's y bound"), "{error}");
        let mut huge = params();
        huge.dim = i32::MAX as u32 + 1;
        let error = huge.validate().expect_err("a row past int is refused");
        assert!(
            error.contains("does not fit the declaration's int"),
            "{error}"
        );
    }

    #[test]
    fn a_span_shorter_than_the_launch_needs_is_refused() {
        let p = params();
        let good = spans_for(&p);
        p.check_spans(&good).expect("the released spans fit");
        for (name, short) in [
            (
                "stream",
                EngramWriteSpans {
                    stream_bytes: good.stream_bytes - 4,
                    ..good
                },
            ),
            (
                "gate",
                EngramWriteSpans {
                    gate_bytes: good.gate_bytes - 4,
                    ..good
                },
            ),
            (
                "value",
                EngramWriteSpans {
                    value_bytes: good.value_bytes - 4,
                    ..good
                },
            ),
            (
                "output",
                EngramWriteSpans {
                    stream_out_bytes: good.stream_out_bytes - 4,
                    ..good
                },
            ),
        ] {
            let error = p.check_spans(&short).expect_err("a short span is refused");
            assert!(
                error.contains(name),
                "the {name} refusal must name its span: {error}"
            );
            assert!(error.contains("need"), "{error}");
        }
    }

    /// The CPU twin: the one rounded multiply, the one rounded add, one bf16 cast - the engine
    /// quantiser's rounding, which is the host reference's own.
    fn write_reference(
        stream: &[f32],
        gates: &[f32],
        value: &[f32],
        dim: usize,
        copies: usize,
    ) -> Vec<f32> {
        assert_eq!(stream.len(), copies * dim);
        assert_eq!(gates.len(), copies);
        assert_eq!(value.len(), dim);
        let mut out = vec![0.0f32; copies * dim];
        for (copy, gate) in gates.iter().enumerate() {
            let base = copy * dim;
            for (column, value) in value.iter().enumerate() {
                let index = base + column;
                out[index] = bf16_round(stream[index] + *gate * *value);
            }
        }
        out
    }

    #[test]
    fn the_write_is_the_landed_twins_write_half() {
        // The landed CPU twin of the gate kernel (crate::engram_native_ref) computes the gates and
        // the same write. Its output is the authority this launcher's arithmetic must reproduce:
        // feed it a synthetic kv/stream/product and check that this module's write reference, given
        // the twin's gates and the twin's value row, produces byte-identical f32 values.
        let dim = 8usize;
        let hc = 2usize;
        let kv: Vec<f32> = (0..hc * dim + dim)
            .map(|index| ((index as f32) * 0.25 - 3.0) / 7.0)
            .collect();
        let stream: Vec<f32> = (0..hc * dim)
            .map(|index| ((index as f32) * 0.5 - 2.0) / 3.0)
            .map(bf16_round)
            .collect();
        let product: Vec<f32> = (0..hc * dim)
            .map(|index| 1.0 + (index as f32) * 0.125)
            .collect();
        let (gates, twin_write) =
            engram_native_ref::gate_and_write(&kv, &stream, &product, dim, hc, 1e-6);
        let value = &kv[hc * dim..hc * dim + dim];
        let this_write = write_reference(&stream, &gates, value, dim, hc);
        assert_eq!(
            this_write.len(),
            twin_write.len(),
            "the write lengths must agree"
        );
        for (index, (mine, twin)) in this_write.iter().zip(&twin_write).enumerate() {
            assert_eq!(
                mine.to_bits(),
                twin.to_bits(),
                "the write differs at value {index}: {mine} vs {twin}"
            );
        }
        // And the rounding is the one released cast: a value halfway between two bf16 neighbours
        // rounds to even, as the twin does.
        let halfway = f32::from_bits(0x3F80_8000); // 1.00390625, the 8th-bit tie below 1.0078125
        assert_eq!(
            bf16_round(halfway).to_bits(),
            write_reference(&[halfway], &[0.0], &[0.0], 1, 1)[0].to_bits()
        );
    }

    #[test]
    fn the_declaration_pin_is_not_vacuous() {
        let mutated = SOURCE.replace("const float *__restrict__ gate,", "const float *gate,");
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        assert!(
            !declaration_matches(&c_decl::kernel_declaration(&mutated, KERNEL)),
            "the qualifier pin is vacuous"
        );
        // The contract's dim is the declaration's `int`: a uint32_t there is a different ABI.
        let mutated = SOURCE.replace("const int dim,", "const uint32_t dim,");
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        assert!(
            !declaration_matches(&c_decl::kernel_declaration(&mutated, KERNEL)),
            "the int pin is vacuous"
        );
    }

    #[test]
    fn the_write_arithmetic_is_the_rounded_one_in_the_source() {
        // The compiled bytes are not available on this host, so the source is the pin: the write
        // must be one rounded multiply, one rounded add and the one bf16 cast, and it must not
        // contract into an FMA (which rounds once where the host twin rounds twice).
        assert!(
            SOURCE.contains("__fadd_rn(stream[index], __fmul_rn(gate_value, value[column]))"),
            "the write must be the explicit rounded multiply-then-add"
        );
        assert!(
            SOURCE.contains("__float2bfloat16_rn"),
            "the write must round through the bf16 conversion intrinsic"
        );
        assert!(
            !SOURCE.contains("__fmaf_rn") && !SOURCE.contains("fmaf("),
            "an FMA would round once where the host twin rounds twice"
        );
        // The trace tail is ABI-uniform and inert: no event, no dereference.
        assert!(
            SOURCE.contains("(void)trace_ring;"),
            "the unused trace words must be stated"
        );
        assert!(
            !SOURCE.contains("trace_ring->"),
            "a null ring must never be dereferenced"
        );
    }
}

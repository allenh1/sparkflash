//! Typed launch capability of the whole-layer device port's seam image (eight kernels)
//! (kernels/layer_seams_sm121.cu).
//!
//! The layers they replace are the ones the served forward loop ran on the host SIMD seam:
//!
//! * mhc_entry_bf16 - the mHC coefficient projection, its tail (rsqrt, sigmoid pre/post and the
//!   split Sinkhorn comb), the hc_pre collapse and the input RMS pre-norm of one token;
//! * mhc_ffn_seams - the same for the FFN predictor, plus the FFN activation quantiser, so the
//!   FFN's E4M3 code row is produced in the same launch that makes the row it encodes;
//! * w8a8_prologue_e4m3 - the attention projection's activation quantiser over the normed row,
//!   with the reference's BF16 narrowing applied before the MX block rule (D39 rule 1: the W8A8
//!   activation is a BF16 module output, so the amax and the codes are the narrowed values');
//! * mhc_attn_residual - the attention sublayer's hc_post over the reduced delta, the entry's hc
//!   rows and its post+comb table;
//! * mhc_ffn_epilogue - the shared expert added onto the reduced routed MoE delta and the FFN's
//!   hc_post over the post-attention stream, in one launch;
//! * w8a8_prologue_bf16_e4m3 - the same quantiser over a BF16 activation row: the reference's
//!   q_norm output is bf16, so the projection quantises the values after that rounding (the f32
//!   variant over-read a half-width span when handed the bf16 row);
//! * rope_rotate_rows_bf16 - the attention output's RoPE over a batch of head rows, with the
//!   rotation's sign as a word: sin_sign = 1 is the forward rotation and -1 the inverse, so one
//!   table pair serves both (a negated table class would state one bit twice);
//! * shared_swiglu_f32 - the shared expert's activation over its gate and up rows, which replaces
//!   the default arm's host SwiGLU between two device GEMVs (it is not a layer stage and carries no
//!   trace ring; the stage that launches it records the trace).
//!
//! The argument arrays are each kernel's declaration in order, and the declarations are pinned here:
//! ENTRY_PARAMS, FFN_PARAMS, PROLOGUE_PARAMS, ATTN_RESIDUAL_PARAMS, FFN_EPILOGUE_PARAMS and
//! SHARED_SWIGLU_PARAMS carry
//! every parameter's name, base type and pointer shape, the *_declaration_matches functions are the
//! comparisons, and a mutation test proves each comparison is not vacuous.
//!
//! Geometry. The released model fixes the shape these kernels are compiled for: one token, four mHC
//! copies, 5,120 hidden values, 2 * 4 + 4^2 = 24 coefficient rows. The seam kernels refuse any
//! other (dim, hc_mult, rows) in the kernel body as well as here, because the shared-memory tables
//! are compile-time sized from the released geometry. The prologue's cols must be a multiple of the
//! 32-value MX block (the released width is) and its rows is the token count; its grid is one
//! thread per (row, MX block), so a 5,120-wide row is exactly 160 threads over five blocks.
//!
//! Tracing. The seven layer-stage kernels carry the trace ring, layer id and step sequence for ABI
//! uniformity (shared_swiglu_f32 does not: it is the only kernel here that is not a stage).
//! The steady-state loop passes no ring, so the contract is that a **null ring writes nothing**:
//! every store through it lives behind an explicit `if (trace_ring != nullptr)` guard, and
//! [tests::every_seam_kernel_guards_its_trace_ring] reads the kernel source and refuses a kernel
//! whose guard disappears (the same shape as the ptx_membar_lines pin: the compiled bytes cannot
//! show a branch that a null ring never takes, so the source is what is pinned). An armed ring gets
//! the parent scope's record - MEGA_MHC for the four mHC seams, DENSE_MXFP8 for the prologue whose
//! quantised row feeds the dense projections. The shared memory and the arithmetic are described in
//! the kernel's header.
//!
//! No cubin pin lives here. The bytes and the digest are in baseline/kernel_baseline.txt
//! (crate::baseline::pin); this module only resolves symbols.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, Module};

/// The cubin stem: the baseline pin's name and the staged file's stem.
pub const CUBIN: &str = "layer_seams_sm121";
/// The kernel sources, for the declaration pins.
pub const SOURCE: &str = include_str!("../../../kernels/layer_seams_sm121.cu");

/// The entry kernel's symbol.
pub const ENTRY_KERNEL: &str = "mhc_entry_bf16";
/// The FFN seam kernel's symbol.
pub const FFN_KERNEL: &str = "mhc_ffn_seams";
/// The activation quantiser's symbol. Its row is narrowed to BF16 with RNE before the MX block
/// rule, which is the reference's composition (`bf16_round` then `quantize_activation_fp8`); the
/// FFN seam's fused quantiser deliberately does not narrow, because the FFN's reference takes the
/// raw f32 row.
pub const PROLOGUE_KERNEL: &str = "w8a8_prologue_e4m3";
/// The BF16-input variant of the activation quantiser: the bf16 q_norm row is the reference's
/// operand, so the projection quantises the values after that rounding rather than over-reading the
/// half-width span as f32.
pub const PROLOGUE_BF16_KERNEL: &str = "w8a8_prologue_bf16_e4m3";
/// The attention residual kernel's symbol: the attention sublayer's hc_post.
pub const ATTN_RESIDUAL_KERNEL: &str = "mhc_attn_residual";
/// The FFN epilogue kernel's symbol: the shared add and the FFN's hc_post.
pub const FFN_EPILOGUE_KERNEL: &str = "mhc_ffn_epilogue";
/// The shared-expert SwiGLU's symbol: the default arm's activation between two device GEMVs.
pub const SHARED_SWIGLU_KERNEL: &str = "shared_swiglu_f32";
/// The attention output's rotated rows: the inverse RoPE over the body's head rows (and the
/// forward rotation, selected by the sign word - one symbol, one table pair).
pub const ROPE_ROWS_KERNEL: &str = "rope_rotate_rows_bf16";

/// The released hidden width the seam kernels are compiled for.
pub const RELEASED_DIM: u32 = 5120;
/// The released mHC multiplier.
pub const RELEASED_HC_MULT: u32 = 4;
/// The released coefficient-row count: 2 * hc_mult + hc_mult^2.
pub const RELEASED_ROWS: u32 = 24;
/// The MX block the activation quantiser scales.
pub const QUANT_BLOCK: u32 = 32;

/// The threads in the seam kernels' block; also the prologue's block.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);
/// The seam kernels' grid: one block owns the token.
pub const SEAM_GRID: (u32, u32, u32) = (1, 1, 1);

/// The entry kernel's argument words: seventeen pointers/scalars plus the trace trio.
pub const ENTRY_ARG_WORDS: usize = 20;
/// The FFN seam kernel's argument words: the hc ABI plus the optional pre-FFN hidden row out.
pub const FFN_ARG_WORDS: usize = 23;
/// The prologue kernel's argument words.
pub const PROLOGUE_ARG_WORDS: usize = 8;
/// The BF16 prologue variant's argument words: the same eight, with a bf16 input pointer.
pub const PROLOGUE_BF16_ARG_WORDS: usize = 8;
/// The attention residual kernel's argument words: four pointers, three geometry words, the
/// native-BF16 flag and the trace trio.
pub const ATTN_RESIDUAL_ARG_WORDS: usize = 11;
/// The FFN epilogue kernel's argument words: five pointers, three geometry words, the shared-row
/// presence and rounding flags and the trace trio.
pub const FFN_EPILOGUE_ARG_WORDS: usize = 14;
/// The shared-expert SwiGLU's argument words: two const pointers, the output pointer, the clamp
/// limit and the row width.
pub const SHARED_SWIGLU_ARG_WORDS: usize = 5;
/// rope_rotate_rows_bf16's argument words: four pointers, the row count, the two widths, the sign
/// word and the trace trio.
pub const ROPE_ROWS_ARG_WORDS: usize = 11;

/// mhc_entry_bf16's declaration in order: (name, base type, pointer, pointee const).
pub const ENTRY_PARAMS: [(&str, &str, bool, bool); ENTRY_ARG_WORDS] = [
    ("residual", "float", true, true),
    ("attn_pre_mix", "float", true, true),
    ("fn_w", "float", true, true),
    ("hc_base", "float", true, true),
    ("hc_scale", "float", true, true),
    ("norm_w", "uint8_t", true, true),
    ("mix_table", "float", true, false),
    ("hc_rows", "float", true, false),
    ("normed", "float", true, false),
    ("dim", "uint32_t", false, false),
    ("hc_mult", "uint32_t", false, false),
    ("rows", "uint32_t", false, false),
    ("norm_bf16", "uint32_t", false, false),
    ("native_bf16", "uint32_t", false, false),
    ("sinkhorn_iterations", "uint32_t", false, false),
    ("norm_eps", "float", false, false),
    ("hc_eps", "float", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// mhc_ffn_seams's declaration in order.
pub const FFN_PARAMS: [(&str, &str, bool, bool); FFN_ARG_WORDS] = [
    ("stream", "float", true, true),
    ("attn_pre_mix", "float", true, true),
    ("fn_w", "float", true, true),
    ("hc_base", "float", true, true),
    ("hc_scale", "float", true, true),
    ("norm_w", "uint8_t", true, true),
    ("mix_table", "float", true, false),
    ("pre_mix", "float", true, false),
    ("seam_row", "float", true, false),
    ("act_codes", "uint8_t", true, false),
    ("act_scales", "uint8_t", true, false),
    ("x_bf16", "__nv_bfloat16", true, false),
    ("dim", "uint32_t", false, false),
    ("hc_mult", "uint32_t", false, false),
    ("rows", "uint32_t", false, false),
    ("norm_bf16", "uint32_t", false, false),
    ("native_bf16", "uint32_t", false, false),
    ("sinkhorn_iterations", "uint32_t", false, false),
    ("norm_eps", "float", false, false),
    ("hc_eps", "float", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// w8a8_prologue_e4m3's declaration in order.
pub const PROLOGUE_PARAMS: [(&str, &str, bool, bool); PROLOGUE_ARG_WORDS] = [
    ("normed", "float", true, true),
    ("codes", "uint8_t", true, false),
    ("scales", "uint8_t", true, false),
    ("cols", "uint32_t", false, false),
    ("rows", "uint32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// mhc_attn_residual's declaration in order.
pub const ATTN_RESIDUAL_PARAMS: [(&str, &str, bool, bool); ATTN_RESIDUAL_ARG_WORDS] = [
    ("hc_rows", "float", true, true),
    ("mix_table", "float", true, true),
    ("reduced_bf16", "void", true, true),
    ("stream_out", "float", true, false),
    ("dim", "uint32_t", false, false),
    ("hc_mult", "uint32_t", false, false),
    ("rows", "uint32_t", false, false),
    ("native_bf16", "uint32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// w8a8_prologue_bf16_e4m3's declaration in order.
pub const PROLOGUE_BF16_PARAMS: [(&str, &str, bool, bool); PROLOGUE_BF16_ARG_WORDS] = [
    ("normed", "__nv_bfloat16", true, true),
    ("codes", "uint8_t", true, false),
    ("scales", "uint8_t", true, false),
    ("cols", "uint32_t", false, false),
    ("rows", "uint32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// shared_swiglu_f32's declaration in order.
pub const SHARED_SWIGLU_PARAMS: [(&str, &str, bool, bool); SHARED_SWIGLU_ARG_WORDS] = [
    ("gate", "float", true, true),
    ("up", "float", true, true),
    ("out", "float", true, false),
    ("limit", "float", false, false),
    ("inter", "uint32_t", false, false),
];

/// rope_rotate_rows_bf16's declaration in order.
pub const ROPE_ROWS_PARAMS: [(&str, &str, bool, bool); ROPE_ROWS_ARG_WORDS] = [
    ("src", "__nv_bfloat16", true, true),
    ("cos", "float", true, true),
    ("sin", "float", true, true),
    ("out", "__nv_bfloat16", true, false),
    ("rows", "uint32_t", false, false),
    ("head_width", "uint32_t", false, false),
    ("rope_head_dim", "uint32_t", false, false),
    ("sin_sign", "int32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// mhc_ffn_epilogue's declaration in order.
pub const FFN_EPILOGUE_PARAMS: [(&str, &str, bool, bool); FFN_EPILOGUE_ARG_WORDS] = [
    ("stream", "float", true, true),
    ("moe_delta", "void", true, true),
    ("shared_row", "void", true, true),
    ("mix_table", "float", true, true),
    ("stream_out", "float", true, false),
    ("dim", "uint32_t", false, false),
    ("hc_mult", "uint32_t", false, false),
    ("rows", "uint32_t", false, false),
    ("shared_present", "uint32_t", false, false),
    ("shared_bf16", "uint32_t", false, false),
    ("native_bf16", "uint32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// Parses the entry kernel's declaration out of the pinned source.
#[must_use]
pub fn entry_declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, ENTRY_KERNEL)
}

/// Parses the FFN seam kernel's declaration.
#[must_use]
pub fn ffn_declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, FFN_KERNEL)
}

/// Parses the prologue kernel's declaration.
#[must_use]
pub fn prologue_declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, PROLOGUE_KERNEL)
}

fn shape_matches(decl: &KernelDecl, kernel: &str, params: &[(&str, &str, bool, bool)]) -> bool {
    decl.name == kernel
        && decl.extern_c
        && decl.global
        && decl.params.len() == params.len()
        && decl.params.iter().zip(params).all(|(param, pin)| {
            param.name == pin.0
                && param.base == pin.1
                && param.pointer == pin.2
                && param.const_pointee == pin.3
                && (!param.pointer || param.restrict)
        })
}

/// Whether the entry declaration has ENTRY_PARAMS' shape and order.
#[must_use]
pub fn entry_declaration_matches(decl: &KernelDecl) -> bool {
    shape_matches(decl, ENTRY_KERNEL, &ENTRY_PARAMS)
}

/// Whether the FFN seam declaration has FFN_PARAMS' shape and order.
#[must_use]
pub fn ffn_declaration_matches(decl: &KernelDecl) -> bool {
    shape_matches(decl, FFN_KERNEL, &FFN_PARAMS)
}

/// Parses the BF16 prologue variant's declaration out of the pinned source.
#[must_use]
pub fn prologue_bf16_declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, PROLOGUE_BF16_KERNEL)
}

/// Whether the BF16 prologue declaration has PROLOGUE_BF16_PARAMS' shape and order.
#[must_use]
pub fn prologue_bf16_declaration_matches(decl: &KernelDecl) -> bool {
    shape_matches(decl, PROLOGUE_BF16_KERNEL, &PROLOGUE_BF16_PARAMS)
}

/// Whether the prologue declaration has PROLOGUE_PARAMS' shape and order.
#[must_use]
pub fn prologue_declaration_matches(decl: &KernelDecl) -> bool {
    shape_matches(decl, PROLOGUE_KERNEL, &PROLOGUE_PARAMS)
}

/// Parses the attention residual kernel's declaration out of the pinned source.
#[must_use]
pub fn attn_residual_declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, ATTN_RESIDUAL_KERNEL)
}

/// Parses the FFN epilogue kernel's declaration.
#[must_use]
pub fn ffn_epilogue_declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, FFN_EPILOGUE_KERNEL)
}

/// Whether the attention residual declaration has ATTN_RESIDUAL_PARAMS' shape and order.
#[must_use]
pub fn attn_residual_declaration_matches(decl: &KernelDecl) -> bool {
    shape_matches(decl, ATTN_RESIDUAL_KERNEL, &ATTN_RESIDUAL_PARAMS)
}

/// Whether the FFN epilogue declaration has FFN_EPILOGUE_PARAMS' shape and order.
#[must_use]
pub fn ffn_epilogue_declaration_matches(decl: &KernelDecl) -> bool {
    shape_matches(decl, FFN_EPILOGUE_KERNEL, &FFN_EPILOGUE_PARAMS)
}

/// Parses the shared-expert SwiGLU's declaration out of the pinned source.
#[must_use]
pub fn shared_swiglu_declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, SHARED_SWIGLU_KERNEL)
}

/// Whether the SwiGLU declaration has SHARED_SWIGLU_PARAMS' shape and order.
#[must_use]
pub fn shared_swiglu_declaration_matches(decl: &KernelDecl) -> bool {
    shape_matches(decl, SHARED_SWIGLU_KERNEL, &SHARED_SWIGLU_PARAMS)
}

/// Parses the rope rows kernel's declaration out of the pinned source.
#[must_use]
pub fn rope_rows_declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, ROPE_ROWS_KERNEL)
}

/// Whether the rope rows declaration has ROPE_ROWS_PARAMS' shape and order.
#[must_use]
pub fn rope_rows_declaration_matches(decl: &KernelDecl) -> bool {
    shape_matches(decl, ROPE_ROWS_KERNEL, &ROPE_ROWS_PARAMS)
}

/// One launch's arguments for mhc_entry_bf16, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntryParams {
    /// [hc * dim] f32: the hc stream the layer reads.
    pub residual: Deviceptr,
    /// [hc] f32: the previous layer's FFN pre mix, or the identity seed.
    pub attn_pre_mix: Deviceptr,
    /// [rows][hc * dim] f32: the released attention predictor.
    pub fn_w: Deviceptr,
    /// [rows] f32.
    pub hc_base: Deviceptr,
    /// [3] f32: pre, post and comb scales.
    pub hc_scale: Deviceptr,
    /// [dim] f32 or BF16 attn_norm weight, selected by norm_bf16.
    pub norm_w: Deviceptr,
    /// [rows] f32 out: the pre, post and comb coefficient tables.
    pub mix_table: Deviceptr,
    /// [hc * dim] f32 out: the entry's read of the stream.
    pub hc_rows: Deviceptr,
    /// [dim] f32 out: the normalized block input.
    pub normed: Deviceptr,
    /// Whether norm_w is BF16 (1) or f32 (0).
    pub norm_bf16: u32,
    /// Whether the released BF16 boundaries are applied (--native-rounding mhc).
    pub native_bf16: u32,
    /// The config's Sinkhorn iteration count.
    pub sinkhorn_iterations: u32,
    /// The config's RMS epsilon, as f32.
    pub norm_eps: f32,
    /// The config's mHC epsilon, as f32.
    pub hc_eps: f32,
    /// The trace ring's device address, or 0 for the trace-disabled cubin.
    pub trace_ring: u64,
    /// The layer id the trace event would record.
    pub layer_id: u32,
    /// The step sequence the trace event would record.
    pub step_seq: u64,
}

impl EntryParams {
    /// The argument array a launch passes, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; ENTRY_ARG_WORDS] {
        [
            self.residual.0,
            self.attn_pre_mix.0,
            self.fn_w.0,
            self.hc_base.0,
            self.hc_scale.0,
            self.norm_w.0,
            self.mix_table.0,
            self.hc_rows.0,
            self.normed.0,
            u64::from(RELEASED_DIM),
            u64::from(RELEASED_HC_MULT),
            u64::from(RELEASED_ROWS),
            u64::from(self.norm_bf16),
            u64::from(self.native_bf16),
            u64::from(self.sinkhorn_iterations),
            u64::from(self.norm_eps.to_bits()),
            u64::from(self.hc_eps.to_bits()),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// One launch's arguments for mhc_ffn_seams, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FfnSeamsParams {
    /// [hc * dim] f32: the post-attention hc stream.
    pub stream: Deviceptr,
    /// [hc] f32: the entry's pre mix, this block's hc_pre input.
    pub attn_pre_mix: Deviceptr,
    /// [rows][hc * dim] f32: the released FFN predictor.
    pub fn_w: Deviceptr,
    /// [rows] f32.
    pub hc_base: Deviceptr,
    /// [3] f32.
    pub hc_scale: Deviceptr,
    /// [dim] f32 or BF16 ffn_norm weight.
    pub norm_w: Deviceptr,
    /// [rows] f32 out: the FFN coefficient tables.
    pub mix_table: Deviceptr,
    /// [hc] f32 out: the FFN pre mix (the next layer's input).
    pub pre_mix: Deviceptr,
    /// [dim] f32 out: the normalized FFN block input (the seam row).
    pub seam_row: Deviceptr,
    /// [dim] u8 out: the FFN activation E4M3 row.
    pub act_codes: Deviceptr,
    /// [dim / 32] u8 out: its ue8m0 block scales.
    pub act_scales: Deviceptr,
    /// [dim] bf16 out, or 0: the FFN block's RMS-normed input narrowed to BF16 with RNE, which is
    /// the router's input (docs/archive/build/moe-stage-contract.md H3b). It is the norm's *output*, not the
    /// pre-norm hc_pre collapse; the kernel writes it in the norm loop beside seam_row. It writes
    /// nothing for a null pointer, so a launcher passes 0 until the layer's buffers carve the span.
    /// The host arm's `pre_ffn_hidden` is this f32 row and the router narrows it at the launch.
    pub x_bf16: Deviceptr,
    /// Whether norm_w is BF16 (1) or f32 (0).
    pub norm_bf16: u32,
    /// Whether the released BF16 boundaries are applied.
    pub native_bf16: u32,
    /// The config's Sinkhorn iteration count.
    pub sinkhorn_iterations: u32,
    /// The config's RMS epsilon, as f32.
    pub norm_eps: f32,
    /// The config's mHC epsilon, as f32.
    pub hc_eps: f32,
    /// The trace ring's device address, or 0.
    pub trace_ring: u64,
    /// The layer id the trace event would record.
    pub layer_id: u32,
    /// The step sequence the trace event would record.
    pub step_seq: u64,
}

impl FfnSeamsParams {
    /// The argument array a launch passes, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; FFN_ARG_WORDS] {
        [
            self.stream.0,
            self.attn_pre_mix.0,
            self.fn_w.0,
            self.hc_base.0,
            self.hc_scale.0,
            self.norm_w.0,
            self.mix_table.0,
            self.pre_mix.0,
            self.seam_row.0,
            self.act_codes.0,
            self.act_scales.0,
            self.x_bf16.0,
            u64::from(RELEASED_DIM),
            u64::from(RELEASED_HC_MULT),
            u64::from(RELEASED_ROWS),
            u64::from(self.norm_bf16),
            u64::from(self.native_bf16),
            u64::from(self.sinkhorn_iterations),
            u64::from(self.norm_eps.to_bits()),
            u64::from(self.hc_eps.to_bits()),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// One launch's arguments for w8a8_prologue_e4m3, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrologueParams {
    /// [rows][cols] f32: the activation row to quantise.
    pub normed: Deviceptr,
    /// [rows][cols] u8 out.
    pub codes: Deviceptr,
    /// [rows][cols / 32] u8 out.
    pub scales: Deviceptr,
    /// The row width (5,120 in the released configuration).
    pub cols: u32,
    /// The token count.
    pub rows: u32,
    /// The trace ring's device address, or 0.
    pub trace_ring: u64,
    /// The layer id the trace event would record.
    pub layer_id: u32,
    /// The step sequence the trace event would record.
    pub step_seq: u64,
}

impl PrologueParams {
    /// The argument array a launch passes, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; PROLOGUE_ARG_WORDS] {
        [
            self.normed.0,
            self.codes.0,
            self.scales.0,
            u64::from(self.cols),
            u64::from(self.rows),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// The prologue's grid: one thread per 32-value MX block.
///
/// # Errors
/// A zero row or column count, or a cols that is not a whole number of MX blocks, is refused
/// rather than launched as a zero or partial grid.
pub fn prologue_grid(cols: u32, rows: u32) -> Result<(u32, u32, u32), String> {
    if cols == 0 || rows == 0 {
        return Err("the activation quantiser needs at least one row and one column".to_owned());
    }
    if !cols.is_multiple_of(QUANT_BLOCK) {
        return Err(format!(
            "the activation quantiser scales 32-value blocks, so cols is a multiple of 32; got {cols}"
        ));
    }
    let blocks = rows.checked_mul(cols / QUANT_BLOCK).ok_or_else(|| {
        format!(
            "the activation quantiser's block count overflows for {rows} rows of {cols} columns"
        )
    })?;
    Ok((blocks.div_ceil(BLOCK.0), 1, 1))
}

/// Loads the five kernels from one cubin, in (entry, ffn_seams, prologue, attn_residual,
/// ffn_epilogue) order.
///
/// # Errors
/// The driver's error when the cubin is missing one of the symbols.
#[allow(clippy::type_complexity)]
pub fn load(
    cuda: &Cuda,
    cubin: &[u8],
) -> Result<
    (
        Module,
        Kernel,
        Module,
        Kernel,
        Module,
        Kernel,
        Module,
        Kernel,
        Module,
        Kernel,
        Module,
        Kernel,
        Module,
        Kernel,
        Module,
        Kernel,
    ),
    String,
> {
    cuda.load_layer_seams(cubin)
}

/// Refuses a seam launch whose geometry is not the released one the kernel is compiled for.
///
/// # Errors
/// A (dim, hc_mult, rows) other than (5,120, 4, 24).
pub fn check_seam_geometry(dim: u32, hc_mult: u32, rows: u32) -> Result<(), String> {
    if dim != RELEASED_DIM || hc_mult != RELEASED_HC_MULT || rows != RELEASED_ROWS {
        return Err(format!(
            "the layer seam kernels are compiled for the released geometry (dim {RELEASED_DIM}, \
             hc_mult {RELEASED_HC_MULT}, rows {RELEASED_ROWS}); got ({dim}, {hc_mult}, {rows})"
        ));
    }
    Ok(())
}

/// Launches mhc_entry_bf16 once for the token.
///
/// # Errors
/// The driver's launch error.
pub fn launch_entry(cuda: &Cuda, kernel: Kernel, params: &EntryParams) -> Result<(), String> {
    cuda.launch(kernel, SEAM_GRID, BLOCK, &params.words())
}

/// Launches mhc_ffn_seams once for the token.
///
/// # Errors
/// The driver's launch error.
pub fn launch_ffn_seams(
    cuda: &Cuda,
    kernel: Kernel,
    params: &FfnSeamsParams,
) -> Result<(), String> {
    cuda.launch(kernel, SEAM_GRID, BLOCK, &params.words())
}

/// Launches w8a8_prologue_e4m3 over a whole activation row set.
///
/// # Errors
/// A zero or non-block-aligned geometry, or the driver's launch error.
pub fn launch_prologue(cuda: &Cuda, kernel: Kernel, params: &PrologueParams) -> Result<(), String> {
    let grid = prologue_grid(params.cols, params.rows)?;
    cuda.launch(kernel, grid, BLOCK, &params.words())
}

/// One launch's arguments for w8a8_prologue_bf16_e4m3, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrologueBf16Params {
    /// [rows][cols] BF16: the activation row to quantise.
    pub normed: Deviceptr,
    /// [rows][cols] u8 out.
    pub codes: Deviceptr,
    /// [rows][cols / 32] u8 out.
    pub scales: Deviceptr,
    /// The row width (1,280 for the q_norm row, 5,120 for the FFN one).
    pub cols: u32,
    /// The token count.
    pub rows: u32,
    /// The trace ring's device address, or 0.
    pub trace_ring: u64,
    /// The layer id the trace event would record.
    pub layer_id: u32,
    /// The step sequence the trace event would record.
    pub step_seq: u64,
}

impl PrologueBf16Params {
    /// The argument array a launch passes, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; PROLOGUE_BF16_ARG_WORDS] {
        [
            self.normed.0,
            self.codes.0,
            self.scales.0,
            u64::from(self.cols),
            u64::from(self.rows),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// Launches w8a8_prologue_bf16_e4m3 over one bf16 row.
///
/// # Errors
/// A zero or non-MX-multiple geometry, or the driver's launch error.
pub fn launch_prologue_bf16(
    cuda: &Cuda,
    kernel: Kernel,
    params: &PrologueBf16Params,
) -> Result<(), String> {
    let grid = prologue_grid(params.cols, params.rows)?;
    cuda.launch(kernel, grid, BLOCK, &params.words())
}

/// One launch's arguments for mhc_attn_residual, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnResidualParams {
    /// [hc * dim] f32: the hc stream as the entry consumed it (its `hc_rows` copy).
    pub hc_rows: Deviceptr,
    /// [rows] f32: the entry's coefficient table (pre, post, comb); hc_post reads post and comb.
    pub mix_table: Deviceptr,
    /// [dim] BF16: the all-reduced attention delta.
    pub reduced_bf16: Deviceptr,
    /// [hc * dim] f32 out: the updated hc stream (`residual.stream`).
    pub stream_out: Deviceptr,
    /// Whether the released BF16 boundary is applied (--native-rounding mhc).
    pub native_bf16: u32,
    /// The trace ring's device address, or 0 for the trace-disabled launch.
    pub trace_ring: u64,
    /// The layer id the trace event would record.
    pub layer_id: u32,
    /// The step sequence the trace event would record.
    pub step_seq: u64,
}

impl AttnResidualParams {
    /// The argument array a launch passes, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; ATTN_RESIDUAL_ARG_WORDS] {
        [
            self.hc_rows.0,
            self.mix_table.0,
            self.reduced_bf16.0,
            self.stream_out.0,
            u64::from(RELEASED_DIM),
            u64::from(RELEASED_HC_MULT),
            u64::from(RELEASED_ROWS),
            u64::from(self.native_bf16),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// One launch's arguments for mhc_ffn_epilogue, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FfnEpilogueParams {
    /// [hc * dim] f32: the post-attention hc stream (read) and its in-place update (written by
    /// `stream_out`; the same span in the layer loop).
    pub stream: Deviceptr,
    /// [dim] BF16: the routed MoE delta after its own reduce.
    pub moe_delta: Deviceptr,
    /// [dim] f32 or BF16: the shared expert's row, selected by `shared_bf16`.
    pub shared_row: Deviceptr,
    /// [rows] f32: the FFN coefficient table; hc_post reads post and comb.
    pub mix_table: Deviceptr,
    /// [hc * dim] f32 out: the next layer's hc stream.
    pub stream_out: Deviceptr,
    /// Whether the arm has a shared-expert row at all (1) or not (0). The port's weight set has no
    /// shared-expert weight yet, so the layer loop passes 0 and the FFN output is the widened routed
    /// delta alone; the reference's shared add is then absent by construction, not by accident.
    pub shared_present: u32,
    /// Whether the shared row at `shared_row` is BF16-stored (1) or f32 (0). 1 is what the layer
    /// loop passes: it is the coded shared-expert GEMV's BF16 store (the M3 store, TP4's
    /// `--native-rounding shared` arm writes the same bytes) and the epilogue widens it before its
    /// f32 add. 0 exists for an f32 producer, of which there is none on device today.
    pub shared_bf16: u32,
    /// Whether the released BF16 boundary is applied (--native-rounding mhc).
    pub native_bf16: u32,
    /// The trace ring's device address, or 0 for the trace-disabled launch.
    pub trace_ring: u64,
    /// The layer id the trace event would record.
    pub layer_id: u32,
    /// The step sequence the trace event would record.
    pub step_seq: u64,
}

impl FfnEpilogueParams {
    /// The argument array a launch passes, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; FFN_EPILOGUE_ARG_WORDS] {
        [
            self.stream.0,
            self.moe_delta.0,
            self.shared_row.0,
            self.mix_table.0,
            self.stream_out.0,
            u64::from(RELEASED_DIM),
            u64::from(RELEASED_HC_MULT),
            u64::from(RELEASED_ROWS),
            u64::from(self.shared_present),
            u64::from(self.shared_bf16),
            u64::from(self.native_bf16),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// Launches mhc_attn_residual once for the token.
///
/// # Errors
/// The driver's launch error.
pub fn launch_attn_residual(
    cuda: &Cuda,
    kernel: Kernel,
    params: &AttnResidualParams,
) -> Result<(), String> {
    cuda.launch(kernel, SEAM_GRID, BLOCK, &params.words())
}

/// One launch's arguments for shared_swiglu_f32, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedSwigluParams {
    /// [inter] f32: the gate projection's row.
    pub gate: Deviceptr,
    /// [inter] f32: the up projection's row.
    pub up: Deviceptr,
    /// [inter] f32 out: silu(gate) * up.
    pub out: Deviceptr,
    /// The checkpoint's clamp limit; 0 disables it.
    pub limit: f32,
    /// The row width; the launch covers it in 256-thread blocks.
    pub inter: u32,
}

impl SharedSwigluParams {
    /// The argument array a launch passes, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; SHARED_SWIGLU_ARG_WORDS] {
        [
            self.gate.0,
            self.up.0,
            self.out.0,
            u64::from(self.limit.to_bits()),
            u64::from(self.inter),
        ]
    }
}

/// The blocks a shared_swiglu_f32 launch needs for one row.
///
/// # Errors
/// A row that is not at least one thread wide.
pub fn shared_swiglu_grid(inter: u32) -> Result<(u32, u32, u32), String> {
    if inter == 0 {
        return Err("shared_swiglu_f32: a row of 0 values has no block".to_string());
    }
    let blocks = inter.div_ceil(BLOCK.0);
    Ok((blocks, 1, 1))
}

/// Launches shared_swiglu_f32 over an inter-wide row.
///
/// # Errors
/// An empty row or the driver's launch error.
pub fn launch_shared_swiglu(
    cuda: &Cuda,
    kernel: Kernel,
    params: &SharedSwigluParams,
) -> Result<(), String> {
    let grid = shared_swiglu_grid(params.inter)?;
    cuda.launch(kernel, grid, BLOCK, &params.words())
}

/// One launch's arguments for rope_rotate_rows_bf16, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeRowsParams {
    /// [rows][head_width] bf16: the head rows to rotate (the attention body's out).
    pub src: Deviceptr,
    /// [rope_head_dim / 2] f32: the position's cos table (already offset to the position).
    pub cos: Deviceptr,
    /// [rope_head_dim / 2] f32: the position's sin table (already offset to the position).
    pub sin: Deviceptr,
    /// [rows][head_width] bf16 out; may alias `src` (each pair reads before it writes).
    pub out: Deviceptr,
    /// The head rows: the launch's grid.x, and the wq_b output's head count for this rank.
    pub rows: u32,
    /// The row width, 512 for the MLA latent.
    pub head_width: u32,
    /// The rotated tail width, 64 in the released config.
    pub rope_head_dim: u32,
    /// 1: rope_rotate (forward); -1: rope_rotate_inverse (the attention output's rotation).
    pub sin_sign: i32,
    /// The trace ring's device address, or 0.
    pub trace_ring: u64,
    /// The layer id the trace event would record.
    pub layer_id: u32,
    /// The step sequence the trace event would record.
    pub step_seq: u64,
}

impl RopeRowsParams {
    /// The argument array a launch passes, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; ROPE_ROWS_ARG_WORDS] {
        [
            self.src.0,
            self.cos.0,
            self.sin.0,
            self.out.0,
            u64::from(self.rows),
            u64::from(self.head_width),
            u64::from(self.rope_head_dim),
            self.sin_sign as i64 as u64,
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }

    /// Refuses a geometry the kernel would read out of bounds on, or a sign it does not implement.
    ///
    /// # Errors
    /// Zero rows or width, a tail that is zero, odd, or wider than the row, or a sign other than
    /// +1/-1.
    pub fn validate(&self) -> Result<(), String> {
        if self.rows == 0 {
            return Err("rope rows: a grid of 0 rows has no head to rotate".to_string());
        }
        if self.head_width == 0 {
            return Err("rope rows: head_width is zero, so the row has no values".to_string());
        }
        if self.rope_head_dim == 0 || self.rope_head_dim > self.head_width {
            return Err(format!(
                "rope rows: a {}-wide tail does not fit a {}-wide row",
                self.rope_head_dim, self.head_width
            ));
        }
        if !self.rope_head_dim.is_multiple_of(2) {
            return Err(format!(
                "rope rows: an odd {}-wide tail is not a pair list",
                self.rope_head_dim
            ));
        }
        if self.sin_sign != 1 && self.sin_sign != -1 {
            return Err(format!(
                "rope rows: sin_sign {} is neither 1 (forward) nor -1 (inverse)",
                self.sin_sign
            ));
        }
        Ok(())
    }
}

/// The grid a rope rows launch needs: one block per head row.
///
/// # Errors
/// A zero row count.
pub fn rope_rows_grid(rows: u32) -> Result<(u32, u32, u32), String> {
    if rows == 0 {
        return Err("rope rows: a grid of 0 rows has no head to rotate".to_string());
    }
    Ok((rows, 1, 1))
}

/// Launches rope_rotate_rows_bf16 over `rows` head rows.
///
/// # Errors
/// A refused geometry or the driver's launch error.
pub fn launch_rope_rows(
    cuda: &Cuda,
    kernel: Kernel,
    params: &RopeRowsParams,
) -> Result<(), String> {
    params.validate()?;
    let grid = rope_rows_grid(params.rows)?;
    cuda.launch(kernel, grid, BLOCK, &params.words())
}

/// Launches mhc_ffn_epilogue once for the token.
///
/// # Errors
/// The driver's launch error.
pub fn launch_ffn_epilogue(
    cuda: &Cuda,
    kernel: Kernel,
    params: &FfnEpilogueParams,
) -> Result<(), String> {
    cuda.launch(kernel, SEAM_GRID, BLOCK, &params.words())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_decl::Consts;

    #[test]
    fn the_entry_declaration_is_the_pinned_one() {
        let decl = entry_declaration();
        assert!(
            entry_declaration_matches(&decl),
            "the entry declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), ENTRY_ARG_WORDS);
        assert_eq!(ENTRY_PARAMS.len(), ENTRY_ARG_WORDS);
    }

    #[test]
    fn the_ffn_seam_declaration_is_the_pinned_one() {
        let decl = ffn_declaration();
        assert!(
            ffn_declaration_matches(&decl),
            "the FFN seam declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), FFN_ARG_WORDS);
        assert_eq!(FFN_PARAMS.len(), FFN_ARG_WORDS);
    }

    #[test]
    fn the_prologue_declaration_is_the_pinned_one() {
        let decl = prologue_declaration();
        assert!(
            prologue_declaration_matches(&decl),
            "the prologue declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), PROLOGUE_ARG_WORDS);
        assert_eq!(PROLOGUE_PARAMS.len(), PROLOGUE_ARG_WORDS);
    }

    #[test]
    fn the_attention_residual_declaration_is_the_pinned_one() {
        let decl = attn_residual_declaration();
        assert!(
            attn_residual_declaration_matches(&decl),
            "the attention residual declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), ATTN_RESIDUAL_ARG_WORDS);
        assert_eq!(ATTN_RESIDUAL_PARAMS.len(), ATTN_RESIDUAL_ARG_WORDS);
    }

    #[test]
    fn the_ffn_epilogue_declaration_is_the_pinned_one() {
        let decl = ffn_epilogue_declaration();
        assert!(
            ffn_epilogue_declaration_matches(&decl),
            "the FFN epilogue declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), FFN_EPILOGUE_ARG_WORDS);
        assert_eq!(FFN_EPILOGUE_PARAMS.len(), FFN_EPILOGUE_ARG_WORDS);
    }

    #[test]
    fn the_declaration_pins_are_not_vacuous() {
        let mutated = SOURCE.replace(
            "const float *__restrict__ hc_scale",
            "const float *hc_scale",
        );
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        assert!(
            !entry_declaration_matches(&c_decl::kernel_declaration(&mutated, ENTRY_KERNEL)),
            "the entry qualifier pin is vacuous"
        );
        let mutated = SOURCE.replace("uint8_t *__restrict__ act_codes", "uint8_t *act_codes");
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        assert!(
            !ffn_declaration_matches(&c_decl::kernel_declaration(&mutated, FFN_KERNEL)),
            "the FFN qualifier pin is vacuous"
        );
        let mutated = SOURCE.replace(
            "const __nv_bfloat16 *__restrict__ normed,  // [rows][cols] the bf16 activation row",
            "const __nv_bfloat16 *normed,  // [rows][cols] the bf16 activation row",
        );
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        assert!(
            !prologue_bf16_declaration_matches(&c_decl::kernel_declaration(
                &mutated,
                PROLOGUE_BF16_KERNEL
            )),
            "the bf16 prologue qualifier pin is vacuous"
        );
        let mutated = SOURCE.replace(
            "const float *__restrict__ up,     // [inter] the up projection's f32 row",
            "const float *up,     // [inter] the up projection's f32 row",
        );
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        assert!(
            !shared_swiglu_declaration_matches(&c_decl::kernel_declaration(
                &mutated,
                SHARED_SWIGLU_KERNEL
            )),
            "the SwiGLU qualifier pin is vacuous"
        );
        let mutated = SOURCE.replace("const float *__restrict__ normed", "const float *normed");
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        assert!(
            !prologue_declaration_matches(&c_decl::kernel_declaration(&mutated, PROLOGUE_KERNEL)),
            "the prologue qualifier pin is vacuous"
        );
        let mutated = SOURCE.replace(
            "const void *__restrict__ reduced_bf16",
            "const void *reduced_bf16",
        );
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        assert!(
            !attn_residual_declaration_matches(&c_decl::kernel_declaration(
                &mutated,
                ATTN_RESIDUAL_KERNEL
            )),
            "the attention residual qualifier pin is vacuous"
        );
        let mutated = SOURCE.replace(
            "const float *__restrict__ cos,            // [rope_head_dim / 2] this position's cos",
            "const float *cos,            // [rope_head_dim / 2] this position's cos",
        );
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        assert!(
            !rope_rows_declaration_matches(&c_decl::kernel_declaration(&mutated, ROPE_ROWS_KERNEL)),
            "the rope rows qualifier pin is vacuous"
        );
        let mutated = SOURCE.replace(
            "const void *__restrict__ shared_row",
            "const void *shared_row",
        );
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        assert!(
            !ffn_epilogue_declaration_matches(&c_decl::kernel_declaration(
                &mutated,
                FFN_EPILOGUE_KERNEL
            )),
            "the FFN epilogue qualifier pin is vacuous"
        );
    }

    fn pointers() -> [Deviceptr; 24] {
        core::array::from_fn(|index| Deviceptr(0x1000 * (index as u64 + 1)))
    }

    #[test]
    fn the_bf16_prologue_declaration_is_the_pinned_one() {
        let decl = prologue_bf16_declaration();
        assert!(
            prologue_bf16_declaration_matches(&decl),
            "the bf16 prologue declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), PROLOGUE_BF16_ARG_WORDS);
        assert_eq!(PROLOGUE_BF16_PARAMS.len(), PROLOGUE_BF16_ARG_WORDS);
    }

    #[test]
    fn the_bf16_prologue_argument_array_is_the_declaration_order() {
        let p = pointers();
        let params = PrologueBf16Params {
            normed: p[0],
            codes: p[1],
            scales: p[2],
            cols: 1280,
            rows: 1,
            trace_ring: 0,
            layer_id: 5,
            step_seq: 11,
        };
        let words = params.words();
        assert_eq!(words.len(), PROLOGUE_BF16_ARG_WORDS);
        assert_eq!(&words[..3], &[0x1000, 0x2000, 0x3000]);
        assert_eq!(words[3], 1280);
        assert_eq!(words[4], 1);
        assert_eq!(words[5..], [0, 5, 11]);
    }

    #[test]
    fn the_shared_swiglu_declaration_is_the_pinned_one() {
        let decl = shared_swiglu_declaration();
        assert!(
            shared_swiglu_declaration_matches(&decl),
            "the shared SwiGLU declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), SHARED_SWIGLU_ARG_WORDS);
        assert_eq!(SHARED_SWIGLU_PARAMS.len(), SHARED_SWIGLU_ARG_WORDS);
    }

    #[test]
    fn the_shared_swiglu_argument_array_is_the_declaration_order() {
        let p = pointers();
        let params = SharedSwigluParams {
            gate: p[0],
            up: p[1],
            out: p[2],
            limit: 30.0f32,
            inter: 2304,
        };
        let words = params.words();
        assert_eq!(words.len(), SHARED_SWIGLU_ARG_WORDS);
        assert_eq!(&words[..3], &[0x1000, 0x2000, 0x3000]);
        assert_eq!(words[3], u64::from(30.0f32.to_bits()));
        assert_eq!(words[4], 2304);
        assert_eq!(shared_swiglu_grid(2304).unwrap(), (9, 1, 1));
        assert_eq!(shared_swiglu_grid(256).unwrap(), (1, 1, 1));
        assert!(shared_swiglu_grid(0).is_err());
    }

    #[test]
    fn the_entry_argument_array_is_the_declaration_order() {
        let p = pointers();
        let params = EntryParams {
            residual: p[0],
            attn_pre_mix: p[1],
            fn_w: p[2],
            hc_base: p[3],
            hc_scale: p[4],
            norm_w: p[5],
            mix_table: p[6],
            hc_rows: p[8],
            normed: p[9],
            norm_bf16: 0,
            native_bf16: 1,
            sinkhorn_iterations: 20,
            norm_eps: 1e-20f32,
            hc_eps: 1e-6f32,
            trace_ring: 0,
            layer_id: 3,
            step_seq: 9,
        };
        let words = params.words();
        assert_eq!(words.len(), ENTRY_ARG_WORDS);
        assert_eq!(
            &words[..9],
            &[
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 0x9000, 0xa000
            ]
        );
        assert_eq!(&words[9..15], &[5120, 4, 24, 0, 1, 20]);
        assert_eq!(words[15], u64::from(1e-20f32.to_bits()));
        assert_eq!(words[16], u64::from(1e-6f32.to_bits()));
        assert_eq!(words[17..], [0, 3, 9]);
    }

    #[test]
    fn the_ffn_argument_array_is_the_declaration_order() {
        let p = pointers();
        let params = FfnSeamsParams {
            stream: p[0],
            attn_pre_mix: p[1],
            fn_w: p[2],
            hc_base: p[3],
            hc_scale: p[4],
            norm_w: p[5],
            mix_table: p[6],
            pre_mix: p[7],
            seam_row: p[8],
            act_codes: p[9],
            act_scales: p[10],
            x_bf16: p[11],
            norm_bf16: 1,
            native_bf16: 1,
            sinkhorn_iterations: 20,
            norm_eps: 1e-20f32,
            hc_eps: 1e-6f32,
            trace_ring: 0,
            layer_id: 2,
            step_seq: 4,
        };
        let words = params.words();
        assert_eq!(words.len(), FFN_ARG_WORDS);
        assert_eq!(words[0], 0x1000);
        assert_eq!(words[8], 0x9000);
        assert_eq!(words[10], 0xb000);
        assert_eq!(words[11], 0xc000);
        assert_eq!(&words[12..18], &[5120, 4, 24, 1, 1, 20]);
        assert_eq!(words[18], u64::from(1e-20f32.to_bits()));
        assert_eq!(words[19], u64::from(1e-6f32.to_bits()));
        assert_eq!(words[20..], [0, 2, 4]);
    }

    #[test]
    fn the_prologue_argument_array_is_the_declaration_order() {
        let p = pointers();
        let params = PrologueParams {
            normed: p[0],
            codes: p[1],
            scales: p[2],
            cols: 5120,
            rows: 1,
            trace_ring: 0,
            layer_id: 5,
            step_seq: 6,
        };
        let words = params.words();
        assert_eq!(words, [0x1000, 0x2000, 0x3000, 5120, 1, 0, 5, 6]);
    }

    #[test]
    fn the_attention_residual_argument_array_is_the_declaration_order() {
        let p = pointers();
        let params = AttnResidualParams {
            hc_rows: p[0],
            mix_table: p[1],
            reduced_bf16: p[2],
            stream_out: p[3],
            native_bf16: 1,
            trace_ring: 0,
            layer_id: 5,
            step_seq: 6,
        };
        let words = params.words();
        assert_eq!(words.len(), ATTN_RESIDUAL_ARG_WORDS);
        assert_eq!(&words[..4], &[0x1000, 0x2000, 0x3000, 0x4000]);
        assert_eq!(&words[4..8], &[5120, 4, 24, 1]);
        assert_eq!(words[8..], [0, 5, 6]);
    }

    #[test]
    fn the_ffn_epilogue_argument_array_is_the_declaration_order() {
        let p = pointers();
        let params = FfnEpilogueParams {
            stream: p[0],
            moe_delta: p[1],
            shared_row: p[2],
            mix_table: p[3],
            stream_out: p[4],
            shared_present: 0,
            shared_bf16: 1,
            native_bf16: 1,
            trace_ring: 0,
            layer_id: 8,
            step_seq: 9,
        };
        let words = params.words();
        assert_eq!(words.len(), FFN_EPILOGUE_ARG_WORDS);
        assert_eq!(&words[..5], &[0x1000, 0x2000, 0x3000, 0x4000, 0x5000]);
        assert_eq!(&words[5..11], &[5120, 4, 24, 0, 1, 1]);
        assert_eq!(words[11..], [0, 8, 9]);
    }

    #[test]
    fn the_rope_rows_declaration_is_the_pinned_one() {
        let decl = rope_rows_declaration();
        assert!(
            rope_rows_declaration_matches(&decl),
            "the rope rows declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), ROPE_ROWS_ARG_WORDS);
        assert_eq!(ROPE_ROWS_PARAMS.len(), ROPE_ROWS_ARG_WORDS);
    }

    #[test]
    fn the_rope_rows_argument_array_is_the_declaration_order() {
        let p = pointers();
        let params = RopeRowsParams {
            src: p[0],
            cos: p[1],
            sin: p[2],
            out: p[3],
            rows: 16,
            head_width: 512,
            rope_head_dim: 64,
            sin_sign: -1,
            trace_ring: 0,
            layer_id: 7,
            step_seq: 13,
        };
        let words = params.words();
        assert_eq!(words.len(), ROPE_ROWS_ARG_WORDS);
        assert_eq!(&words[..4], &[0x1000, 0x2000, 0x3000, 0x4000]);
        assert_eq!(&words[4..8], &[16, 512, 64, u64::MAX]);
        assert_eq!(words[8..], [0, 7, 13]);
        assert_eq!(rope_rows_grid(16).unwrap(), (16, 1, 1));
        assert!(rope_rows_grid(0).is_err());
        assert!(params.validate().is_ok());
        assert_eq!(params.words()[7], (-1i64) as u64);
    }

    #[test]
    fn the_rope_rows_geometry_refusals_are_not_vacuous() {
        let p = pointers();
        let base = RopeRowsParams {
            src: p[0],
            cos: p[1],
            sin: p[2],
            out: p[3],
            rows: 16,
            head_width: 512,
            rope_head_dim: 64,
            sin_sign: 1,
            trace_ring: 0,
            layer_id: 0,
            step_seq: 0,
        };
        assert!(base.validate().is_ok());
        for (mutate, what) in [
            (RopeRowsParams { rows: 0, ..base }, "zero rows"),
            (
                RopeRowsParams {
                    head_width: 0,
                    ..base
                },
                "zero width",
            ),
            (
                RopeRowsParams {
                    rope_head_dim: 0,
                    ..base
                },
                "zero tail",
            ),
            (
                RopeRowsParams {
                    rope_head_dim: 1,
                    ..base
                },
                "odd tail",
            ),
            (
                RopeRowsParams {
                    rope_head_dim: 513,
                    ..base
                },
                "tail wider than the row",
            ),
            (
                RopeRowsParams {
                    sin_sign: 0,
                    ..base
                },
                "sign 0",
            ),
            (
                RopeRowsParams {
                    sin_sign: 2,
                    ..base
                },
                "sign 2",
            ),
        ] {
            assert!(mutate.validate().is_err(), "{what} must be refused");
        }
    }

    #[test]
    fn the_quantiser_geometry_is_the_block_layout() {
        // One thread per MX block: 160 blocks for a 5,120-wide row fit in one 256-thread block,
        // and two rows need two.
        assert_eq!(prologue_grid(5120, 1).expect("valid"), (1, 1, 1));
        assert!(prologue_grid(5120, 0).is_err());
        assert!(prologue_grid(0, 1).is_err());
        assert!(prologue_grid(5121, 1).is_err());
        assert_eq!(prologue_grid(5120, 2).expect("valid"), (2, 1, 1));
        assert_eq!(prologue_grid(32, 8).expect("valid"), (1, 1, 1));
        assert_eq!(prologue_grid(32, 300).expect("valid"), (2, 1, 1));
    }

    /// The source text every seam kernel must keep around its trace store.
    const TRACE_GUARD: &str = "if (trace_ring != nullptr)";

    /// One kernel's body from \`source\`, from its opening brace to the matching close.
    fn kernel_body_of<'a>(name: &str, source: &'a str) -> Option<&'a str> {
        let needle = format!("__global__ void {name}(");
        let start = source.find(&needle)?;
        let open = start + source[start..].find('{')?;
        let mut depth = 0usize;
        for (offset, character) in source[open..].char_indices() {
            match character {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&source[open..=open + offset]);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// The ways the trace-ring guard can be wrong, empty when the contract holds.
    fn guard_violations(source: &str) -> Vec<&'static str> {
        let mut violations = Vec::new();
        for kernel in [
            ENTRY_KERNEL,
            FFN_KERNEL,
            PROLOGUE_KERNEL,
            PROLOGUE_BF16_KERNEL,
            ATTN_RESIDUAL_KERNEL,
            FFN_EPILOGUE_KERNEL,
            ROPE_ROWS_KERNEL,
        ] {
            let Some(body) = kernel_body_of(kernel, source) else {
                violations.push("a seam kernel lost its body");
                continue;
            };
            if body.matches(TRACE_GUARD).count() != 2 {
                violations.push("a seam kernel's trace store is not behind two null guards");
            }
            if !(body.contains("SPARK_GPU_TRACE_ENTER") && body.contains("SPARK_GPU_TRACE_EXIT")) {
                violations.push("a seam kernel does not record its enter/exit inside the guard");
            }
        }
        violations
    }

    #[test]
    fn every_seam_kernel_guards_its_trace_ring() {
        // A null ring must mean "write nothing": the loop passes no ring in the steady state, and
        // these kernels run once per stage per layer per position. The compiled cubin cannot show
        // the difference (trace is compiled out and the branch folds away), so the pin is the
        // source text the kernels are compiled from.
        assert_eq!(guard_violations(SOURCE), Vec::<&'static str>::new());
        for kernel in [
            ENTRY_KERNEL,
            FFN_KERNEL,
            PROLOGUE_KERNEL,
            PROLOGUE_BF16_KERNEL,
            ATTN_RESIDUAL_KERNEL,
            FFN_EPILOGUE_KERNEL,
            ROPE_ROWS_KERNEL,
        ] {
            let body = kernel_body_of(kernel, SOURCE).expect("every seam kernel has a body");
            assert!(body.contains(TRACE_GUARD), "{kernel} must guard its ring");
        }
        // Mutation: dropping one guard has to be caught, or the pin is vacuous. The mutation is made
        // inside a kernel body, because the header's TRACE paragraph also spells the guard out.
        let body = kernel_body_of(ENTRY_KERNEL, SOURCE).expect("the entry kernel has a body");
        let mutated_body = body.replacen(TRACE_GUARD, "if (true)", 1);
        assert_ne!(
            mutated_body, body,
            "the mutation must change the kernel body"
        );
        let mutated = SOURCE.replacen(body, &mutated_body, 1);
        assert_ne!(mutated, SOURCE, "the mutation must change the source");
        assert!(
            !guard_violations(&mutated).is_empty(),
            "a seam kernel that lost its null guard must be refused"
        );
    }

    #[test]
    fn the_released_geometry_is_the_kernel_header_s() {
        let mut table = Consts::parse(SOURCE);
        table.extend(crate::m4_abi::header_source());
        assert_eq!(table.value("SPARK_M4_HIDDEN"), i128::from(RELEASED_DIM));
        assert_eq!(
            table.value("SPARK_M4_HC_MULT"),
            i128::from(RELEASED_HC_MULT)
        );
        assert_eq!(
            table.value("SPARK_M4_HC_PREDICTOR_ROWS"),
            i128::from(RELEASED_ROWS)
        );
        assert_eq!(table.value("SPARK_M4_QUANT_BLOCK"), i128::from(QUANT_BLOCK));
        assert!(check_seam_geometry(5120, 4, 24).is_ok());
        assert!(check_seam_geometry(5120, 4, 25).is_err());
        assert!(check_seam_geometry(2560, 4, 24).is_err());
    }
}

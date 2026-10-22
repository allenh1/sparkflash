//! The typed launch capabilities of the kv-path kernels (kernels/kv_path_sm121.cu).
//!
//! The argument arrays are the kernels' declaration parameter lists in order, so the only way to
//! fill a word wrongly is to fill a field wrongly, and the fields are named. Every kernel here is one
//! row per launch: the position's single MLA latent row, 512 values wide, normalised by kv_norm and
//! rotated on its last 64, quantised in 32-value blocks, then appended to the layer's cache ring.
//!
//! Span guards, not just geometry guards. Every entry point takes the byte length of the span it will
//! write and refuses a span shorter than the launch needs, naming the requirement. The port passes
//! the [crate::kv_path_ref] span sizes and never a computed pointer arithmetic: an overrun on a
//! device span is a silently corrupted workspace, and a refusal is a stage error the loop can print.
//!
//! Tracing. The pinned contract (KERNEL_CONTRACTS in crates/sparkflash-cli/src/layer_device.rs) does
//! not carry the trace ring/layer/step words for these kernels, and the protocol's 16 scopes do not
//! describe a kv append, so there is no trace argument to pass.
//!
//! Kernels. All four kv-path kernels are here: the fused kv_norm + rope (one bf16 rounding, the host
//! order), rope on a stored row's tail, the whole-row fp8 round-trip, and the cache append. The append's geometry and visibility contract are the frozen
//! DeviceKvCache interface in crates/sparkflash-cli/src/layer_device.rs (win/layer-device @ ee0a0daa):
//! one row is 2 * head_dim bf16 values (k then v), the row for position p lives at slot p % rows, and
//! publication is stream order after the position's window read.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, Module};
use crate::kv_path_ref::{MLA_LATENT, WINDOW_QUANT_BLOCK};

/// Threads per block: one row's tail fits one wave (32 pairs), and the head copy is a strided loop.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);

/// The grid for one row: the kernel loops over the row's values, so one block is the launch.
pub const GRID: (u32, u32, u32) = (1, 1, 1);

/// The released rope tail width in the device port's rows (config.rope_head_dim).
pub const RELEASED_ROPE_HEAD_DIM: usize = 64;

/// One rope launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeRotateParams {
    /// The stored row the port rotates: [head_width] bf16.
    pub row: Deviceptr,
    /// The position's cos table: [rope_head_dim / 2] f32.
    pub cos: Deviceptr,
    /// The position's sin table: [rope_head_dim / 2] f32.
    pub sin: Deviceptr,
    /// The rotated row: [head_width] bf16 (out of place; the stored row is not clobbered).
    pub out: Deviceptr,
    /// The row width, 512 for the MLA latent.
    pub head_width: u32,
    /// The rotated tail width, 64 in the released config.
    pub rope_head_dim: u32,
    /// The position the tables were built for (an ABI word).
    pub position: u32,
}

impl RopeRotateParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; 7] {
        [
            self.row.0,
            self.cos.0,
            self.sin.0,
            self.out.0,
            u64::from(self.head_width),
            u64::from(self.rope_head_dim),
            u64::from(self.position),
        ]
    }

    /// Refuses a geometry the kernel would read out of bounds on.
    ///
    /// # Errors
    /// A zero width, a tail longer than the row or not a multiple of two, or a tail whose tables
    /// would be a different length than the kernel indexes.
    pub fn validate(&self) -> Result<(), String> {
        if self.head_width == 0 {
            return Err("rope: head_width is zero, so the row has no values to rotate".to_owned());
        }
        if self.rope_head_dim == 0
            || self.rope_head_dim > self.head_width
            || !self.rope_head_dim.is_multiple_of(2)
        {
            return Err(format!(
                "rope: a {}-long tail does not fit a {}-wide row (it must be even and no wider than the row)",
                self.rope_head_dim, self.head_width
            ));
        }
        Ok(())
    }

    /// Refuses spans shorter than the launch writes or reads.
    ///
    /// The port has the spans' byte lengths, so the guard is stated in bytes: row and out are
    /// head_width bf16, cos and sin are rope_head_dim / 2 f32.
    ///
    /// # Errors
    /// Names the first span that is too short, with the length the launch needs.
    pub fn check_spans(
        &self,
        row_bytes: usize,
        out_bytes: usize,
        table_bytes: usize,
    ) -> Result<(), String> {
        let need_row = usize::try_from(self.head_width)
            .map_err(|_| "rope: head_width does not fit a usize".to_owned())?
            * 2;
        let need_table = usize::try_from(self.rope_head_dim)
            .map_err(|_| "rope: rope_head_dim does not fit a usize".to_owned())?
            / 2
            * 4;
        if row_bytes < need_row || out_bytes < need_row {
            return Err(format!(
                "rope: the row span is {row_bytes} bytes in and {out_bytes} out, a {}-wide bf16 row is {need_row}",
                self.head_width
            ));
        }
        if table_bytes < need_table {
            return Err(format!(
                "rope: the table span is {table_bytes} bytes, a {}-long tail needs {need_table} bytes of f32 per table",
                self.rope_head_dim
            ));
        }
        Ok(())
    }
}

/// One window-quantiser launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowQuantParams {
    /// The rotated row, already bf16-stored: [cols] bf16.
    pub row: Deviceptr,
    /// The E4M3FN codes: [cols] bytes.
    pub codes: Deviceptr,
    /// The E8M0 scales: [cols / block] bytes.
    pub scales: Deviceptr,
    /// The row width, 512 for the MLA latent.
    pub cols: u32,
    /// The block size: the format's 32 (a different block would not be the released format).
    pub block: u32,
}

impl WindowQuantParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; 5] {
        [
            self.row.0,
            self.codes.0,
            self.scales.0,
            u64::from(self.cols),
            u64::from(self.block),
        ]
    }

    /// Refuses a geometry the kernel would read or write out of bounds on.
    ///
    /// # Errors
    /// A zero width, a block that is not the format's 32, or a width that is not a whole number of
    /// blocks.
    pub fn validate(&self) -> Result<(), String> {
        if self.cols == 0 {
            return Err(
                "window quantiser: cols is zero, so there is no row to quantise".to_owned(),
            );
        }
        if self.block as usize != WINDOW_QUANT_BLOCK {
            return Err(format!(
                "window quantiser: block {} is not the released format's {WINDOW_QUANT_BLOCK}; a different block is a different format, not a tuning choice",
                self.block
            ));
        }
        if !self.cols.is_multiple_of(self.block) {
            return Err(format!(
                "window quantiser: {}-wide row is not a whole number of {}-value blocks",
                self.cols, self.block
            ));
        }
        Ok(())
    }

    /// Refuses spans shorter than the launch writes or reads.
    ///
    /// # Errors
    /// Names the first span that is too short, with the length the launch needs.
    pub fn check_spans(
        &self,
        row_bytes: usize,
        codes_bytes: usize,
        scales_bytes: usize,
    ) -> Result<(), String> {
        let need_row = usize::try_from(self.cols)
            .map_err(|_| "window quantiser: cols does not fit a usize".to_owned())?
            * 2;
        let need_codes = usize::try_from(self.cols)
            .map_err(|_| "window quantiser: cols does not fit a usize".to_owned())?;
        let need_scales = need_codes
            / usize::try_from(self.block)
                .map_err(|_| "window quantiser: block does not fit a usize".to_owned())?;
        if row_bytes < need_row {
            return Err(format!(
                "window quantiser: the row span is {row_bytes} bytes, a {}-wide bf16 row is {need_row}",
                self.cols
            ));
        }
        if codes_bytes < need_codes || scales_bytes < need_scales {
            return Err(format!(
                "window quantiser: the code span is {codes_bytes} bytes and the scale span {scales_bytes}, a {}-wide row needs {need_codes} and {need_scales}",
                self.cols
            ));
        }
        Ok(())
    }
}

/// Loads the rope kernel from its cubin, resolving the pinned symbol by name.
///
/// The symbol has no KernelId: one cubin carries the four kv-path symbols and one baseline entry,
/// and the by-name loader is the repo's existing route for a kernel with its own cubin and no id.
/// The cubin itself is resolved by name through the driver's KV_PATH_CUBIN constant.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain the symbol.
pub fn load_rope_rotate(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_rope_rotate(cubin)
}

/// Loads the window quantiser from its cubin by name.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain the symbol.
pub fn load_window_quant(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_window_quant(cubin)
}

/// Launches the rope rotation for one row.
///
/// # Errors
/// The geometry or span guard's refusal, or the driver's launch error.
pub fn launch_rope_rotate(
    cuda: &Cuda,
    kernel: Kernel,
    params: &RopeRotateParams,
) -> Result<(), String> {
    params.validate()?;
    cuda.launch(kernel, GRID, BLOCK, &params.words())
}

/// The fused kv_norm + rope symbol in the kv-path cubin.
pub const KV_NORM_ROPE_KERNEL: &str = "kv_norm_rope_bf16";

/// The kv-path translation unit the declaration pins are parsed from.
pub const SOURCE: &str = include_str!("../../../kernels/kv_path_sm121.cu");

/// The number of argument words the fused norm+rope launch fills.
pub const KV_NORM_ROPE_ARG_WORDS: usize = 8;

/// The fused declaration's parameters in order: (name, base type, pointer, pointee const).
pub const KV_NORM_ROPE_PARAMS: [(&str, &str, bool, bool); KV_NORM_ROPE_ARG_WORDS] = [
    ("row", "__nv_bfloat16", true, true),
    ("norm", "__nv_bfloat16", true, true),
    ("eps", "float", false, false),
    ("cos", "float", true, true),
    ("sin", "float", true, true),
    ("out", "__nv_bfloat16", true, false),
    ("head_width", "uint32_t", false, false),
    ("rope_head_dim", "uint32_t", false, false),
];

/// Parses the fused norm+rope declaration out of the pinned translation unit.
#[must_use]
pub fn kv_norm_rope_declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, KV_NORM_ROPE_KERNEL)
}

/// Whether a parsed declaration has [KV_NORM_ROPE_PARAMS]' shape and order.
#[must_use]
pub fn kv_norm_rope_declaration_matches(decl: &KernelDecl) -> bool {
    decl.name == KV_NORM_ROPE_KERNEL
        && decl.extern_c
        && decl.global
        && decl.params.len() == KV_NORM_ROPE_ARG_WORDS
        && decl
            .params
            .iter()
            .zip(KV_NORM_ROPE_PARAMS)
            .all(|(param, pin)| {
                param.name == pin.0
                    && param.base == pin.1
                    && param.pointer == pin.2
                    && param.const_pointee == pin.3
                    && (!param.pointer || param.restrict)
            })
}

/// One fused kv_norm + rope launch's arguments, in the kernel declaration's order.
///
/// The host order this reproduces is rms_norm(kv, kv_norm, 512, norm_eps) then rope_rotate on the
/// last rope_head_dim values, with ONE bf16 rounding at the end (tp4_forward.rs:3032-3038). The
/// declared precision deviation is the operand: the device latent is already bf16 where the host
/// keeps f32, so this kernel widens bf16 inputs where the host had f32. Everything after that
/// widening is the host's arithmetic in the host's order; see the kernel file's header and
/// [crate::kv_path_ref::kv_norm_rope_ref]'s twin tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvNormRopeParams {
    /// The bf16 wkv latent row: [head_width] bf16 (the attention projection's output).
    pub row: Deviceptr,
    /// The bf16 kv_norm weight: [head_width] bf16.
    pub norm: Deviceptr,
    /// The config's norm_eps as f32.
    pub eps: f32,
    /// The position's cos table: [rope_head_dim / 2] f32.
    pub cos: Deviceptr,
    /// The position's sin table: [rope_head_dim / 2] f32.
    pub sin: Deviceptr,
    /// The normed, rotated row: [head_width] bf16, written once.
    pub out: Deviceptr,
    /// Values in the row (512 for the MLA latent).
    pub head_width: u32,
    /// The rotated tail width (64 in the released config).
    pub rope_head_dim: u32,
}

impl KvNormRopeParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; KV_NORM_ROPE_ARG_WORDS] {
        [
            self.row.0,
            self.norm.0,
            u64::from(self.eps.to_bits()),
            self.cos.0,
            self.sin.0,
            self.out.0,
            u64::from(self.head_width),
            u64::from(self.rope_head_dim),
        ]
    }

    /// Refuses a geometry the kernel cannot execute.
    ///
    /// # Errors
    /// Names the field that does not fit: a zero row, a tail that is zero, odd or wider than the
    /// row.
    pub fn validate(&self) -> Result<(), String> {
        if self.head_width == 0 {
            return Err("kv norm rope: head_width is zero, so the row has no values".to_owned());
        }
        if self.rope_head_dim == 0
            || self.rope_head_dim > self.head_width
            || !self.rope_head_dim.is_multiple_of(2)
        {
            return Err(format!(
                "kv norm rope: a {}-long tail does not fit a {}-wide row as an even run",
                self.rope_head_dim, self.head_width
            ));
        }
        Ok(())
    }

    /// Refuses a span shorter than the launch reads or writes.
    ///
    /// # Errors
    /// Names the span and the bytes the launch needs.
    pub fn check_spans(
        &self,
        row_bytes: usize,
        norm_bytes: usize,
        cos_bytes: usize,
        sin_bytes: usize,
        out_bytes: usize,
    ) -> Result<(), String> {
        let need_row = usize::try_from(self.head_width)
            .map_err(|error| format!("kv norm rope: head_width: {error}"))?
            * 2;
        let need_table = usize::try_from(self.rope_head_dim / 2)
            .map_err(|error| format!("kv norm rope: rope_head_dim: {error}"))?
            * 4;
        if row_bytes < need_row || norm_bytes < need_row || out_bytes < need_row {
            return Err(format!(
                "kv norm rope: spans row {row_bytes}, norm {norm_bytes}, out {out_bytes} bytes; a {}-wide bf16 row needs {need_row} each",
                self.head_width
            ));
        }
        if cos_bytes < need_table || sin_bytes < need_table {
            return Err(format!(
                "kv norm rope: spans cos {cos_bytes}, sin {sin_bytes} bytes; a {}-long tail needs {need_table} each",
                self.rope_head_dim
            ));
        }
        Ok(())
    }
}

/// Loads the fused kv_norm + rope kernel from the kv-path cubin by name.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain the symbol.
pub fn load_kv_norm_rope(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kv_norm_rope(cubin)
}

/// Launches the fused kv_norm + rope for one row.
///
/// # Errors
/// The geometry or span guard's refusal, or the driver's launch error.
pub fn launch_kv_norm_rope(
    cuda: &Cuda,
    kernel: Kernel,
    params: &KvNormRopeParams,
) -> Result<(), String> {
    params.validate()?;
    cuda.launch(kernel, GRID, BLOCK, &params.words())
}

/// Loads the cache append from its cubin by name.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain the symbol.
pub fn load_kv_append(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kv_append(cubin)
}

/// Launches the window quantiser for one row.
///
/// # Errors
/// The geometry or span guard's refusal, or the driver's launch error.
pub fn launch_window_quant(
    cuda: &Cuda,
    kernel: Kernel,
    params: &WindowQuantParams,
) -> Result<(), String> {
    params.validate()?;
    cuda.launch(kernel, GRID, BLOCK, &params.words())
}

/// The alignment the device KV cache guarantees: its base and every layer's row run (the frozen
/// DeviceKvCache.ALIGN in crates/sparkflash-cli/src/layer_device.rs).
pub const CACHE_ALIGN: usize = 256;

/// One cache-append launch's arguments, in the kernel declaration's order.
///
/// The cache pointer is the LAYER's run (the port passes DeviceKvCache::layer_span(layer).device),
/// not the cache base: layers are offsets through span arithmetic, so the kernel needs no layer
/// argument and cannot address another layer's rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvAppendParams {
    /// This position's k row: [head_dim] bf16.
    pub k_row: Deviceptr,
    /// This position's v row: [head_dim] bf16.
    pub v_row: Deviceptr,
    /// The layer's ring: [rows][2 * head_dim] bf16.
    pub cache: Deviceptr,
    /// Ring capacity in rows (DeviceKvCache::capacity_rows).
    pub rows: u32,
    /// Values per k or v row (the head width; DeviceKvCache::row_values = 2 * head_dim).
    pub head_dim: u32,
    /// The position being appended; the slot is position % rows.
    pub position: u32,
}

impl KvAppendParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; 6] {
        [
            self.k_row.0,
            self.v_row.0,
            self.cache.0,
            u64::from(self.rows),
            u64::from(self.head_dim),
            u64::from(self.position),
        ]
    }

    /// The ring slot this append writes, as the kernel computes it.
    ///
    /// # Panics
    /// Panics on a zero ring; call [KvAppendParams::validate] first. The kernel would divide by zero
    /// on the same input, which is why the guard exists at this layer rather than inside the kernel.
    #[must_use]
    pub fn slot(&self) -> u32 {
        self.position % self.rows
    }

    /// Refuses a geometry the kernel would read or write out of bounds on.
    ///
    /// # Errors
    /// A zero ring or a zero head width; a row pair whose value count does not fit a u32 span.
    pub fn validate(&self) -> Result<(), String> {
        if self.rows == 0 {
            return Err("kv append: the ring has zero rows, so the slot is undefined".to_owned());
        }
        if self.head_dim == 0 {
            return Err("kv append: head_dim is zero, so there is no row to append".to_owned());
        }
        if self.head_dim.checked_mul(2).is_none() {
            return Err(format!(
                "kv append: 2 * head_dim overflows a u32 at {}",
                self.head_dim
            ));
        }
        Ok(())
    }

    /// Refuses spans shorter than the launch reads or writes, and a cache base the vector loads
    /// would fault on.
    ///
    /// The port has the spans' byte lengths and the cache's device address, so the guard is stated
    /// in bytes and in alignment: k and v are head_dim bf16, the cache is rows * 2 * head_dim bf16,
    /// and the cache base must carry DeviceKvCache's 256-byte alignment.
    ///
    /// # Errors
    /// Names the span and the length the launch needs, or the unaligned base.
    pub fn check_spans(
        &self,
        k_bytes: usize,
        v_bytes: usize,
        cache_bytes: usize,
    ) -> Result<(), String> {
        let need_row = usize::try_from(self.head_dim)
            .map_err(|_| "kv append: head_dim does not fit a usize".to_owned())?
            * 2;
        let need_cache = usize::try_from(self.rows)
            .map_err(|_| "kv append: rows does not fit a usize".to_owned())?
            * usize::try_from(self.head_dim)
                .map_err(|_| "kv append: head_dim does not fit a usize".to_owned())?
            * 2
            * 2;
        if k_bytes < need_row || v_bytes < need_row {
            return Err(format!(
                "kv append: the k span is {k_bytes} bytes and the v span {v_bytes}, a {}-wide bf16 row is {need_row}",
                self.head_dim
            ));
        }
        if cache_bytes < need_cache {
            return Err(format!(
                "kv append: the cache span is {cache_bytes} bytes, a {}-row ring of {}-wide k/v rows is {need_cache}",
                self.rows, self.head_dim
            ));
        }
        if !self.cache.0.is_multiple_of(CACHE_ALIGN as u64) {
            return Err(format!(
                "kv append: the cache base {:#x} is not {CACHE_ALIGN}-byte aligned",
                self.cache.0
            ));
        }
        Ok(())
    }
}

/// Launches the cache append for one position.
///
/// # Errors
/// The geometry or span guard's refusal, or the driver's launch error.
pub fn launch_kv_append(
    cuda: &Cuda,
    kernel: Kernel,
    params: &KvAppendParams,
) -> Result<(), String> {
    params.validate()?;
    cuda.launch(kernel, GRID, BLOCK, &params.words())
}

/// One position's kv-path row as the stage enqueues it: the three launches, in the order the
/// semantics require, over spans the stage owns.
///
/// The order is not a convention to remember: the quantiser reads the ROTATED row (the whole row is
/// quantised once, and the window is a slice of already-quantised rows), and the cache append stores
/// the same rotated row the attention body's slice-copy reads. [KvPathRow::validate_sequence] states
/// those two dependencies as pointer equalities and the position as one value, so a stage that
/// reorders the launches or feeds the quantiser the pre-rotation row is refused by name instead of
/// scoring a tiny logits difference.
///
/// What this helper deliberately does NOT do: decide the window. The frozen contract places the group
/// inside the kv_path stage, at this position, before the seam and the slice-copy (run_layer runs
/// kv_path first), and that placement is also the visibility the append needs: the window at position
/// p is rows [p - window_width, p) - the cache holds exactly p rows at the top of the position and the
/// push happens at the bottom - so row p is never in this position's window and first appears at
/// p + 1. The append is thus before every window read that could see it, and "after the window read"
/// is not a requirement this helper can impose: no read of position p's window can contain row p.
/// Enqueueing the group after the slice-copy would be numerically identical, but it would state the
/// opposite of the contract's placement, and a stage must be readable against the contract. In the
/// MLA release model the k and v rows are the same rotated latent, so k_row and v_row are both the
/// rope output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPathRow {
    /// The rope rotation for this position's latent row.
    pub rope: RopeRotateParams,
    /// The whole-row window quantisation, reading the rotated row.
    pub quant: WindowQuantParams,
    /// The cache append of this position's k/v pair.
    pub append: KvAppendParams,
}

impl KvPathRow {
    /// Refuses a sequence whose launches do not compose: the quantiser must read the rotated row,
    /// the appended k row must be that same row, all three must be one position and one width.
    ///
    /// # Errors
    /// Names the first dependency that does not hold.
    pub fn validate_sequence(&self) -> Result<(), String> {
        self.rope.validate()?;
        self.quant.validate()?;
        self.append.validate()?;
        if self.quant.row != self.rope.out {
            return Err(
                "kv path: the quantiser must read the ROTATED row (rope.out); the whole row is                  quantised once, the window is a slice of already-quantised rows"
                    .to_owned(),
            );
        }
        if self.append.k_row != self.rope.out {
            return Err(
                "kv path: the append's k row must be the same rotated row the quantiser read, so                  the window and the cache hold identical values"
                    .to_owned(),
            );
        }
        if self.rope.position != self.append.position {
            return Err(format!(
                "kv path: the rope is for position {} and the append for position {}",
                self.rope.position, self.append.position
            ));
        }
        if self.rope.head_width != self.append.head_dim {
            return Err(format!(
                "kv path: the rope row is {} wide and the append row {}-wide",
                self.rope.head_width, self.append.head_dim
            ));
        }
        if self.quant.cols != self.rope.head_width {
            return Err(format!(
                "kv path: the quantiser row is {} wide and the rotated row {}-wide",
                self.quant.cols, self.rope.head_width
            ));
        }
        Ok(())
    }

    /// The byte spans one row needs, for a stage carving its workspace: these are the guards'
    /// inputs, not a second sizing rule.
    ///
    /// # Errors
    /// A geometry the guards refuse.
    pub fn span_requirements(&self) -> Result<RowSpanRequirements, String> {
        self.validate_sequence()?;
        let row = usize::try_from(self.rope.head_width)
            .map_err(|_| "kv path: head_width does not fit a usize".to_owned())?;
        let pairs = usize::try_from(self.rope.rope_head_dim)
            .map_err(|_| "kv path: rope_head_dim does not fit a usize".to_owned())?
            / 2;
        let rows = usize::try_from(self.append.rows)
            .map_err(|_| "kv path: rows does not fit a usize".to_owned())?;
        Ok(RowSpanRequirements {
            row_bf16_bytes: row * 2,
            table_f32_bytes: pairs * 4,
            codes_bytes: row,
            scales_bytes: row / WINDOW_QUANT_BLOCK,
            ring_bytes: rows * row * 2 * 2,
        })
    }
}

/// The byte spans one kv-path row touches, as [KvPathRow::span_requirements] derives them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowSpanRequirements {
    /// One bf16 row: the rotated row (and, aliased, its input).
    pub row_bf16_bytes: usize,
    /// One f32 table: cos and sin are each this long, per pair of the rotated tail.
    pub table_f32_bytes: usize,
    /// The window codes the quantiser writes over one row.
    pub codes_bytes: usize,
    /// The window scales the quantiser writes over one row.
    pub scales_bytes: usize,
    /// The layer's cache ring: rows * 2 * head_width bf16.
    pub ring_bytes: usize,
}

/// Enqueues one position's kv-path row: rope, then the window quantisation of the rotated row, then
/// the cache append. No synchronise, no readback: the three launches are ordered by the stream.
///
/// # Errors
/// [KvPathRow::validate_sequence]'s refusal, or the driver's launch error.
pub fn launch_kv_path_row(
    cuda: &Cuda,
    rope_kernel: Kernel,
    quant_kernel: Kernel,
    append_kernel: Kernel,
    row: &KvPathRow,
) -> Result<(), String> {
    row.validate_sequence()?;
    launch_rope_rotate(cuda, rope_kernel, &row.rope)?;
    launch_window_quant(cuda, quant_kernel, &row.quant)?;
    launch_kv_append(cuda, append_kernel, &row.append)
}

/// The released MLA latent width, re-exported so a call site does not import two modules for it.
pub const RELEASED_ROW_WIDTH: usize = MLA_LATENT;

#[cfg(test)]
mod tests {
    use super::*;

    fn pointer(address: u64) -> Deviceptr {
        Deviceptr(address)
    }

    fn rope_params(head_width: u32, rope_head_dim: u32) -> RopeRotateParams {
        RopeRotateParams {
            row: pointer(0x1000),
            cos: pointer(0x2000),
            sin: pointer(0x3000),
            out: pointer(0x4000),
            head_width,
            rope_head_dim,
            position: 37,
        }
    }

    #[test]
    fn the_rope_words_are_the_kernel_declaration_order() {
        let params = rope_params(512, 64);
        assert_eq!(
            params.words(),
            [0x1000, 0x2000, 0x3000, 0x4000, 512, 64, 37]
        );
        assert_eq!(params.words().len(), 7);
    }

    #[test]
    fn the_window_words_are_the_kernel_declaration_order() {
        let params = WindowQuantParams {
            row: pointer(0x1000),
            codes: pointer(0x2000),
            scales: pointer(0x3000),
            cols: 512,
            block: WINDOW_QUANT_BLOCK as u32,
        };
        assert_eq!(params.words(), [0x1000, 0x2000, 0x3000, 512, 32]);
        assert_eq!(params.words().len(), 5);
    }

    #[test]
    fn the_rope_geometry_guard_refuses_a_ragged_tail() {
        assert!(rope_params(512, 64).validate().is_ok());
        assert!(rope_params(512, 0).validate().is_err());
        assert!(rope_params(512, 63).validate().is_err());
        assert!(rope_params(512, 514).validate().is_err());
        assert!(rope_params(0, 64).validate().is_err());
    }

    #[test]
    fn the_window_geometry_guard_refuses_a_foreign_block_or_a_partial_block() {
        let good = WindowQuantParams {
            row: pointer(0),
            codes: pointer(0),
            scales: pointer(0),
            cols: 512,
            block: 32,
        };
        assert!(good.validate().is_ok());
        let mut wrong_block = good;
        wrong_block.block = 16;
        assert!(wrong_block.validate().is_err());
        let mut partial = good;
        partial.cols = 500;
        assert!(partial.validate().is_err());
        let mut zero = good;
        zero.cols = 0;
        assert!(zero.validate().is_err());
    }

    #[test]
    fn the_rope_span_guard_refuses_a_short_span_with_the_requirement() {
        let params = rope_params(512, 64);
        assert!(params.check_spans(1024, 1024, 128).is_ok());
        let error = params.check_spans(1023, 1024, 128).expect_err("short row");
        assert!(error.contains("1024"), "{error}");
        let error = params.check_spans(1024, 512, 128).expect_err("short out");
        assert!(error.contains("out"), "{error}");
        let error = params
            .check_spans(1024, 1024, 127)
            .expect_err("short table");
        assert!(error.contains("127"), "{error}");
    }

    #[test]
    fn the_window_span_guard_refuses_a_short_span_with_the_requirement() {
        let params = WindowQuantParams {
            row: pointer(0),
            codes: pointer(0),
            scales: pointer(0),
            cols: 512,
            block: 32,
        };
        assert!(params.check_spans(1024, 512, 16).is_ok());
        assert!(params.check_spans(1023, 512, 16).is_err());
        assert!(params.check_spans(1024, 511, 16).is_err());
        assert!(params.check_spans(1024, 512, 15).is_err());
    }

    fn append_params(rows: u32, head_dim: u32, position: u32) -> KvAppendParams {
        KvAppendParams {
            k_row: pointer(0x1000),
            v_row: pointer(0x2000),
            cache: pointer(0x1_0000),
            rows,
            head_dim,
            position,
        }
    }

    #[test]
    fn the_append_words_are_the_kernel_declaration_order() {
        let params = append_params(128, 512, 37);
        assert_eq!(params.words(), [0x1000, 0x2000, 0x1_0000, 128, 512, 37]);
        assert_eq!(params.words().len(), 6);
    }

    #[test]
    fn the_append_slot_is_the_ring_modulo() {
        let params = append_params(128, 512, 37);
        assert_eq!(params.slot(), 37);
        assert_eq!(append_params(128, 512, 128 + 5).slot(), 5);
        assert_eq!(append_params(128, 512, 0).slot(), 0);
        assert_eq!(append_params(1, 512, 999).slot(), 0);
    }

    #[test]
    fn the_append_geometry_guard_refuses_a_bad_ring() {
        assert!(append_params(128, 512, 0).validate().is_ok());
        assert!(append_params(0, 512, 0).validate().is_err());
        assert!(append_params(128, 0, 0).validate().is_err());
    }

    #[test]
    fn the_append_span_guard_refuses_a_short_span_and_a_misaligned_cache() {
        let params = append_params(128, 512, 3);
        let row_bytes = 512 * 2;
        let cache_bytes = 128 * 512 * 2 * 2;
        assert!(
            params
                .check_spans(row_bytes, row_bytes, cache_bytes)
                .is_ok()
        );
        assert!(
            params
                .check_spans(row_bytes - 1, row_bytes, cache_bytes)
                .is_err()
        );
        assert!(
            params
                .check_spans(row_bytes, row_bytes - 1, cache_bytes)
                .is_err()
        );
        let error = params
            .check_spans(row_bytes, row_bytes, cache_bytes - 1)
            .expect_err("short cache");
        assert!(error.contains(&cache_bytes.to_string()), "{error}");
        // The 256-byte DeviceKvCache alignment: a base of 0x100 is fine, 0x108 is refused by address.
        assert!(
            params
                .check_spans(row_bytes, row_bytes, cache_bytes)
                .is_ok()
        );
        let mut misaligned = params;
        misaligned.cache = pointer(0x1_0008);
        let error = misaligned
            .check_spans(row_bytes, row_bytes, cache_bytes)
            .expect_err("misaligned");
        assert!(error.contains("aligned"), "{error}");
    }

    #[test]
    fn the_released_geometry_is_the_port_geometry() {
        assert_eq!(RELEASED_ROW_WIDTH, 512);
        assert_eq!(RELEASED_ROPE_HEAD_DIM, 64);
        assert_eq!(BLOCK.0, 256);
        assert_eq!(GRID, (1, 1, 1));
        assert!(
            rope_params(RELEASED_ROW_WIDTH as u32, RELEASED_ROPE_HEAD_DIM as u32)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn the_fused_norm_rope_words_follow_the_declaration() {
        let params = KvNormRopeParams {
            row: pointer(0x1000),
            norm: pointer(0x2000),
            eps: 1e-20_f32,
            cos: pointer(0x3000),
            sin: pointer(0x4000),
            out: pointer(0x5000),
            head_width: 512,
            rope_head_dim: 64,
        };
        assert_eq!(
            params.words(),
            [
                0x1000,
                0x2000,
                u64::from(1e-20_f32.to_bits()),
                0x3000,
                0x4000,
                0x5000,
                512,
                64
            ]
        );
    }

    #[test]
    fn the_fused_norm_rope_guards_refuse_a_bad_row_and_a_short_span() {
        let good = KvNormRopeParams {
            row: pointer(0x1000),
            norm: pointer(0x2000),
            eps: 1e-20_f32,
            cos: pointer(0x3000),
            sin: pointer(0x4000),
            out: pointer(0x5000),
            head_width: 512,
            rope_head_dim: 64,
        };
        assert!(good.validate().is_ok());
        assert!(good.check_spans(1024, 1024, 128, 128, 1024).is_ok());
        assert!(good.check_spans(1023, 1024, 128, 128, 1024).is_err());
        assert!(good.check_spans(1024, 1024, 127, 128, 1024).is_err());

        let mut bad = good;
        bad.head_width = 0;
        assert!(bad.validate().is_err());
        let mut bad = good;
        bad.rope_head_dim = 63;
        assert!(bad.validate().is_err());
        let mut bad = good;
        bad.rope_head_dim = 514;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn the_fused_norm_rope_declaration_is_pinned_to_the_source() {
        let decl = kv_norm_rope_declaration();
        assert!(
            kv_norm_rope_declaration_matches(&decl),
            "the parsed declaration drifted from KV_NORM_ROPE_PARAMS: {decl:?}"
        );
        let mut mutated = decl.clone();
        mutated.params[0].const_pointee = false;
        assert!(
            !kv_norm_rope_declaration_matches(&mutated),
            "the pin must notice a dropped const on the latent"
        );
    }
}

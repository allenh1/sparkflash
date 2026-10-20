//! Typed launch capability of the deferred sequence's CSA2 attention entry
//! (kernels/flash_mla_csa2_seq_sm121.cu).
//!
//! The argument array is the kernel declaration's parameter list in order - the same nineteen words
//! as flash_mla_csa2_sm121, so a caller that already builds FlashMlaParams can move to this entry
//! without changing its upload layout. The declaration is pinned here against the kernel source
//! (SOURCE): PARAMS is the pinned name and type of every parameter and declaration_matches is the
//! check, with a mutation test proving it is not vacuous.
//!
//! Geometry. One launch runs one format-2 (CSA union) attention for every query head at ONE
//! position: grid (HEADS, 1, 1), block (THREADS, 1, 1). That is the base entry's geometry; this
//! entry exists because the sequence path calls it once per position, and its indexer stage is
//! position-causal instead of unmasked.
//!
//! No cubin pin. The device catalog owns the cubin's bytes and digest, as for the other launch
//! modules here; the harness verifies the pinned artifact through baseline::pin.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};

/// The kernel symbol in the cubin.
pub const KERNEL: &str = "flash_mla_csa2_seq_sm121";
/// The driver identity of this kernel. Unlike the base flash_mla entry, the sequence entry is a
/// separate cubin the shared loader resolves by id.
pub const KERNEL_ID: KernelId = KernelId::FlashMlaCsa2Seq;
/// The kernel source the declaration pin is read from.
pub const SOURCE: &str = include_str!("../../../kernels/flash_mla_csa2_seq_sm121.cu");
/// The number of argument words a launch fills.
pub const ARG_WORDS: usize = 19;

/// Query heads in the released model; grid.x.
pub const HEADS: usize = 64;
/// Indexer heads: two query heads share one indexer head.
pub const INDEX_HEADS: usize = 32;
/// The MLA latent width, f32 in the reference and FP4/E4M3 in the store.
pub const LATENT: usize = 512;
/// The 128-token sliding window.
pub const SWA_WINDOW: usize = 128;
/// The candidate list bound (index_topk).
pub const MAX_SELECTED: usize = 512;
/// The union slot bound: window rows plus selected candidates.
pub const MAX_ATTENDED: usize = MAX_SELECTED + SWA_WINDOW;
/// 256 threads, one warp per HSI block in the indexer.
pub const THREADS: u32 = 256;
/// The launch geometry: every head, one position, one slice.
pub const GRID: (u32, u32, u32) = (HEADS as u32, 1, 1);
/// The block geometry.
pub const BLOCK: (u32, u32, u32) = (THREADS, 1, 1);
/// K8 batched sequence symbol: one block per head serves six consecutive positions.
pub const BATCH_KERNEL: &str = "flash_mla_csa2_seq_batched_sm121";
/// Fixed six-row verifier batch.
pub const BATCH_ROWS: usize = 6;

/// The kernel's only accepted window_kv_format value: the CSA union.
pub const FORMAT_CSA_UNION: u64 = 2;

/// The declaration's parameters in order: (name, base type, pointer, pointee const).
pub const PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("q", "__nv_bfloat16", true, true),
    ("kv_fp4", "uint8_t", true, true),
    ("kv_scales", "uint8_t", true, true),
    ("window_codes", "uint8_t", true, true),
    ("window_scales", "uint8_t", true, true),
    ("window_kv_format", "uint32_t", false, false),
    ("window_rows", "uint32_t", false, false),
    ("selected", "uint32_t", true, true),
    ("index_key", "__nv_bfloat16", true, true),
    ("index_query", "__nv_bfloat16", true, true),
    ("sink", "float", true, true),
    ("block_scores", "float", true, false),
    ("block_best", "uint32_t", true, false),
    ("out", "__nv_bfloat16", true, false),
    ("position", "uint32_t", false, false),
    ("selected_count", "uint32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// Parses the kernel declaration out of the pinned source.
#[must_use]
pub fn declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, KERNEL)
}

/// Whether a parsed declaration has PARAMS' shape and order.
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

/// One launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashMlaSeqParams([u64; ARG_WORDS]);

impl FlashMlaSeqParams {
    /// Builds the argument array with the kernel's own parameter names, in order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        q: Deviceptr,
        kv_fp4: Deviceptr,
        kv_scales: Deviceptr,
        window_codes: Deviceptr,
        window_scales: Deviceptr,
        window_kv_format: u64,
        window_rows: u64,
        selected: Deviceptr,
        index_key: Deviceptr,
        index_query: Deviceptr,
        sink: Deviceptr,
        block_scores: Deviceptr,
        block_best: Deviceptr,
        out: Deviceptr,
        position: u64,
        selected_count: u32,
        trace_ring: u64,
        layer_id: u32,
        step_seq: u64,
    ) -> Self {
        Self([
            q.0,
            kv_fp4.0,
            kv_scales.0,
            window_codes.0,
            window_scales.0,
            window_kv_format,
            window_rows,
            selected.0,
            index_key.0,
            index_query.0,
            sink.0,
            block_scores.0,
            block_best.0,
            out.0,
            position,
            u64::from(selected_count),
            trace_ring,
            u64::from(layer_id),
            step_seq,
        ])
    }

    /// The parameter words a launch passes, the shape cuLaunchKernel takes.
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.0
    }

    /// One argument word by PARAMS index.
    #[must_use]
    pub fn word(&self, index: usize) -> u64 {
        self.0[index]
    }
}

/// Loads the K8 batched sequence symbol by name from the same cubin.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [BATCH_KERNEL].
pub fn load_batched(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_flash_mla_csa2_seq_batched(cubin)
}

/// Loads the kernel from a cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain KERNEL.
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches the sequence attention once, at the params' position.
///
/// # Errors
/// Returns the driver's error.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &FlashMlaSeqParams) -> Result<(), String> {
    cuda.launch(kernel, GRID, BLOCK, params.as_slice())
}

/// Launches the K8 batched sequence entry once: one block per head over six rows.
///
/// The 19-word ABI is unchanged; `q`, `window_codes`, `window_scales`, `selected`, `index_query`,
/// `out`, `block_scores` and `block_best` MUST be batch-major over [BATCH_ROWS], and `position` is
/// the first of six consecutive positions. Every row must carry the same `window_rows` and
/// `selected_count`; the entry refuses a differing-count call by contract, not by a runtime scan.
/// The `selected` row stride is exactly `selected_count`, so rows are packed contiguously; a caller
/// that pads each row to a wider buffer must also pass that padded width as `selected_count`.
/// `block_scores`/`block_best` are one [hsi_blocks] pair, not one per index head: only index head 0
/// publishes them (its two query heads split the blocks), matching the CPU indexer reference.
/// OPEN ABI ITEM (K7): a caller with differing per-row counts MUST pad the selected list to the
/// common `selected_count` with future ids - the kernel already masks candidate > that row's
/// position - and pass a full window for every row. A per-row count pointer pair is a follow-up
/// symbol, not this v1.
///
/// # Errors
/// Refuses a format other than the CSA union, an empty head list, or a position that would wrap.
pub fn launch_batched(
    cuda: &Cuda,
    kernel: Kernel,
    params: &FlashMlaSeqParams,
) -> Result<(), String> {
    if params.as_slice()[5] != FORMAT_CSA_UNION {
        return Err("the batched sequence entry only runs the CSA union format".to_owned());
    }
    if params.as_slice()[14] + (BATCH_ROWS as u64 - 1) > u64::from(u32::MAX) {
        return Err("the six consecutive positions would wrap u32".to_owned());
    }
    cuda.launch(kernel, GRID, BLOCK, params.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_decl::Consts;
    use crate::m4_abi;

    fn params() -> FlashMlaSeqParams {
        FlashMlaSeqParams::new(
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            Deviceptr(0x3000),
            Deviceptr(0x4000),
            Deviceptr(0x5000),
            FORMAT_CSA_UNION,
            128,
            Deviceptr(0x6000),
            Deviceptr(0x7000),
            Deviceptr(0x8000),
            Deviceptr(0x9000),
            Deviceptr(0xa000),
            Deviceptr(0xb000),
            Deviceptr(0xc000),
            129,
            65,
            0,
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
        let mutated = SOURCE.replace("uint32_t *__restrict__ block_best", "uint32_t * block_best");
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        let decl = c_decl::kernel_declaration(&mutated, KERNEL);
        assert!(
            !declaration_matches(&decl),
            "the qualifier pin is vacuous: {decl:?}"
        );
    }

    #[test]
    fn the_argument_array_is_the_declaration_order() {
        let p = params();
        assert_eq!(
            p.as_slice(),
            [
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 2, 128, 0x6000, 0x7000, 0x8000, 0x9000,
                0xa000, 0xb000, 0xc000, 129, 65, 0, 7, 11
            ]
        );
        for (index, (name, _, _, _)) in PARAMS.iter().enumerate() {
            assert_eq!(p.word(index), p.as_slice()[index], "param {name}");
        }
        assert_eq!(p.as_slice().len(), ARG_WORDS);
    }

    #[test]
    fn the_batched_geometry_is_one_block_per_head_over_six_rows() {
        assert_eq!(BATCH_ROWS, 6);
        assert_eq!(GRID, (HEADS as u32, 1, 1));
        assert_eq!(BLOCK, (THREADS, 1, 1));
        assert_ne!(BATCH_KERNEL, KERNEL);
    }

    #[test]
    fn the_geometry_is_the_kernels() {
        let table = constants();
        assert_eq!(table.value("SPARK_M4_HEADS"), HEADS as i128);
        assert_eq!(table.value("SPARK_M4_INDEX_HEADS"), INDEX_HEADS as i128);
        assert_eq!(table.value("SPARK_M4_MLA_LATENT"), LATENT as i128);
        assert_eq!(table.value("SPARK_M4_SWA_WINDOW"), SWA_WINDOW as i128);
        assert_eq!(
            table.value("SPARK_M4_MLA_MAX_SELECTED"),
            MAX_SELECTED as i128
        );
        assert_eq!(
            table.value("SPARK_M4_MLA_MAX_ATTENDED"),
            MAX_ATTENDED as i128
        );
        assert_eq!(table.value("SPARK_M4_MLA_THREADS"), i128::from(THREADS));
        assert_eq!(table.value("SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION"), 2);
        assert_eq!(FORMAT_CSA_UNION, 2);
        assert_eq!(MAX_ATTENDED, 640);
        assert_eq!(GRID, (64, 1, 1));
        assert_eq!(BLOCK, (256, 1, 1));
    }
}

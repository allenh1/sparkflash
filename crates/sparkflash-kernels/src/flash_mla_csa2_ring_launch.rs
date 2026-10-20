//! Typed launch capability of the D49 ring-capacity CSA2 attention entry
//! (kernels/flash_mla_csa2_ring_sm121.cu).
//!
//! The argument array is the kernel declaration's parameter list in order: the frozen sequence
//! entry's nineteen words plus the two D49 arguments, ring_capacity and compression_ratio. The
//! declaration is pinned here against the kernel source (SOURCE): PARAMS is the pinned name and type
//! of every parameter and declaration_matches is the check, with a mutation test proving it is not
//! vacuous.
//!
//! D49 semantics. The KV ring holds 134 slots (window 128 + nextn + 1), so the caller's slot is
//! position % 134, but the arithmetic window stays 128 and the union score array is bounded by
//! 512 candidates + 134 ring slots = 646. Capacity and compression ratio are explicit arguments;
//! neither is inferred from a candidate count. The frozen flash_mla_csa2_seq_sm121 entry, its
//! nineteen-word ABI, its source file and its cubin pin are NOT changed.
//!
//! Geometry. One launch runs one format-2 (CSA union) attention for every query head at ONE
//! position: grid (HEADS, 1, 1), block (THREADS, 1, 1), the same geometry as the sequence entry.
//!
//! No cubin pin here. baseline/kernel_baseline.txt records the compiled artifact; the harness reads
//! it through baseline::pin, as for the other launch modules.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};

/// The kernel symbol in the cubin.
pub const KERNEL: &str = "flash_mla_csa2_ring_sm121";
/// The driver identity of this kernel: a separate cubin with its own pin and KernelId.
pub const KERNEL_ID: KernelId = KernelId::FlashMlaCsa2Ring;
/// The kernel source the declaration pin is read from.
pub const SOURCE: &str = include_str!("../../../kernels/flash_mla_csa2_ring_sm121.cu");
/// The number of argument words a launch fills: 19 frozen words + ring_capacity + compression_ratio.
pub const ARG_WORDS: usize = 21;

/// Query heads in the released model; grid.x.
pub const HEADS: usize = 64;
/// Indexer heads: two query heads share one indexer head.
pub const INDEX_HEADS: usize = 32;
/// The MLA latent width, f32 in the reference and FP4/E4M3 in the store.
pub const LATENT: usize = 512;
/// The 128-token arithmetic sliding window; it is NOT the ring capacity.
pub const SWA_WINDOW: usize = 128;
/// D49 ring slots: window 128 + nextn + 1.
pub const RING_CAPACITY: usize = 134;
/// The candidate list bound (index_topk).
pub const MAX_SELECTED: usize = 512;
/// The union slot bound: the 512 candidates plus the whole ring.
pub const MAX_ATTENDED: usize = MAX_SELECTED + RING_CAPACITY;
/// 256 threads, one warp per HSI block in the indexer.
pub const THREADS: u32 = 256;
/// The launch geometry: every head, one position, one slice.
pub const GRID: (u32, u32, u32) = (HEADS as u32, 1, 1);
/// The block geometry.
pub const BLOCK: (u32, u32, u32) = (THREADS, 1, 1);
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
    ("ring_capacity", "uint32_t", false, false),
    ("compression_ratio", "uint32_t", false, false),
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
        ring_capacity: u64,
        compression_ratio: u64,
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
            ring_capacity,
            compression_ratio,
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

/// Loads the kernel from a cubin, resolving the pinned ring symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain KERNEL.
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches one ring-capacity attention: grid (64, 1, 1), block (256, 1, 1).
///
/// Parity with the sequence entry: `word(7)` is ring_capacity (1..=134) and `word(8)` is the
/// explicit compression ratio. The kernel refuses a capacity outside the D49 ring, a zero ratio, a
/// window that does not fit the ring, or a union that does not fit the score array, writing NaN into
/// out[] and no trace pair.
///
/// # Errors
/// Refuses a format other than the CSA union, a capacity outside 1..=134, a zero ratio, a window
/// that does not fit the ring, or a union that does not fit the score array.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &FlashMlaSeqParams) -> Result<(), String> {
    let words = params.as_slice();
    if words[5] != FORMAT_CSA_UNION {
        return Err("the ring entry only runs the CSA union format".to_owned());
    }
    let ring_capacity = words[7];
    let ratio = words[8];
    if ring_capacity == 0 || ring_capacity > RING_CAPACITY as u64 {
        return Err(format!(
            "ring_capacity must be 1..={RING_CAPACITY}, got {ring_capacity}"
        ));
    }
    if words[6] > ring_capacity {
        return Err(format!(
            "window_rows {} does not fit ring_capacity {ring_capacity}",
            words[6]
        ));
    }
    if ratio == 0 {
        return Err("compression_ratio must be non-zero".to_owned());
    }
    if words[6] + words[17] > MAX_ATTENDED as u64 {
        return Err(format!(
            "window_rows {} + selected_count {} exceeds the {MAX_ATTENDED}-slot union array",
            words[6], words[17]
        ));
    }
    cuda.launch(kernel, GRID, BLOCK, words)
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
            RING_CAPACITY as u64,
            2,
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
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 2, 128, 134, 2, 0x6000, 0x7000, 0x8000,
                0x9000, 0xa000, 0xb000, 0xc000, 129, 65, 0, 7, 11
            ]
        );
        for (index, (name, _, _, _)) in PARAMS.iter().enumerate() {
            assert_eq!(p.word(index), p.as_slice()[index], "param {name}");
        }
        assert_eq!(p.as_slice().len(), ARG_WORDS);
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
            table.value("SPARK_M4_MLA_RING_CAPACITY"),
            RING_CAPACITY as i128
        );
        assert_eq!(
            table.value("SPARK_M4_MLA_RING_MAX_ATTENDED"),
            MAX_ATTENDED as i128
        );
        assert_eq!(table.value("SPARK_M4_MLA_THREADS"), i128::from(THREADS));
        assert_eq!(table.value("SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION"), 2);
        assert_eq!(FORMAT_CSA_UNION, 2);
        assert_eq!(MAX_ATTENDED, 646);
        assert_eq!(GRID, (64, 1, 1));
        assert_eq!(BLOCK, (256, 1, 1));
    }
}

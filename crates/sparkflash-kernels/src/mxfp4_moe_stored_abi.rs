//! The ABI and constant pins of the stored-layout down projection
//! (kernels/b12x_mxfp4_moe_stored_sm121.cu).
//!
//! This is a separate module from m4_abi.rs for the same reason router_abi.rs and
//! routed_gate_up_abi.rs are: m4_abi is frozen by the D26 size ratchet and its KERNELS array is the
//! six Milestone 4 kernels, so a new, non-M4 kernel is pinned here instead of growing it. Nothing in
//! this module is read by the engine; it exists so that the kernel's signature, its constants, its
//! stored row contract and its pinned cubin cannot drift without a test failing.
//!
//! What is pinned and against what:
//!
//! * the kernel declaration, parsed out of the kernel source by [crate::c_decl] (the same parser
//!   m4_abi uses), against [STORED_PARAMS] - the launch code in
//!   [crate::mxfp4_moe_stored_launch] fills the argument array in exactly this order, and the
//!   parameter list is literally the incumbent combine's list, so the two kernels cannot drift apart
//!   while the chain switches between them;
//! * the STORED row contract (288-byte code rows, 18-byte scale rows, 306 bytes per output row
//!   across the two tensors) against the host twin's [crate::mxfp4_ref::stored_moe_layout];
//! * the A8 activation operand strides (640 / 20), unchanged from the incumbent, against
//!   [crate::mxfp4_ref::MOE_ACT_BYTES_PER_ROW] and [crate::mxfp4_ref::MOE_SCALE_BYTES_PER_ROW];
//! * the geometry static_asserts in the kernel, each evaluated through [Consts];
//! * the cubin pin from baseline/kernel_baseline.txt - bytes, sha256 and PTX membar count, measured
//!   by the pinned toolkit (nvcc V13.0.88, CUDA 13.0.3, sm_121a).
//!
//! The numbers here are measurements of the pinned toolkit, not arithmetic: they are re-derived by
//! crates/sparkflash-kernels/scripts/check-kernel-baseline.sh, which tests/kernel_baseline.rs runs
//! on a CUDA host.

use crate::c_decl::{self, Consts, KernelDecl};

/// The kernel symbol, which is also the cubin stem and the manifest key.
pub const KERNEL: &str = "b12x_mxfp4_moe_stored_sm121";

/// The kernel source, as the parser and the pins see it.
pub const SOURCE: &str = include_str!("../../../kernels/b12x_mxfp4_moe_stored_sm121.cu");

/// The shared M4 header the kernel includes; parsed after the kernel so its SPARK_M4_* names resolve.
const COMMON_HEADER: &str = include_str!("../../../kernels/m4_common_sm121.cuh");

/// The hidden size (config.json hidden_size).
pub const HIDDEN: u32 = 5120;
/// The real K width of one tensor-parallel rank: moe_inter_dim 2304 / TP 4.
pub const K_VALUES: u32 = 576;
/// MX block width: one FP4 or FP8 scale per 32 values along K.
pub const QUANT_BLOCK: u32 = 32;
/// FP4 values per byte.
pub const FP4_PER_BYTE: u32 = 2;
/// MX blocks the stored kernel walks: 576 / 32, no padding block.
pub const K_BLOCKS: u32 = K_VALUES / QUANT_BLOCK;
/// Packed FP4 code bytes per stored output row: 576 / 2.
pub const CODE_ROW_BYTES: u32 = K_VALUES / FP4_PER_BYTE;
/// ue8m0 scale bytes per stored output row: one per MX block.
pub const SCALE_ROW_BYTES: u32 = K_BLOCKS;
/// The cost of one stored output row across the two SEPARATE tensors (not a stride).
pub const ROW_BYTES_TOTAL: u32 = CODE_ROW_BYTES + SCALE_ROW_BYTES;
/// Routed experts selected per token.
pub const TOP_K: u32 = 6;
/// Activation bytes per routed row: the A8 chain layout, unchanged (640 real+pad E4M3 values).
pub const ACT_ROW_BYTES: u32 = 640;
/// Activation scale bytes per routed row: the A8 chain layout, unchanged (20 MX blocks).
pub const ACT_SCALE_ROW_BYTES: u32 = 20;
/// Hidden values one block writes: 32 output columns.
pub const TILE_D: u32 = 32;
/// Parallel K groups a block's 256 threads form.
pub const K_GROUPS: u32 = 8;
/// Threads per block: TILE_D * K_GROUPS.
pub const THREADS: u32 = 256;
/// Blocks along the hidden axis: HIDDEN / TILE_D.
pub const GRID_Y: u32 = 160;

/// The number of argument words a launch fills.
pub const ARG_WORDS: usize = 10;

/// The kernel's parameters, in declaration order: (name, base type, is a pointer, pointee const).
///
/// This IS the incumbent combine's list ([crate::mxfp4_moe_launch::PARAMS]), not a copy: the stored
/// kernel exists to be swapped in behind the same call site, so the word order must be one
/// declaration and not two. The launch's parameter array is the incumbent's type
/// ([crate::mxfp4_moe_launch::Mxfp4MoeParams]) for the same reason.
pub const STORED_PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = crate::mxfp4_moe_launch::PARAMS;

/// Pinned cubin size, measured with nvcc V13.0.88 (CUDA 13.0.3) for sm_121a.
pub const CUBIN_BYTES: usize = 30_760;
/// Pinned cubin sha256, from the same measurement.
pub const CUBIN_SHA256: &str = "1dbfd764f11e7378ecc1790b8f5bca2e97871c6a6ad3092cd16c6a5df767ba08";
/// PTX membar lines in the pinned PTX: the kernel emits none (no fence, no acquire).
pub const CUBIN_MEMBAR_LINES: usize = 0;
/// The toolkit the pin was measured with.
pub const TOOLKIT_RELEASE: &str = "V13.0.88";

/// Parses the kernel declaration out of the pinned source.
#[must_use]
pub fn declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, KERNEL)
}

/// The kernel's constants with the shared M4 header's names resolvable.
#[must_use]
pub fn constants() -> Consts {
    let mut table = Consts::parse(SOURCE);
    table.extend(COMMON_HEADER);
    table
}

/// Whether a parsed declaration has [STORED_PARAMS]' shape and order.
#[must_use]
pub fn declaration_matches(decl: &KernelDecl) -> bool {
    decl.name == KERNEL
        && decl.extern_c
        && decl.global
        && decl.params.len() == ARG_WORDS
        && decl.params.iter().zip(STORED_PARAMS).all(|(param, pin)| {
            param.name == pin.0
                && param.base == pin.1
                && param.pointer == pin.2
                && param.const_pointee == pin.3
                && (!param.pointer || param.restrict)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed baseline manifest, read where the cubin pin is checked.
    const BASELINE: &str = include_str!("../baseline/kernel_baseline.txt");

    #[test]
    fn the_kernel_declaration_is_the_pinned_one() {
        let decl = declaration();
        assert!(
            declaration_matches(&decl),
            "the declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), ARG_WORDS);
        assert!(!decl.params[6].pointer, "tokens is a value");
        assert!(decl.params[7].pointer && decl.params[7].base == "GpuTraceRing");
        assert!(!decl.params[8].pointer && !decl.params[9].pointer);
    }

    #[test]
    fn the_declaration_pin_is_not_vacuous() {
        let mutated = SOURCE.replace(
            "    const uint8_t *__restrict__ w_fp4,",
            "    const uint8_t * w_fp4,",
        );
        assert_ne!(mutated, SOURCE, "the mutation must apply");
        let decl = c_decl::kernel_declaration(&mutated, KERNEL);
        assert!(
            !declaration_matches(&decl),
            "the dropped qualifier must fail the ABI pin"
        );
    }

    #[test]
    fn the_parameter_list_is_the_incumbent_combine_list() {
        // The stored kernel is a drop-in for the same call site, so the word order is pinned by
        // identity against the incumbent's list, not by a second transcription that could drift.
        assert_eq!(STORED_PARAMS, crate::mxfp4_moe_launch::PARAMS);
        assert_eq!(
            crate::mxfp4_moe_launch::ARG_WORDS,
            ARG_WORDS,
            "the two kernels must fill the same number of ABI words"
        );
    }

    #[test]
    fn the_stored_row_contract_is_the_twins() {
        let table = constants();
        assert_eq!(table.value("SPARK_M4_STORED_K_VALUES"), K_VALUES as i128);
        assert_eq!(table.value("SPARK_M4_STORED_K_BLOCKS"), K_BLOCKS as i128);
        assert_eq!(
            table.value("SPARK_M4_STORED_CODE_ROW"),
            CODE_ROW_BYTES as i128
        );
        assert_eq!(
            table.value("SPARK_M4_STORED_SCALE_ROW"),
            SCALE_ROW_BYTES as i128
        );
        assert_eq!(
            table.value("SPARK_M4_STORED_ROW_TOTAL"),
            ROW_BYTES_TOTAL as i128
        );
        assert_eq!(
            table.value("SPARK_M4_STORED_ACT_ROW"),
            ACT_ROW_BYTES as i128
        );
        assert_eq!(
            table.value("SPARK_M4_STORED_ACT_SCALE_ROW"),
            ACT_SCALE_ROW_BYTES as i128
        );
        assert_eq!(table.value("SPARK_M4_STORED_TILE_D"), TILE_D as i128);
        assert_eq!(table.value("SPARK_M4_STORED_K_GROUPS"), K_GROUPS as i128);
        assert_eq!(table.value("SPARK_M4_STORED_THREADS"), THREADS as i128);
        assert_eq!(table.value("SPARK_M4_STORED_GRID_Y"), GRID_Y as i128);
        // The same numbers, from the host twin side: this is the contract check_stored_moe_layout
        // enforces on any loader.
        let layout = crate::mxfp4_ref::stored_moe_layout();
        assert_eq!(layout.weight_row_bytes, CODE_ROW_BYTES as usize);
        assert_eq!(layout.weight_scale_row_bytes, SCALE_ROW_BYTES as usize);
        assert_eq!(layout.activation_row_bytes, ACT_ROW_BYTES as usize);
        assert_eq!(
            layout.activation_scale_row_bytes,
            ACT_SCALE_ROW_BYTES as usize
        );
        assert_eq!(crate::mxfp4_ref::MOE_STORED_K_BLOCKS, K_BLOCKS as usize);
        assert_eq!(crate::mxfp4_ref::MOE_TOP_K, TOP_K as usize);
        assert_eq!(crate::mxfp4_ref::MOE_WIDTH, K_VALUES as usize);
        assert_eq!(
            CODE_ROW_BYTES + SCALE_ROW_BYTES,
            crate::mxfp4_ref::MOE_STORED_CODE_ROW as u32
                + crate::mxfp4_ref::MOE_STORED_SCALE_ROW as u32
        );
        assert_eq!(ROW_BYTES_TOTAL, 306);
    }

    #[test]
    fn the_stored_layout_is_distinguishable_from_the_padded_one() {
        // The chain selects the cubin by plan layout, so the two contracts must not be confused: a
        // loader that announces the padded 320/20 row is rejected by the stored check, and the stored
        // 288/18 row is rejected by the incumbent's check.
        let stored = crate::mxfp4_ref::stored_moe_layout();
        assert!(crate::mxfp4_ref::check_stored_moe_layout(stored).is_ok());
        assert!(
            crate::mxfp4_ref::check_moe_layout(stored).is_err(),
            "the stored layout must be rejected by the A8 check"
        );
        let padded = crate::mxfp4_ref::pre_padded_moe_layout();
        assert!(crate::mxfp4_ref::check_moe_layout(padded).is_ok());
        assert!(
            crate::mxfp4_ref::check_stored_moe_layout(padded).is_err(),
            "the A8 layout must be rejected by the stored check"
        );
        // And the two named layouts are not equal, so a selector cannot match both branches.
        assert_ne!(stored, padded);
    }

    #[test]
    fn every_static_assert_names_a_project_symbol_and_evaluates_true() {
        let table = constants();
        let asserts = c_decl::static_asserts(SOURCE);
        assert!(
            asserts.len() >= 10,
            "the kernel pins its geometry with asserts: {asserts:?}"
        );
        for assertion in &asserts {
            assert!(
                assertion.condition.contains("SPARK_"),
                "a tautology is not a geometry assert: {}",
                assertion.condition
            );
            assert_eq!(
                table.eval(&assertion.condition),
                1,
                "static_assert must evaluate true: {}",
                assertion.condition
            );
        }
    }

    #[test]
    fn the_manifest_pins_the_stored_cubin() {
        let line = BASELINE
            .lines()
            .find(|line| line.starts_with("kernel b12x_mxfp4_moe_stored_sm121 "))
            .expect("the baseline manifest must name the stored combine kernel");
        let expected = format!(
            "kernel b12x_mxfp4_moe_stored_sm121 cubin_bytes={CUBIN_BYTES} cubin_sha256={CUBIN_SHA256} ptx_membar_lines={CUBIN_MEMBAR_LINES}"
        );
        assert_eq!(
            line.trim(),
            expected,
            "the manifest entry must be the pinned cubin"
        );
        // The header count and toolkit are part of the same claim: this is one of twenty-one
        // entries.
        assert!(
            BASELINE.contains("thirty-one baseline kernels"),
            "the manifest header must say thirty-one"
        );
        assert!(BASELINE.contains(&format!("nvcc_release={TOOLKIT_RELEASE}")));
        assert!(BASELINE.contains("arch=sm_121a"));
    }
}

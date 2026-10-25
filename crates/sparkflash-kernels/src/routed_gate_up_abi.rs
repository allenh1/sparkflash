//! The ABI and constant pins of the routed-expert gate/up projection
//! (kernels/routed_gate_up_sm121.cu).
//!
//! This is a separate module from m4_abi.rs for the same reason router_abi.rs is: that module is
//! frozen by the D26 size ratchet and its KERNELS array is the six Milestone 4 kernels plus the M7
//! SwiGLU epilogue, so a new, non-M4 kernel is pinned here instead of growing it. Nothing in this
//! module is read by the engine; it exists so that the kernel's signature, its constants, its
//! geometry asserts and its pinned cubin cannot drift without a test failing.
//!
//! What is pinned and against what:
//!
//! * the kernel declaration, parsed out of the kernel source by [crate::c_decl] (the same parser
//!   m4_abi uses), against [GATE_UP_PARAMS] - the launch code in [crate::routed_gate_up_launch]
//!   fills the argument array in exactly this order;
//! * the kernel's constants, against [crate::routed_gate_up_ref]'s (the CPU twin that defines the
//!   arithmetic);
//! * the geometry static_asserts in the kernel, each evaluated through [Consts];
//! * the cubin pin from baseline/kernel_baseline.txt - bytes, sha256 and PTX membar count, measured
//!   by the pinned toolkit (nvcc V13.0.88, CUDA 13.0.3, sm_121a).
//!
//! The numbers here are measurements of the pinned toolkit, not arithmetic: they are re-derived by
//! crates/sparkflash-kernels/scripts/check-kernel-baseline.sh, which tests/kernel_baseline.rs runs
//! on a CUDA host.

use crate::c_decl::{self, Consts, KernelDecl};

/// The kernel symbol, which is also the cubin stem and the manifest key.
pub const GATE_UP_KERNEL: &str = "routed_gate_up_sm121";

/// The kernel source, as the parser and the pins see it.
pub const GATE_UP_SOURCE: &str = include_str!("../../../kernels/routed_gate_up_sm121.cu");

/// The shared M4 header the kernel includes; parsed after the kernel so its SPARK_M4_* names resolve.
const COMMON_HEADER: &str = include_str!("../../../kernels/m4_common_sm121.cuh");

/// The hidden size the GEMM reduces (config.json hidden_size).
pub const HIDDEN: u32 = 5120;
/// The routed width per rank before padding (moe_inter_dim 2304 / four ranks).
pub const WIDTH: u32 = 576;
/// The routed width the kernel walks, padding included: the pinned SwiGLU routed shape.
pub const WIDTH_PADDED: u32 = 640;
/// Routed experts selected per token (config.json num_experts_per_tok).
pub const TOP_K: u32 = 6;
/// MX block width: one FP8 or FP4 scale per 32 values along K.
pub const QUANT_BLOCK: u32 = 32;
/// MX blocks per hidden row: 5120 / 32.
pub const K_BLOCKS: u32 = HIDDEN / QUANT_BLOCK;
/// Threads per block: one compute thread per (column, K group), one reducer per column.
pub const THREADS: u32 = 256;
/// Columns one block covers (the reducer threads are one per column).
pub const TILE: u32 = 32;
/// Blocks along the output tile axis: ceil(640 / 32).
pub const TILES: u32 = WIDTH_PADDED.div_ceil(TILE);
/// Packed FP4 bytes per output row: 5120 / 2.
pub const PACKED_ROW_BYTES: u32 = HIDDEN / 2;
/// ue8m0 scale bytes per activation row and per weight row: 5120 / 32.
pub const SCALE_ROW_BYTES: u32 = K_BLOCKS;

/// The argument array a launch fills, in words: six pointers, tokens, trace ring, layer id, step.
pub const ARG_WORDS: usize = 10;

/// Pinned cubin size, measured with nvcc V13.0.88 (CUDA 13.0.3) for sm_121a.
pub const CUBIN_BYTES: usize = 35_656;
/// Pinned cubin sha256, from the same measurement.
pub const CUBIN_SHA256: &str = "b108f7fa1a21623e876f17a8e2da5634257a6745e1d2f1233cc7d325c4a360f8";
/// PTX membar lines in the pinned PTX: the kernel emits none (no fence, no acquire).
pub const CUBIN_MEMBAR_LINES: usize = 0;
/// The toolkit the pin was measured with.
pub const TOOLKIT_RELEASE: &str = "V13.0.88";

/// The kernel's parameters, in declaration order: (name, base type, is a pointer, pointee const).
pub const GATE_UP_PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("act_fp8", "uint8_t", true, true),
    ("act_scales", "uint8_t", true, true),
    ("w_fp4", "uint8_t", true, true),
    ("w_scales", "uint8_t", true, true),
    ("expert_ids", "uint32_t", true, true),
    ("out", "__nv_bfloat16", true, false),
    ("tokens", "uint32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// Parses the gate/up declaration out of the kernel source.
#[must_use]
pub fn declaration() -> KernelDecl {
    c_decl::kernel_declaration(GATE_UP_SOURCE, GATE_UP_KERNEL)
}

/// The kernel's constants with the shared M4 header's names resolvable.
#[must_use]
pub fn constants() -> Consts {
    let mut table = Consts::parse(GATE_UP_SOURCE);
    table.extend(COMMON_HEADER);
    table
}

/// Whether a parsed declaration has [GATE_UP_PARAMS]' shape and order. Used by the mutation test,
/// which proves the comparison above is not vacuous.
#[must_use]
pub fn declaration_matches(decl: &KernelDecl) -> bool {
    decl.name == GATE_UP_KERNEL
        && decl.extern_c
        && decl.global
        && decl.params.len() == ARG_WORDS
        && decl.params.iter().zip(GATE_UP_PARAMS).all(|(param, pin)| {
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
        assert!(!decl.params[6].pointer, "tokens is a value");
        assert!(decl.params[7].pointer && decl.params[7].base == "GpuTraceRing");
        assert!(!decl.params[8].pointer && !decl.params[9].pointer);
    }

    #[test]
    fn the_declaration_pin_is_not_vacuous() {
        // Swap the two weight operands' pointee constness the way a careless ABI edit would; the
        // pin must see it. (A row-major/column-major swap cannot be seen by the declaration, which
        // is why the twin's layout contract covers the strides.)
        let mutated = GATE_UP_SOURCE.replace(
            "    const uint8_t *__restrict__ w_fp4,",
            "    uint8_t *__restrict__ w_fp4,",
        );
        assert_ne!(mutated, GATE_UP_SOURCE, "the mutation must apply");
        let decl = c_decl::kernel_declaration(&mutated, GATE_UP_KERNEL);
        assert!(
            !declaration_matches(&decl),
            "a non-const weight operand must fail the ABI pin"
        );
    }

    #[test]
    fn the_kernel_constants_are_the_twin_constants() {
        let table = constants();
        assert_eq!(table.value("SPARK_M4_HIDDEN"), HIDDEN as i128);
        assert_eq!(table.value("SPARK_M4_MOE_WIDTH"), WIDTH as i128);
        assert_eq!(
            table.value("SPARK_M4_MOE_WIDTH_PADDED"),
            WIDTH_PADDED as i128
        );
        assert_eq!(table.value("SPARK_M4_MOE_TOP_K"), TOP_K as i128);
        assert_eq!(table.value("SPARK_M4_QUANT_BLOCK"), QUANT_BLOCK as i128);
        assert_eq!(table.value("SPARK_GATE_UP_K_BLOCKS"), K_BLOCKS as i128);
        assert_eq!(table.value("SPARK_GATE_UP_THREADS"), THREADS as i128);
        assert_eq!(table.value("SPARK_GATE_UP_TILE"), TILE as i128);
        assert_eq!(table.value("SPARK_GATE_UP_TILES"), TILES as i128);
        assert_eq!(
            table.value("SPARK_GATE_UP_PACKED_ROW"),
            PACKED_ROW_BYTES as i128
        );
        assert_eq!(
            table.value("SPARK_GATE_UP_SCALE_ROW"),
            SCALE_ROW_BYTES as i128
        );
        assert_eq!(crate::routed_gate_up_ref::HIDDEN, HIDDEN as usize);
        assert_eq!(crate::routed_gate_up_ref::WIDTH, WIDTH as usize);
        assert_eq!(
            crate::routed_gate_up_ref::WIDTH_PADDED,
            WIDTH_PADDED as usize
        );
        assert_eq!(crate::routed_gate_up_ref::TOP_K, TOP_K as usize);
        assert_eq!(crate::routed_gate_up_ref::K_BLOCKS, K_BLOCKS as usize);
        assert_eq!(
            crate::routed_gate_up_ref::WEIGHT_ROW_BYTES,
            PACKED_ROW_BYTES as usize
        );
        assert_eq!(
            crate::routed_gate_up_ref::SCALE_ROW_BYTES,
            SCALE_ROW_BYTES as usize
        );
    }

    #[test]
    fn every_static_assert_names_a_project_symbol_and_evaluates_true() {
        let table = constants();
        let asserts = c_decl::static_asserts(GATE_UP_SOURCE);
        assert!(
            asserts.len() >= 8,
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
    fn the_manifest_pins_the_gate_up_cubin() {
        let line = BASELINE
            .lines()
            .find(|line| line.starts_with("kernel routed_gate_up_sm121 "))
            .expect("the baseline manifest must name the gate/up kernel");
        let expected = format!(
            "kernel routed_gate_up_sm121 cubin_bytes={CUBIN_BYTES} cubin_sha256={CUBIN_SHA256} ptx_membar_lines={CUBIN_MEMBAR_LINES}"
        );
        assert_eq!(
            line.trim(),
            expected,
            "the manifest entry must be the pinned cubin"
        );
        // The header count and toolkit are part of the same claim: this is one of twenty-one
        // entries.
        assert!(
            BASELINE.contains("forty-two baseline kernels"),
            "the manifest header must say forty-two"
        );
        assert!(BASELINE.contains(&format!("nvcc_release={TOOLKIT_RELEASE}")));
        assert!(BASELINE.contains("arch=sm_121a"));
    }
}

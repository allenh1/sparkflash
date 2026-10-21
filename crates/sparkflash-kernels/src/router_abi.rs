//! The ABI and constant pins of the device MoE router (kernels/moe_router_sm121.cu).
//!
//! This is a separate module from m4_abi.rs on purpose: that module is frozen by the D26 size
//! ratchet and its KERNELS array is the six Milestone 4 kernels plus the M7 SwiGLU epilogue, so a
//! new, non-M4 kernel is pinned here instead of growing it. Nothing in this module is read by the
//! engine; it exists so that the kernel's signature, its released constants, its geometry asserts
//! and its pinned cubin cannot drift without a test failing.
//!
//! What is pinned and against what:
//!
//! * the kernel declaration, parsed out of the kernel source by [crate::c_decl] (the same parser
//!   m4_abi uses), against [ROUTER_PARAMS] - the launch code in [crate::router_launch] fills the
//!   argument array in exactly this order;
//! * the released constants, against [crate::router_ref]'s (the CPU twin that is itself compared
//!   bit for bit to the engine's host path in that module's tests);
//! * the geometry static_asserts in the kernel, each evaluated through [Consts];
//! * the cubin pin from baseline/kernel_baseline.txt - bytes, sha256 and PTX membar count, measured
//!   by the pinned toolkit (nvcc V13.0.88, CUDA 13.0.3, sm_121a).
//!
//! The numbers here are measurements of the pinned toolkit, not arithmetic: they are re-derived by
//! scripts/check-kernel-baseline.sh, which tests/kernel_baseline.rs runs on a CUDA host.

use crate::c_decl::{self, Consts, KernelDecl};

/// The stage-one kernel symbol, which is also the cubin stem and the manifest key.
pub const ROUTER_KERNEL: &str = "moe_router_sm121";
/// The stage-two symbol: a second kernel in the same cubin, not a manifest key of its own.
pub const TOPK_KERNEL: &str = "moe_router_topk_sm121";

/// The kernel source, as the parser and the pins see it.
pub const ROUTER_SOURCE: &str = include_str!("../../../kernels/moe_router_sm121.cu");

/// The shared M4 header the kernel includes; parsed after the kernel so its SPARK_M4_* names resolve.
const COMMON_HEADER: &str = include_str!("../../../kernels/m4_common_sm121.cuh");

/// Experts the released router selects among (config.json n_routed_experts).
pub const N_ROUTED: u32 = 384;
/// Hidden values per token and per gate row (config.json hidden_size).
pub const HIDDEN: u32 = 5120;
/// Experts activated per token (config.json num_experts_per_tok).
pub const TOP_K: u32 = 6;
/// Tokens one stage-one thread block routes (one token per active lane).
pub const TOKENS_PER_BLOCK: u32 = 8;
/// Stage-one threads per block (eight warps of 32 lanes).
pub const THREADS: u32 = 256;
/// The warp width: lanes 0..7 each own one token's serial reduction.
pub const LANES: u32 = 32;
/// Warps per stage-one block: one warp owns one expert.
pub const WARPS: u32 = THREADS / LANES;
/// Consecutive experts one stage-one block owns (one per warp).
pub const EXPERTS_PER_BLOCK: u32 = WARPS;
/// The MX block size the hidden width divides into (the stage-one loop itself is serial).
pub const K_TILE: u32 = 32;
/// Stage-two threads per block (one token per thread).
pub const TOPK_THREADS: u32 = 256;
/// Gate temperature (config.json does not carry it; the released default is 1.0).
pub const GATE_TEMP: f32 = 1.0;
/// PyTorch F.softplus's threshold (model.py:816).
pub const SOFTPLUS_THRESHOLD: f32 = 20.0;
/// The norm_topk_prob denominator floor (model.py:825).
pub const NORM_EPS: f32 = 1e-20;
/// The routed scaling factor (model.py:826).
pub const ROUTE_SCALE: f32 = 1.5;

/// The stage-one argument array, in words: six pointers, tokens, trace ring, layer id, step.
pub const ARG_WORDS: usize = 10;
/// The stage-two argument array, in words: four pointers, tokens, trace ring, layer id, step.
pub const TOPK_ARG_WORDS: usize = 8;

/// Pinned cubin size, measured with nvcc V13.0.88 (CUDA 13.0.3) for sm_121a.
pub const CUBIN_BYTES: usize = 27_376;
/// Pinned cubin sha256, from the same measurement.
pub const CUBIN_SHA256: &str = "3b8823e84158e2956b19509e90f6aed02dbe68235c1a7f5d76ab438c51e53eb0";
/// PTX membar lines in the pinned PTX: the kernel emits none (no fence, no acquire).
pub const CUBIN_MEMBAR_LINES: usize = 0;
/// The toolkit the pin was measured with.
pub const TOOLKIT_RELEASE: &str = "V13.0.88";

/// The kernel's parameters, in declaration order: (name, base type, is a pointer, pointee const).
pub const ROUTER_PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("x", "__nv_bfloat16", true, true),
    ("gate", "__nv_bfloat16", true, true),
    ("bias", "float", true, true),
    ("ids", "uint32_t", true, false),
    ("weights", "float", true, false),
    ("scores", "float", true, false),
    ("tokens", "uint32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// Parses the router declaration out of the kernel source.
#[must_use]
pub fn declaration() -> KernelDecl {
    c_decl::kernel_declaration(ROUTER_SOURCE, ROUTER_KERNEL)
}

/// The kernel's constants with the shared M4 header's names resolvable.
#[must_use]
pub fn constants() -> Consts {
    let mut table = Consts::parse(ROUTER_SOURCE);
    table.extend(COMMON_HEADER);
    table
}

/// The stage-two kernel's parameters, in declaration order.
pub const TOPK_PARAMS: [(&str, &str, bool, bool); TOPK_ARG_WORDS] = [
    ("bias", "float", true, true),
    ("scores", "float", true, true),
    ("ids", "uint32_t", true, false),
    ("weights", "float", true, false),
    ("tokens", "uint32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// Parses the stage-two declaration out of the kernel source.
#[must_use]
pub fn topk_declaration() -> KernelDecl {
    c_decl::kernel_declaration(ROUTER_SOURCE, TOPK_KERNEL)
}

/// Whether a parsed declaration has [TOPK_PARAMS]' shape and order.
#[must_use]
pub fn topk_declaration_matches(decl: &KernelDecl) -> bool {
    decl.name == TOPK_KERNEL
        && decl.extern_c
        && decl.global
        && decl.params.len() == TOPK_ARG_WORDS
        && decl.params.iter().zip(TOPK_PARAMS).all(|(param, pin)| {
            param.name == pin.0
                && param.base == pin.1
                && param.pointer == pin.2
                && param.const_pointee == pin.3
                && (!param.pointer || param.restrict)
        })
}

/// Whether a parsed declaration has [ROUTER_PARAMS]' shape and order. Used by the mutation test,
/// which proves the comparison above is not vacuous.
#[must_use]
pub fn declaration_matches(decl: &KernelDecl) -> bool {
    decl.name == ROUTER_KERNEL
        && decl.extern_c
        && decl.global
        && decl.params.len() == ARG_WORDS
        && decl.params.iter().zip(ROUTER_PARAMS).all(|(param, pin)| {
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
        // The two non-pointer words at the end are the launch's scalar parameters; the trace ring
        // sits between them, which is the order the trace macros and the launcher both assume.
        assert!(!decl.params[6].pointer, "tokens is a value");
        assert!(decl.params[7].pointer, "trace_ring is a pointer");
        assert_eq!(decl.params[7].base, "GpuTraceRing");
        assert!(!decl.params[8].pointer && !decl.params[9].pointer);
    }

    #[test]
    fn the_stage_two_declaration_is_the_pinned_one() {
        let decl = topk_declaration();
        assert!(
            topk_declaration_matches(&decl),
            "the stage-two declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), TOPK_ARG_WORDS);
        assert!(!decl.params[4].pointer, "tokens is a value");
        assert!(decl.params[5].pointer, "trace_ring is a pointer");
        assert_eq!(decl.params[5].base, "GpuTraceRing");
        assert!(!decl.params[6].pointer && !decl.params[7].pointer);
    }

    #[test]
    fn the_declaration_pin_is_not_vacuous() {
        // Swap two adjacent parameter types the way a careless ABI edit would; the pin must see it.
        let mutated = ROUTER_SOURCE.replacen(
            "    uint32_t *__restrict__ ids,                 // [tokens][top_k] selected expert ids (stage 2)\n    float *__restrict__ weights,                // [tokens][top_k] normalised weights (stage 2)",
            "    float *__restrict__ ids,                    // [tokens][top_k] selected expert ids (stage 2)\n    uint32_t *__restrict__ weights,             // [tokens][top_k] normalised weights (stage 2)",
            1,
        );
        assert_ne!(mutated, ROUTER_SOURCE, "the mutation must apply");
        let decl = c_decl::kernel_declaration(&mutated, ROUTER_KERNEL);
        assert!(
            !declaration_matches(&decl),
            "a swapped ids/weights type must fail the ABI pin"
        );
    }

    #[test]
    fn the_kernel_constants_are_the_released_ones() {
        let table = constants();
        assert_eq!(table.value("SPARK_ROUTER_N_ROUTED"), N_ROUTED as i128);
        assert_eq!(table.value("SPARK_ROUTER_HIDDEN"), HIDDEN as i128);
        assert_eq!(table.value("SPARK_ROUTER_TOP_K"), TOP_K as i128);
        assert_eq!(
            table.value("SPARK_ROUTER_TOKENS_PER_BLOCK"),
            TOKENS_PER_BLOCK as i128
        );
        assert_eq!(table.value("SPARK_ROUTER_THREADS"), THREADS as i128);
        assert_eq!(table.value("SPARK_ROUTER_LANES"), LANES as i128);
        assert_eq!(table.value("SPARK_ROUTER_WARPS"), WARPS as i128);
        assert_eq!(
            table.value("SPARK_ROUTER_EXPERTS_PER_BLOCK"),
            EXPERTS_PER_BLOCK as i128
        );
        assert_eq!(
            table.value("SPARK_ROUTER_TOPK_THREADS"),
            TOPK_THREADS as i128
        );
        assert_eq!(table.value("SPARK_ROUTER_K_TILE"), K_TILE as i128);
        assert_eq!(
            table.value("SPARK_ROUTER_K_TILE"),
            table.value("SPARK_M4_QUANT_BLOCK")
        );
        // The released constants are spelled in the source as exactly these literals; the CPU twin
        // in crate::router_ref uses the same values, and its released test checks the config.
        assert_eq!(table.raw("SPARK_ROUTER_GATE_TEMP"), Some("1.0f"));
        assert_eq!(table.raw("SPARK_ROUTER_SOFTPLUS_THRESHOLD"), Some("20.0f"));
        assert_eq!(table.raw("SPARK_ROUTER_NORM_EPS"), Some("1e-20f"));
        assert_eq!(table.raw("SPARK_ROUTER_ROUTE_SCALE"), Some("1.5f"));
        assert_eq!(crate::router_ref::N_ROUTED as u32, N_ROUTED);
        assert_eq!(crate::router_ref::HIDDEN as u32, HIDDEN);
        assert_eq!(crate::router_ref::TOP_K as u32, TOP_K);
        assert_eq!(crate::router_ref::GATE_TEMP, GATE_TEMP);
        assert_eq!(crate::router_ref::SOFTPLUS_THRESHOLD, SOFTPLUS_THRESHOLD);
        assert_eq!(crate::router_ref::NORM_EPS, NORM_EPS);
        assert_eq!(crate::router_ref::ROUTE_SCALE, ROUTE_SCALE);
    }

    #[test]
    fn every_static_assert_names_a_project_symbol_and_evaluates_true() {
        let table = constants();
        let asserts = c_decl::static_asserts(ROUTER_SOURCE);
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
    fn the_manifest_pins_the_router_cubin() {
        let line = BASELINE
            .lines()
            .find(|line| line.starts_with("kernel moe_router_sm121 "))
            .expect("the baseline manifest must name the router kernel");
        let expected = format!(
            "kernel moe_router_sm121 cubin_bytes={CUBIN_BYTES} cubin_sha256={CUBIN_SHA256} ptx_membar_lines={CUBIN_MEMBAR_LINES}"
        );
        assert_eq!(
            line.trim(),
            expected,
            "the manifest entry must be the pinned cubin"
        );
        // The header count and toolkit are part of the same claim: twenty entries, this toolkit.
        assert!(
            BASELINE.contains("twenty baseline kernels"),
            "the manifest header must say twenty"
        );
        assert!(BASELINE.contains(&format!("nvcc_release={TOOLKIT_RELEASE}")));
        assert!(BASELINE.contains("arch=sm_121a"));
    }
}

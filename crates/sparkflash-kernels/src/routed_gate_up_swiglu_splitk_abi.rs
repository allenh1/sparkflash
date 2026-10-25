//! The ABI, geometry and pin checklist of the split-K grouped gate/up kernel
//! (pending_kernels/routed_gate_up_swiglu_splitk_sm121.cu).
//!
//! This module is the new-`KernelId` row of the standing kernel rule for
//! `KernelId::RoutedGateUpSwigluSplitK`: the declaration pin, the typed parameter row, the arity
//! pin, the geometry constants (pinned against the CPU twin), the joined-signature row, the
//! baseline 20 -> 21 expectation and the documented pin checklist. It is deliberately NOT in
//! m4_abi.rs: that module family is frozen by D26 (its `KERNELS` table is the six Milestone 4
//! kernels and its source-text scanners are size-ratcheted), which is the same reason
//! router_abi.rs and routed_gate_up_abi.rs are separate modules. The joined-signature row below is
//! the m4_abi row in form (`m4_abi::SIGNATURES`' `(file, symbol, "params joined by '; '")`), placed
//! where a non-M4 kernel belongs.
//!
//! What is pinned and against what:
//!
//! * the kernel declaration, parsed out of the kernel source by [crate::c_decl] (the same parser
//!   m4_abi uses), against [PARAMS] - [crate::routed_gate_up_swiglu_splitk_launch::words] fills the
//!   argument array in exactly this order, and the arity pin is [ARG_WORDS];
//! * the joined declaration signature, against [SIGNATURE] (a qualifier edit is visible);
//! * the kernel's constants, against the CPU twin
//!   ([crate::routed_gate_up_swiglu_splitk_ref]) - the module that defines the arithmetic and the
//!   D51 tolerance row;
//! * the geometry `static_assert`s in the kernel, each evaluated through [Consts].
//!
//! What is deliberately NOT pinned here yet: the cubin. [PIN_RECORDED] is false, [CUBIN_BYTES] and
//! [CUBIN_SHA256] are zero/empty, and the kernel source lives under this crate's
//! `pending_kernels/` rather than the repository's `kernels/`. A pin is a measurement of the
//! compiled artifact (nvcc V13.0.88, CUDA 13.0.3, sm_121a in the sibling pins) and no CUDA host
//! was available to this branch, so the pin job is NOT run: [PIN_CHECKLIST] is the ordered list a
//! CUDA host executes, and `the_pin_is_recorded_before_the_manifest_names_it` fails if the manifest
//! half of that job is applied without the rest.
//!
//! Default-off (D51 amendment A, rule 5). The kernel is not wired into any chain by this crate: the
//! engine and the launch sites belong to the Lead's host-side writer. Even the typed launcher
//! refuses to run unless the process arms [crate::routed_gate_up_swiglu_splitk_launch::gate_armed],
//! so nothing can reach the new arithmetic by accident.

use crate::c_decl::{self, Consts, KernelDecl};

/// The kernel symbol, which is also the cubin stem and the future manifest key.
pub const KERNEL: &str = "routed_gate_up_swiglu_splitk_sm121";

/// The kernel source, as the parser and the pins see it.
///
/// The include path moves to `../../../kernels/` in the same commit that applies [PIN_CHECKLIST]
/// step 1; the path is the one thing the move changes in this module.
pub const SOURCE: &str = include_str!("../pending_kernels/routed_gate_up_swiglu_splitk_sm121.cu");

/// The shared M4 header the kernel includes; parsed after the kernel so its SPARK_M4_* names resolve.
const COMMON_HEADER: &str = include_str!("../../../kernels/m4_common_sm121.cuh");
/// The activation-quantiser header the fused epilogue includes (the released activation scale rule,
/// which is NOT the weight-grid rule in the common header).
const ACT_HEADER: &str = include_str!("../../../kernels/m4_activation_quant_sm121.cuh");

/// The hidden size both projections reduce (config.json hidden_size). Mirrors SPARK_M4_HIDDEN.
pub const HIDDEN: u32 = 5120;
/// The routed width per rank before padding (moe_inter_dim 2304 / four ranks).
pub const WIDTH: u32 = 576;
/// The padded routed width the activation operand is 640 wide over (D30).
pub const WIDTH_PADDED: u32 = 640;
/// Routed experts selected per token (config.json num_experts_per_tok).
pub const TOP_K: u32 = 6;
/// MX block width: one E4M3 or E2M1 scale per 32 values along K.
pub const QUANT_BLOCK: u32 = 32;
/// MX blocks per hidden row: 5120 / 32.
pub const K_BLOCKS: u32 = HIDDEN / QUANT_BLOCK;
/// Threads per block: eight warps.
pub const THREADS: u32 = 256;
/// Warps per block, and the number of K shards the split-K schedule creates.
pub const WARPS: u32 = 8;
/// K blocks one warp-shard folds: 160 / 8.
pub const BLOCKS_PER_SHARD: u32 = K_BLOCKS / WARPS;
/// Output columns one CTA owns: exactly one MX block.
pub const TILE: u32 = 32;
/// Blocks along the output tile axis: 640 / 32.
pub const TILES: u32 = WIDTH_PADDED / TILE;
/// Columns one warp owns in the shuffle fold: 32 / 8.
pub const COLS_PER_GROUP: u32 = TILE / WARPS;
/// Accumulators per (shard, column): gate and up for each of the six slots.
pub const ROWS: u32 = TOP_K * 2;
/// Packed FP4 bytes per output row: 5120 / 2.
pub const PACKED_ROW_BYTES: u32 = HIDDEN / 2;
/// ue8m0 scale bytes per activation row and per weight row: 5120 / 32.
pub const SCALE_ROW_BYTES: u32 = K_BLOCKS;
/// ue8m0 scale bytes per quantised activation row: 640 / 32.
pub const OUT_SCALE_ROW_BYTES: u32 = WIDTH_PADDED / QUANT_BLOCK;

/// The argument array a launch fills, in words: ten pointers, `tokens`, `limit`, trace ring,
/// layer id, step sequence. This is the arity pin.
pub const ARG_WORDS: usize = 15;

/// The activation-operand rows a token produces: one quantised row per routed slot.
pub const ROUTED_ROWS_PER_TOKEN: u32 = TOP_K;

/// The kernel's parameters, in declaration order: (name, base type, is a pointer, pointee const).
pub const PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("act_fp8", "uint8_t", true, true),
    ("act_scales", "uint8_t", true, true),
    ("w1_fp4", "uint8_t", true, true),
    ("w1_scales", "uint8_t", true, true),
    ("w3_fp4", "uint8_t", true, true),
    ("w3_scales", "uint8_t", true, true),
    ("expert_ids", "uint32_t", true, true),
    ("routing_weights", "float", true, true),
    ("out_fp8", "uint8_t", true, false),
    ("out_scales", "uint8_t", true, false),
    ("tokens", "uint32_t", false, false),
    ("limit", "float", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// The joined declaration signature, in [crate::m4_abi]'s `SIGNATURES` form: every parameter's
/// spelling with its name, joined by `"; "`. A dropped `__restrict__`, a widened scalar or a
/// swapped operand order is visible here even when the (type, name) row is not enough.
pub const SIGNATURE: &str = "const uint8_t *__restrict__ act_fp8; const uint8_t *__restrict__ act_scales; const uint8_t *__restrict__ w1_fp4; const uint8_t *__restrict__ w1_scales; const uint8_t *__restrict__ w3_fp4; const uint8_t *__restrict__ w3_scales; const uint32_t *__restrict__ expert_ids; const float *__restrict__ routing_weights; uint8_t *__restrict__ out_fp8; uint8_t *__restrict__ out_scales; const uint32_t tokens; const float limit; GpuTraceRing *__restrict__ trace_ring; const uint32_t layer_id; const uint64_t step_seq";

/// The number of pinned baseline kernels today (`baseline/kernel_baseline.txt`).
pub const BASELINE_ENTRIES_NOW: usize = 41;
/// The baseline entry count once this kernel is pinned: the standing rule's 20 -> 21 step.
pub const BASELINE_ENTRIES_AFTER_PIN: usize = BASELINE_ENTRIES_NOW + 1;

/// Whether the cubin pin has been recorded. False on this branch: the pin job needs a CUDA host.
pub const PIN_RECORDED: bool = false;
/// Pinned cubin size in bytes, from `scripts/regen-kernel-baseline.sh`. 0 = not measured yet.
pub const CUBIN_BYTES: usize = 0;
/// Pinned cubin sha256, lower-case hex. Empty = not measured yet.
pub const CUBIN_SHA256: &str = "";
/// PTX membar lines in the pinned PTX. 0 until measured; this kernel emits no fence.
pub const CUBIN_MEMBAR_LINES: usize = 0;

/// The ordered job that turns this pending module into a pinned kernel, on a CUDA host.
///
/// Every step is a separate artifact and the order matters: the manifest and the probe table are
/// compared by `tests/kernel_baseline.rs`, so a half-applied job fails there instead of leaving a
/// kernel that is pinned but never compiled, or compiled but never pinned.
pub const PIN_CHECKLIST: [&str; 8] = [
    "1. move the source: git mv crates/sparkflash-kernels/pending_kernels/routed_gate_up_swiglu_splitk_sm121.cu kernels/routed_gate_up_swiglu_splitk_sm121.cu, and change SOURCE's include path in this module to ../../../kernels/",
    "2. add (\"routed_gate_up_swiglu_splitk_sm121.cu\", \"routed_gate_up_swiglu_splitk_sm121\") to build.rs PROBED_KERNELS and raise its declared length 20 -> 21",
    "3. add routed_gate_up_swiglu_splitk_sm121 to the kernels= list in scripts/check-kernel-baseline.sh (20 -> 21)",
    "4. run scripts/regen-kernel-baseline.sh on the pinned toolkit, append the new kernel line to baseline/kernel_baseline.txt, and move the manifest's entry count to 23 (the header text that says twenty-two becomes twenty-three)",
    "5. fill CUBIN_BYTES, CUBIN_SHA256 and CUBIN_MEMBAR_LINES from that measurement and set PIN_RECORDED = true in the same commit - the_pin_is_recorded_before_the_manifest_names_it fails otherwise",
    "6. raise the expectations that count kernels: tests/kernel_baseline.rs (listed.len() 22 -> 23, the manifest/lookup count 19 -> 20) and this module's BASELINE_ENTRIES_NOW 22 -> 23",
    "7. re-run the gates (fmt, clippy -D warnings, cargo test --workspace, scripts tests, cargo check -p sparkflash-kernels) and record the cubin sha256 in the D49 pin row",
    "8. run GPU_VERIFICATION_PLAN on the device and record its rows next to the kernel's tolerance row; only then may a launch site arm the kernel (the Lead's host-side writer owns that edit)",
];

/// The device rows this branch could not produce. Every row is a measurement to record, not a
/// prediction; the CPU arm can produce only the tolerance and the traffic arithmetic.
pub const GPU_VERIFICATION_PLAN: [&str; 7] = [
    "1. stage the pinned cubin (after PIN_CHECKLIST) under the harness kernel dir and run the m10-shaped gate/up harness with SPARKFLASH_ENABLE_MOE_SPLITK=1; it must print the tolerance row (max/mean deviation against the bit-exact twin) before any timing, and the row must match the CPU arm's declared tolerance",
    "2. bit-exactness is NOT required of this kernel (D51 amendment A): the pass condition is the declared tolerance in routed_gate_up_swiglu_splitk_ref::DECLARED_TOLERANCE plus the D41 suite row, not the twin's sha",
    "3. occupancy sweep: TILE blocks per CTA (1, 2, 4 MX blocks) and a second grid axis over expert slots (grid (tokens, tiles, top_k) with one CTA per expert) - the 20-CTA geometry at TILE=1 is the launch-fusion arm, not the bandwidth arm; record each variant's achieved GB/s",
    "4. record achieved GB/s against the 273 GB/s per-node floor for the whole launch (traffic_bytes = the arguments' own bytes: 6 experts x 2 projections x 576 rows x 2560 B packed weights + 1,105,920 B of weight scales + the activation row + the two outputs), with the percentage, not the time alone",
    "5. compare against the incumbent 12-launch path on the same layer bytes: 8 x routed_gate_up_sm121 + 4 x swiglu_mxfp8_quant_weighted_sm121 device_us from the b68 ledger, and the 12-launch -> 4-launch count change",
    "6. probe run-to-run determinism (the shuffle fold is order-fixed, so two launches on the same inputs must produce the same codes and scales byte for byte) and record the sha of both outputs",
    "7. run the D41 oracle suite because D51 rule 4 rejects a perf merge that regresses it, and record the logits-bin sha row; a default-on wiring is allowed only after both rows exist (D51 amendment A, rule 5)",
];

/// Parses the kernel declaration out of the pending source.
///
/// # Panics
/// Panics when the declaration cannot be parsed, which is a defect the tests must not paper over.
#[must_use]
pub fn declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, KERNEL)
}

/// The kernel's constants with the shared headers' names resolvable.
#[must_use]
pub fn constants() -> Consts {
    let mut table = Consts::parse(SOURCE);
    table.extend(COMMON_HEADER);
    table.extend(ACT_HEADER);
    table
}

/// The declaration's parameters joined in [SIGNATURE]'s form.
#[must_use]
pub fn parsed_signature() -> String {
    c_decl::kernel_declaration(&c_decl::strip_comments(SOURCE), KERNEL)
        .params
        .iter()
        .map(|param| param.raw_type.as_str())
        .collect::<Vec<_>>()
        .join("; ")
}

/// Whether a parsed declaration has [PARAMS]' shape and order. Used by the mutation test, which
/// proves the comparison is not vacuous.
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
        assert_eq!(decl.params.len(), ARG_WORDS, "the arity pin");
        assert!(!decl.params[10].pointer, "tokens is a value");
        assert!(!decl.params[11].pointer, "limit is a value");
        assert!(decl.params[12].pointer && decl.params[12].base == "GpuTraceRing");
        assert!(!decl.params[13].pointer && !decl.params[14].pointer);
    }

    #[test]
    fn the_declaration_pin_is_not_vacuous() {
        // A dropped pointee const on a weight operand is what a careless ABI edit looks like.
        let mutated = SOURCE.replace(
            "    const uint8_t *__restrict__ w1_fp4,",
            "    uint8_t *__restrict__ w1_fp4,",
        );
        assert_ne!(mutated, SOURCE, "the mutation must apply");
        let decl = c_decl::kernel_declaration(&mutated, KERNEL);
        assert!(
            !declaration_matches(&decl),
            "a non-const weight operand must fail the ABI pin"
        );
    }

    #[test]
    fn the_joined_signature_row_is_the_declarations() {
        assert_eq!(
            parsed_signature(),
            SIGNATURE,
            "the m4_abi-style signature row must be the declaration's parameters"
        );
        // The signature row is not vacuous: dropping one __restrict__ changes it.
        let mutated = SOURCE.replace(
            "const uint32_t *__restrict__ expert_ids",
            "const uint32_t *expert_ids",
        );
        assert_ne!(mutated, SOURCE, "the mutation must apply");
        let joined = c_decl::kernel_declaration(&mutated, KERNEL)
            .params
            .iter()
            .map(|param| param.raw_type.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        assert_ne!(joined, SIGNATURE);
    }

    #[test]
    fn the_kernel_constants_are_the_twin_constants() {
        let table = constants();
        for (name, value) in [
            ("SPARK_M4_HIDDEN", HIDDEN),
            ("SPARK_M4_MOE_WIDTH", WIDTH),
            ("SPARK_M4_MOE_WIDTH_PADDED", WIDTH_PADDED),
            ("SPARK_M4_MOE_TOP_K", TOP_K),
            ("SPARK_M4_QUANT_BLOCK", QUANT_BLOCK),
            ("SPARK_SPLITK_K_BLOCKS", K_BLOCKS),
            ("SPARK_SPLITK_BLOCKS_PER_SHARD", BLOCKS_PER_SHARD),
            ("SPARK_SPLITK_TILE", TILE),
            ("SPARK_SPLITK_TILES", TILES),
            ("SPARK_SPLITK_WARPS", WARPS),
            ("SPARK_SPLITK_SHARDS", WARPS),
            ("SPARK_SPLITK_COLS_PER_GROUP", COLS_PER_GROUP),
            ("SPARK_SPLITK_ROWS", ROWS),
            ("SPARK_SPLITK_PACKED_ROW", PACKED_ROW_BYTES),
            ("SPARK_SPLITK_SCALE_ROW", SCALE_ROW_BYTES),
            ("SPARK_SPLITK_OUT_SCALE_ROW", OUT_SCALE_ROW_BYTES),
            ("SPARK_SPLITK_THREADS", THREADS),
        ] {
            assert_eq!(table.value(name), value as i128, "{name}");
        }
        use crate::routed_gate_up_swiglu_splitk_ref as twin;
        assert_eq!(twin::HIDDEN, HIDDEN as usize);
        assert_eq!(twin::WIDTH, WIDTH as usize);
        assert_eq!(twin::WIDTH_PADDED, WIDTH_PADDED as usize);
        assert_eq!(twin::TOP_K, TOP_K as usize);
        assert_eq!(twin::QUANT_BLOCK, QUANT_BLOCK as usize);
        assert_eq!(twin::K_BLOCKS, K_BLOCKS as usize);
        assert_eq!(twin::WARPS, WARPS as usize);
        assert_eq!(twin::BLOCKS_PER_SHARD, BLOCKS_PER_SHARD as usize);
        assert_eq!(twin::TILE, TILE as usize);
        assert_eq!(twin::TILES, TILES as usize);
        assert_eq!(twin::COLS_PER_GROUP, COLS_PER_GROUP as usize);
        assert_eq!(twin::OUT_SCALE_ROW_BYTES, OUT_SCALE_ROW_BYTES as usize);
        assert_eq!(twin::WEIGHT_ROW_BYTES, PACKED_ROW_BYTES as usize);
        assert_eq!(twin::SCALE_ROW_BYTES, SCALE_ROW_BYTES as usize);
    }

    #[test]
    fn every_static_assert_names_a_project_symbol_and_evaluates_true() {
        let table = constants();
        let asserts = c_decl::static_asserts(SOURCE);
        assert!(
            asserts.len() >= 12,
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

    /// The half-applied-pin guard, as a function of values a test reads at run time.
    ///
    /// It is a helper rather than an inline `assert!` so the compiler does not fold the constants:
    /// the point is the RELATION between the manifest (read from disk) and this module's pin fields,
    /// which must move in one commit. Splitting the pin across two commits would leave either a
    /// manifest entry with no artifact or a recorded pin the manifest cannot name.
    fn check_pin_and_manifest(
        manifest_names_kernel: bool,
        entry_count: usize,
        pin_recorded: bool,
        cubin_bytes: usize,
        cubin_sha256: &str,
        entries_now: usize,
        entries_after_pin: usize,
    ) -> Result<(), String> {
        if manifest_names_kernel {
            if !pin_recorded || cubin_bytes == 0 || cubin_sha256.len() != 64 {
                return Err(format!(
                    "the manifest names {KERNEL} but the pin fields are unset (PIN_RECORDED={pin_recorded} CUBIN_BYTES={cubin_bytes} sha_len={}): apply PIN_CHECKLIST steps 4-5 together",
                    cubin_sha256.len()
                ));
            }
            if entry_count != entries_after_pin {
                return Err(format!(
                    "the pin job adds one entry: manifest has {entry_count}, expected {entries_after_pin}"
                ));
            }
        } else {
            if pin_recorded {
                return Err(format!(
                    "PIN_RECORDED is true but the manifest does not pin {KERNEL}"
                ));
            }
            if cubin_bytes != 0 || !cubin_sha256.is_empty() {
                return Err("an unrecorded pin has no byte count and no digest".to_owned());
            }
            if entry_count != entries_now {
                return Err(format!(
                    "before the pin job the manifest holds {entries_now} entries, found {entry_count}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn the_pin_is_recorded_before_the_manifest_names_it() {
        // Read the manifest from disk, not through the compile-time include, so the relation below
        // is checked at run time against whatever the working tree holds.
        let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let manifest = std::fs::read_to_string(crate_dir.join("baseline/kernel_baseline.txt"))
            .expect("the baseline manifest must be readable");
        let named = manifest
            .lines()
            .any(|line| line.starts_with(&format!("kernel {KERNEL} ")));
        let entries = manifest
            .lines()
            .filter(|line| line.starts_with("kernel "))
            .count();
        check_pin_and_manifest(
            named,
            entries,
            PIN_RECORDED,
            CUBIN_BYTES,
            CUBIN_SHA256,
            BASELINE_ENTRIES_NOW,
            BASELINE_ENTRIES_AFTER_PIN,
        )
        .expect("the manifest entry and the pin fields move in one commit");
        // The compile-time copy and the working tree must be the same file.
        assert_eq!(BASELINE.lines().count(), manifest.lines().count());
        // The standing rule the checklist quotes: 41 entries today (the opt bulk-prefill MMA cubin is
        // the latest addition), 42 after the pin job.
        assert_eq!(BASELINE_ENTRIES_NOW, 41);
        assert_eq!(BASELINE_ENTRIES_AFTER_PIN, 42);
        // The pin job's other artifacts must agree with the manifest's state.
        let script = include_str!("../scripts/check-kernel-baseline.sh");
        let list = script
            .lines()
            .find(|line| line.starts_with("kernels="))
            .expect("the check script must name its kernel list");
        assert_eq!(
            list.contains(KERNEL),
            named,
            "the manifest and the check script's kernel list must be pinned in the same commit"
        );
        let build = include_str!("../build.rs");
        assert_eq!(
            build.contains(KERNEL),
            named,
            "the manifest and build.rs PROBED_KERNELS must be pinned in the same commit"
        );
    }

    #[test]
    fn the_pin_checklist_and_the_gpu_plan_name_every_step() {
        let checklist = PIN_CHECKLIST.join("\n");
        for needle in [
            "pending_kernels/",
            "build.rs PROBED_KERNELS",
            "check-kernel-baseline.sh",
            "regen-kernel-baseline.sh",
            "CUBIN_SHA256",
            "kernel_baseline.rs",
            "D49 pin row",
            "GPU_VERIFICATION_PLAN",
        ] {
            assert!(
                checklist.contains(needle),
                "the checklist must name {needle}: {checklist}"
            );
        }
        let plan = GPU_VERIFICATION_PLAN.join("\n");
        for needle in [
            "tolerance row",
            "273 GB/s",
            "occupancy sweep",
            "D41 oracle suite",
            "determinism",
        ] {
            assert!(
                plan.contains(needle),
                "the GPU plan must name {needle}: {plan}"
            );
        }
    }

    #[test]
    fn the_new_kernel_does_not_touch_the_frozen_flash_mla_entry() {
        // The 19-word flash_mla_csa2_seq_sm121 ABI is frozen: this task adds a NEW symbol and must
        // not mutate or reuse it.
        assert_ne!(KERNEL, crate::cuda_driver::FLASH_MLA_CUBIN);
        assert_eq!(
            crate::flash_mla_csa2_seq_launch::ARG_WORDS,
            19,
            "the frozen flash_mla sequence entry keeps its nineteen words"
        );
        assert_eq!(
            crate::m4_abi::KERNELS.len(),
            6,
            "the D26-frozen m4_abi table must not grow: a new kernel's row lives in its own module"
        );
        // And the new KernelId is a new variant, appended rather than substituted.
        use crate::cuda_driver::KernelId;
        assert_eq!(KernelId::ALL.len(), 20);
        assert_eq!(
            KernelId::ALL[19],
            KernelId::RoutedGateUpSwigluSplitK,
            "the new kernel is appended after the nineteen existing driver kernels"
        );
        assert_eq!(KernelId::ALL[12], KernelId::FlashMlaCsa2Seq);
        assert!(BASELINE.contains("kernel flash_mla_csa2_seq_sm121 "));
    }
}

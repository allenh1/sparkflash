//! The determinism contract (DECISIONS.md D21), checked instead of described.
//!
//! D21 records three statements - D-a kernel determinism, D-b reduction determinism, D-c no assumed
//! cross-boundary equality - and the M7 harness cites them from its printed output. A decision that
//! only lives in a document is a description; this test fails when either half disappears, so the
//! contract cannot rot into prose.

#[test]
fn the_determinism_contract_is_written_down_and_the_harness_cites_it() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let decisions = std::fs::read_to_string(root.join("docs/DECISIONS.md"))
        .expect("docs/DECISIONS.md must be readable from the crate");
    for statement in [
        "D-a: kernel determinism.",
        "D-b: reduction determinism.",
        "D-c: no assumed cross-boundary equality.",
    ] {
        assert!(
            decisions.contains(statement),
            "the determinism contract is missing this statement: {statement}"
        );
    }
    let harness =
        std::fs::read_to_string(root.join("crates/sparkflash-kernels/examples/m7_layer0.rs"))
            .expect("the M7 harness must be readable from the crate");
    for citation in [
        "D-a (kernel determinism)",
        "D-b (reduction determinism)",
        "D-c (no assumed cross-boundary equality)",
    ] {
        assert!(
            harness.contains(citation),
            "the harness does not cite the contract from its printed output: {citation}"
        );
    }
}

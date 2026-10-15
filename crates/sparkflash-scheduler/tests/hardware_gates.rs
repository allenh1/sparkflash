//! The Milestone 5 hardware gates, checked from outside the crate.
//!
//! The gates themselves live in `sparkflash_scheduler::gates`, where they share the run
//! summaries and the metric parser with the crate's other tests; this file only pins the public
//! surface an operator relies on when they run them on the target.

/// The gates are addressable from outside the crate, with their thresholds intact.
#[test]
fn the_gate_list_is_public_and_complete() {
    let gates = sparkflash_scheduler::gates::GATES;
    assert_eq!(gates.len(), 6, "Milestone 5 has six hardware gates");
    assert_eq!(sparkflash_scheduler::gates::QEVAL_MIN_CORRECT, 72);
    assert_eq!(sparkflash_scheduler::gates::QEVAL_TOTAL, 75);
    assert_eq!(
        sparkflash_scheduler::gates::NEEDLE_CONTEXT_TOKENS,
        1_048_576
    );
    assert_eq!(sparkflash_scheduler::gates::SPARKDASH_MAX_CONCURRENCY, 16);
    assert_eq!(sparkflash_scheduler::gates::CUDA_GRAPH_MAX_BATCH, 16);
    assert_eq!(sparkflash_scheduler::gates::TRACE_GATE_STEPS, 45);
    let summary = "correct=73 total=75";
    assert_eq!(
        sparkflash_scheduler::gates::parse_metric(summary, "correct"),
        Some(73.0)
    );
    for gate in gates {
        assert!(
            gate.needs.contains("needs") || gate.needs.len() > 10,
            "{} must say what it needs",
            gate.name
        );
    }
}

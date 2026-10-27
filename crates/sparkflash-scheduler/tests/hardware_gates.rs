// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

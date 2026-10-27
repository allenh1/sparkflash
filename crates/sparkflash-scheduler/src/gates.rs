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

//! The Milestone 5 hardware gates, and the thresholds they check.
//!
//! Every gate in this module needs the target runtime - weights, a GPU, the cluster or the
//! trace device - so every gate body is `#[ignore]`d and CI stays green on a workstation. An
//! operator runs them with:
//!
//! `SPARKFLASH_QEVAL_RESULTS="correct=73 total=75" cargo test -p sparkflash-scheduler -- --ignored`
//!
//! Each gate reads the summary its run produced from an environment variable instead of
//! inventing a number, so the gate checks the run that actually happened. The unignored tests
//! below pin the thresholds to the specification's numbers and exercise the summary parser, so
//! a drifting constant still fails where the gate itself cannot run.

use std::env;

/// qeval questions answered correctly for a pass.
pub const QEVAL_MIN_CORRECT: u32 = 72;
/// Questions in the qeval set.
pub const QEVAL_TOTAL: u32 = 75;
/// KL divergence ceiling for the DSpark acceptance run.
pub const DSPARK_KL_MAX: f64 = 0.007;
/// Context length of the needle gate.
pub const NEEDLE_CONTEXT_TOKENS: usize = 1_048_576;
/// Highest sparkDash concurrency the cluster gate drives.
pub const SPARKDASH_MAX_CONCURRENCY: u32 = 16;
/// Highest CUDA-graph batch the runtime gate replays.
pub const CUDA_GRAPH_MAX_BATCH: u32 = 16;
/// Steps the scoped-trace gate captures.
pub const TRACE_GATE_STEPS: u64 = 45;

/// Environment variable holding the qeval run summary.
pub const QEVAL_RESULTS_ENV: &str = "SPARKFLASH_QEVAL_RESULTS";
/// Environment variable holding the DSpark acceptance summary.
pub const DSPARK_RESULTS_ENV: &str = "SPARKFLASH_DSPARK_RESULTS";
/// Environment variable holding the long-context needle summary.
pub const NEEDLE_RESULTS_ENV: &str = "SPARKFLASH_NEEDLE_RESULTS";
/// Environment variable holding the sparkDash cluster summary.
pub const SPARKDASH_RESULTS_ENV: &str = "SPARKFLASH_SPARKDASH_RESULTS";
/// Environment variable holding the CUDA-graph replay summary.
pub const CUDA_GRAPH_RESULTS_ENV: &str = "SPARKFLASH_CUDA_GRAPH_RESULTS";
/// Environment variable holding the scoped-trace summary.
pub const TRACE_RESULTS_ENV: &str = "SPARKFLASH_TRACE_RESULTS";

/// One gate: what it proves and what it needs before it can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateSpec {
    /// Gate name, as the acceptance list calls it.
    pub name: &'static str,
    /// What a pass proves.
    pub proves: &'static str,
    /// What the gate needs that a workstation does not have.
    pub needs: &'static str,
}

/// The six hardware gates of Milestone 5.
pub const GATES: [GateSpec; 6] = [
    GateSpec {
        name: "qeval",
        proves: "quality does not regress: at least 72 of 75 questions",
        needs: "a qeval run over the weights",
    },
    GateSpec {
        name: "dspark-kl",
        proves: "the DSpark draft stays within 0.007 KL of the target distribution",
        needs: "an acceptance run over the verification model",
    },
    GateSpec {
        name: "needle-1m",
        proves: "a needle is found with a full 1M-token context",
        needs: "the long-context serve path and 1M tokens of KV",
    },
    GateSpec {
        name: "sparkdash-1-16",
        proves: "sparkDash holds concurrency 1 through 16",
        needs: "the four-node cluster",
    },
    GateSpec {
        name: "cuda-graph-1-16",
        proves: "the captured graph replays for batch 1 through 16",
        needs: "a CUDA device and the captured graph",
    },
    GateSpec {
        name: "trace-45-scopes",
        proves: "a 45-step scoped trace records every one of the 16 scopes",
        needs: "the trace device and the target runtime",
    },
];

/// Reads a run summary that a gate needs.
///
/// # Panics
/// Panics with the variable name and the expected format when the operator did not provide a
/// summary, which is the correct outcome for an ignored gate that was asked to run.
#[must_use]
pub fn run_summary(variable: &str) -> String {
    env::var(variable).unwrap_or_else(|_| {
        panic!("{variable} is not set; set it to the run's summary, e.g. correct=73 total=75")
    })
}

/// Parses a `key=value` metric out of a run summary.
///
/// Separators are whitespace, commas and newlines. Returns `None` when the key is absent or its
/// value is not a number.
#[must_use]
pub fn parse_metric(summary: &str, key: &str) -> Option<f64> {
    summary
        .split(|character: char| character.is_whitespace() || character == ',')
        .filter_map(|field| field.split_once('='))
        .find(|(name, _)| name.trim() == key)
        .and_then(|(_, value)| value.trim().parse::<f64>().ok())
}

/// A metric a gate needs: a missing key fails the gate rather than defaulting to a value that
/// cannot be told apart from a healthy run (review M5 A3).
///
/// # Panics
/// Panics, naming the key, when the summary does not report it.
#[must_use]
pub fn required_metric(summary: &str, key: &str) -> f64 {
    parse_metric(summary, key)
        .unwrap_or_else(|| panic!("the summary must report {key}=<value>, not: {summary}"))
}

/// A metric a gate needs that has to be a whole number of events.
///
/// # Panics
/// Panics when the key is missing, or when its value is not a whole non-negative number: a
/// fractional value used to be truncated by a cast, so `max_concurrency=16.9` passed a gate that
/// asked for 16 (review M5 A3).
#[must_use]
pub fn required_count(summary: &str, key: &str) -> u64 {
    let value = required_metric(summary, key);
    // 2^64 exactly, rather than u64::MAX as f64, which rounds up to the same value and would
    // let a saturated cast through.
    if value < 0.0 || value.fract() != 0.0 || value >= 18_446_744_073_709_551_616.0 {
        panic!("the summary must report {key}=<whole number>, not: {summary}");
    }
    value as u64
}

/// The qeval gate's numbers: questions correct and questions asked, both required.
#[must_use]
pub fn qeval_numbers(summary: &str) -> (u64, u64) {
    (
        required_count(summary, "correct"),
        required_count(summary, "total"),
    )
}

/// The sparkDash gate's numbers: the highest concurrency held and the requests that failed. The
/// failure count is required, because a run that never reported it cannot be told apart from one
/// that failed requests and stayed quiet.
#[must_use]
pub fn sparkdash_numbers(summary: &str) -> (u64, u64) {
    (
        required_count(summary, "max_concurrency"),
        required_count(summary, "failed_requests"),
    )
}

/// The CUDA graph gate's numbers: the highest batch replayed and the replays that disagreed.
#[must_use]
pub fn cuda_graph_numbers(summary: &str) -> (u64, u64) {
    (
        required_count(summary, "max_batch"),
        required_count(summary, "mismatches"),
    )
}

/// The needle gate's context length, which has to be exact.
#[must_use]
pub fn needle_context(summary: &str) -> u64 {
    required_count(summary, "context")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace_cli;

    /// Every gate declares what it proves and what it cannot run without.
    #[test]
    fn every_gate_declares_what_it_proves_and_needs() {
        assert_eq!(GATES.len(), 6);
        for gate in GATES {
            assert!(!gate.name.is_empty());
            assert!(
                gate.needs.len() > 10,
                "{} needs a real requirement",
                gate.name
            );
            assert!(gate.proves.len() > 10, "{} proves something", gate.name);
        }
        let names: Vec<&str> = GATES.iter().map(|gate| gate.name).collect();
        assert!(names.contains(&"qeval"));
        assert!(names.contains(&"trace-45-scopes"));
    }

    /// A drift guard, not a spec check: every assertion below compares a constant in this file
    /// to a literal in this file, so it catches an accidental edit of a threshold and nothing
    /// else. Whether the numbers match the specification is checked against docs/, once
    /// (review M5 test-quality).
    #[test]
    fn the_threshold_literals_do_not_drift() {
        assert_eq!(QEVAL_MIN_CORRECT, 72);
        assert_eq!(QEVAL_TOTAL, 75);
        assert!((DSPARK_KL_MAX - 0.007).abs() < 1e-12);
        assert_eq!(NEEDLE_CONTEXT_TOKENS, 1_048_576);
        assert_eq!(SPARKDASH_MAX_CONCURRENCY, 16);
        assert_eq!(CUDA_GRAPH_MAX_BATCH, 16);
        assert_eq!(TRACE_GATE_STEPS, trace_cli::GATE_STEPS);
        assert_eq!(TRACE_GATE_STEPS, 45);
    }

    /// The parser the gates rely on, exercised where CI can run it.
    #[test]
    fn a_metric_is_parsed_out_of_a_run_summary() {
        assert_eq!(parse_metric("correct=73 total=75", "correct"), Some(73.0));
        assert_eq!(parse_metric("correct=73,total=75", "total"), Some(75.0));
        assert_eq!(
            parse_metric(
                "kl=0.0062
steps=45",
                "kl"
            ),
            Some(0.0062)
        );
        assert_eq!(
            parse_metric(
                "steps=45
",
                "steps"
            ),
            Some(45.0)
        );
        assert_eq!(parse_metric("correct=73", "missing"), None);
        assert_eq!(parse_metric("correct=many", "correct"), None);
        assert_eq!(parse_metric("", "correct"), None);
    }

    /// The parsing a gate does before it asserts anything, exercised where CI can run it: the
    /// load-bearing negative evidence is required, and a fractional value is not a count. Both
    /// used to slip through, which let a partial summary pass six gates (review M5 A3).
    #[test]
    fn a_partial_or_fractional_summary_cannot_pass_a_gate() {
        assert_eq!(qeval_numbers("correct=72 total=75"), (72, 75));
        assert_eq!(
            sparkdash_numbers("max_concurrency=16 failed_requests=0"),
            (16, 0)
        );
        assert_eq!(cuda_graph_numbers("max_batch=16 mismatches=0"), (16, 0));
        assert_eq!(needle_context("context=1048576"), 1_048_576);
        assert!(
            std::panic::catch_unwind(|| qeval_numbers("correct=72")).is_err(),
            "a qeval summary without total must not pass on a default"
        );
        assert!(
            std::panic::catch_unwind(|| qeval_numbers("total=75")).is_err(),
            "a qeval summary without correct must not pass on a default"
        );
        assert!(
            std::panic::catch_unwind(|| sparkdash_numbers("max_concurrency=16")).is_err(),
            "a sparkDash summary without failed_requests must not pass on a default"
        );
        assert!(
            std::panic::catch_unwind(|| cuda_graph_numbers("max_batch=16")).is_err(),
            "a CUDA graph summary without mismatches must not pass on a default"
        );
        assert!(
            std::panic::catch_unwind(|| qeval_numbers("correct=72 total=75.5")).is_err(),
            "75.5 questions is not 75"
        );
        assert!(
            std::panic::catch_unwind(|| sparkdash_numbers(
                "max_concurrency=16.9 failed_requests=0"
            ))
            .is_err(),
            "concurrency 16.9 used to truncate to 16"
        );
        assert!(
            std::panic::catch_unwind(|| cuda_graph_numbers("max_batch=16.9 mismatches=0")).is_err(),
            "batch 16.9 used to truncate to 16"
        );
        assert!(
            std::panic::catch_unwind(|| needle_context("context=1048576.9")).is_err(),
            "a 1M context is 1048576 tokens exactly"
        );
    }

    #[ignore = "hardware gate: needs a qeval run; set SPARKFLASH_QEVAL_RESULTS"]
    #[test]
    fn qeval_scores_at_least_72_of_75() {
        let summary = run_summary(QEVAL_RESULTS_ENV);
        let (correct, total) = qeval_numbers(&summary);
        assert_eq!(
            total, QEVAL_TOTAL as u64,
            "the gate is over {QEVAL_TOTAL} questions, the summary says {total}"
        );
        assert!(
            correct >= u64::from(QEVAL_MIN_CORRECT),
            "qeval scored {correct}/{total}, the floor is {QEVAL_MIN_CORRECT}/{QEVAL_TOTAL}"
        );
    }

    #[ignore = "hardware gate: needs an acceptance run; set SPARKFLASH_DSPARK_RESULTS"]
    #[test]
    fn the_dspark_draft_stays_below_the_kl_ceiling() {
        let summary = run_summary(DSPARK_RESULTS_ENV);
        let kl = required_metric(&summary, "kl");
        // A negative KL is not a good run, it is a summary that measured something else: the
        // ceiling is a strict upper bound, so without this a report of kl=-1 would pass it.
        assert!(
            kl >= 0.0,
            "the DSpark draft reported {kl} KL, which cannot be a divergence"
        );
        assert!(
            kl < DSPARK_KL_MAX,
            "the DSpark draft diverged by {kl} KL, the ceiling is {DSPARK_KL_MAX}"
        );
    }

    #[ignore = "hardware gate: needs a 1M-token run; set SPARKFLASH_NEEDLE_RESULTS"]
    #[test]
    fn a_needle_is_found_at_one_million_tokens() {
        let summary = run_summary(NEEDLE_RESULTS_ENV);
        let context = needle_context(&summary);
        let found = required_metric(&summary, "found");
        assert_eq!(
            context, NEEDLE_CONTEXT_TOKENS as u64,
            "the gate runs at {NEEDLE_CONTEXT_TOKENS} tokens"
        );
        assert_eq!(found, 1.0, "the needle was not found at 1M tokens");
    }

    #[ignore = "hardware gate: needs the four-node cluster; set SPARKFLASH_SPARKDASH_RESULTS"]
    #[test]
    fn sparkdash_holds_concurrency_one_through_sixteen() {
        let summary = run_summary(SPARKDASH_RESULTS_ENV);
        let (highest, failed) = sparkdash_numbers(&summary);
        assert_eq!(
            highest, SPARKDASH_MAX_CONCURRENCY as u64,
            "the gate drives concurrency up to {SPARKDASH_MAX_CONCURRENCY}"
        );
        assert_eq!(failed, 0, "{failed} requests failed under load");
    }

    #[ignore = "hardware gate: needs a CUDA device; set SPARKFLASH_CUDA_GRAPH_RESULTS"]
    #[test]
    fn the_cuda_graph_replays_for_batch_one_through_sixteen() {
        let summary = run_summary(CUDA_GRAPH_RESULTS_ENV);
        let (highest, mismatches) = cuda_graph_numbers(&summary);
        assert_eq!(highest, CUDA_GRAPH_MAX_BATCH as u64);
        assert_eq!(
            mismatches, 0,
            "{mismatches} replays disagreed with the eager path"
        );
    }

    #[ignore = "hardware gate: needs the trace device; set SPARKFLASH_TRACE_RESULTS"]
    #[test]
    fn the_45_step_scoped_trace_records_every_scope() {
        let summary = run_summary(TRACE_RESULTS_ENV);
        let steps = required_count(&summary, "steps");
        let scopes = required_count(&summary, "scopes");
        assert_eq!(
            steps, TRACE_GATE_STEPS,
            "the gate captures {TRACE_GATE_STEPS} steps"
        );
        assert_eq!(
            scopes as usize,
            sparkflash_trace::TraceScope::COUNT,
            "every one of the 16 scopes must record at least one event"
        );
    }
}

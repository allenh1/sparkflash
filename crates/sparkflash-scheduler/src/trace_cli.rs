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

//! `sparkflash trace`: the CLI surface for the scoped trace capture.
//!
//! The specification asks for a scoped capture of "a specific step count or request ID", and
//! the qualification gate in the milestone text is the 45-step capture of every scope:
//! `sparkflash trace --scopes all --steps 45`. This module owns the argument parsing, the
//! scope-mask resolution and the rendering, so the command is testable without a GPU, a
//! cluster or a trace device. Only the plan is produced here; running the capture is a
//! hardware gate (see [crate::gates]).

use std::process::ExitCode;

use sparkflash_trace::TraceScope;

use crate::error::SchedulerError;

/// Scope name that selects every defined scope.
pub const ALL_SCOPES: &str = "all";

/// Steps the qualification gate captures.
pub const GATE_STEPS: u64 = 45;

/// Largest step count the CLI accepts.
///
/// The capture is a diagnostic and its ring is finite, so a runaway step count is refused
/// rather than silently truncated.
pub const MAX_TRACE_STEPS: u64 = 10_000;

/// The capture the CLI was asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceCliPlan {
    /// Scope mask to capture.
    pub scopes: u64,
    /// Step count to capture.
    pub steps: u64,
}

impl TraceCliPlan {
    /// The qualification gate's capture: every scope, 45 steps.
    #[must_use]
    pub const fn gate() -> Self {
        Self {
            scopes: TraceScope::ALL,
            steps: GATE_STEPS,
        }
    }

    /// True when every defined scope is captured.
    #[must_use]
    pub const fn captures_every_scope(&self) -> bool {
        self.scopes == TraceScope::ALL
    }

    /// Scope names in bit order.
    #[must_use]
    pub fn scope_names(&self) -> Vec<&'static str> {
        TraceScope::TABLE
            .iter()
            .filter(|(_, bit)| self.scopes & bit != 0)
            .map(|(name, _)| *name)
            .collect()
    }

    /// The human-readable plan the command prints.
    #[must_use]
    pub fn render(&self) -> String {
        let names = self.scope_names();
        format!(
            "trace plan: {} steps, {} scopes (mask 0x{:04X})\n  scopes: {}\n  capture: hardware gate (needs the target runtime and the trace device)",
            self.steps,
            names.len(),
            self.scopes,
            names.join(",")
        )
    }
}

/// Parses `--scopes` and `--steps`.
///
/// The defaults are the qualification gate: every scope and 45 steps.
///
/// # Errors
/// `SchedulerError::InvalidArgument` for an unknown scope name, a step count outside
/// `1..=MAX_TRACE_STEPS`, a missing value or an unknown flag.
pub fn parse(args: impl IntoIterator<Item = String>) -> Result<TraceCliPlan, SchedulerError> {
    let mut plan = TraceCliPlan::gate();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scopes" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--scopes needs a value"))?;
                plan.scopes = resolve_scopes(&value)?;
            }
            "--steps" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--steps needs a value"))?;
                let steps: u64 = value
                    .parse()
                    .map_err(|_| invalid(format!("--steps {value} is not a step count")))?;
                if steps == 0 || steps > MAX_TRACE_STEPS {
                    return Err(invalid(format!(
                        "--steps must be inside 1..={MAX_TRACE_STEPS}, got {steps}"
                    )));
                }
                plan.steps = steps;
            }
            other => return Err(invalid(format!("unknown trace argument {other}"))),
        }
    }
    Ok(plan)
}

/// Resolves `all` or a comma-separated list of scope names, case-insensitively.
///
/// # Errors
/// `SchedulerError::InvalidArgument` for a name that is not in `TraceScope::TABLE`.
pub fn resolve_scopes(value: &str) -> Result<u64, SchedulerError> {
    if value.eq_ignore_ascii_case(ALL_SCOPES) {
        return Ok(TraceScope::ALL);
    }
    let mut mask = 0u64;
    for part in value.split(',') {
        let part = part.trim();
        let (_, bit) = TraceScope::TABLE
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(part))
            .ok_or_else(|| {
                invalid(format!(
                    "{part} is not a trace scope; names: {}",
                    scope_name_list()
                ))
            })?;
        mask |= bit;
    }
    if mask == 0 {
        return Err(invalid("--scopes selected no scope".to_owned()));
    }
    Ok(mask)
}

/// Every scope name, for the error message.
#[must_use]
pub fn scope_name_list() -> String {
    TraceScope::TABLE
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(",")
}

fn invalid(message: impl Into<String>) -> SchedulerError {
    SchedulerError::InvalidArgument(message.into())
}

/// Runs `sparkflash trace` and returns its exit code.
pub fn run(args: impl IntoIterator<Item = String>) -> ExitCode {
    match parse(args) {
        Ok(plan) => {
            println!("{}", plan.render());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("sparkflash trace: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|value| (*value).to_owned()).collect()
    }

    /// The defaults are exactly the qualification gate.
    #[test]
    fn the_defaults_are_the_qualification_gate() {
        let plan = parse(args(&[])).expect("plan");
        assert_eq!(plan, TraceCliPlan::gate());
        assert_eq!(plan.steps, GATE_STEPS);
        assert!(plan.captures_every_scope());
        assert_eq!(plan.scope_names().len(), TraceScope::COUNT);
    }

    /// `all` and named subsets both resolve, whatever their case.
    #[test]
    fn all_and_named_scopes_resolve_case_insensitively() {
        let plan = parse(args(&["--scopes", "all", "--steps", "45"])).expect("plan");
        assert_eq!(plan.scopes, TraceScope::ALL);
        assert_eq!(plan.steps, 45);
        let plan = parse(args(&["--scopes", "sched_batch,moe_mxfp4"])).expect("plan");
        assert_eq!(plan.scopes, TraceScope::SCHED_BATCH | TraceScope::MOE_MXFP4);
        assert_eq!(plan.scope_names(), vec!["SCHED_BATCH", "MOE_MXFP4"]);
        assert_eq!(plan.steps, GATE_STEPS, "the step default survives");
    }

    /// The rendered plan is what the operator reads, so it has to name the scopes and the steps.
    #[test]
    fn a_rendered_plan_names_every_scope_and_the_step_count() {
        let text = TraceCliPlan::gate().render();
        for name in [
            "SCHED_BATCH",
            "DSPARK_DRAFT",
            "L2_PREFETCH",
            "ENGRAM_IO_URING",
        ] {
            assert!(text.contains(name), "{text}");
        }
        assert!(text.contains("45 steps"), "{text}");
        assert!(text.contains("0xFFFF"), "{text}");
        assert!(text.contains("hardware gate"), "{text}");
    }

    /// Every malformed argument is refused with a decision, not a panic.
    #[test]
    fn an_unknown_scope_or_step_count_is_refused() {
        assert!(parse(args(&["--scopes", "nope"])).is_err());
        assert!(parse(args(&["--scopes", ""])).is_err());
        assert!(parse(args(&["--scopes", "sched_batch,nope"])).is_err());
        assert!(parse(args(&["--steps", "0"])).is_err());
        assert!(parse(args(&["--steps", "10001"])).is_err());
        assert!(parse(args(&["--steps", "x"])).is_err());
        assert!(parse(args(&["--steps"])).is_err());
        assert!(parse(args(&["--scopes"])).is_err());
        assert!(parse(args(&["--verbose"])).is_err());
        assert!(
            parse(args(&["--steps", "10000"]))
                .expect("the largest accepted step count")
                .steps
                == MAX_TRACE_STEPS
        );
    }

    /// The command reports a bad argument as a failure exit code.
    #[test]
    fn run_reports_failure_for_a_bad_argument() {
        assert_eq!(run(args(&["--scopes", "nope"])), ExitCode::FAILURE);
        assert_eq!(run(args(&["--steps", "45"])), ExitCode::SUCCESS);
    }
}

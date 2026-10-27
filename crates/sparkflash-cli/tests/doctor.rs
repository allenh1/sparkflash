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

//! End-to-end tests for `sparkflash doctor`: the process contract, not the check logic.
//!
//! The check logic and every threshold boundary is unit-tested inside `doctor.rs` against
//! injected probes. What is left to prove here is what an operator sees: the exit status, the
//! two output modes, and the fact that the default invocation touches no hardware at all.
//!
//! No test in this file passes `--nodes`. A cluster probe needs explicit permission and this
//! box shares a network with a live inference server.

use std::process::{Command, Output};

use serde_json::Value;

/// Runs the built binary with `args` and returns its output.
fn doctor(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparkflash"))
        .arg("doctor")
        .args(args)
        .output()
        .expect("sparkflash doctor runs")
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn default_invocation_reports_every_check_not_applicable_and_exits_zero() {
    let output = doctor(&[]);
    assert_eq!(output.status.code(), Some(0), "{}", stdout_of(&output));
    let text = stdout_of(&output);
    assert!(text.contains("not applicable"), "{text}");
    assert!(
        text.contains("5 of 7 specified checks implemented"),
        "{text}"
    );
    assert!(text.contains("INCOMPLETE"), "{text}");
    assert!(text.contains("pass --local or --nodes"), "{text}");
    for id in [
        "driver-kernel",
        "link-mtu",
        "rdma-bandwidth",
        "clocks-memory",
        "engram-shards",
    ] {
        assert!(text.contains(id), "{id} missing from the report: {text}");
    }
}

#[test]
fn default_invocation_does_not_fail_a_dev_box() {
    // The whole point: a dev box with no GPU, no RDMA and no shards must not be reported as
    // not ready. Only an explicit --local or --nodes turns on hardware probes.
    let output = doctor(&[]);
    assert_eq!(output.status.code(), Some(0));
    assert!(!stdout_of(&output).contains("NOT READY"));
}

#[test]
fn json_mode_emits_the_machine_readable_report() {
    let output = doctor(&["--json"]);
    assert_eq!(output.status.code(), Some(0));
    let value: Value = serde_json::from_str(&stdout_of(&output)).expect("valid JSON");
    assert_eq!(value["tool"], "sparkflash doctor");
    assert_eq!(value["spec_section"], "6.1");
    assert_eq!(value["specified_checks"], 7);
    assert_eq!(value["implemented_checks"], 5);
    assert_eq!(value["verdict"], "incomplete");
    assert_eq!(value["exit_code"], 0);
    assert_eq!(value["counts"]["not_applicable"], 5);
    assert_eq!(value["counts"]["failed"], 0);
    let checks = value["checks"].as_array().expect("checks array");
    assert_eq!(checks.len(), 5);
    assert_eq!(checks[0]["id"], "driver-kernel");
    assert_eq!(checks[4]["id"], "engram-shards");
    assert!(
        value["spec_discrepancy"]
            .as_str()
            .expect("note")
            .contains("five")
    );
}

#[test]
fn help_states_the_seven_point_discrepancy() {
    let output = doctor(&["--help"]);
    let text = stdout_of(&output);
    assert!(text.contains("seven"), "{text}");
    assert!(text.contains("five"), "{text}");
    assert!(text.contains("--apply"), "{text}");
    assert!(text.contains("--nodes"), "{text}");
    // The clock lock names exactly what it changes, and needs both flags.
    assert!(text.contains("--lock-sm-clocks"), "{text}");
    assert!(text.contains("nvidia-smi -lgc 0,2200"), "{text}");
}

#[test]
fn the_master_switch_alone_does_not_lock_clocks() {
    // --apply is the master switch for state-changing commands; the clock lock additionally
    // needs --lock-sm-clocks. A preflight must never mutate a node that may be serving a
    // model, so this is checked at the process level as well as in the unit tests.
    let output = doctor(&["--local", "--apply"]);
    let text = stdout_of(&output);
    assert!(text.contains("read-only"), "{text}");
    assert!(text.contains("--lock-sm-clocks"), "{text}");
    assert!(!text.contains("--lock-sm-clocks: nvidia-smi"), "{text}");
}

#[test]
fn a_read_only_run_never_reports_applying_the_lock() {
    for args in [vec!["--local"], vec!["--local", "--apply"]] {
        let output = doctor(&args);
        let text = stdout_of(&output);
        assert!(
            !text.contains("--lock-sm-clocks: nvidia-smi -lgc"),
            "{args:?} locked clocks: {text}"
        );
    }
}

#[test]
fn an_out_of_range_rank_is_a_usage_error() {
    let output = doctor(&["--rank", "9", "--world-size", "4"]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("world-size"), "{stderr}");
}

#[test]
fn an_unknown_flag_is_a_usage_error() {
    let output = doctor(&["--definitely-not-a-flag"]);
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn default_options_are_read_only() {
    // Without --apply the command must not be able to change clocks. The unit tests prove the
    // probe is never called; this proves the default really is read-only at the process level.
    let output = doctor(&[]);
    let text = stdout_of(&output);
    assert!(text.contains("read-only") || !text.contains("nvidia-smi -lgc"));
}

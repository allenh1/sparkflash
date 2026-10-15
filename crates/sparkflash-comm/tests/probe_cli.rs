//! Binary-level contract tests for rocenante-probe.
//!
//! These drive the compiled binary the doctor shells out to: the doctor's exact invocation, the
//! refusal to print any number when there is no fabric to measure, and a single-object JSON on
//! stdout. None of them needs a device.

use std::process::{Command, Output};

use serde_json::Value;

const PROBE: &str = env!("CARGO_BIN_EXE_rocenante-probe");

fn run_probe(args: &[&str]) -> Output {
    Command::new(PROBE)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("spawning {PROBE}: {error}"))
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn help_lists_every_flag_the_doctor_uses() {
    let output = run_probe(&["--help"]);
    assert!(output.status.success(), "{}", stderr_of(&output));
    let help = stdout_of(&output);
    for flag in [
        "--bytes",
        "--peer",
        "--json",
        "--samples",
        "--warmup",
        "--transport",
    ] {
        assert!(help.contains(flag), "{flag} is missing from:\n{help}");
    }
}

#[test]
fn a_missing_peer_is_a_usage_error_with_empty_stdout() {
    let output = run_probe(&["--bytes", "10240", "--json"]);
    assert!(
        !output.status.success(),
        "the fabric transport needs a peer"
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(
        stdout_of(&output).is_empty(),
        "a usage error must not put anything on stdout"
    );
    assert!(
        stderr_of(&output).contains("--peer"),
        "{}",
        stderr_of(&output)
    );
}

#[cfg(not(feature = "rdma"))]
#[test]
fn without_the_fabric_feature_the_probe_refuses_and_prints_no_number() {
    let output = run_probe(&["--bytes", "10240", "--peer", "127.0.0.1", "--json"]);
    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(3));
    let stdout = stdout_of(&output);
    assert!(
        !stdout.contains("oneshot_allreduce_us") && stdout.is_empty(),
        "a build with no fabric path must print no number: {stdout}"
    );
    assert!(
        stderr_of(&output).contains("--features rdma"),
        "the refusal must name the rebuild: {}",
        stderr_of(&output)
    );
}

#[cfg(feature = "rdma")]
#[test]
fn the_rdma_path_refuses_rather_than_inventing_a_latency() {
    let output = run_probe(&["--bytes", "10240", "--peer", "127.0.0.1", "--json"]);
    let stderr = stderr_of(&output);
    assert_eq!(
        output.status.code(),
        Some(3),
        "the fabric refusal is a measurement-unavailable exit, never a generic failure: {stderr}"
    );
    assert!(
        stdout_of(&output).is_empty(),
        "a refused measurement must not put a number on stdout"
    );
    assert!(
        stderr.contains("refused to open"),
        "the refusal must say the fabric path refused: {stderr}"
    );
    assert!(
        stderr.contains("backend"),
        "the refusal must name the missing backend prerequisite, not just fail: {stderr}"
    );
}

#[test]
fn loopback_json_is_a_single_object_the_doctor_can_parse() {
    let output = run_probe(&[
        "--transport",
        "loopback",
        "--world",
        "4",
        "--bytes",
        "1024",
        "--samples",
        "2",
        "--warmup",
        "1",
        "--json",
    ]);
    assert!(output.status.success(), "{}", stderr_of(&output));
    let stdout = stdout_of(&output);
    let trimmed = stdout.trim();
    assert!(
        trimmed.starts_with('{') && trimmed.ends_with('}'),
        "{stdout}"
    );
    assert_eq!(
        trimmed.lines().count(),
        1,
        "the doctor parses stdout as one JSON document: {stdout}"
    );
    let value: Value =
        serde_json::from_str(trimmed).unwrap_or_else(|error| panic!("stdout is not JSON: {error}"));
    let micros = value
        .get("oneshot_allreduce_us")
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("the doctor's field is missing: {stdout}"));
    assert!(micros > 0.0, "{stdout}");
    assert_eq!(value["samples"], 2);
    assert_eq!(value["fabric"], false);
    assert_eq!(value["transport"], "loopback");
}

#[test]
fn loopback_human_output_states_the_statistic_and_the_sample_count() {
    let output = run_probe(&[
        "--transport",
        "loopback",
        "--world",
        "4",
        "--bytes",
        "1024",
        "--samples",
        "2",
        "--warmup",
        "1",
    ]);
    assert!(output.status.success(), "{}", stderr_of(&output));
    let stdout = stdout_of(&output);
    assert!(stdout.contains("median"), "{stdout}");
    assert!(stdout.contains("2 samples"), "{stdout}");
    assert!(stdout.contains("loopback"), "{stdout}");
}

#[test]
fn a_failed_run_never_prints_json_even_when_json_was_asked_for() {
    let output = run_probe(&["--transport", "loopback", "--world", "1", "--json"]);
    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(2));
    assert!(
        stdout_of(&output).is_empty(),
        "an invalid request must not print JSON"
    );
}

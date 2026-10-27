#!/usr/bin/env python3
# Copyright (c) 2026 Hunter L. Allen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the host gate set and write one machine-readable artifact per run.

The host gate set is the part of .gitlab-ci.yml that needs no GPU, no checkpoint, no RDMA fabric
and no network: fmt, check, clippy, rustdoc, the D26 freeze ratchet, the m4_abi split proof, the
workspace tests, the trace no-default tests, the example-harness tests, and the feature sweep. This
script runs exactly those commands, in order, and records what happened:

  * the exact argv and cwd of every gate, plus the environment overrides used, so the run can be
    reproduced from the artifact alone;
  * each gate's exit code and wall time, with one full log per gate beside the JSON;
  * the suite counts parsed from libtest's "test result:" lines;
  * the toolchain strings (rustc, cargo, rustc -vV host/release/LLVM, rustfmt, python3, uname) and
    the git sha/subject/dirty state the run was taken at.

Nothing depends on an agent's shell history: if it is not in the JSON, it did not affect the run.

Usage:
  python3 ci/host-gates.py                       # every gate, artifact under target/gate-artifacts
  python3 ci/host-gates.py --only fmt,check      # a subset (names from --list)
  python3 ci/host-gates.py --out /tmp/gates      # somewhere else
  python3 ci/host-gates.py --list                # names, commands and env overrides, no run

Exit code: 0 when every selected gate exits 0, 1 when any does not. The artifact is written either
way, because a failed run is exactly the record a batch report needs.

The artifact is a JSON object:
  kind, schema_version, generated_at_utc, cwd, argv, duration_s,
  git {sha, short, subject, dirty, status},
  toolchain {rustc, cargo, rustc_verbose {release, host, llvm}, rustfmt, python, uname},
  gates [ {name, argv, env, exit_code, duration_s, suites {passed, failed, ignored, measured,
           filtered_out, test_binaries}, log, output_tail} ],
  verdict, failures
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "target" / "gate-artifacts"
SUITE_RE = re.compile(
    r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; "
    r"(\d+) filtered out;",
    re.M,
)

GATES = [
    {
        "name": "fmt",
        "argv": ["cargo", "fmt", "--all", "--", "--check"],
        "env": {},
    },
    {
        "name": "check",
        "argv": ["cargo", "check", "--workspace", "--all-targets"],
        "env": {},
    },
    {
        "name": "clippy",
        "argv": ["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"],
        "env": {},
    },
    {
        "name": "rustdoc",
        "argv": ["cargo", "doc", "--workspace", "--no-deps"],
        "env": {"RUSTDOCFLAGS": "-D missing_docs"},
    },
    {
        "name": "m4-abi-freeze",
        "argv": ["sh", "ci/check-m4-abi-size.sh"],
        "env": {},
    },
    {
        "name": "m4-abi-split",
        "argv": ["python3", "ci/verify-m4-abi-split.py"],
        "env": {},
    },
    {
        "name": "test",
        "argv": ["cargo", "test", "--workspace"],
        "env": {},
    },
    {
        "name": "trace-no-default",
        "argv": ["cargo", "test", "-p", "sparkflash-trace", "--no-default-features"],
        "env": {},
    },
    {
        "name": "test-examples",
        "argv": ["cargo", "test", "-p", "sparkflash-kernels", "--examples"],
        "env": {},
    },
    {
        "name": "features-self-test",
        "argv": ["python3", "ci/test_check_features.py"],
        "env": {},
    },
    {
        "name": "features-sweep",
        "argv": ["python3", "ci/check_features.py"],
        "env": {},
    },
]


def capture(argv):
    try:
        result = subprocess.run(argv, cwd=str(REPO_ROOT), capture_output=True, text=True)
    except OSError as error:
        return None, str(error)
    return result.stdout.strip(), result.stderr.strip()


def toolchain():
    rustc, rustc_error = capture(["rustc", "--version"])
    cargo, cargo_error = capture(["cargo", "--version"])
    verbose, verbose_error = capture(["rustc", "-vV"])
    rustfmt, _ = capture(["rustfmt", "--version"])
    python, _ = capture(["python3", "--version"])
    uname, _ = capture(["uname", "-srm"])
    verbose_fields = {}
    for line in (verbose or "").splitlines():
        key, _, value = line.partition(":")
        if key in ("release", "host", "LLVM version"):
            verbose_fields[key.replace(" ", "_").lower()] = value.strip()
    return {
        "rustc": rustc,
        "cargo": cargo,
        "rustc_verbose": verbose_fields,
        "rustfmt": rustfmt,
        "python": python,
        "uname": uname,
        "errors": {
            "rustc": rustc_error,
            "cargo": cargo_error,
            "rustc_vv": verbose_error,
        },
    }


def git_state():
    def one(*argv):
        value, _ = capture(["git", *argv])
        return value

    status = one("status", "--porcelain")
    lines = [line for line in status.splitlines() if line.strip()]
    return {
        "sha": one("rev-parse", "HEAD"),
        "short": one("rev-parse", "--short", "HEAD"),
        "branch": one("rev-parse", "--abbrev-ref", "HEAD"),
        "subject": one("log", "-1", "--format=%s"),
        "dirty": bool(lines),
        "status": lines,
    }


def parse_suites(text):
    counts = {"passed": 0, "failed": 0, "ignored": 0, "measured": 0, "filtered_out": 0, "test_binaries": 0}
    for match in SUITE_RE.finditer(text):
        counts["test_binaries"] += 1
        for index, key in enumerate(("passed", "failed", "ignored", "measured", "filtered_out"), start=1):
            counts[key] += int(match.group(index))
    return counts


def run_gate(gate, artifact_path, logs_dir):
    env = dict(os.environ)
    env.update(gate.get("env", {}))
    started = time.monotonic()
    try:
        result = subprocess.run(
            gate["argv"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
    except OSError as error:
        stdout, stderr, exit_code = "", str(error), 127
    duration = round(time.monotonic() - started, 3)
    log_path = logs_dir / (gate["name"] + ".log")
    log_path.write_text(
        "$ " + " ".join(gate["argv"]) + "\n"
        + "cwd: " + str(REPO_ROOT) + "\n"
        + "env: " + json.dumps(gate.get("env", {}), sort_keys=True) + "\n"
        + "exit: " + str(exit_code) + "\n"
        + "duration_s: " + str(duration) + "\n\n"
        + "--- stdout ---\n" + stdout + "\n\n--- stderr ---\n" + stderr + "\n"
    )
    combined = stdout + "\n" + stderr
    tail = [line for line in combined.splitlines() if line.strip()][-20:]
    return {
        "name": gate["name"],
        "argv": gate["argv"],
        "env": gate.get("env", {}),
        "exit_code": exit_code,
        "duration_s": duration,
        "suites": parse_suites(combined),
        "log": str(log_path.relative_to(REPO_ROOT)) if log_path.is_relative_to(REPO_ROOT) else str(log_path),
        "output_tail": tail,
    }


def diff_artifacts(old_path, new_path):
    old = json.loads(Path(old_path).read_text())
    new = json.loads(Path(new_path).read_text())
    print("old: " + old.get("generated_at_utc", "?") + " " + (old.get("git", {}).get("sha") or "?") + " " + old.get("verdict", "?"))
    print("new: " + new.get("generated_at_utc", "?") + " " + (new.get("git", {}).get("sha") or "?") + " " + new.get("verdict", "?"))
    changed = 0
    old_gates = {gate["name"]: gate for gate in old.get("gates", [])}
    new_gates = {gate["name"]: gate for gate in new.get("gates", [])}
    for name in list(old_gates) + [entry for entry in new_gates if entry not in old_gates]:
        before = old_gates.get(name)
        after = new_gates.get(name)
        if before is None or after is None:
            print("gate present in only one artifact: " + name)
            changed += 1
            continue
        if before["exit_code"] != after["exit_code"]:
            print("changed: " + name + " exit " + str(before["exit_code"]) + " -> " + str(after["exit_code"]))
            changed += 1
        before_counts = before.get("suites", {})
        after_counts = after.get("suites", {})
        keys = sorted(set(before_counts) | set(after_counts))
        delta = [key + " " + str(before_counts.get(key)) + " -> " + str(after_counts.get(key)) for key in keys if before_counts.get(key) != after_counts.get(key)]
        if delta:
            print("changed: " + name + " suites " + ", ".join(delta))
            changed += 1
    old_toolchain = old.get("toolchain", {})
    new_toolchain = new.get("toolchain", {})
    for key in ("rustc", "cargo", "rustfmt"):
        if old_toolchain.get(key) != new_toolchain.get(key):
            print("toolchain changed: " + key + " " + str(old_toolchain.get(key)) + " -> " + str(new_toolchain.get(key)))
    if changed:
        print(str(changed) + " gate difference(s)")
        return 1
    print("no gate exit-code or suite-count difference")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="artifact directory (default: target/gate-artifacts)")
    parser.add_argument("--only", default="", help="comma-separated gate names; default is every gate")
    parser.add_argument("--list", action="store_true", help="print the gate names and commands, then exit")
    parser.add_argument("--diff", nargs=2, metavar=("OLD", "NEW"), default=None, help="compare two artifacts instead of running gates")
    args = parser.parse_args()

    if args.diff:
        return diff_artifacts(args.diff[0], args.diff[1])

    if args.list:
        for gate in GATES:
            overrides = " ".join(k + "=" + v for k, v in gate.get("env", {}).items())
            print(gate["name"] + ": " + " ".join(gate["argv"]) + ((" [" + overrides + "]") if overrides else ""))
        return 0

    wanted = [name for name in args.only.split(",") if name]
    known = {gate["name"] for gate in GATES}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        sys.exit("error: unknown gate(s): " + ", ".join(unknown) + " (--list shows the names)")
    selected = [gate for gate in GATES if not wanted or gate["name"] in wanted]

    started = time.monotonic()
    generated = datetime.datetime.now(datetime.timezone.utc)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    git = git_state()
    stem = generated.strftime("%Y%m%dT%H%M%SZ") + "-" + (git.get("short") or "unknown")
    logs_dir = out_dir / (stem + "-logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "kind": "sparkflash-host-gates",
        "schema_version": 1,
        "generated_at_utc": generated.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": None,
        "cwd": str(REPO_ROOT),
        "argv": sys.argv,
        "git": git,
        "toolchain": toolchain(),
        "gates": [],
    }
    failures = []
    for gate in selected:
        print("running " + gate["name"] + "...", flush=True)
        entry = run_gate(gate, out_dir / (stem + ".json"), logs_dir)
        record["gates"].append(entry)
        status = "ok" if entry["exit_code"] == 0 else "FAILED"
        print(
            "  {}: {} ({:.1f}s, {} test binaries, {} passed, {} failed)".format(
                gate["name"],
                status,
                entry["duration_s"],
                entry["suites"]["test_binaries"],
                entry["suites"]["passed"],
                entry["suites"]["failed"],
            ),
            flush=True,
        )
        if entry["exit_code"] != 0:
            failures.append(gate["name"])

    record["duration_s"] = round(time.monotonic() - started, 3)
    record["verdict"] = "pass" if not failures else "fail"
    record["failures"] = failures
    artifact_path = out_dir / (stem + ".json")
    artifact_path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
    print("")
    print("verdict: " + record["verdict"] + ("; failed: " + ", ".join(failures) if failures else ""))
    print("artifact: " + str(artifact_path))
    print("logs:     " + str(logs_dir))
    print("git:      " + (git.get("sha") or "unknown") + (" (dirty)" if git.get("dirty") else ""))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

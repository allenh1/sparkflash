# Machine-readable host-gate artifacts

Status: added for task-27 (backlog: gate-artifact JSON). The CI gate is unchanged; this is the
record producer the batch reports and the future release gate read instead of prose.

## Why

A batch report used to be a sentence like "gates green: fmt, clippy, workspace tests, 87 test
binaries". That cannot be diffed, cannot be replayed, and does not say which toolchain produced it.
`ci/host-gates.py` runs the host-safe part of `.gitlab-ci.yml` and writes one JSON artifact per run:

* the exact `argv`, working directory and environment overrides of every gate, so the run is
  reproducible from the artifact alone;
* each gate's exit code and wall time, with one full log per gate next to the JSON;
* the suite counts parsed from libtest's `test result:` lines;
* the toolchain strings (`rustc`, `cargo`, `rustc -vV` host/release/LLVM, `rustfmt`, `python3`,
  `uname`) and the git sha/subject/dirty state the run was taken at.

Nothing that only exists in an agent's shell history is part of the verdict. If a value is not in
the JSON, it did not affect the run.

## How to run

```sh
python3 ci/host-gates.py                         # every gate, artifact under target/gate-artifacts/
python3 ci/host-gates.py --only fmt,check,clippy # a subset
python3 ci/host-gates.py --out /tmp/gates        # somewhere else
python3 ci/host-gates.py --list                  # names, commands and env overrides, no run
python3 ci/host-gates.py --diff OLD.json NEW.json # compare two artifacts
```

Exit code is 0 when every selected gate exits 0 and 1 when any does not. The JSON is written either
way: a failed run is exactly the record a batch report needs.

The default gate set is the host-safe part of the pipeline: `fmt`, `check`, `clippy`, `rustdoc`,
`m4-abi-freeze`, `m4-abi-split`, `test`, `trace-no-default`, `test-examples`, `features-self-test`,
`features-sweep`. The CUDA job is not included because it needs a toolkit and a GPU-tagged runner;
no gate in this set touches the network, a node, the GPU or a checkpoint.

## Artifact layout and schema

```text
target/gate-artifacts/
  20261019T214714Z-0639413.json          # the artifact
  20261019T214714Z-0639413-logs/         # one full log per gate
    fmt.log
    test.log
    ...
```

`target/` is git-ignored: artifacts are evidence for a batch, not source. The `gate-artifact` CI job
(manual, `.gitlab-ci.yml`) runs the same script and keeps the directory as a pipeline artifact for
90 days.

The JSON object (schema_version 1):

| field | meaning |
| --- | --- |
| `kind`, `schema_version` | `sparkflash-host-gates`, 1 |
| `generated_at_utc`, `duration_s` | when the run started and how long it took |
| `cwd`, `argv` | the working directory and the exact invocation |
| `git` | `sha`, `short`, `branch`, `subject`, `dirty`, `status` (porcelain lines) |
| `toolchain` | `rustc`, `cargo`, `rustc_verbose` (release/host/llvm), `rustfmt`, `python`, `uname` |
| `gates` | one entry per gate: `name`, `argv`, `env`, `exit_code`, `duration_s`, `suites`, `log`, `output_tail` |
| `verdict`, `failures` | `pass`/`fail` and the failing gate names |

`suites` sums libtest across the gate's test binaries: `passed`, `failed`, `ignored`, `measured`,
`filtered_out`, `test_binaries`.

## Reproducing and diffing a run

To replay a record: check out the recorded `git.sha`, `cd` to the recorded `cwd`, run the recorded
`argv` with the recorded `env` overrides. The recorded toolchain strings say whether the replay can
match: a different rustc can change a warning or a test count.

A batch record should be taken on a clean tree. The JSON still records `dirty` and the porcelain
`status` lines when it is not, so an artifact cannot silently claim a sha it was not built from.

To compare two batches (or two release candidates):

```sh
python3 ci/host-gates.py --diff old.json new.json
```

which prints per-gate exit-code and suite-count changes, toolchain changes, and exits 0 only when
no gate's verdict or counts moved. That is the machine-readable version of the prose line a batch
report used to carry.

## Verification of the emitter itself

Run on 2026-10-19 in the task-27 worktree: the pass path (`--only fmt,m4-abi-freeze,m4-abi-split`)
wrote an artifact with `verdict: pass`; the fail path (the freeze pin temporarily off by one) wrote
`verdict: fail`, `failures: [m4-abi-freeze]`, exit code 1, with the gate's full log; and
`--diff` reported the `m4-abi-freeze` exit `0 -> 1` change and exited 1. The full gate set was then
run once on the task-27 tip and recorded as `target/gate-artifacts/20261019T*-*.json`.

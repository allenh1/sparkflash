# Qualification runbook — the Milestone 5 gates and the All-Reduce latency gate

This document is the executable procedure for the correctness and throughput gates of this build:
the All-Reduce latency gate (spec lines 1323 and 1336) and the qualification and capture gates of
Milestone 5 (spec line 1340). It replaces no test. Every gate below already exists in the tree as an
ignored test or as a check inside `sparkflash doctor`; this file says what to run, what the run
proves, what a pass looks like as a number with a unit, and what has to be kept as evidence.

Two rules, both learned in this repository:

1. **A gate is not evidence until its command has been run and its artifact kept.** An ignored test is
   a plan; a passing ignored test is a measurement. Five of the six Milestone 5 gates read their
   numbers from an environment variable instead of measuring them (crates/sparkflash-scheduler/src/gates.rs:95-99),
   so the cargo log alone does **not** contain the measurement: the artifact column below always names
   the raw run output as well.
2. **A gate with no instrument is a finding, not a paragraph.** Where no command in this tree can
   produce the number a gate checks, the gate's own section says so and section 5 collects them.

State vocabulary: **never run** means no command has ever executed the gate against real hardware;
**run and failed** and **run and passed** mean exactly that, on the date given.

State as of 2026-10-15. The cluster is occupied by another instance, no gate below has been executed,
and every cluster value quoted in section 2 was measured read-only on 2026-10-15 and is carried
forward from that earlier reading rather than re-measured for this document.

---

## 1. State today

| Gate | Spec | Instrument in the merged tree | State today |
| :--- | :--- | :--- | :--- |
| 10 KiB one-shot All-Reduce latency | 1336 (20 µs requirement), 1323 (22 µs preflight threshold) | `sparkflash doctor` check 3, which shells out to `rocenante-probe` (merged, labels checked, refuses rather than inventing a number); the crate's own test is a panic stub | **never run** (the instrument exists but the RDMA backend is a skeleton, so it can only refuse; see 3.1) |
| Bit-identical collectives over 10,000 replays | 1336 | crates/sparkflash-comm/tests/hardware_gates.rs:52 is a panic stub; only the CPU reference side runs in CI | **never run** (no instrument; see 3.2) |
| Dual-rail counters carry half the bytes each | 190, 1336 | crates/sparkflash-comm/tests/hardware_gates.rs:62 is a panic stub; doctor measures per-rail Gb/s instead | **never run** (no instrument; see 3.3) |
| qeval 72/75 | 1340 | gates.rs:293 + `SPARKFLASH_QEVAL_RESULTS` | **never run** (no producer; see 3.4) |
| DSpark KL < 0.007 | 1340 | gates.rs:308 + `SPARKFLASH_DSPARK_RESULTS` | **never run** (no producer; see 3.5) |
| 1M-token needle | 1340 | gates.rs:325 + `SPARKFLASH_NEEDLE_RESULTS` | **never run** (no producer; see 3.6) |
| sparkDash c = 1..16 | 1340 | gates.rs:338 + `SPARKFLASH_SPARKDASH_RESULTS` | **never run** (no producer; see 3.7) |
| CUDA graph replay, batch 1..16 | 1340, 1336 | gates.rs:350 + `SPARKFLASH_CUDA_GRAPH_RESULTS` | **never run** (no producer and no graph-capture code; see 3.8) |
| 45-step scoped trace, every scope | 1340 | `sparkflash trace` planner (trace_cli.rs) + gates.rs:362 | **never run** (planner only, no capture runner; see 3.9) |
| Preflight, checks 1-5 | 1321-1325 | `sparkflash doctor` (real probes) | **measured read-only 2026-10-15**, not through this runbook: 3 checks measured failing (driver, MTU, memory), a 4th cannot pass (the Engram shards are not packed here: the packer exists but has not been run against the checkpoint), and check 3 has no instrument (section 2) |

What has actually run: CI (.gitlab-ci.yml:52) executes `cargo test --workspace`, which excludes every
ignored test, so the gates themselves never run in CI. The tests that do run in CI are the threshold
drift guards and the summary parser in crates/sparkflash-scheduler/src/gates.rs:183-289, the CPU
collective reference in crates/sparkflash-comm/tests/reduction_bits.rs, and the doctor check logic
against injected probes (crates/sparkflash-cli/tests/doctor.rs). Those pin constants and parsers; none
of them touches hardware, and none of them is one of the gates below.

---

## 2. Preconditions

Nothing in section 3 may be run until all of this holds. Each item names what it is checked by, so a
reader can see which precondition a future run skipped.

### 2.1 The four nodes and their addresses

- The four ranks are addressed as `192.0.2.80, .84, .68, .59` (crates/sparkflash-comm/tests/hardware_gates.rs:18).
- Both rails share `192.0.2.0/24` and each node holds **two** addresses, one per rail; the head node
  is `192.0.2.80` on rail 1 and `192.0.2.85` on rail 2 (docs/DECISIONS.md D12).
- Rail 1 / rail 2 address pairs for all four nodes, measured read-only on 2026-10-18 (Lead; this is a
  configuration listing, not a gate): node0 `.80`/`.85`, node1 `.84`/`.88`, node2
  `.64`/`.68`, node3 `.59`/`.63`. The full table, the ACTIVE f1 device names and the NCCL
  environment of the deployed reference are in `docs/build/rdma-backend.md`. Recovering these was a
  precondition; recording them is not the same as a dual-rail run, and no dual-rail gate has used
  them yet (see 3.1, where one address for `--nodes` drives both rails' probes).
- The deployed vLLM reference is **single-rail** (`NCCL_IB_HCA=rocep1s0f1`, read from the live
  container on 2026-10-18), while our gates target the dual-rail aggregate. Do not compare a
  single-rail reference number with the dual-rail bar (`docs/build/rdma-backend.md`, section 3).

### 2.2 Permission and an exclusive window

- The cluster gates refuse to run without an explicit per-run variable:
  `SPARK_COMM_HW_PERMISSION=1` (hardware_gates.rs:10-21). Without it they panic before touching the
  fabric, so a mis-typed `--ignored` cannot reach the cluster quietly.
- The cluster is **currently occupied by another instance**, which is why this runbook has not been
  executed: the last read-only measurement found `MemAvailable 5 GiB` on a node whose floor is 114 GiB
  (docs/DECISIONS.md D12; measured 2026-10-15). Any throughput or latency number taken while another
  tenant is resident is not attributable to this build. Every gate in section 3 needs an exclusive
  window; the preflight's `clocks-memory` check is only meaningful while the node is idle.

### 2.3 Version and resource floors: required versus measured

| Requirement | Required | Source | Measured 2026-10-15 (D12) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| NVIDIA driver | >= 580.178.04; 580.142 is rejected | crates/sparkflash-cli/src/doctor.rs:64, 71; spec 1321 | 580.173.02 | **FAIL** |
| Kernel | >= 6.17.0-1018, or 7.0.0-1019-nvidia-64k | doctor.rs:74-83; spec 1321 | 6.17.0-1029-nvidia | pass |
| `nvidia-uvm` 64K page packing | active | spec 1321 | not verifiable: no parameter of that meaning among the module's 56 parameters | unknown |
| Rail link speed | both rails ACTIVE at 200000 Mb/s | doctor.rs:86; spec 1322 | the f1 siblings are ACTIVE at 200 Gb/s; the f0 functions the spec names are DOWN at 40 Gb/s | **FAIL as written**, pass with siblings (doctor.rs:1694-1745) |
| Rail MTU | 9000, proven by a 8972-byte do-not-fragment ping | doctor.rs:89, 92; spec 1322 | 1500 | **FAIL** |
| Aggregate RDMA bandwidth | >= 180 Gbps over both rails | doctor.rs:95; spec 1323 | not measured | never run |
| 10 KiB one-shot All-Reduce | < 22 µs (doctor) or < 20 µs (spec 1336) | doctor.rs:98-101 | not measured; the probe binary is not on PATH | never run |
| SM clock cap | [0, 2200] MHz | doctor.rs:104-107; spec 1324 | not measured | never run |
| Available memory | `MemAvailable >= 114 GiB` per node | doctor.rs:110; spec 1324 | 5 GiB (other tenant resident) | **FAIL** |
| Engram shards | `engram-l1-r{rank}of4.bin`, `engram-l14-r{rank}of4.bin` plus manifest, 512-byte aligned, ~48 GiB per node | doctor.rs:113-119, 2079-2135; spec 1325 | absent; `sparkflash pack-engram` exists and reads the checkpoint's safetensors shards (crates/sparkflash-cli/src/main.rs:42, crates/sparkflash-engram/src/packer.rs) but has not been run against the checkpoint, so no shard has been written | **FAIL** |

Section 2.4's second command is the one that produces this table; the "measured" column above is the
2026-10-15 reading, not a run of that command.

### 2.4 The preflight command

```bash
cargo build --release -p sparkflash-cli
./target/release/sparkflash doctor --local --json
./target/release/sparkflash doctor --nodes 192.0.2.80,192.0.2.84,192.0.2.68,192.0.2.59 --rank 0 --world-size 4 --rdma-seconds 1 --json
```

- Flags are the ones the binary parses: `--nodes`, `--json`, `--apply`, `--lock-sm-clocks`,
  `--local`, `--engram-dir`, `--rank`, `--world-size`, `--rdma-seconds`, `--skip-hash`
  (doctor.rs:1442-1476). `--local` probes one node; `--nodes` turns on the cluster checks.
- **Pass criterion: the report's `verdict` is `"ready"`**, meaning every check ran and passed
  (doctor.rs:1221-1281). The process exit code is **not** sufficient: `Verdict::Incomplete` (nothing
  failed, at least one check did not run) also exits 0, and only `Verdict::NotReady` exits 1
  (doctor.rs:1244-1252, 1270-1271).
- `--apply` and `--lock-sm-clocks` are the only mutating flags and both are refused while a node may
  be serving traffic (doctor.rs:1449-1457). Preflight is read-only: do not pass them during this
  runbook.
- Check 3 needs `ibv_write_bw` (perftest, still an external package) and `rocenante-probe` on PATH.
  The probe is a binary target of this workspace now - build it with `--features rdma` - and doctor
  fails the check loudly rather than pretending it ran when either command is missing.
- The RDMA probe uses only the **first** node in `--nodes` (doctor.rs:743-745) and runs one
  `ibv_write_bw` per local rail against it, so the "aggregate" it reports is two rails to one peer, not
  a four-rank ring (doctor.rs:742-769). The same is true of the All-Reduce probe (doctor.rs:773-775).

### 2.5 What the preflight cannot tell you

Specification 6.1 is titled "Seven-Point DGX Spark Preflight Verification" (spec line 1317) and
enumerates **five** checks (lines 1321-1325). Doctor implements those five, records the discrepancy as
a constant, and refuses to invent the missing two (doctor.rs:51-57). A green preflight therefore
proves the five written checks and nothing about the two the title promises; "seven-point" is not
satisfiable by any command in this tree until the specification says what the other two are.
---

## 3. The gates

Each gate below gives the spec line it comes from, the test or check that implements it, the exact
command, the pass criterion as a number with its unit, the artifact to keep, and the state today.
Run every command from the repository root. `<peer>` and the like are the only placeholders.

Two mechanical facts apply to all six Milestone 5 gates:

- They live in the library test module `gates::tests` (crates/sparkflash-scheduler/src/gates.rs:177-376),
  so the test target is `--lib` and the full test path is `gates::tests::<name>`.
- They read the run's summary from an environment variable and **print nothing on success**
  (gates.rs:95-99); asking one to run without its variable panics. Capture the raw summary as well as
  the cargo log, or the successful run leaves no measurement behind.

### 3.1 10 KiB one-shot All-Reduce latency

**Spec.** Line 1336: "Validate < 20 µs 10 KiB All-Reduce latency and bit-identical outputs across
10,000 CUDA graph replays." Line 1323 (preflight check 3): aggregate bandwidth >= 180 Gbps and
"10 KiB One-Shot All-Reduce latency < 22 µs".

**Instrument.** `sparkflash doctor` check 3 (crates/sparkflash-cli/src/doctor.rs) invokes
`rocenante-probe --bytes 10240 --peer <peer> --json` and reads the `oneshot_allreduce_us` field, with
the byte size fixed by `ONESHOT_AR_BYTES = 10 * 1024`. The probe is a binary target of this workspace
now (`crates/sparkflash-comm/Cargo.toml`, pipeline in `crates/sparkflash-comm/src/probe.rs`), and the
JSON read is a pure function that requires `transport == "rdma"` and `fabric == true`: a loopback
pipeline run is labelled and refused by name rather than scored against the fabric budget, which was
a real exploit until the label check was added and is pinned by three tests. **The instrument exists
and cannot measure on this hardware yet**: with the `rdma` feature the backend is a skeleton, so the
probe refuses - on a host with no RDMA subsystem at `FabricUnavailable`, and only when the rails
resolve at `RdmaNotBuilt` - and without the feature it names the exact rebuild command. The refusal
is the honest output; a number from it would not be one.

**Command.**

```bash
# both rails to one peer, aggregate >= 180 Gbps and the 10 KiB one-shot latency
./target/release/sparkflash doctor --nodes <peer> --rdma-seconds 1 --json

# the latency number on its own, once the probe binary is on PATH
rocenante-probe --bytes 10240 --peer <peer> --json

# the crate's own gate, which today cannot measure (see Status)
SPARK_COMM_HW_PERMISSION=1 cargo test -p sparkflash-comm --features rdma \
  --test hardware_gates -- --ignored --exact roce_oneshot_two_mib_all_reduce_meets_the_latency_budget
```

**Pass criterion.** `oneshot_allreduce_us < 22.0` µs (doctor.rs:101) and aggregate
`ibv_write_bw` >= 180.0 Gbps across the two rails (doctor.rs:95). The stricter `< 20 µs` of spec
line 1336 is **not encoded anywhere** in the tree: no constant, no test, no probe field. A run must
state which of the two it is claiming.

**Artifact.** The doctor JSON report (`tool`, `verdict` and the per-check evidence strings) plus the
raw `rocenante-probe` JSON documents and the two `ibv_write_bw` outputs. Keep the peer address and the
rail device names with them: the aggregate is meaningless without them (section 2.4).

**State: never run**, and the test that exists cannot run: its body is
`panic!("not implemented on this host: ...")` (hardware_gates.rs:44-47). Note the payload mismatch:
the test is named for **2 MiB** while the spec's budget and doctor's probe are for 10 KiB.

### 3.2 Bit-identical collectives over 10,000 replays

**Spec.** Line 1336, second half: "bit-identical outputs across 10,000 CUDA graph replays".

**Instrument.** crates/sparkflash-comm/tests/hardware_gates.rs:52
`roce_bits_match_the_cpu_reference_over_ten_thousand_replays`, whose body is a `panic!` (lines 55-57).
The CPU side of the comparison does exist and does run in CI:
`sparkflash_comm::reduce::canonical_all_reduce_into`, exercised by
crates/sparkflash-comm/tests/reduction_bits.rs. What is missing is the hardware side.

**Command.**

```bash
SPARK_COMM_HW_PERMISSION=1 cargo test -p sparkflash-comm --features rdma \
  --test hardware_gates -- --ignored --exact roce_bits_match_the_cpu_reference_over_ten_thousand_replays
```

**Pass criterion.** Zero mismatches over 10,000 replays. No constant and no assertion for this exists
in the tree; the number is only in the spec, so a run has to report its replay count and mismatch
count as text.

**Artifact.** The test log carrying the replay count and the first mismatching replay, plus the CPU
reference digest it compared against.

**State: never run**; the instrument does not exist (section 5, finding 3).

### 3.3 Dual-rail counters carry half the bytes each

**Spec.** Line 190 (dual-rail topology: both halves active in parallel) and line 1336 (the transport
being validated). The check exists as a test name, not as a spec sentence.

**Instrument.** hardware_gates.rs:62
`dual_rail_counters_show_both_interfaces_carrying_half_the_bytes`, body a `panic!` (lines 65-67).
Doctor measures something adjacent but different: per-rail `ibv_write_bw` Gb/s, summed
(doctor.rs:742-770). It does not read the counters.

**Command.**

```bash
SPARK_COMM_HW_PERMISSION=1 cargo test -p sparkflash-comm --features rdma \
  --test hardware_gates -- --ignored --exact dual_rail_counters_show_both_interfaces_carrying_half_the_bytes
```

**Pass criterion.** The two rails' byte counters differ by at most the documented truncation:
spec line 758-760 halves the payload with integer division, so each counter is within 3 bytes of
exactly half (compare docs/DECISIONS.md D13, which keeps that formula and pins the undercount with a
test). No hardware-side constant exists.

**Artifact.** The two rails' counter readings before and after one dispatch, with the dispatch size.

**State: never run**; the instrument does not exist.

### 3.4 qeval 72/75

**Spec.** Line 1340: "run the correctness and throughput qualification gates (`qeval` 72/75, ...)".

**Instrument.** gates.rs:291-304, `#[ignore = "hardware gate: needs a qeval run; set SPARKFLASH_QEVAL_RESULTS"]`,
test `qeval_scores_at_least_72_of_75`. Thresholds: `QEVAL_MIN_CORRECT = 72`,
`QEVAL_TOTAL = 75` (gates.rs:17-19).

**Command.**

```bash
SPARKFLASH_QEVAL_RESULTS="correct=73 total=75" cargo test -p sparkflash-scheduler --lib \
  gates::tests::qeval_scores_at_least_72_of_75 -- --ignored --exact
```

**Pass criterion.** `correct >= 72` **and** `total == 75` exactly (gates.rs:296-303); unit: questions.
A summary missing either key, or reporting a fractional count, panics the gate rather than passing it
(gates.rs:120-140, 246-289).

**Artifact.** The qeval harness's own output: the 75 questions with the graded answers, plus its
summary line, plus the cargo log. The gate does not print the summary it read.

**State: never run.** There is **no producer**: see section 5, finding 2.

### 3.5 DSpark draft KL divergence

**Spec.** Line 1340: "KL divergence < 0.007".

**Instrument.** gates.rs:306-321, test `the_dspark_draft_stays_below_the_kl_ceiling`;
`DSPARK_KL_MAX: f64 = 0.007` (gates.rs:21). The gate also rejects a negative KL, because a negative
value would otherwise satisfy a strict upper bound while measuring something else (gates.rs:311-316).

**Command.**

```bash
SPARKFLASH_DSPARK_RESULTS="kl=0.0062" cargo test -p sparkflash-scheduler --lib \
  gates::tests::the_dspark_draft_stays_below_the_kl_ceiling -- --ignored --exact
```

**Pass criterion.** `0 <= kl < 0.007` (dimensionless, nats or bits must be stated with the number).

**Artifact.** The acceptance run's per-step draft and target distributions, the KL computation, and
the summary line. Without the distributions, a passing number cannot be checked against its own
definition of KL base.

**State: never run**; no producer (finding 2). Note that docs/DECISIONS.md D6 records that the
reference implements no acceptance loop at all, so this gate is measuring our loop, not a port.

### 3.6 1M-token needle

**Spec.** Line 1340: "1M needle test"; the context length is pinned by
`NEEDLE_CONTEXT_TOKENS = 1_048_576` (gates.rs:23).

**Instrument.** gates.rs:323-334, test `a_needle_is_found_at_one_million_tokens`. It requires the
exact context (1,048,576 tokens) and `found=1`; a fractional context such as 1048576.9 is rejected
(gates.rs:286-288).

**Command.**

```bash
SPARKFLASH_NEEDLE_RESULTS="context=1048576 found=1" cargo test -p sparkflash-scheduler --lib \
  gates::tests::a_needle_is_found_at_one_million_tokens -- --ignored --exact
```

**Pass criterion.** `context == 1,048,576` tokens exactly, `found == 1` (gates.rs:329-333).

**Artifact.** The full prompt hash, the needle's depth and text, the model's answer, the KV cache
telemetry for the run (pages per tier), and the summary line. The gate reads only `context` and
`found`, so what the run actually asked is not recoverable from the gate.

**State: never run**; no producer (finding 2), and the reference serve path could not hold 1M
tokens of KV even if it were driven: it has no KV pool and no persistent tier, and decodes one chunk
at a time on the CPU (crates/sparkflash-cli/src/serve.rs:12-17, 490-492; finding 1).

### 3.7 sparkDash concurrency 1..16

**Spec.** Line 1340: "`sparkDash` c=1..16 sweeps"; `SPARKDASH_MAX_CONCURRENCY = 16` (gates.rs:25).

**Instrument.** gates.rs:336-346, test `sparkdash_holds_concurrency_one_through_sixteen`. The failed
request count is required, not optional, because a run that omits it cannot be told apart from one
that failed requests and stayed quiet (gates.rs:151-160).

**Command.**

```bash
SPARKFLASH_SPARKDASH_RESULTS="max_concurrency=16 failed_requests=0" cargo test -p sparkflash-scheduler --lib \
  gates::tests::sparkdash_holds_concurrency_one_through_sixteen -- --ignored --exact
```

**Pass criterion.** `max_concurrency == 16` and `failed_requests == 0` (gates.rs:341-345).

**Artifact.** The sweep's per-concurrency results (16 levels, each with its own latency and failure
count), the server-side telemetry with the run, and the summary line. A single `max_concurrency=16`
line without the 16 levels does not show that the sweep happened.

**State: never run**; there is no sweep client in the tree (finding 2), and the one server in the
tree is not a concurrency-16 target: the reference serve path serialises every request through a
single engine thread, one sequence at a time (crates/sparkflash-cli/src/serve.rs:24-31; finding 1).

### 3.8 CUDA graph replay, batch 1..16

**Spec.** Line 1340: "Capture full CUDA graphs for batch sizes `1..16`"; line 1336 for the replay
validation. Threshold `CUDA_GRAPH_MAX_BATCH = 16` (gates.rs:27).

**Instrument.** gates.rs:348-358, test `the_cuda_graph_replays_for_batch_one_through_sixteen`.

**Command.**

```bash
SPARKFLASH_CUDA_GRAPH_RESULTS="max_batch=16 mismatches=0" cargo test -p sparkflash-scheduler --lib \
  gates::tests::the_cuda_graph_replays_for_batch_one_through_sixteen -- --ignored --exact
```

**Pass criterion.** `max_batch == 16` and `mismatches == 0` replays disagreeing with the eager path
(gates.rs:353-357); unit: replays.

**Artifact.** The capture log for batches 1 through 16, the eager-versus-graph comparison, and the
summary line.

**State: never run**; no producer (finding 2) **and no capture implementation**: `cudaGraph` and
`cuda_graph` appear in `crates/` only inside gates.rs itself (finding 9).

### 3.9 45-step scoped trace, every scope

**Spec.** Line 1340: "capture a 45-step scoped trace via `sparkflash trace --scopes all --steps 45`".

**Instrument.** Two halves, only one of which exists. The command exists and parses:
crates/sparkflash-cli/src/main.rs:41 dispatches `trace` to
`sparkflash_scheduler::trace_cli::run` (trace_cli.rs:158), with `GATE_STEPS = 45` (trace_cli.rs:20).
The capture does not: trace_cli.rs:6-8 says "Only the plan is produced here; running the capture is a
hardware gate". The gate reads the result: gates.rs:360-374, test
`the_45_step_scoped_trace_records_every_scope`, requiring `steps == 45` and
`scopes == TraceScope::COUNT`, which is 16 (crates/sparkflash-trace/src/scope.rs:56).

**Command.**

```bash
# the plan the CLI prints today; this is the command the spec names
./target/release/sparkflash trace --scopes all --steps 45

# the gate on the captured result
SPARKFLASH_TRACE_RESULTS="steps=45 scopes=16" cargo test -p sparkflash-scheduler --lib \
  gates::tests::the_45_step_scoped_trace_records_every_scope -- --ignored --exact
```

**Pass criterion.** `steps == 45` and `scopes == 16`, i.e. every one of the 16 scopes recorded at
least one event (gates.rs:366-374). The scope count is `TraceScope::COUNT`, not a literal, so adding
a scope raises the bar automatically.

**Artifact.** The captured trace file exported for Chrome/Perfetto (the exporter is
crates/sparkflash-trace/src/export.rs), the per-scope event counts, and the summary line. A plan
printout is not an artifact: it proves the arguments parsed, not that a trace was captured.

**State: never run**; the capture runner does not exist (finding 7).

---

## 4. What a pass would not prove

If every gate above passed tomorrow, the following would still be unproven. This is the part of the
runbook that keeps a green board from being read as "defect-free".

1. **Sampling, not coverage.** qeval is a floor on one 75-question set: 72/75 is 96.0%, and a run at
   the floor cannot be distinguished from a much weaker one, nor from one at 100%, by this gate alone.
   KL < 0.007 bounds one acceptance run's sampled distributions; it is not a per-token bound and says
   nothing about other prompts. The needle is one retrieval at one context length, and the gate reads
   only `context` and `found`, so it cannot even report how deep the needle was.
2. **The gates measure our engine, not the released model.** There is no bit-exactness gate against
   the Python reference. The kernels are reimplementations (docs/DECISIONS.md D14 fixes one such
   divergence; the FP4/FP8 layouts and the sharding are others), the only equivalence instruments are
   CPU references such as wt/m4k/crates/sparkflash-kernels/src/mxfp4_ref.rs, and a passing qeval run
   is compatible with an implementation that is numerically different everywhere the metric averages
   out.
3. **Hardware state is one sample.** The latency and bandwidth gates measure one payload size
   (10 KiB), one peer, one fabric state, and no contention. They do not sample the 2 MiB router
   threshold, the straggler path (docs/DECISIONS.md D11's bounded wait, NaN result and timeout trace
   record have never fired on hardware), or a rail failing mid-run.
4. **Determinism over replays is not determinism over inputs.** 10,000 identical replays would show
   the collective is reproducible for one operation order; it would not cover the interleavings a
   serving engine actually produces.
5. **The KV, cache and storage decisions are outside every gate.** Dequantise-on-read (D1), the Engram
   row layout (D4), the SWA bounded replay's approximate states (docs/architecture/08-kv-footprint.md
   section 4), the `io_uring` prefetch depth, the DRM carveout and the 72-hour TTL are not sampled by
   any of these nine gates. The needle test would touch a few of them at 1M; it has no producer.
6. **The memory guard is a snapshot.** `MemAvailable >= 114 GiB` is read before loading; no gate bounds
   peak allocation during a 1M-token run, which is where an OOM would actually appear.
7. **Concurrency depth is not soak.** `c=16` with zero failed requests does not sample longer runs,
   cancellation, or the abort paths (crates/sparkflash-api/src/loop_abort.rs has no hardware gate).
8. **The encoding surface is exercised by CI, not here.** The golden vectors, the reasoning-effort
   tables and the mid-conversation system prompt are covered by unignored tests
   (crates/sparkflash-api/tests/encoding_golden.rs and friends) and are therefore outside this
   runbook - passing them is not evidence about the GPU path, and passing the GPU gates is not
   evidence about them.

---

## 5. Findings: gates whose instrument is missing

These are the most load-bearing statements in this document. Each one is a place where the tree
contains an expectation and no instrument.

1. **There is no GPU engine, and that is what the five model gates need.** `sparkflash serve`
   is implemented and wired (crates/sparkflash-cli/src/serve.rs, dispatched at
   crates/sparkflash-cli/src/main.rs:43): it builds a `sparkflash_engine` reference engine over either
   the synthetic tiny plan (`--synthetic`, no checkpoint on disk) or a real sharded checkpoint
   (`--checkpoint DIR`), presents it to `sparkflash_api::server::router` through a local driver, binds
   a `TcpListener` (serve.rs:892) and serves it with `axum::serve` (serve.rs:907) on a `tokio`
   runtime (serve.rs:874), printing the address it actually bound (serve.rs:906). It is one process,
   `f32`, on the CPU, one sequence at a time, with requests serialised through a single engine
   thread and no batching (serve.rs:12-17, 24-31, 552-554). So `MockEngine`
   (crates/sparkflash-api/src/engine.rs:460, 731) is no longer the only implementation of the api
   crate's `InferenceEngine` - `ReferenceDriver` implements it too (serve.rs:363). Its failure
   semantics are a decision recorded in `docs/DECISIONS.md` D19: an engine panic is caught per request
   only where the build unwinds, and the release profile's `panic = "abort"` (Cargo.toml:85) makes a
   release `sparkflash` abort the process instead. The reference
   forward pass is implemented and runs: all five modules are present and filled in (`attention`,
   `engram`, `mhc`, `moe`, `mtp`; crates/sparkflash-engine/src/forward/mod.rs:21-25), no
   `unimplemented!()` or `todo!()` remains anywhere in crates/sparkflash-engine/src, and the engine runs
   a whole prefill and decode in `f32` on the CPU (crates/sparkflash-engine/src/forward/mod.rs:1-3;
   crates/sparkflash-engine/tests/forward.rs:102-108, 168-177, 206-225). What has run end to end is
   the synthetic tiny plan, the path every serve test takes (serve.rs:950-951). The code's route from
   a checkpoint to a token is compile-verified and refusal-tested only - a directory without
   `model.safetensors.index.json` is refused by name (serve.rs:285, 1318-1322), and shards that do not
   match the plan are refused at load (serve.rs:728-733) - and it has never run against real shards,
   because this host has none, so the numbers the reference path could print are still unusable for
   the model gates, which measure the GPU path. The first revision of this finding was right that no
   such route existed when the survey was taken. Two corrections to it: it predated milestone 2's
   merge, which
   wired `pack-engram` (crates/sparkflash-engram/src/packer.rs), and it predated the serve slice, so
   it said the packer and the server did not exist. The consequence is unchanged: gates 3.4 through
   3.8 cannot be run by any command in this tree, not merely "have not been run" - they measure the
   GPU and cluster path (a qeval harness, a 16-way concurrent server, graph capture), and none of
   that exists here.
2. **Six run summaries have no producer.** `SPARKFLASH_QEVAL_RESULTS`, `SPARKFLASH_DSPARK_RESULTS`,
   `SPARKFLASH_NEEDLE_RESULTS`, `SPARKFLASH_SPARKDASH_RESULTS`, `SPARKFLASH_CUDA_GRAPH_RESULTS` and
   `SPARKFLASH_TRACE_RESULTS` appear **only** in crates/sparkflash-scheduler/src/gates.rs (declared at
   lines 32-42, used in the six gate bodies at 291-374). Nothing in the repository writes them, so the
   operator has to obtain each summary from a harness that does not exist here. A gate that reads its
   result from an environment variable is only as good as the producer of that variable.
3. **The three cluster gates are panic stubs.** crates/sparkflash-comm/tests/hardware_gates.rs:44-47,
   55-57 and 65-67 all `panic!` with "not implemented on this host", after the permission and RDMA
   backend checks. Even with `SPARK_COMM_HW_PERMISSION=1` and `--features rdma` they cannot measure
   anything, so the All-Reduce latency gate has no measurement in the test tree.
4. **The one real All-Reduce instrument is merged and still cannot measure.** `rocenante-probe` is
   a binary target of `sparkflash-comm` now, and the doctor invokes it, so check 3 has an instrument
   where it previously had a missing command. What the instrument can do is refuse precisely: the
   RDMA backend resolves the live rails and then returns `CommError::RdmaNotBuilt`, and on a host
   without `/sys/class/infiniband` it fails earlier at `FabricUnavailable`. Its hardware gate test
   is `#[ignore]`d with a reason saying it is blocked on the backend. So the preflight still cannot
   pass, but for the right reason, and the failure mode changed from "no such command" to a
   prerequisite named in the message - which is the difference between an unrunnable gate and an
   honest one.
5. **The spec carries two latency budgets, and they are different things.** 22 µs at line 1323
   (preflight check 3) and 20 µs at line 1336 (milestone 3). Doctor encodes 22.0 (doctor.rs:101) and
   nothing encoded 20. D17 records the separation: 20 µs at 10 KiB is the requirement, the preflight's
   22 µs is a deliberately looser readiness threshold, and a run reporting 21 µs has passed the
   preflight and failed the gate. The 2 MiB in one test's name is the router threshold, not a third
   budget. Nothing in the tree has to change for this; what has to change is a report that cites one
   number as the other.
6. **The preflight's own title is unsatisfiable.** Spec 6.1 is titled "Seven-Point" (line 1317) and
   enumerates five checks (lines 1321-1325); doctor implements five and records the discrepancy
   (doctor.rs:51-57). The missing two are not described anywhere in the tree.
7. **The trace command is a planner.** trace_cli.rs:6-8 and `render()` (trace_cli.rs:65-74) produce a
   plan that ends with "capture: hardware gate (needs the target runtime and the trace device)". The
   spec names this command as the gate (line 1340); the command cannot perform it.
8. **The latency gate's payload is inconsistent.** The test is named for **2 MiB**
   (hardware_gates.rs:41) while the spec's latency budget and doctor's probe are for 10 KiB
   (spec 1323, 1336; doctor.rs:98). 2 MiB is the router threshold, not the latency payload.
9. **No CUDA graph capture exists.** `cudaGraph` and `cuda_graph` appear in `crates/` only inside
   gates.rs (the parser at 162-169, the gate at 348-358). There is no capture, no replay and no
   eager-versus-graph comparison to run.
10. **Doctor's collective probes do not see the ring.** Both the bandwidth probe and the All-Reduce
    probe use only the first `--nodes` address (doctor.rs:743-745, 773-775), so the reported aggregate
    is two local rails to one peer. Spec line 1323 asks for the probe "between all 4 ranks"; a green
    check 3 would not establish that.
11. **A green run leaves no measurement in the log.** The six gates assert on the summary and print
    nothing on success (gates.rs:95-99). Artifact capture is therefore a deliberate step, not a
    side effect of running the tests - which is why every gate above has an artifact column.
12. **Positive finding: the only ignored tests that really exercise hardware are the KV ones, and they
    are on another branch.** wt/m4kv/crates/sparkflash-kv/tests/hardware.rs:25 (NVMe `O_DIRECT` round
    trip at the device's own alignment, needs `SPARKFLASH_KV_NVME_PATH`) and line 58 (instantiate the
    default pool for real) have real bodies rather than panics. They are not in the merged tree
    (`crates/sparkflash-kv` has no `tests/` directory here), and they are not qualification gates, but
    they are the pattern the three comm stubs should follow.

---

## 6. Copy-paste order for a qualification window

Run in this order. Stop at the first failure: a gate whose precondition fails produces a number that
cannot be attributed. Every line assumes an exclusive window, `SPARK_COMM_HW_PERMISSION=1` exported
only with the Lead's approval, and the repository root as the working directory.

```bash
# 0. build
cargo build --release -p sparkflash-cli

# 1. preflight, this node, then the cluster. Pass = verdict "ready", not exit code 0.
./target/release/sparkflash doctor --local --json
./target/release/sparkflash doctor --nodes 192.0.2.80,192.0.2.84,192.0.2.68,192.0.2.59 \
  --rank 0 --world-size 4 --rdma-seconds 1 --json

# 2. collectives (needs the probe binary on PATH and the fabric permission)
SPARK_COMM_HW_PERMISSION=1 cargo test -p sparkflash-comm --features rdma \
  --test hardware_gates -- --ignored --exact roce_oneshot_two_mib_all_reduce_meets_the_latency_budget
SPARK_COMM_HW_PERMISSION=1 cargo test -p sparkflash-comm --features rdma \
  --test hardware_gates -- --ignored --exact roce_bits_match_the_cpu_reference_over_ten_thousand_replays
SPARK_COMM_HW_PERMISSION=1 cargo test -p sparkflash-comm --features rdma \
  --test hardware_gates -- --ignored --exact dual_rail_counters_show_both_interfaces_carrying_half_the_bytes

# 3. the six Milestone 5 gates, each with the summary its run produced
SPARKFLASH_QEVAL_RESULTS="correct=<n> total=75" cargo test -p sparkflash-scheduler --lib \
  gates::tests::qeval_scores_at_least_72_of_75 -- --ignored --exact
SPARKFLASH_DSPARK_RESULTS="kl=<x>" cargo test -p sparkflash-scheduler --lib \
  gates::tests::the_dspark_draft_stays_below_the_kl_ceiling -- --ignored --exact
SPARKFLASH_NEEDLE_RESULTS="context=1048576 found=<0|1>" cargo test -p sparkflash-scheduler --lib \
  gates::tests::a_needle_is_found_at_one_million_tokens -- --ignored --exact
SPARKFLASH_SPARKDASH_RESULTS="max_concurrency=<n> failed_requests=<n>" cargo test -p sparkflash-scheduler --lib \
  gates::tests::sparkdash_holds_concurrency_one_through_sixteen -- --ignored --exact
SPARKFLASH_CUDA_GRAPH_RESULTS="max_batch=<n> mismatches=<n>" cargo test -p sparkflash-scheduler --lib \
  gates::tests::the_cuda_graph_replays_for_batch_one_through_sixteen -- --ignored --exact
SPARKFLASH_TRACE_RESULTS="steps=45 scopes=16" cargo test -p sparkflash-scheduler --lib \
  gates::tests::the_45_step_scoped_trace_records_every_scope -- --ignored --exact

# 4. the capture itself
./target/release/sparkflash trace --scopes all --steps 45

# 5. keep, per gate: the raw run output, the summary used, and the cargo log
```

Steps 2 and 3 cannot currently succeed: step 2's gates are panic stubs (finding 3) and the probe,
now merged, can only refuse because the RDMA backend is a skeleton (finding 4); step 3's six gates
have no producer of their summaries (finding 2)
and five of them additionally need a GPU engine this tree does not have (finding 1); step 4 prints a plan rather than a
capture (finding 7). Until those findings are closed, this runbook's honest state is section 1: every
gate **never run**.

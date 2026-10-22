# Operations

How to run a four-node window, collect evidence, and close it without leaving the live service down.
Process rules (clean row, batching, gates) are in [`../RULES.md`](../RULES.md); build and deployment
commands are in [`build-deploy.md`](build-deploy.md); artifact schemas are in
[`evidence.md`](evidence.md).

## 1. Ownership and preconditions

- The window is **Lead-owned**. Teammates request it; they do not stop the live vLLM instance.
- The live client uses `http://192.0.2.80:8000`; while vLLM is stopped it has no model. Keep the
  window short and always restore before closing.
- Stage the binary and all evidence tooling **before** stopping vLLM. Staging during the window is
  wasted GPU time.
- A row needs the four nodes idle: zero `sparkflash-*` processes and `MemFree >= ~110 GiB` each.
- Record the **staged binary sha256** and the **source commit** in the row name. A binary rebuilt on
  the node is not the staged artifact.
- The serving control scripts are `scripts/serve-four-node.sh` and `scripts/window-control.sh`;
  the window wrappers are `scripts/stage-window.sh` and `scripts/run-fourrank.sh`.

## 2. Window sequence

### 2.1 Before the stop

1. Build aarch64 from the exact commit that will run (see `build-deploy.md`), or reuse a staged
   binary whose sha256 is recorded.
2. Check the kernel baseline locally:
   `SPARKFLASH_NVCC=<nvcc> scripts/check-kernel-baseline.sh` (from
   `crates/sparkflash-kernels`). A missing `nvcc` is a skip, not a pass.
3. Run the host gates on the commit: `cargo fmt --all -- --check`,
   `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
4. Stage the binary, cubins, checkpoints and run scripts on all four nodes.
5. Confirm the frozen argv for the row (token ids, steps, flags) and write it into the row directory.

### 2.2 Stop the service

```sh
scripts/window-control.sh stop        # docker stop vllm_node on all four nodes
```

Confirm port 8000 is closed and `MemAvailable` has recovered on all four nodes. The window is not
started until the service is actually gone; a half-stopped cluster produces unattributable rows.

### 2.3 Clean and run

1. On **all four nodes**, drop the checkpoint from the page cache
   (`posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` over the checkpoint shards; `/tmp/dropcache.py`).
2. Zero `sparkflash-*` processes; verify `MemFree >= ~110 GiB` per node.
3. Start one process per rank. Use `ssh -n` (or a fully detached child), or the ssh session holds
   the channel and the next rank misses its connect deadline. For the four-rank loop:
   ```sh
   scripts/run-fourrank.sh <tag> <row> token
   ```
   with `SPARKFLASH_SYNC_LAUNCHES=1` for a sync control and `SPARKFLASH_FORWARD_EVIDENCE=1` for an
   evidence row. The device arm is opt-in with `SPARKFLASH_LAYER_DEVICE=1`.
4. Never stage/copy while a row is running. Never run two rows at once on the same nodes.

### 2.4 Gate the row

```sh
scripts/numerical-gate.sh --run <row-dir>      # product gate (D54)
scripts/identity-gate.sh --run <row-dir>       # strict mode, only if bit identity is claimed
scripts/device-frontier.py --row-dir <row-dir> # device-arm rows: stage frontier / refusal
scripts/check-collective-count.py              # device-arm rows: layers x 2 on all four ranks
```

A failed numerical gate is a port bug, not a tolerance question. A row that ends at open is an
**OPEN REFUSAL** (device-frontier exit 3), not a speed number.

### 2.5 Restore and close

Restart vLLM with the recorded recipe (not a hand-typed variant), then:

```sh
curl -fsS -o /dev/null -w '%{http_code}\n' http://192.0.2.80:8000/v1/models
```

Expect `200` and the expected served root. The window is finished only after the health check and a
client request succeed. If health fails, restart again; do not leave the window open.

## 3. Qualification gates

The Milestone-5 gates and the All-Reduce latency gate are listed here with their current state. The
full historical gate table is in the archive. A gate is evidence only when its command ran and its
artifact was kept; an ignored test is a plan.

| Gate | Instrument | Current state |
| :-- | :-- | :-- |
| 10 KiB one-shot All-Reduce p50 | `sparkflash doctor` check 3 / `scripts/ws-b-allreduce-bench.sh` | **missed**: 35.70 us p50 dual-rail at MTU 9000 vs 25 us WS-B / 22 us doctor bar |
| Bit-identical collectives, 10,000 replays | `sparkflash-comm` hardware gate | one dual-rail run passed 10,000/10,000; four other runs failed on retry-exceeded (transport completions, not numeric corruption) |
| Dual-rail counters half bytes each | hardware gate + `scripts/ws-b-fabric-verify.sh` | measured accounting passes; single-rail mode explicitly refuses a dual-rail claim |
| qeval 72/75 | `SPARKFLASH_QEVAL_RESULTS` env | no producer yet |
| DSpark KL < 0.007 | `SPARKFLASH_DSPARK_RESULTS` env | no producer yet; host-twin acceptance 1.571 is separate |
| 1M-token needle | `SPARKFLASH_NEEDLE_RESULTS` env | no producer yet |
| sparkDash concurrency 1..16 | `SPARKFLASH_SPARKDASH_RESULTS` env | no producer yet |
| CUDA graph replay, batch 1..16 | `SPARKFLASH_CUDA_GRAPH_RESULTS` env | no capture code yet |
| 45-step scoped trace | trace planner + `SPARKFLASH_*_RESULTS` | no capture runner yet |
| Preflight | `sparkflash doctor` | driver/MTU/memory checked; SM clock and UVM not verifiable |

Preconditions for any hardware gate: `SPARK_COMM_HW_PERMISSION=1` and an exclusive window. Do not
compare a single-rail reference number to the dual-rail bar. The deployed vLLM is single-rail
(`NCCL_IB_HCA=rocep1s0f1`); our gate is dual-rail aggregate.

## 4. Evidence collection

One run directory per row, under the window root:

```text
<row>/
  r0/ .. r3/                 per-rank evidence
  argv.txt                   frozen argv, one line per rank
  summary.txt                the run summary
```

Keep, per row: the raw per-rank records, the gate output, the binary sha256 and source commit, the
served root (if an oracle was captured), and the clean-row notes (cache drop, MemFree). The schemas
are in [`evidence.md`](evidence.md).

## 5. Review procedure

The review protocol is in [`../RULES.md`](../RULES.md) section 6. The short form:

1. Fresh adversarial critic, no authorship.
2. Every claim tied to a test, measurement or spec line; measured means it ran.
3. Iterate until the critic approves; only the Lead merges.
4. `fmt`, `clippy -D warnings` and `cargo test --workspace` pass on the branch.
5. Documentation-only findings are non-blocking and recorded as owned follow-ups.

## 6. Common failures

| Symptom | Cause / fix |
| :-- | :-- |
| rank 1+ times out, rank 0 logs a truncated hello | ssh holds the channel: use `ssh -n` or a detached child. |
| a step is ~250 ms slower than its band | cold page cache after O_DIRECT boot: run the cache drop on all four nodes. |
| one rank collector count differs | a rank-conditional path skipped a reduce; fix the sequence, not the counter. |
| a device row stops at a stage | expected open refusal; record it with `device-frontier.py`, do not add a fallback. |
| the service is down after the window | restart was not health-checked; rerun the restore recipe until `/v1/models` is 200. |

## Archived sources

`window-runbook.md`, `QUALIFICATION.md`, `acceptance-gate-rehearsal.md`, `review-protocol.md`,
`d48-window-protocol.md`, `d41-capture-protocol-v2.md`, `ws-b-infra-tickets-20261021.md`.

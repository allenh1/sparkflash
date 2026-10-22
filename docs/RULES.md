# Rules

**Status: in force from 2026-10-22.** This is the single source of truth for how work is measured,
gated, reviewed and documented on the speed track. It supersedes, in full:

- `docs/archive/RULES-speed.md` (the old operational copy),
- D53 (operating rules),
- `docs/archive/build/review-protocol.md` (the review section below is the current copy),
- every measurement/process rule stated in an individual `docs/build/` record.

Conflict order: **this file > `docs/DECISIONS.md` IDs > topic handbooks > archived records**.
Decision IDs (D1-D54) remain stable and are cited from source comments; this file does not renumber
them. Code-level contracts (ABI, formats, flags) still win over prose.

## 1. The one number

- The speed metric is **lean decode ms/step, mean over four ranks**, with the frozen argv, a clean
  row, and the same base commit. No other speed number is published.
- It is appended to [`docs/build/speed-log.csv`](build/speed-log.csv) with
  `scripts/speed-log.sh append --run <window run dir>`; `print` renders the table.
- **Current: 179.81 ms/step** (2026-10-22, integration `cd233423`, four-rank mean, identity PASS;
  replicate 180.00). 90 tok/s = 11.1 ms/step.
- Historical numbers are not comparable across base commits, and a row only counts in the window
  and on the base it was measured on. Quote history only with its commit.
- Every merge request states its **expected effect on ms/step before work starts**. A change with no
  expected effect is not speed work.

## 2. Gates and targets

- Speed gates, in order: **<= 400 ms (cleared) -> <= 150 -> <= 50 -> <= 20 -> <= 11.1 ms/step**.
  The next gate is 150.
- Product target: **50-60 tok/s** at vLLM parity with DSpark. Personal target: **90 tok/s**
  single-stream.
- With the measured MTP acceptance length of **1.571 accepted/verify step**, 90 tok/s needs a
  **17.46 ms verify step** (`1000 * 1.571 / 90`). At a lower acceptance length the budget tightens.
  D48's 42 ms/verify-step budget assumed ~3.8 accepted tokens; at the measured 1.571 the 90 tok/s
  budget is 17.46 ms, not 42 ms.
- A gate is cleared only by a gated row (section 3), not by an estimate.

## 3. Correctness gate (D54)

The product gate is **numerical**, not byte-identical:

```bash
scripts/numerical-gate.sh --run <row-dir> [more rows...]
```

Each rank is compared against the frozen reference artifact for the same fixture
(`docs/build/reference/speed-gate/<fixture>/`, keyed by sha256 of positions, prompt length and
token ids). Token equality against vLLM stays retired (D39); the token bar below is against the
frozen CPU twin. The declared bars:

| check | bar |
| :-- | :-- |
| greedy argmax tokens | identical to the reference |
| top-1 agreement | 1.00 of decode positions |
| top-5 set overlap | >= 0.80 mean Jaccard |
| top-50 id overlap | >= 0.96 mean |
| max abs delta logprob | <= 2.0e-2 nats over the reference top-100 |
| mean abs delta logprob | <= 2.0e-3 nats |
| KL(reference || run) | <= 1.0e-4 nats over the renormalised top-100 |
| finiteness | every logit finite |

The gate prints the reference's smallest top1-top2 margin with every row. A token flip inside that
margin is a tie, not a defect, but the logprob bars must still pass.

Rules:

- **Byte identity is the strict mode, not the product gate.** Run
  `scripts/identity-gate.sh --run <row>` only when a slice claims bit identity (a pure layout,
  pointer or geometry change). Do not claim byte identity for a slice that deliberately reorders a
  sum.
- **The reference is an artifact, not a re-run.** The bit-exact CPU twin is the reference
  implementation; the product may differ inside the bars. Freeze a new fixture with
  `scripts/numerical-gate.sh --freeze <run> <dir>`; never edit a frozen reference to fit a run.
- A tolerance arm is declared in the row and in the merge request, with its measured distance.
  Admissible: split reductions, tree sums, kernels whose parallel dimension divides a sum.
  Not admissible: changed quantisation/dtype/terms, non-finite values, a bar loosened without a
  measured noise floor.
- A result the gate already proves gets **no document, no decision number and no re-run**.

## 4. Measurement protocol

### 4.1 Clean row (mandatory)

Before every four-rank row, on all four nodes:

1. Drop the checkpoint from the page cache:
   `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` over the checkpoint shards
   (`/tmp/dropcache.py`; no root needed). The O_DIRECT loader does not populate the page cache,
   and a cold read can add ~250 ms/step.
2. Zero all `sparkflash-*` processes and confirm `MemFree >= ~110 GiB` per node.
3. Do not stage or copy files while a row is running.

Record the **median and the clean-step band**, not only the mean. One mode step moves a 9-step mean
by ~25 ms; three move it by ~200 ms.

### 4.2 Batching and noise

- A fresh four-rank process costs ~6-7 minutes of checkpoint upload regardless of step count. Batch
  slices into one window when their accumulated expected movement is above **~20 ms/step**.
- Otherwise use **two interleaved rows (A/B/A/B)**. A single row is about +/-5 ms on a clean cluster;
  the mode shift is what the 20 ms threshold protects against.
- Do not file a row whose answer is already known from CI or construction. A row exists to decide
  something.
- The probe row is for picking the next slice, not for reporting a delta. Run it only when the block
  table is stale.
- A row that ends at open is **OPEN REFUSAL**, not a number: record it with
  `scripts/device-frontier.py --row-dir <row>` (exit 3) and never write it to `speed-log.csv`.
- Every completed row is one line in `speed-log.csv`. **No per-row documents.**

### 4.3 Estimates

- Estimates come from the **hardware floor** (weight bytes / peak bandwidth, MACs / peak MACs),
  never by scaling the previous kernel's measured rate.
- Report GB/s achieved against peak next to ms. A single-token launch should reach **>= 60 % of
  peak memory bandwidth**; beating our own previous kernel is not a result.
- "Host path closed" means **zero host reference arithmetic per token** (the linalg calls and the
  `forward/attention.rs` / `forward/engram.rs` reference paths reached from
  `tp4_forward.rs`). Host arithmetic is a floor no kernel work can remove.

## 5. Decisions and documentation

- **Decisions** are recorded only for hard-to-reverse choices, at most about one per week. IDs are
  stable and never renumbered. The current register is [`DECISIONS.md`](DECISIONS.md); full
  historical rationale is in `docs/archive/decisions-full.md`.
- **No STATUS documents.** Current state lives in this file, the speed log, and the topic handbooks.
- **One topic handbook per area**, describing current truth. No chronological journaling, no
  "window X then window Y" narratives, no per-row records. Historical records go to
  `docs/archive/` and are not maintained.
- **Investigations are time-boxed to two days**, then ship a change or drop the thread.
- A measured but unwired kernel is **inventory**: wire it or delete it.
- **No new host/CPU twins of device paths.** The device path is the product; the CPU twin is the
  frozen reference artifact, not a second implementation to maintain.

## 6. Review and merge

1. **Every merge request gets a fresh adversarial critic**: a reviewer with independent context, no
   authorship, starting from the diff and the task, whose job is to falsify the claims.
2. **Iterate until the critic approves.** A "will fix later" answer is either fixed or recorded as
   an open, owned follow-up.
3. **Only the Lead merges.**
4. The critic checks, at minimum:
   - every claim is tied to a test, a measurement or a specification line; measured means it ran;
   - numbers carry the commit/run they were measured at; hypotheses are labelled as hypotheses;
   - a non-trivial behaviour change updates the topic handbook; an architectural change adds a
     DECISIONS entry;
   - the diff is inside the declared write scope;
   - `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
     `cargo test --workspace` pass on the branch;
   - the row passed `scripts/numerical-gate.sh` (or the declared identity mode).
5. **Documentation-only findings are non-blocking.** A doc-only delta changes no code, test
   assertion, output string or CI behaviour. Record it as an owned follow-up; never let a doc change
   carry a code change.

## 7. Current focus and workstreams

- **Plan of record: the whole-layer device port** on `win/layer-device`. One branch, one goal: a
  decoder layer that runs entirely on the GPU, with no D2H in the steady state, tree reductions on
  device, and the cross-rank all-reduce on the GPU.
- Intermediate commits on that branch may be slower than 179.81. **No published number until the
  whole layer runs and the branch is measured as a whole.** 179.81 remains the standing number.
- Next gates, in order: first device-only prefix runs; one full device-only layer runs; the default
  flips to the device arm; then graph capture and the GEMV re-tile.
- Parked until then: host-seam micro-optimisation, partial-port measurement, and WS-C/D/E/F/G speed
  work. Serving/API parity work continues on its own track and does not consume speed windows.
- Each workstream has **one owner and one exit gate**. Unblock, don't work around.
- Fabric: MTU 9000 is applied on all four nodes; `nvidia-peermem` is blocked on a root rebuild, so
  the port starts on NCCL and switches to GPU-initiated RDMA only if a measured fabric row says it
  must.

## 8. Windows

A GPU window is Lead-owned. The non-negotiables:

1. Stop the live vLLM deployment on all four nodes (`window-control.sh stop`); confirm memory and
   port 8000 are released.
2. Run clean rows (section 4.1), one row per decision.
3. Restore vLLM with the documented recipe and verify health (`/v1/models` 200 and the expected
   served root) before the window closes.

The operational commands are in [`docs/build/operations.md`](build/operations.md).

# Rules

**Status: in force from 2026-10-22.** This is the single source of truth for how work is measured,
gated, reviewed and documented on the speed track. It supersedes, in full:

- `docs/archive/RULES-speed.md` (the old operational copy),
- D53 (operating rules),
- `docs/archive/build/review-protocol.md` (the review section below is the current copy),
- every measurement/process rule stated in an individual `docs/build/` record.

Conflict order: **this file > `docs/DECISIONS.md` IDs > topic handbooks > archived records**.
Decision IDs (D1-D56) remain stable and are cited from source comments; this file does not renumber
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

## 3. Correctness gate (D54, amended by D55 and D56)

The product gate is **numerical**, not byte-identical:

```bash
scripts/numerical-gate.sh --run <device-row-dir> --host-baseline <host-row-dir>
```

**Teacher-forced only.** Both arms are fed the reference's token sequence and their logits are
compared position by position. The gate no longer requires the run's own argmax tokens to match:
one flipped token makes every later position a different question, and the flip is usually a
near-tie, not a defect. What the gate does require is that the run *was* fed the reference
sequence - a run on a different prompt is refused as "not teacher-forced", because every number
below it would be measuring a different question.

**Bars are calibrated to ~1.5x the measured 40-layer device noise floor (D56).** Because the frozen
reference `b08cbf341ad85320` is a snapshot of the host arm itself (`host_arm_floor = 0.0`), `1.5 x
host` degenerates to zero and the declared ceilings in
`docs/build/reference/speed-gate/bars.json` bind:
- `max |delta logprob| <= 8.5` (measured 40-layer device floor: `3.48..5.61`)
- `mean |delta logprob| <= 0.85` (measured floor: `0.51..0.55`)
- `KL(reference || run) <= 0.35` (measured floor: `0.20..0.23`)
- `top-1 >= 0.65`, `top-5 Jaccard >= 0.60`, `top-50 Jaccard >= 0.60` (on fixture `b08cbf341ad85320`,
  4 of 9 positions have reference top1-top2 margins `< 0.40` nats, down to `0.0038` nats at
  position 133)
- Teacher-forced perplexity evaluated on the **reference target sequence** `ref["tokens"]` (never
  each arm's own argmax choices) within **20%** of the host arm (`perplexity_rel_max = 0.20`;
  measured on the coalesced device arm `s1-c2`: `13.4955` vs host `13.7747`, i.e. `-2.03%`)
- Every logit finite.

### Measured result (D56): the gate is PASS

Device arm (`s1-c2` coalesced parallel `layer_seams_sm121` and `s1-devq`) against the frozen twin on
the teacher-forced 138-token fixture `b08cbf341ad85320`, with `s1-hostq` as the host baseline
([`build/s1-window-evidence/README.md`](build/s1-window-evidence/README.md)):

| metric | host arm (`s1-hostq`) | device arm (`s1-c2`, 191 ms locked) | device arm (`s1-devq`) | D56 bar |
| :-- | --: | --: | --: | :-- |
| max abs delta logprob | 0.00e+00 | **5.61** | **4.05** | `<= 8.50` |
| mean abs delta logprob | 0.00e+00 | **0.547** | **0.516** | `<= 0.85` |
| KL(reference \|\| run) | 0.00e+00 | **0.230** | **0.216** | `<= 0.35` |
| top-1 / top-5 / top-50 | 1.000 / 1.000 / 1.000 | 0.667 / 0.668 / 0.627 | 0.889 / 0.688 / 0.664 | `>= 0.65 / 0.60 / 0.60` |
| teacher-forced perplexity (on `ref["tokens"]`) | 13.7747 | **13.4955 (−2.0%)** | **11.9356 (−13.4%)** | within 20% of host (`11.0198..16.5296`) |

Why the 40-layer device floor is ~0.5 nats mean |Δlogprob| and why serialising GPU reductions was
rejected (D56): each token step executes 80 native `ncclAllReduce(ncclSum, ncclBfloat16)`
collectives (rounding in bf16 at each pairwise ring/tree addition rather than once after a host f32
sum) and 80 parallel 256-thread warp/block tree reductions in `mhc_entry_bf16` and `mhc_ffn_seams`.
Across 138 positions x 40 quantized W4A8/W8A8 MoE+MLA layers (5,520 blocks), 1-ULP bf16 differences
flip ~1-2% of the 3-bit-mantissa FP8 E4M3 activation codes and perturb discrete top-6 MoE routing.
Replacing the parallel warp/block tree reductions in `layer_seams_sm121.cu` with single-thread
serial loops to chase CPU bit-identity regressed step latency by **+127 ms/step (+66%: 191 ms ->
318 ms at 1469 MHz)** without changing KL (`0.217` vs `0.230`). The parallel tree reductions are
therefore kept as the canonical S1 implementation.

### Measured host-arm noise floor (D55 step 2)

The host arm was run through the amended gate against the frozen twin reference for fixture
`fab52c00a0dc6f6c` (the existing gate prompt, teacher-forced, all four ranks):

| metric | host arm vs frozen CPU twin |
| :-- | :-- |
| max abs delta logprob | 0.00e+00 |
| mean abs delta logprob | 0.00e+00 |
| KL(reference || run) | 0.00e+00 |
| top-1 / top-5 / top-50 | 1.000 / 1.000 / 1.000 |

Run: the `s1-host` node row, whose path and identity are in
[`build/s1-window-evidence/README.md`](build/s1-window-evidence/README.md) (locked clocks, `scripts/numerical-gate.sh --run
row s1-host`). **The host arm's own distance from the twin is exactly zero**: it is
the same arithmetic, so this fixture cannot produce a nonzero floor. That is a fact about the
fixture, not permission to loosen anything, and it makes the relative rule degenerate on it - with
a zero floor `1.5 x host` is zero and only the frozen ceilings bind. The ceilings are therefore
recorded from the device arm's measured distance on the same fixture (a reassociated
implementation, D51), and they may not grow; the perplexity bar is what keeps the long chain
honest, because a genuine arithmetic defect moves perplexity while a reassociation does not.

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
- **The boundary-diff tooling is a debugger, not a gate** (D55). `scripts/s1-first-diff.py`, the
  per-tensor bounds file `docs/build/reference/s1-device-bounds.json` and the per-rank row dumps
  exist to find the layer that broke *after* the amended gate fails, or when a slice claims bit
  identity. They judge no row on their own and no row is filed against them.
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

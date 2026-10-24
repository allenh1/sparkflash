# Performance

The one published number is **lean decode ms/step, mean over four ranks**, on the frozen argv and a
clean row. Process rules are in [`../RULES.md`](../RULES.md). This page holds the measurement facts
and the cost model; the authoritative row history is [`speed-log.csv`](speed-log.csv).

## Current

| | |
| :-- | :-- |
| Published | **146.03 ms/step** (2026-10-23, `a42b1aa1`, `s2-d`, four-rank mean, D56 numerical gate PASS) |
| Replicate | 145.28 ms/step (`s2-d-rep2`, delta 0.75 ms) |
| Median / clean-step band | 139.00 / 138.50 ms; steady-state steps 1..8 at 138-141 ms |
| Implied rate | 6.85 tok/s mean, 7.19 tok/s median (one token/step) |
| Sustained (1,000-step soak, `s2-e`) | median 151.0 ms, last-100 median 159 ms; the ramp is +13.6% on the **median** (first-100 median 140 -> last-100 median 159). The first-100 *mean* is not usable - it is 273 ms because of the single 13,274 ms stall |
| 90 tok/s | 11.1 ms/step |
| Next gate | **<= 50 ms/step** - 150 is cleared on the mean (`146.03`, and `s1-c1-boost` 142.94 before it), but not on the sustained figure |
| Kernel time bound (what capture cannot remove) | `113-139 ms/step` at boost: floor from the `1500 MHz` probe (`186.397 ms/step`, `s1-result.md` §8.2) scaled by the clock ratio, ceiling from the `139.00 ms` step. `3.2-4.0x` the `35 ms` target - why the step is kernel-bound (D58) |

The number is not comparable across base commits. The published row is `s2-d` on `a42b1aa1` (the S2
device-by-default tree: `SPARK_COMM_REDUCE=nccl`, registered collective buffers, whole layer on
device). That base is **not** main's current tree: !173 squash-merged the S2 branch as `f361c350`
after review fixes that landed on that branch past `a42b1aa1`, so main carries code the row was not
measured on and its effect on ms/step is unmeasured. The S4 probe row re-measures on main, and until
it does, `146.03` is the row for `a42b1aa1` and main's number is unmeasured rather than assumed. The previous published row, `179.81` on `cd233423`, is the host-arm tree before the S2
port, and `s1-c1-boost` (`142.94`) is the S1 40-layer device arm at unlocked boost clocks. Any
candidate must be measured in the same window on the same base.

**Where the time is** (D58): kernel time. *Cannot reach the target before the kernels do.* A step is at least as long as its GPU kernel work.
Measured at `1500 MHz` (`s1-c-probe`, rank 0 at 1482 MHz) the probe gives `186.397 ms/step` over 40
layers + head, but that is not a like-for-like comparison with a boost step: the same kernels cannot
beat the clock ratio (`1.62-1.65x`, most favourably `1.649x`), so at boost their time is at least
`186.397 / 1.649 = 113 ms` (`3.2x` the `35 ms` target) and at most the `139.00 ms` step (`4.0x`).
Either bound keeps `35 ms` out of reach until kernel time falls by at least `3.2x`. The `1500 MHz` row
is not re-measured - no locked run is taken any more, the hardware sits at 2405-2418 MHz - so S4's
boost-clock probe row supersedes it.
So capture cannot reach the target before the kernels do
([`s3-capture-audit.md`](s3-capture-audit.md)).

Two numbers that look like they belong in that argument do **not**, and are labelled here so they are
not quoted as measurements: the collectives' `~3.87 ms` share is an estimate from the sync-mode bench
(`s2-result.md` §3.1 item 1), and per-step host-busy time is **unmeasured** - the step counters record
`d2h_copies` at one call site and assert the allocation/sync counts rather than hooking them
(`device_counters.rs`; instrumenting the driver's copy and stream calls is a prerequisite for any
host-free claim).

## Trajectory

Measured four-rank lean rows. Older rows are listed for shape only; use `speed-log.csv` for the
complete record.

| ms/step | commit | What moved it |
| --: | :-- | :-- |
| 1201.81 | `807bb218` | pre-WS-A baseline, evidence ON |
| 599.94 | `4e4cea50` | WS-A2 async device host path |
| 469.81 | `0d8797db` | M=1 dense re-tile |
| 356.39 | `528b39fb` | decode GEMV for wq_a/wkv/wq_b/wo_b |
| 350.39 | `4f83eee5` | shared-expert wiring (partial; device arm gated) |
| 249.25 | `3813ecbe` | shared expert's f32 reference arm on device, bit-identical |
| 227.31 | `7ecec4c2` | Batch 4 + seam warm pass default-on |
| 224.53 | `54abfd6c` | host-weight GEMV seam arm |
| 213.78 | `7d176935` | attention staging slot |
| 198.78 | `7bf7ec78` | mHC seam host-SIMD projection |
| 186.86 | `3620ac77` | one-pass activation quantiser |
| 179.81 | `cd233423` | seam tails: bit-encoder FP8, cached thread count |

## Where the time goes

### Device/host split

At the 249.25 tree (probe), total device kernel time was **71.2 ms of 249.25** (~29 %). The step is
host-bound per block: `attention_output` was 0.837 ms/layer wall against 0.202 kernel;
`moe_gate_up` 1.894 against 0.637. Only the two GEMV blocks are close to device-bound (62-72 % of
peak bandwidth).

### Block table

Host-only probe on the 224.53 tree, per call, x40 per step. The 179.81 tree removes further seam
tail costs; use this as a ranking, not as current absolute values.

| Block | ms/call | Notes |
| :-- | --: | :-- |
| `ffn_seams` | 0.524 | host SIMD / seam |
| `attention_output` | 0.512 host + 0.552 wait | the wait is the drain |
| `moe_reduce` | 0.378 | cross-rank exchange |
| `mhc_entry` | 0.323 | host SIMD |
| `stage_attention` | 0.321 (later ~0.031) | staging slot fixed most of it |
| `attention_projections` | 0.234 | host quantise dominates |
| `query_path` | 0.094 | |
| `ref_seam` | 0.077 | |

The lean loop also had **12 D2H observation points per layer-position** before the port, and two
cross-rank exchanges whose host machinery was ~29.5 ms/step. The device port removes both.

### What the 90 tok/s target needs

Measured MTP acceptance length is **1.571 accepted/verify step** (host twin, 24/42, 95 % CI
1.333-1.810; p50 1.000, p90 3.000). At that AL, 90 tok/s needs **17.46 ms/verify step**
(`1000 * 1.571 / 90`). At the standing `146.03 ms/step`, MTP would give ~10.8 tok/s (`1000 * 1.571 / 146.03`); the remaining work is the
layer port, not MTP tuning.

## Constraints the measurements established

- **Host arithmetic rework makes the step worse on GB10.** Interleaving independent f32 chains in
  `mhc.rs`/`linalg.rs` (bit-identical) produced 378.4 / 321.7 / 291.8 ms against a 246.7 baseline,
  with a ~230 ms rank-wide stall. CPU and GPU share the memory system and power envelope.
- **A kernel win moves the step only if the block's host share is already below its kernel time.**
  `wo_a` went 0.4879 -> 0.2022 ms/layer (the largest single-kernel win) and the four-rank step stayed
  flat at median 249 vs baseline 246.7, because the host share was 0.60.
- **Driver call count is not wall.** Removing 198,720 D2D copies/step and small-operand H2D changes
  moved nothing on the async path; only calls that force a drain are visible.
- **Remaining levers** are (a) pipelining layer L+1's host seam against layer L's device work, or
  (b) removing host seams/bytes. Speeding them up in place is not a lever.
- **The fabric is not the current limiter.** A sequential-read arm on the host path was +8 ms/step;
  the reader-thread default wins.

## Measurement facts

- **Clean row is mandatory.** Drop the checkpoint from the page cache on all four nodes, zero
  `sparkflash-*`, `MemFree >= ~110 GiB`, no staging during a row. A cold read added ~250 ms/step in
  one measured mode. Record the median and the clean-step band, not only the mean.
- **Noise:** ~+/-5 ms on a clean cluster; a single mode step moves a 9-step mean by ~25 ms. Batch
  changes above ~20 ms expected movement; otherwise A/B/A/B.
- **Warm pass:** `SPARKFLASH_SEAM_WARM` (default on since `d7ffd9f7`) reads seam tensors and Engram
  rows into process memory before position 0. Same tag/argv, cache dropped: 229.33/228.44 with the
  pass vs 255.11/256.44 without (-25.8/-28.0 ms/step).
- **Rejected rows are still data:** posted-reduce lane +18.5 ms/step; mHC device projection
  +46.4 ms/step against its host path; forced sequential fabric read +8 ms/step. They are in
  `speed-log.csv` and should not be re-tried without a new mechanism.

## Estimation rules

- Derive from the hardware floor: weight bytes / peak bandwidth, MACs / peak MACs. Do not scale the
  previous kernel's measured rate.
- Report GB/s achieved against peak next to ms. Single-token launch target: **>= 60 % of peak memory
  bandwidth**.
- "Host path closed" = zero host reference arithmetic per token. That is the precondition for graph
  capture, and no kernel work can remove it.

## Open items

- Whole-layer device port (see [`device-chain.md`](device-chain.md)); no number is published from
  `win/layer-device` until one full layer runs.
- After the layer: CUDA graph capture, GEMV re-tile, then the 150 ms gate.
- MTP acceptance remains a separate lever; 1.571 is the current measured AL.

## Archived sources

`RULES-speed.md` (measurement facts), `perf-model-20261021.md`, `p0-timing-table-20261021.md`,
`p0-timing-preliminary-20261021.md`, `m2-critical-path-profile-20261021.md`,
`m2-execution-plan-20261021.md`, `ws-a-host-path-20261021.md`, `ws-d-proposal-20261021.md`,
`ws-pipeline-design-20261022.md`, `ws-host-bound-night-20261022.md`,
`ws-gb10-host-device-contention-20261022.md`, `device-chain-timing.md`,
`kernel-occupancy-inventory.md`, `d48-window-protocol.md`, `ws-whole-layer-device-port-20261022.md`.

# S2 result: the whole decoder layer on device with the NCCL all-reduce, as the default arm

**Status: CLOSED at the measured kernel-bound figure.** Every acceptance criterion of issue #5 is
met, or met as re-scoped by `D57` where the criterion as written was unattainable on this hardware.
The re-scope is recorded here and in [`../DECISIONS.md`](../DECISIONS.md).

The figure this closes at is stated carefully, because the short rows and the long soak disagree:
**`146.03 / 145.28 ms` mean and `139.00 / 138.50 ms` median over the two nine-step replicates, but
the 1,000-step soak measures a median of `151 ms` and an end-state of `159 ms`** — step time ramps
`140 -> 159 ms` (+13.6%) across that soak on all four ranks (§3.1 item 3). The sustained number is
`151-159 ms`; the ramp is an open finding, not a footnote.

- **Clock protocol (D57).** Issue #5 asks for the S2-a gate and the S2-d replicates under a
  `1500 MHz` clock lock. That lock was retired by operator decision on 2026-10-23: these four GB10
  parts hold `2405-2411 MHz` continuously and rarely throttle, so a 1500 MHz lock measures a clock
  the product never runs at. Every S2-a row is a native-boost row; the S2-d and S2-e rows are
  labelled boost rows but **the row directories do not record their clocks**, so that label comes
  from the window record, not from the artifact. The gate validates the window's clock by requiring
  the **median of the non-idle `clocks.sm` samples** (idle = below 1500 MHz) to reach 90% of
  2411 MHz; across 40 rank-samples those medians are `2398-2554 MHz`. That is a **window-level
  sanity check** — the part was working and nothing pinned it low — and **not** a measurement of the
  clock during the timed collectives, which the samples cannot attribute (10,000 iterations is
  ~0.48 s of a multi-second window sampled at ~4 Hz). Per-collective clock evidence needs an
  instrumented step row; that is an open follow-up.
- **Gate bars and the GDR clause (D57).** The `30 us` / `60 us` ceilings had never been met by any
  arm on this fabric, including the host-staged one-shot RDMA path at `35.70 us`, so they were not a
  regression signal. The gate carries three ceilings, each `1.5x` the measured worst of 40
  rank-samples: `p50 <= 75 us`, `p90 <= 104 us`, `p99 <= 251 us`. Three, because the p99 here is
  **heavy-tailed and not reproducible** — it lands at `78.7-89.6`, `118.9-126.2` and `141.6-167.2 us`
  on healthy fabric with p50 inside 4% and the clock unchanged — so p50 detects a central shift, p90
  is the stable tail bar (`58.35-69.39 us`), and p99 is a catastrophic alarm only, for the
  ARP-broken fabric of this window (`p90 402-448`, `p99 583-668 us`), which fails all three
  ([`fabric.md`](fabric.md) §5.2). GDR is required only on a platform whose driver
  reports support; GB10 reports `GPU_DIRECT_RDMA_SUPPORTED = 0` and `DMA_BUF_SUPPORTED = 0`, so GDR
  is recorded as undrivable rather than failed.

What is **not** re-scoped: the ≤ 60 ms/step target. It stays open and belongs to S4 (§4.2). S2 closes
at the measured kernel-bound figure, because the step's `d2h = 1 / alloc = 0 / sync = 0` invariants
hold and the remainder is GPU kernel time, not communication or host handoff.

Every hardware row in this report was executed on the 4-node DGX Spark GB10 (`sm_121a`, driver
`580.178.04`, CUDA `13.0`) cluster inside window `/home/user/ev-session-window/s2w01` at commit
`a42b1aa1` (the code exactly as measured; the gate script changed after the S2-d rows were taken and
its own row is at `e17f433b`), at native boost clocks with no clock lock. Rows:
the S2-a 10 KiB collective gate (`10,000` samples, 4 ranks, 13 runs at the two post-fix protocol
versions plus the 10 that set the floor), the two teacher-forced
replicates (`s2-d`, `s2-d-rep2`), the S2-e 1,000-step soak (`s2-e`), and the `s2-d` /
`s2-d-rep2` rows in [`speed-log.csv`](speed-log.csv).

## 1. Acceptance criteria

| Criterion (issue #5) | State | Where |
| :-- | :-- | :-- |
| Startup guard: `SPARK_COMM_REDUCE` must be set; NCCL transport config and build in the run header | **DONE** (code + unit tests + verified in `tp4-token-r*.json` and `tp4-forward-embed-r*.json`) | `crates/sparkflash-cli/src/run_header.rs`, `tp4_chain.rs` (`run_loop`), `tp4_forward.rs` (`run_header_record`) |
| Collective buffers registered once at open (`ncclCommRegister`) | **DONE** (`reduce_registered_buffers = 4`, `40,960 B` per rank in `s2-d`, `s2-d-rep2`, and `s2-e`; `2` buffers, `20,480 B` in `gpu-reduce-bench`) | `crates/sparkflash-comm/src/device_reduce.rs` (`RunReduce::register_buffers`), `tp4_forward.rs` (`attach_reduce`, `DeviceLayer` Drop) |
| S2-a gate: `scripts/gpu-reduce-bench.sh gate` passes with buffers registered (clock protocol and bars per D57: native boost, `p50 <= 75 us` / `p90 <= 104 us` / `p99 <= 251 us`, GDR required only where the platform supports it) | **PASS** (`20261024T033025Z`, `valid: true`, `pass: true`, `sparkflash-reduce-gate/3`, `provenance.git_dirty: false`, `admissible_as_floor_row: true`): `10,000` iterations × 4 ranks, `registered_buffers = 2` (`20,480 B`), `differing = 0`, `max_ulp = 0`, `p50 = 47.82-48.03 us`, `p90 = 58.37-58.56 us`, `p99 = 81.81-83.10 us`, enqueue `5.7-5.9 us`; clocks hold a non-idle median of `2411-2418 MHz`; five further clean-tree runs under the same gate (`20261024T031917Z`, `031959Z`, `032040Z`, `032944Z`, `033107Z`) also PASS. `gdr_active = false` and `gdr_required = false`: the driver reports `GPU_DIRECT_RDMA_SUPPORTED = 0` / `DMA_BUF_SUPPORTED = 0`, so GDR is undrivable on GB10 (a note, printed on PASS too, §3.1) | `scripts/gpu-reduce-bench.sh gate`, `target/gpu-reduce/20261024T033025Z/gate-summary.json` |
| `s2-extent-args.md` committed (including DSpark aux layer 37/38/39 tap spans) | **DONE** | [`s2-extent-args.md`](s2-extent-args.md), `crates/sparkflash-cli/src/layer_device.rs` (`DSPARK_AUX_LAYERS`, `dspark_aux_tap_span`) |
| Default arm flip to `device`; `host` stays selectable | **DONE** | `crates/sparkflash-cli/src/layer_device.rs` (`parse_device_switch`, `select_layer_arm`) |
| S2-d: two lean replicates on the default arm within ±2 ms, numerical gate PASS, invariants hold, `s2-d` row in `speed-log.csv` (boost clocks per D57, not the retired 1500 MHz lock) | **PASS**: `s2-d = 146.03 ms` / median `139.00 ms`; `s2-d-rep2 = 145.28 ms` / median `138.50 ms`; **`|Δ| = 0.75 ms ≤ 2.0 ms`**; byte-identical `logits_sha256 = 12a03a86…d17623` on 4/4 ranks × both runs; `numerical-gate.sh` PASS on 4/4 ranks × both runs (`max|d| 5.61 ≤ 8.50`, `mean|d| 0.547 ≤ 0.85`, `KL 0.230 ≤ 0.35`, `ppl 13.4955` vs host `13.7747`); `d2h=1 / alloc=0 / sync=0` on 9/9 steps × 4/4 ranks; both rows appended to [`speed-log.csv`](speed-log.csv) | `scripts/s2-verdict.py replicates` (`target/s2-runs/s2-d-verdict.json`) |
| S2-e: 1,000-step soak, p99 ≤ 1.10× median, 0 NCCL timeouts | **PASS** at boost clocks (`steps = 1000` on all 4 ranks; `median = 151.0 ms`, `p99 = 162.0 ms`, `p99/median = 1.0728× ≤ 1.10×`; `nccl_faults = 0`). Two findings stay open: an **unexplained stall** (step index `87` on rank 0, `88` on ranks 1-3; position `217`) took `13,274 ms`, so the `163.75 ms` mean sits above the p99 and `s2-verdict.py` prints that as a note on every rank; and the soak **ramps** (`140 -> 159 ms`, +13.6%, first 100 steps to last 100, monotonic on all 4 ranks), so S2's sustained figure is `151-159 ms`, not the `139 ms` of the nine-step rows. The bar passes as written; both findings are filed with owners, not waved through (§3.1) | `scripts/s2-verdict.py soak` (`target/s2-runs/s2-e-verdict.json`) |
| `s2-result.md`: gate result, soak summary, S3/S4 handoff | **DONE** (this file) | [`s2-result.md`](s2-result.md) |

## 2. What changed

### 2.1 The default arm is the device arm

`SPARKFLASH_LAYER_DEVICE` now selects the device arm when it is unset, empty, `1`, `device`, `true`
or `yes`. `0`, `host`, `false` or `no` select the bit-exact host arm. Any other spelling, including
one with surrounding whitespace, refuses at startup instead of guessing. Evidence recording
(`SPARKFLASH_FORWARD_EVIDENCE=1`) still forces the host arm, because only the host arm reads back
per position. The arm reasons the record carries changed with the flip. `device-frontier.py`
accepts both the S2 spelling and the S1 opt-in spelling, so archived S1 rows still verify.

### 2.2 The run header and the startup guard

`RunHeader::decide` reads `SPARKFLASH_LAYER_DEVICE`, `SPARKFLASH_FORWARD_EVIDENCE` and
`SPARK_COMM_REDUCE` exactly once, at the top of `run_loop`. It runs before setup and before any
device work. It decides the arm and the collective path together:

| Arm switch | Evidence | `SPARK_COMM_REDUCE` | Outcome |
| :-- | :-- | :-- | :-- |
| unset / `1` / `device` | off | `nccl` | device arm; both per-layer collectives on the run's NCCL communicator |
| `0` / `host` | off | `tcp` or `roce` | host arm; the reference exchange, or the RoCE one-shot |
| any | `1` | `tcp` or `roce` | host arm (evidence forces it) |
| unset / `1` / `device` | off | `tcp` or `roce` | **refused**: the device arm's collectives are NCCL enqueues |
| `0` / `host`, or evidence on | any | `nccl` | **refused**, quoting the host arm's reason |
| any | any | unset, empty or unknown | **refused**: a forward run names its collective path |
| a spelling that names neither arm | - | - | **refused** |

A refusal still writes the rank's `tp4-token-rN.json` with `forward: "refused"`. That file carries a
`run_header` object holding `decided: false` and the raw values the decision saw, so a refused row
records what it was given. On success:

- each rank prints three header lines before setup (arm, path, evidence and the reason; the three
  switches; the NCCL transport variables);
- the token envelope and the forward record both carry `run_header` (`sparkflash-run-header/1`):
  - the arm and its reason, evidence, `reduce_path`;
  - every switch's raw value;
  - every `sparkflash_comm::nccl::TRANSPORT_ENV` variable (`NCCL_IB_HCA`, `NCCL_NET_GDR_LEVEL`,
    `NCCL_NET_GDR_READ`, `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME`,
    `NCCL_DMABUF_ENABLE`, `NCCL_LOCAL_REGISTER`), `null` when unset;
  - `nccl_build`, which is `{version, library}` from the opened communicator, or `null` on the host
    arm.

The binding never sets these variables; NCCL reads them itself at init. The forwarding list in the
gate script is pinned to the same constant by a unit test, so the gate and the run cannot record
different sets.

The staged library is unchanged: `libnccl.so.2` 2.31.2 built against CUDA 13.0 (sha256
`aeda7a87…`). vLLM's container ships 2.30.7+cuda13.3. The header records the version the run
actually loaded, so a later library swap is visible in the row rather than inferred.

### 2.3 Registered collective buffers

`DeviceLayer::attach_reduce` registers the four collective spans with the communicator once at open,
before position 0: `attn_out_bf16`, `reduced_bf16`, `moe_delta_bf16` and `moe_reduced_bf16`, which
are 10,240 B each. `register_buffers` refuses an empty list, a second registration while one is
held, and a null or empty span. If one registration fails part-way, the ones already made are
released first, so a refused open holds nothing. NCCL refusing a registration refuses the run. The
only opt-out is NCCL's own `NCCL_LOCAL_REGISTER=0`, and the header records it.

`DeviceLayer`'s Drop synchronises, deregisters, and only then frees the slab. `RunReduce`'s Drop
deregisters anything left before it destroys the communicator.

The forward record's `layer_device_memory` gains `reduce_registered_buffers` (expected 4) and
`reduce_registered_bytes` (expected 40,960). They are not in the envelope, because the envelope is
written after the layer has dropped and deregistered.

**Expected step effect: about 0 ms/step.** At 10 KiB NCCL runs the LL/LL128 protocols, which stage
through its own FIFO buffers, so user-buffer registration buys nothing at this payload. The reason
to do it is S3: a captured graph's NCCL node needs fixed, registered addresses.

### 2.4 The S2-a gate (`scripts/gpu-reduce-bench.sh gate`)

The gate measures the device arm only. Its protocol is D57's: **native boost clocks, no clock
lock**.

- 10,240 B per rank, 10,000 timed iterations at the nodes' own boost clock. It clears any lingering
  artificial lock (`nvidia-smi -rgc`), records each rank's `clocks.max.graphics`, and samples
  `clocks.sm` at ~4 Hz through the window. It refuses nothing for being unlocked; instead it is
  **INVALID** (not PASS) when a rank never leaves its idle clock, or when a rank's **median of the
  non-idle samples** (`≥ GATE_IDLE_MHZ = 1500`) falls below 90% of `GATE_BOOST_MHZ = 2411`. The
  mean-of-all-samples statistic is deliberately not used: a boost-clock window is mostly NCCL init
  and idle (`201–1150 MHz` samples), and averaging those in failed a good run (rank 2, mean
  `1949 MHz` against a `2411 MHz` target). Measured: non-idle medians of `2398–2554 MHz` across 24
  rank-windows.
- `gpu-reduce-bench` registers its send and receive buffers before the warm-up, which is the run's
  shape. It records `registered_buffers` and `registered_bytes` in `stats` and the transport
  environment in `nccl.env`.
- The launcher forwards the operator's `NCCL_*` transport variables verbatim, and dies on a value it
  cannot quote. It runs each rank with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` written to
  `NCCL_DEBUG_FILE`, never stdout, and collects `nccl-rank<r>.log`.
- Rank 0 is launched first and the launcher **waits for its control port** before starting ranks
  1-3: a peer that lost that race used to get `ECONNREFUSED` and exit while three ranks waited.
- Before anything is timed, every rank pings every peer on **both** rails and the gate settles for
  3 s. Measured in this window: two runs taken immediately after an ARP flush had p99 `161-167 us`
  where the same script on a settled fabric had `78.7-87.5 us`. A 2x tail difference caused by
  neighbour state is an instrument defect, not a fabric number, so the gate controls that state and
  records `fabric.warmed` in the summary.
- The verdict `gate-summary.json` (`sparkflash-reduce-gate/3`) is PASS only when all of these hold:
  - **p50 ≤ 75 µs, p90 ≤ 104 µs and p99 ≤ 251 µs**, each `1.5x` the measured worst of 40
    rank-samples (D57). The verdict **re-derives all three** from the reported numbers and never
    trusts the bench binary's own `pass` boolean; a binary that disagrees with its own numbers is
    reported as such.
  - the reduction is correct, not just fast: the bench's check must read `differing = 0`,
    `max_ulp = 0` and `finite`, and the rank must have timed at least the requested number of
    collectives (a truncated sample is not a percentile);
  - **registered:** ≥ 2 buffers and ≥ 2 × 10,240 B on every rank;
  - **GDR:** active on every rank whose driver reports `GPU_DIRECT_RDMA_SUPPORTED` (the gate probes
    attribute 116 and 124 per rank and records them in `platform-rank<r>.json`). A rank that does
    not support it records `gdr_required: false` in the summary instead of failing a clause its
    hardware cannot satisfy; an **unprobed** rank keeps the clause required;
  - one NCCL build and one transport environment across the four ranks;
  - clock validity as described above;
  - every rank logged at least one NET channel line (otherwise the transport that carried the
    collective is unconfirmed).
- Exit codes are 0 PASS, 1 FAIL, 2 INVALID. Notes about what the gate could **not** establish are
  printed on PASS as well as on FAIL: a green line must not hide a limitation.
- The `EXIT` trap reaps any bench left on any node, whatever way the gate exits.

**GDR is a hardware fact on this rig, not a tuning gap.** [`fabric.md`](fabric.md) records that
`nvidia-peermem` is present but the driver rejects it (`EINVAL`), and window `s2w01` confirmed the
DMA-BUF path is closed too: `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED == 0` and `libmlx5.so` has no
`mlx5dv_reg_dmabuf_mr`. So GDR cannot be activated by configuration, `gdr_active` is `false` on
every row, and D57 moved the clause onto platform capability. What the gate still proves is the
part that can regress in code: registration, one build and one transport env, the latency floor,
`max_ulp = 0`, and the clock the collectives ran at.

**History in this window, because the branch carries it.** Commit `a42b1aa` first made the gate
boost-clock (that part is kept as D57), and a review commit `de90726` restored the 1500 MHz lock;
`dc3a034` then reinstated the boost protocol, and `e17f433b` set the measured bars and the
capability-gated GDR clause. The S2-d rows were measured before the gate script moved and are not
affected by it. They do **not** record their clocks: a row directory starts with the step JSONL and
the verdict, and neither carries a clock, so the boost-clock label on those rows comes from the
window record rather than from the artifact (D57's gate clause applies to gate rows only).

## 3. Results

All rows below were measured on the 4-node DGX Spark GB10 (`sm_121a`, driver `580.178.04`, CUDA
`13.0`, `libnccl.so.2` `2.31.2+cuda13.0` SHA-256
`aeda7a87ab4c3707216d428c6f4dee0f178ed87a6ed76f5bd6be9639e737a51a`) cluster inside window
`/home/user/ev-session-window/s2w01` at commit `a42b1aa1` on `sparkflash` binary SHA-256
`d427c39aede12880385cc9206f9711d4d57b5950693eba08c354808b2f82365f` and `gpu-reduce-bench` binary
SHA-256 `c908a13f907c9be074700d7bf08c843c7cd625351ddc322228c5426e423c67e0`, in the standard
two-default, two-`/24` routing on `192.0.2.0/24` with RoCEv2 GID 3 active on `rocep1s0f1`
(`enp1s0f1np1`) and `roceP2p1s0f1` (`enP2p1s0f1np1`), and native boost clocks (`2405–2411 MHz`, no
1500 MHz clock lock).

| Row | What | Result |
| :-- | :-- | :-- |
| S2-a gate (`20261024T033025Z`) | 10,240 B (`5,120` bf16) NCCL sum, 4 ranks, `10,000` samples (`batch=40`), registered buffers (`2` × `10,240 B = 20,480 B`), native boost clocks, warm fabric, `sparkflash-reduce-gate/3`, `provenance.git_dirty: false` | **`PASS` — `valid: true`, `pass: true`, `gdr_required: false`, `registered_buffers: 2` (`20,480 B`), `differing: 0`, `max_ulp: 0`, `p50_us: 47.84 / 47.82 / 48.03 / 47.84` (enqueue `5.7-5.9 us`), `p90_us: 58.48 / 58.37 / 58.56 / 58.50`, `p99_us: 81.81 / 81.83 / 83.09 / 83.10`, non-idle clock median `2411 / 2418 / 2411 / 2411 MHz`, `gdr_active: false` (`0/128` NET channels via GDRDMA) with the driver reporting `GPU_DIRECT_RDMA_SUPPORTED = 0`, `DMA_BUF_SUPPORTED = 0`** |
| S2-a gate floor | 10 runs under the post-fix protocol, 40 rank-samples ([`fabric.md`](fabric.md) §5.2) | **`p50 = 47.89-49.76 us`, `p90 = 58.51-69.39 us`, `p99 = 78.72-167.20 us`; D57 ceilings `p50 <= 75` / `p90 <= 104` / `p99 <= 251` = `1.5x` the measured worst. p90 is the stable tail statistic (`58.35-69.39 us` over all 16 rows); p99 is heavy-tailed (three levels, `78.7` to `167.2`, not reproducible) and is an alarm only** |
| `s2-d` | device arm by default (`SPARKFLASH_LAYER_DEVICE` unset), `SPARK_COMM_REDUCE=nccl`, 138-token teacher-forced list (`token-ids.txt` SHA-256 `458a05925a3a8c2e8f59b4860d3dfdfb2622a3eae959dac4367e0b6c0777f177`), 9 decode positions, `--forward-lean`, boost clocks (`2405–2411 MHz`) | **`146.03 ms` four-rank mean, `139.00 ms` four-rank median** (rank means `146.22 / 146.00 / 146.00 / 145.89 ms`; rank 0 samples `[200, 139, 139, 139, 141, 139, 139, 140, 140] ms`; steady-state `138–141 ms`) |
| `s2-d-rep2` | the same, back to back | **`145.28 ms` four-rank mean, `138.50 ms` four-rank median** (rank means `145.22 / 145.33 / 145.33 / 145.22 ms`; rank 0 samples `[202, 137, 138, 140, 138, 137, 139, 139, 137] ms`; steady-state `137–140 ms`) |
| replicate agreement | `python3 scripts/s2-verdict.py replicates --run target/s2-runs/s2-d --run target/s2-runs/s2-d-rep2` | **`PASS` (`\|Δ four-rank mean\| = 0.75 ms ≤ 2.0 ms`; byte-identical `logits_sha256 = 12a03a861bb6f4644e65a525bb55870c6b528581c28845a371c48dcd4ed17623` across all 4 ranks × 2 runs)** |
| numerical gate | `scripts/numerical-gate.sh --run target/s2-runs/s2-d target/s2-runs/s2-d-rep2 --host-baseline docs/archive/build/s1-hostq` | **`PASS` on all 4 ranks × 2 runs (`max\|d\| = 5.61 ≤ 8.50`, `mean\|d\| = 0.547 ≤ 0.85`, `KL = 0.230 ≤ 0.35`, `ppl = 13.4955` vs host `13.7747`, `top1 = 0.667`, `top5 = 0.668`, `top50 = 0.627`)** |
| step invariants | `d2h=1 / alloc=0 / sync=0` on every head position, all ranks, plus `reduce_registered_buffers = 4` (`40,960 B`) | **`PASS` (`d2h_copies=1 (129,280 B), allocations=0, sync_waits=0` on 9/9 decode steps × 4/4 ranks in `s2-d` and `s2-d-rep2`, and on 1,000/1,000 decode steps × 4/4 ranks in `s2-e`)** |
| S2-e soak | `python3 scripts/s2-verdict.py soak --run target/s2-runs/s2-e` (`--decode-steps 1000`, `token-ids-1129.txt` SHA-256 `3af6e148c9f8c8048d85dceecfe0c46d24be3428362c79186bf08a53be08bdf8`: 130-token prompt + 1,000 timed decode steps = `1,129` positions, `89,840` NCCL collectives/rank) | **`PASS` (`steps = 1000`, `nccl_faults = 0`, `median_ms = 151.0 ms`, `p99_ms = 162.0 ms`, `p99/median = 1.0728× ≤ 1.10×` on all 4 ranks; `step_mean_ms = 163.75 ms`)** |

### 3.1 Analysis of the S2-a gate and S2-d / S2-e results

1. **The S2-a gate now passes, and GDR on GB10 is a hardware fact (D57).** The gate run at
   `20261024T023439Z` (`valid: true`, `pass: true`, `git_head e17f433b`) puts four ranks through
   `10,000` iterations of the `10,240 B` collective with both buffers registered:
   `p50 = 48.37–48.45 us`, `p99 = 79.95–81.02 us`, enqueue `5.7–5.9 us`, `max_ulp = 0`,
   `registered_buffers = 2` (`20,480 B`), and a non-idle clock median of `2489–2554 MHz`.

   The two clauses that used to fail were re-scoped by `D57`, and both re-scopes are about the bar
   being wrong rather than the path being slow:

   - **Latency ceilings.** `30 us` / `60 us` were never met by *any* arm on this fabric, including
     the host-staged one-shot RDMA path at `35.70 us` p50 (D29). Ten 10,000-sample runs across four
     ranks (40 rank-samples) measure `p50 = 47.89-49.76 us`, `p90 = 58.51-69.39 us` and
     `p99 = 78.72-167.20 us`; the ceilings are `1.5x` the measured worst case — `p50 <= 75 us`,
     `p90 <= 104 us`, `p99 <= 251 us` — the rule D56 used. Three bars, because the p99 is heavy-tailed
     here (three levels from `78.7` to `167.2 us` with p50 inside 4% and the clock unchanged): p50
     catches a central shift, p90 is the stable tail bar, p99 is a catastrophic alarm. The broken state this
     window actually produced (cross-interface ARP: `p50 56.7-58.9`, `p90 402-448`,
     `p99 583-668 us`) fails all three by 4-5x, so the bar still bites. The full floor table, with the
     rows that are context rather than floor, is in [`fabric.md`](fabric.md) §5.2.
   - **GDR.** `cuDeviceGetAttribute` reports `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED == 0`
     (attribute 116) and `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED == 0` (124) on every GB10 node; the
     system `/lib/aarch64-linux-gnu/libmlx5.so` lacks `mlx5dv_reg_dmabuf_mr` (`version MLX5_1.25`),
     and `nvidia-peermem` is rejected by `modprobe` (`EINVAL`). No NCCL setting can produce a
     `/GDRDMA` channel here, so the gate now reads those attributes per rank and requires GDR only
     where support is reported. GB10 records `gdr_required: false` **with the probe output beside
     it**; a rank whose probe does not run keeps the clause required, and a supporting platform with
     inactive GDR still fails. The collectives therefore run over dual-rail RoCEv2 (`NET/IB/0`,
     `NET/IB/1`) through the UMA host proxy (`RMA_IB_PROXY`).

   **Cost of the proxy, in the step.** Across 80 collectives per decode step, measured collective
   time is about `80 × 48.4 us ≈ 3.87 ms/step`, `~2.8%` of the `138.5 ms` steady-state step (an
   estimate from the sync-mode bench, not a step-level measurement). The collective is not what
   bounds S2; the kernels are (§3.1 item 2).

2. **Kernel-bound step latency (`S2-d`).** At native boost clocks, the two lean replicates on the
   default `device` arm measure **`146.03 ms`** (`s2-d`, median **`139.00 ms`**, steady-state steps
   1..8 at `138–141 ms`) and **`145.28 ms`** (`s2-d-rep2`, median **`138.50 ms`**, steady-state steps
   1..8 at `137–140 ms`), agreeing within **`0.75 ms`** (`≤ 2.0 ms` bar). They are `2.3–3.1 ms`
   (mean) and `3.5–4.0 ms` (median) slower than S1's boost row (`s1-c1-boost`: `142.94 ms` mean /
   `135.00 ms` median); both are boost-clock rows, so the gap is the S2 change and is not yet
   explained. `d2h = 1 / alloc = 0 / sync = 0` holds on every decode step, so the step is kernel-bound
   exactly as in S1 §8.2. S2 therefore closes at the nine-step `139 ms`, with the caveat that the
   1,000-step soak measures `151-159 ms` (§3.1 item 3): the short rows are the kernel-bound figure,
   the soak is the sustained one, and the gap is the ramp. The
   `≤ 60 ms/step` target is S4's (§4.2).
3. **1,000-step stability soak (`S2-e`).** Sizing `DeviceLayer`'s compressed KV `store_codes` and
   `store_scales` slabs to `max_positions.max(EXPANDED_ROWS)` (instead of a fixed 512-row cap, now
   funnelled through one `seam_store_owner_bytes` helper so the three call sites cannot disagree on
   the stride) allows the 1,000-step soak (`1,129` total positions, `89,840` NCCL collectives per
   rank) to complete with **`0` NCCL errors/warnings/timeouts** and
   **`p99 = 162.0 ms = 1.0728×` the `151.0 ms` median** (`≤ 1.10×` ceiling) on all four ranks.

   **One real stall, filed not waved.** The bar passes, but one step took **`13,274 ms`**: step index
   `87` on rank 0 and `88` on ranks 1-3 (the ranks' step counters are offset by one at that point),
   position `217` in the `1,129`-token sequence while the other 999 steps stayed in `137–172 ms`, so `step_ms.mean_ms` is
   `163.75 ms` and now sits above the p99. `s2-verdict.py` reports that as a **note on every rank**
   rather than a failure, because the bar as written does not look past the p99:
   `(mean - median) × steps` ≈ `12.75 s` of excess over the median in that series, and the ≤ 10 steps
   above the p99 carry ≥ 7 s of it. It is not a cold first step (step 0 is `211 ms`), it does not
   recur in 999 later steps, and the kernel log around the window shows no OOM, reset or thermal
   event. **S2-e passes as specified; the stall is open as a follow-up** and S3/S4 must not treat the
   soak as stall-free until it is explained. Candidate causes not yet tested: page-cache eviction on
   the disk-resident shards, host writeback, and an allocator stall in unified memory.

   **The soak also ramps, and that changes the figure S2 closes at.** Step time is not stationary over
   the 1,000 steps: the first 100 average `140 ms`, steps 400-499 average `150 ms`, and the last 100
   average `159 ms` (+13.6%), monotonically and on all four ranks. The all-soak median is `151 ms` and
   the end state is `159 ms`, so the honest sustained figure is **`151-159 ms`**, not the `139 ms` the
   two nine-step replicates suggest (those are short, warm, and start from a clean KV store). The
   `p99 = 162 ms` bar still passes because it is measured against the same drifting series; a bar that
   passes *because* the series drifts is not evidence of stability, so the ramp is recorded as a
   finding rather than absorbed into the headline. Candidate causes: the compressed KV store growing
   with position (more seam traffic per token), and allocator/page behaviour under a 1,129-position
   store. **Owner: S3 (the seam/attention work), which is the stage that can instrument the split
   between seam traffic and kernel time; the stall is filed with S4's window queue** - both are open
   items in §4.2, not hand-waved notes, because `docs/RULES.md` §5 requires an owner for anything a
   record leaves open.

   **Fabric note for any future window.** Several gate runs in this window hung in NCCL rendezvous
   (`0 of 4 ranks finished`) and were reaped by the gate's `EXIT` trap. The cause is the pre-existing
   dual-rail fragility: both RoCEv2 rails share `192.0.2.0/24`, so a stale ARP entry sends a peer's
   rail-2 traffic out the rail-1 interface. `gate_cleanup` now reaps orphaned benches on every exit
   path, `gate_launch` waits for rank 0's control plane before the peers dial it (a peer that lost
   that race got `ECONNREFUSED` and left three ranks waiting), and the two hung runs recovered after
   an ARP flush on all four nodes. Also corrected in this window: node `.80` was missing its rail-1
   `192.0.2.0/24` route while the other three had both, and the per-rail `/32` overrides used
   earlier in the window were removed so all four nodes are back to the standard two-default,
   two-`/24` routing.

## 4. Handoff

### 4.1 To S3 (graph capture)

[`s2-extent-args.md`](s2-extent-args.md) is the entry list. In order:

1. **Stream (H1).** Move every launch and both NCCL enqueues onto one run-owned non-blocking stream.
   The driver's capture segment already moves kernel launches onto its private stream. The NCCL
   enqueue is passed stream 0 explicitly, so a capture today would miss both collectives.
2. **Position word (H4).** Put the step counter in a device word. Then give the pointer and word
   kernels (rope offsets, ring slots, `n_pos`, `window_rows`, `selected_count`, `position`) the
   device word to read.
3. **Topology.** Hold the graph's topology fixed across positions: `reference_seam`'s compress
   branch and `ref_select`'s `n_pos == 0` skip are host branches.
4. **Input and output.** Replace the begin-position H2D with a device embedding gather (H2). End the
   graph before the head's D2H (H3).

Registration (2.3) and the header guard (2.2) are already in place.

### 4.2 To S4 (GEMV / MoE re-tiling)

The S1 clock-locked kernel table is the baseline. It is the S1 measurement at `1500 MHz`, not
rescaled; S2 changed no kernel, and its own rows are boost-clock rows, so the shares below are the
right guide to *where* the time is but the absolute figures are the 1500 MHz ones. At boost clocks
the whole step is `138.5–139.0 ms` (steady-state median).

| Stage | ms/step (40 L, 1500 MHz) | Share |
| :-- | --: | --: |
| `moe_routed` | 69.20 | 38.6% |
| `attention_output` (`wo_a` + quant + `wo_b`) | 33.61 | 18.8% |
| `moe_router` | 13.72 | 7.7% |
| `attention_body` | 11.71 | 6.5% |
| `moe_shared` | 10.31 | 5.8% |
| `query_path` | 9.12 | 5.1% |
| everything else | 31.54 | 17.5% |
| **40-layer total** | **179.22** | 100% |
| head (`screen`) | 7.18 | - |

**Open items S2 hands forward, each with an owner** (`docs/RULES.md` §5):

| item | evidence | owner |
| :-- | :-- | :-- |
| step time ramps `140 -> 159 ms` over 1,000 steps | `s2-e`, monotonic on all 4 ranks (§3.1 item 3) | **S3** - instrument the seam/attention split; S4 re-tiles what S3 shows |
| one `13,274 ms` stall (step `87`/`88`, position `217`), cause unknown | `s2-e`, not reproducible in 999 later steps | **S4** - the next window that can afford a fault-injection run |
| S2-d rows are `2.3-3.1 ms` (mean) slower than S1's boost row `s1-c1-boost` | both boost rows; gap unexplained (§3.1 item 2) | **S3** - `device` arm seam vs S1 host seam |
| the S2-a clock clause is window-level, not per-collective | `clock.scope` in every gate summary (§2.4) | **S4** - an instrumented step row with per-collective clock |

Reaching 60 ms at boost clocks means removing about `78 ms` of the measured `138.5–139.0 ms`
step, and `moe_routed` and `attention_output` together account for `57.4%` of the 40-layer kernel
time, so those two stages are most of the target. The clock-locked equivalent is ~`126 ms` of the
`179.22 ms` table.

### 4.3 To S5 (DSpark)

`LayerBuffers` now carves a fixed 3-row `workspace.dspark_aux_taps` device span (`DSPARK_AUX_LAYERS
= [37, 38, 39]`, `3 × dim × 4 = 61,440 B` at `dim = 5120`, exposed via
`LayerBuffers::dspark_aux_tap_span(layer, dim)`; see [`s2-extent-args.md`](s2-extent-args.md) §3).
S5 only needs to enqueue a fixed-offset `d2d` copy (or tap kernel) at the end of layers 37, 38, and
39 into those pre-carved addresses, and widen the allocation by `K_verify` when multi-position
verification blocks are captured.

## 5. Reproduction commands used in window `s2w01`

All rows were executed on the four nodes inside window `/home/user/ev-session-window/s2w01`
([`operations.md`](operations.md) §2) with `sparkflash` SHA-256
`d427c39aede12880385cc9206f9711d4d57b5950693eba08c354808b2f82365f` and `gpu-reduce-bench` SHA-256
`c908a13f907c9be074700d7bf08c843c7cd625351ddc322228c5426e423c67e0`.

**S2-a gate** (native boost clocks, no lock; the gate clears any lingering lock itself):

```bash
NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  NCCL_NET_GDR_LEVEL=5 NCCL_DMABUF_ENABLE=1 \
  scripts/gpu-reduce-bench.sh gate
```

**S2-d replicates (`s2-d` and `s2-d-rep2`):**

```bash
python3 scripts/s2-verdict.py replicates \
  --run target/s2-runs/s2-d \
  --run target/s2-runs/s2-d-rep2 \
  --out target/s2-runs/s2-d-verdict.json
scripts/numerical-gate.sh \
  --run target/s2-runs/s2-d target/s2-runs/s2-d-rep2 \
  --host-baseline docs/archive/build/s1-hostq
```

**S2-e 1,000-step soak (`s2-e`):**

```bash
python3 scripts/s2-verdict.py soak \
  --run target/s2-runs/s2-e \
  --out target/s2-runs/s2-e-verdict.json
```

## 6. Verified on the host (no GPU)

- `cargo clippy` with `-D warnings` is clean for `sparkflash-cli` (default, `cuda`, `rdma`) and
  `sparkflash-comm` (default, `rdma`). The `clippy::single_element_loop` in `ffn_seams` that
  predates this branch is fixed in the review commit.
- `RUSTDOCFLAGS=-Dmissing_docs cargo doc` is clean. The remaining intra-doc-link warnings all
  predate this branch.
- Unit tests:
  - `run_header` (the refusal matrix above);
  - the registration account (setup only, never the collective counters);
  - the NCCL register/deregister wrappers (round trip, a refused or null registration is an error,
    an empty buffer never reaches NCCL);
  - the transport-env pin against the gate script;
  - the arm switch spellings;
  - the compressed-store sizing (`seam_store_owner_bytes`: the kernel module's 512-row store at or
    below 512 positions, one row per position above, overflow refused);
  - the DSpark tap spans (three fixed rows for layers 37, 38 and 39, none for any other layer);
  - `scripts/tests` (the device-frontier fixtures regenerated for the S2 arm string;
    `test_s2_verdict.py`, plus `s2-verdict.py --self-test`).
- `RunReduce::register_buffers`' own refusals and partial-failure release need a live communicator,
  so the first hardware open is their first run.
- The gate verdict logic itself has no committed unit test. It was exercised on hardware (9 real
  gate runs, plus 2 that exercised the INVALID and the hang/timeout paths) and on synthetic rank
  JSONs and logs: PASS, GDR-required-but-inactive, GDR-unsupported-but-inactive (the GB10 case),
  missing probe (clause stays required), socket-only, unregistered, env mismatch, clock
  below-target, and no-non-idle-samples. The `s2-verdict.py` notes are covered by that script's
  `--self-test`.
- Failures not caused by this branch (they fail identically on `main`'s tip on this host):
  - the `nccl_bootstrap` and TCP-loopback tests in `sparkflash-comm` and the `tp4_host` /
    `tp4_smoke` tests in `sparkflash-cli`, which time out on macOS loopback;
  - `test_serve_output_probe.py::test_proc_io_parsing_and_sampling`, which needs Linux `/proc`.

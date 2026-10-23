# S1 layer collectives: device-arm audit, the declared host-bridge, and the 10 KiB gate

Milestone S1 ("first device-only layer"), workstream P4. This note carries the three P4 deliverables:
the audit of the landed per-layer all-reduce against the pinned contract, the **declared host-bridge
fallback** (typed, counted, opt-in, never a fallback), and the **explicit microbench gate** for a
10 KiB GPU→GPU all-reduce.

**Status of the numbers in this note: none of them is a measurement.** This machine has no GPU (no
`/dev/nvidia*`), no ssh route to the four nodes and no `libnccl`, so no collective was run, no gate
was run and no latency is quoted. What *was* run is listed in [§5](#5-what-was-run-here), and every
run-only claim below is marked **UNRUN**. Read-only audit means exactly that: `crates/sparkflash-cli/**`
and `crates/sparkflash-kernels/**` were read, not modified by P4; all P4 edits are inside
`crates/sparkflash-comm/**`, `scripts/gpu-reduce-bench.sh` and this file.

## 1. Provenance

- F1's fix (the run-boundary async-failure check) was added after the audit, on the same working
  tree; §2.2 F1 says so and the audit table marks it F1-fixed.
- Working tree commit at audit time: `d330e5b8` ("P0/F1b: derive the mhc_seam bind from the expected
  backbone name set"). The collective contract itself was pinned at `60bc0646` ("!168 Device arm
  batch") and is unchanged since: `tp4_forward.rs` differs from `60bc0646` by a five-line
  `ref_select` staging fix at `tp4_forward.rs:11827`, far from the reduce path. Line numbers below are
  the `d330e5b8` working tree.
- Contract sources: the module docs of `crates/sparkflash-cli/src/layer_device.rs`
  ("Reduce init, warm-up and accounting (comm-lead's API, pinned here)", lines 87-121) and
  `docs/archive/build/ws-whole-layer-device-port-20261022.md` §"All-reduce contract" (line 143) and
  §"warm-up/arena accounting" (lines 232-270).
- The port's collectives are exactly two per layer (`layer_device.rs:167`
  `COLLECTIVES_PER_LAYER = 2`), 5,120 bf16 elements each (`layer_device.rs:170`
  `REDUCE_ELEMENTS = 5_120`, i.e. 10,240 B per rank per collective): the attention-output delta and the
  MoE routed-row delta. A 40-layer position set is 80 collectives per step.

## 2. Audit: does the landed device arm match the contract?

### 2.1 Verdict

Static audit of the reduce path, against the eleven pinned requirements:

| # | Contract item | Verdict | Evidence |
|---|---|---|---|
| 1 | Enqueue-only: one `ncclAllReduce` per collective on the caller's device handle, no sync, no readback | **Met** | `tp4_forward.rs:11455-11490` → `device_reduce.rs:618-639` → `nccl.rs:739-766`; the only entry points are the two stage bodies and the bench |
| 2 | Count in bf16 **elements** (never bytes), `ncclBfloat16`, `ncclSum` | **Met** | `tp4_forward.rs:11466` (`count = self.dim`, checked against span bytes at 11467-11477); `nccl.rs:745-755` |
| 3 | Out-of-place over the layer's own carved spans | **Met** | `tp4_forward.rs:12267-12268` (`attn_out_bf16` → `reduced_bf16`) and `12733-12734` (`moe_delta_bf16` → `moe_reduced_bf16`); consumers read the written span (`12332`, `12791`) |
| 4 | No per-call allocation, copy or host sync | **Met** | `reduce_row` does borrow + span checks + one enqueue; `probe::count` allocates only on the first insert of a stage (`tp4_forward.rs:11488`, map created once at start) |
| 5 | One communicator per run, opened at setup, closed once at teardown | **Met** | `tp4_chain.rs:974-1054` opens once inside the run closure, attaches it via `attach_reduce` (which refuses a second handle, `tp4_forward.rs:11531-11542`); `RunReduce::Drop` destroys the comm; `close(handle)` is never called on it |
| 6 | Warm-up: 3 in-place dummy reduces on the loop's stream + one setup sync, outside the timed region | **Met** | `device_reduce.rs:655-676` with `WARM_UP_COLLECTIVES = 3` (542); `tp4_reduce.rs:143-145`; not counted (`device_reduce.rs:645`) |
| 7 | Identical collective order on every rank (no rank-conditional skip) | **Met in code**, guarded locally | fixed stage sequence `layer_device.rs:1579-1602`; the stage bodies have no rank branch; per-rank count refused at the run boundary (`tp4_forward.rs:14293-14303`, `layer_device.rs:1187`); cross-rank comparison is offline only — see **F7** |
| 8 | Arm fixed at open, never switched mid-run; no silent fallback | **Met** | `ReduceArm` is a field fixed by `open_with_arm` (`device_reduce.rs:581-603`); a missing handle refuses typed at the stage (`tp4_forward.rs:11461-11465`) — see **F8** for when that refusal fires |
| 9 | Only the logits cross PCIe in steady state | **Met** | no D2H in the reduce path; stats are counters (`tp4_forward.rs:14345-14352`); the declared bridge is the single opt-in exception (§3) |
| 10 | Counters are what the port reports (`collectives`, `elements`, `bytes`, `arena_bytes`) | **Partially met** — counters are produced but the run record prints a constant instead of `arena_bytes` | **F2** |
| 11 | Run boundary: step sync, **NCCL async-failure check**, then stats read, then close | **API met (F1-fixed)**: the check is landed and tested in `sparkflash-comm`; the one-line boundary call is the Lead's pending `tp4_forward` change | **F1** |

### 2.2 Findings, with exact fixes

**F1 (FIXED in `sparkflash-comm`) — the run-boundary NCCL async-failure check now exists.**
The contract says (item 4) "Run boundary: the step's own sync (the only one), an NCCL async-failure
check (failure detection, not completion), then the stats read" (`device_reduce.rs:36-46`), and the
same is written into `layer_device.rs:65-71`. It was absent: `ncclCommGetAsyncError` was not in
`REQUIRED_SYMBOLS` and there was no call site anywhere. Landed fix (Lead decision: implement the
promise rather than amend it):

- `crates/sparkflash-comm/src/nccl.rs`: new `ASYNC_ERROR_SYMBOL` ("ncclCommGetAsyncError"), resolved
  **optionally** in `RawApi` — a library without it still loads and runs, and
  `NcclApi::has_async_failure_check()` reports the check as unavailable. New `NCCL_IN_PROGRESS = 7`
  (ABI-pinned). New `NcclApi::check_async_failure(handle)`: absent symbol → `Ok(())` (documented
  no-op, never a failure); the query call itself failing → `NcclError::Call`; communicator status
  `ncclSuccess` or `ncclInProgress` → `Ok`; anything else → the new typed
  `NcclError::AsyncFailure { result, detail }`, whose Display text states "failure detection, not
  completion" and the all-rank abort rule.
- `crates/sparkflash-comm/src/device_reduce.rs`: `RunReduce::check_async_failure()` +
  `RunReduce::async_failure_check_available()`, and the same pair on `DeviceReduce`, documented as
  *call after the step's sync and before the stats read*.
- CPU tests, no library and no GPU: `a_missing_async_error_symbol_is_a_documented_no_op` (a library
  without the symbol loads and the check is a no-op), and
  `the_async_failure_check_reports_only_real_asynchronous_failures` (success and `ncclInProgress` are
  `Ok`; asynchronous code 2 becomes a typed `AsyncFailure` whose text says so; the query's own failure
  is `Call`). The fake symbol table supplies callable `ncclCommGetAsyncError`/`ncclGetErrorString`
  stand-ins, so the mapping is exercised without NCCL.

*Remaining for the Lead (`tp4_forward`, not P4's to edit):* call `run.check_async_failure()` at the
run boundary after the teardown sync and before the stats/record read, and state
`async_failure_check_available()` in the record so a library without the symbol is reported as
"step sync only" instead of being silently credited with a check it could not perform. The API and
its tests are landed; that boundary call is F1's last mile.

**F2 — the run record prints the arena *reserve constant*, not the handle's `arena_bytes`. (P2)**
`tp4_forward.rs:14351` reports `"reduce_arena_reserve_bytes": layer_device::REDUCE_ARENA_RESERVE_BYTES`
(the constant `16 << 20` at `layer_device.rs:331`), while the contract says the run's memory model
charges "the communicator's init-time device reserve (arena_bytes)" as read from the stats
(`layer_device.rs:109-111`, `ws-...md:263-268`). Today `RunReduce::open` is called with the same constant
(`device_reduce.rs:563-575`, `573`), so the two numbers agree by coincidence, not by construction: an
arm opened with a different reserve would silently drift. The stats *are* readable
(`device_reduce.rs:1119`, `arena_bytes()` at 1143) — they are simply never read by the CLI.
*Fix:* at the run boundary read `unsafe { device_reduce::stats(handle) }` once and print
`arena_bytes()` (plus `collectives`/`elements`/`bytes`, and the bridge counters from §3) in the
`layer_device_memory` record, and refuse if `stats.collectives()` disagrees with the probe's
`enqueued_collectives` — the two counters then cross-check each other instead of one being decorative.

**F3 — the stream is the literal `0`, not a value the run owns; graph capture would break it. (P2, latent)**
`tp4_forward.rs:11486` calls `reduce_bf16(0, …)` and the doc comment calls 0 "the stream every launch
of this run uses". That is true today: the launch path passes a null stream
(`cuda_driver.rs:3379-3448`) and only drains synchronously when the run asks for it
(`queued_dispatch_wanted()`, `cuda_driver.rs:1381`; async path 3328-3367). But clause 3 of the
contract says the reduce is "on the caller's stream", and CUDA graph capture runs on a *private*
non-blocking stream (`cuda_driver.rs:85-87`, `1828-1840`, graph segment path `2140`): an NCCL enqueue on the NULL
default stream is not captured by a private-stream capture, so the day the loop captures its segments
the reduce would sit outside the graph and the ordering would silently leave the stream. Also,
`SPARKFLASH_SYNC_LAUNCHES=1`/stage timing/probe metering make every launch a full
`cuCtxSynchronize`, which keeps correctness but destroys the ≤120 ms/step budget — that is a gate
concern, not a collective one.
*Fix (cheap and testable):* give `DeviceLayer` one `reduce_stream: u64` field set by `attach_reduce`
from the request (0 today, the capture stream when capture lands) and pass `self.reduce_stream` at
`11486`; or, if the Lead prefers to keep the literal, record in the contract that capture must thread
the stream, so the hazard is named in the place a future editor reads.

**F4 — the producers are still refusals, so the device arm's collectives cannot be exercised end to end. (P1 for the milestone, not for P4's code)**
`attention_output` (`tp4_forward.rs:12254`) and `moe_routed` (`tp4_forward.rs:12523`) still return
`Err(Self::refusal(...))`; the stages that would fill `attn_out_bf16` and `moe_delta_bf16` are not
ported yet, so `all_reduce`/`all_reduce_moe` enqueue only once P2/P3 land. Every claim in §2.1 about
the *call shape* is static; **no device collective has been executed**. Nothing in P4 changes this,
and the Lead must not quote an executed per-layer collective until those two stages land.

**F5 — a doc comment is attached to the wrong method. (P3, hygiene)**
`tp4_forward.rs:11491-11494` is the doc comment for `attach_reduce` but sits immediately above
`attach_router`'s own doc (11494-11497), so rustdoc merges both onto `attach_router` and `attach_reduce`
(`11531`) is undocumented. *Fix:* move the four lines down to `11531`.

**F6 — the warm-up payload is a constant while the reduce payload is `self.dim`. (P3, latent coupling)**
The bootstrap passes `REDUCE_PAYLOAD_ELEMENTS` (= `layer_device::REDUCE_ELEMENTS`, 5,120) as the
warm-up payload (`tp4_chain.rs:563-565`, `tp4_reduce.rs:140-144`), while the loop enqueues
`self.dim` (`tp4_forward.rs:11466`). They agree today because `DeviceLayer::open` refuses any
`config.dim != HIDDEN` (`tp4_forward.rs:10749-10757`, `HIDDEN = 5120` at 43) and `REDUCE_ELEMENTS` is
the same 5,120. Nothing enforces the equality; a future plan width would warm a different shape than
the loop enqueues. *Fix:* make the bootstrap's payload come from the same single source as the layer
(`HIDDEN`/plan dim), or assert `REDUCE_ELEMENTS == HIDDEN` next to `REDUCE_ELEMENTS`
(`layer_device.rs:170`).

**F7 — the cross-rank count agreement has no in-run caller. (P2, risk)**
`verify_collective_count` refuses a *local* mismatch at the boundary (`tp4_forward.rs:14293-14303`),
but `verify_collective_counts_agree` (`layer_device.rs:1208`) is only ever called by
`scripts/check-collective-count.py` offline; the boundary comment says so itself
(`tp4_forward.rs:14291`). A rank that enqueued a different number of collectives will therefore hang in
the next collective (or be voided by the outer timeout) rather than fail at the boundary. The contract
accepts this ("a skip hangs the communicator instead of failing a check", `device_reduce.rs:40-46`),
but the gate report must not imply a distributed count check happened in-run. *Fix (optional):* gather
the four counts on the control channel at the boundary and call `verify_collective_counts_agree`; if
the channel is not in scope there, state "offline-only" in the contract explicitly.

**F8 — the missing-handle refusal fires in the timed region, not at setup. (P3, diagnostic quality)**
`reduce_row` refuses when `self.reduce` is `None` (`tp4_forward.rs:11461-11465`). The handle's presence
is a run-level property decided at bootstrap, and the refusal is deterministic on all four ranks (the
doc says so), so it is *safe*; but the failure surfaces at layer 0's twelfth stage instead of before
layer 0. *Fix:* in `run_with_arena` after `attach_reduce` (`tp4_forward.rs:13392`), if
`request.reduce.is_none()` and the run opted into the device arm, refuse before the loop starts; keep
the stage-body check as defence.

### 2.3 What is already right (worth keeping)

- The enqueue path is genuinely one call deep: stage body → `reduce_row` → `RunReduce::reduce_bf16` →
  `NcclApi::enqueue_all_reduce_bf16` → `ncclAllReduce`, with the count converted to bf16 elements once
  and a single `NcclDataType::Bfloat16`/`NcclRedOp::Sum`.
- The spans are the layer's own carved rows, disjoint by construction: `layer_device.rs:22-38` is the
  ownership table and the two reduces write `reduced_bf16` / `moe_reduced_bf16`, neither of which is an
  input to any other reduce; the consumers (`attention_residual`, `ffn_epilogue`) read exactly those.
- Setup order is exactly one communicator: id exchange → `open` → 3 warm-ups → one sync → setup
  barrier (`tp4_reduce.rs:116-148`), all before layer 0; the rank's `RunReduce` is moved into the
  request for the whole run and its `Drop` is the only destroy.
- The counters are advanced only on a successful enqueue (`device_reduce.rs:637`), warm-ups are
  excluded, and the bridge counters (§3) share the same `record_completed` path so they cannot drift
  from the device counters.

## 3. The declared host-bridge fallback (P4 deliverable)

### 3.1 What it is

The S1 exception allowed by the milestone: if a node cannot run NCCL, one run may be configured to
reduce the two per-layer collectives **device→host, host reduce, host→device**. It is a *declared*
row in the report, not a fallback path inside the device implementation:

- **Typed**: `ReduceArm::HostBridge` (`device_reduce.rs:129-197`), `as_str() == "host-bridge"`
  (`HOST_BRIDGE_LABEL`, `host_bridge.rs:65`), and it is *never* a device arm:
  `is_device()` is false for it (`device_reduce.rs:159-163`), so the CLI's device-arm checks treat it
  as a different row.
- **Not reachable through the NCCL API**: `RunReduce::open_with_arm(…, ReduceArm::HostBridge, …)`
  returns `NcclError::ArmUnavailable` before the library is even loaded
  (`device_reduce.rs:183-197`; test `opening_the_host_bridge_through_the_device_api_refuses_before_any_library`).
  The bridge has its own type.
- **Build opt-in**: cargo feature `s1-host-bridge`, off by default (`crates/sparkflash-comm/Cargo.toml`).
  With the feature off, `HostBridgeReduce::open` refuses with `ArmUnavailable` and a message naming the
  feature; with it on, construction still has to be explicit.
- **Runtime opt-in**: constructing `HostBridgeReduce` *is* the opt-in — no env var flips it, and no
  `Err` from a device path ever produces it. The selection must happen in the bootstrap *before*
  `RunReduce::open`, so the two arms are mutually exclusive run modes, never a retry.
- **Scoped to the two layer collectives**: only `BridgeStage::{AttentionDelta, MoeDelta}` exist
  (`host_bridge.rs:73-121`); `BridgeStage::parse` refuses every other stage name and says it is "not a
  general copy path". The bridge refuses any name that is not one of the two.
- **Counted**: every completed bridged reduce moves the shared counters exactly as a device reduce
  would, and additionally records the copies (`device_reduce.rs:96-99`, `253-343`):
  `bridge_collectives`, `bridge_d2h_bytes`, `bridge_h2d_bytes`, and
  `bridge_copies() == 2 * bridge_collectives`. A *failed* copy or failed host reduce is **not**
  counted (`NcclError::HostBridge`, `nccl.rs:313-318` and its Display at `nccl.rs:361-364`; the rule is in the error text: "a failed bridge
  is not counted and the run must abort on every rank").

### 3.2 The API

`crates/sparkflash-comm/src/host_bridge.rs` (new, exported from `lib.rs`):

```rust
pub const HOST_BRIDGE_FEATURE: &str = "s1-host-bridge";
pub const HOST_BRIDGE_LABEL: &str = "host-bridge";

pub enum BridgeStage { AttentionDelta, MoeDelta }          // ::ALL, ::as_str, ::parse
pub trait BridgeIo {                                       // the caller supplies CUDA copies
    fn device_to_host(&mut self, device: u64, host: &mut [u16]) -> Result<(), String>;
    fn host_to_device(&mut self, host: &[u16], device: u64) -> Result<(), String>;
}

pub struct HostBridgeReduce<'transport> { /* transport, world, rank, layout, staging, stats, count */ }

impl<'transport> HostBridgeReduce<'transport> {
    pub fn open(transport: &'transport dyn Transport, world: usize, rank: usize, count: usize)
        -> Result<Self, NcclError>;
    pub fn arm(&self) -> ReduceArm;       // == ReduceArm::HostBridge
    pub fn label(&self) -> &'static str;  // == "host-bridge"
    pub fn stats(&self) -> DeviceReduceStats;
    pub fn reduce(&mut self, io: &mut dyn BridgeIo, stage: BridgeStage,
                  input: u64, output: u64, sequence: u64) -> Result<(), NcclError>;
}
```

`reduce` performs D2H into the one staging buffer allocated at `open`, then the host reduce
(`Transport::all_reduce_bf16` over a one-shot `CollectiveLayout`, the canonical rank-order f32 sum with
one bf16 round — the host arm's existing arithmetic, which over-satisfies D54), then H2D into the
output span. Sequence numbers are per-collective and only used by the transport's own framing.

### 3.3 Counter semantics QA can assert

Per **completed** bridged collective at count = 5,120 bf16 elements (10,240 B):

| counter | delta |
|---|---|
| `collectives` / `elements` / `bytes` | +1 / +5,120 / +10,240 |
| `bridge_collectives` | +1 |
| `bridge_d2h_bytes` | +10,240 |
| `bridge_h2d_bytes` | +10,240 |
| `bridge_copies` | +2 |

A full 40-layer step on the declared bridge is therefore **one D2H + one H2D per collective: 80 D2H
and 80 H2D per step** (160 copies, 819,200 B each way), and a device-only run reports
`bridge_collectives == 0` and `bridge_copies == 0`. The qa-owner counter expectation for S1 is
"d2h == 1 (+80 only when the declared host-bridge reduce is on)" — that is these fields, and the
"declared" part is the `ReduceArm::HostBridge`/`"host-bridge"` row, not an env flag.

### 3.4 Wiring the bridge as an S1 run mode (Lead owns the CLI side)

P4 was forbidden to touch `crates/sparkflash-cli/src/*.rs`, so the bridge is a library capability with
CPU tests, not yet a CLI feature. To make it usable:

1. Forward the feature: in `crates/sparkflash-cli/Cargo.toml`, add
   `s1-host-bridge = ["sparkflash-comm/s1-host-bridge"]`, then build/run with
   `cargo build --features s1-host-bridge` (or `--features sparkflash-cli/s1-host-bridge` from the
   workspace root). With the feature off, selecting the bridge refuses at `open`.
2. Choose the arm **explicitly before** the bootstrap opens NCCL (e.g. a new run option
   `--reduce-arm host-bridge`, which must be reflected in `BootstrapConfig`'s agreement words so a
   four-rank mismatch is refused at setup). If the bridge is selected, the bootstrap must not call
   `open_run_reduce` at all; every rank then constructs exactly one `HostBridgeReduce` and
   `attach_reduce(None)` is not used — the port needs a small trait object or enum so a stage can call
   either the device handle or the bridge. The bridge must **never** be entered because
   `open_run_reduce` returned an error: that is the silent fallback the milestone forbids, and it would
   also let one rank bridge while the others reduce on NCCL (all-four abort rule, `tp4_reduce.rs:96-100`).
3. Provide the `BridgeIo` implementation with the run's CUDA driver (`cuda.upload`/`cuda.download`,
   one staging vector already owned by `HostBridgeReduce`): it must be synchronous and error-returning;
   a failed copy is a run abort on all ranks (the bridge does not retry and does not count).
4. In the run record, print the bridge row as `reduce="host-bridge"` and the counters of §3.3; add the
   source-bytes to the memory model (`staging` is PCIe-pinned host memory, not device arena).

## 4. The 10 KiB GPU→GPU all-reduce gate (P4 deliverable)

### 4.1 What the gate measures, and why these rules

The S1 sweep's collective budget is met only if a **10,240-byte GPU→GPU all-reduce** costs at most
**p50 ≤ 30 µs and p99 ≤ 60 µs** per call on the four-node fabric, clock-locked, 10,000 timed
iterations after 200 warm-up iterations, measured as the **synchronous per-call total** (enqueue plus
the stream sync that completes it — the number the layer's host thread would pay if it waited, and the
only self-contained per-iteration number). The gate is **device-only**: it never opens the host
transport, so it measures one collective path, not a host IPC path.

A clock-locked measurement is the only comparable one, so a gate run is **invalid** unless:

- the gate locked **every** rank's GPU clock to that rank's `clocks.max.graphics` before the run
  (one rank failing to lock makes the run INVALID, not PASS), and
- every clock sampled during the measurement window (10 Hz) is ≥ **90 %** of that rank's max, and
- every sampled power draw is recorded (mean) so the window's state is stated, not assumed.

The runner restores the clocks (`nvidia-smi -rgc`) in an EXIT trap, including on failure.
`SPARK_REDUCE_ALLOW_UNLOCKED=1` exists only to label a window invalid and let the harness finish; it
must never be used for a number that is quoted.

### 4.2 The bench binary

`crates/sparkflash-comm/src/bin/gpu-reduce-bench.rs` gained the gate mode (existing modes untouched):

- `--device-only`: skips the host transport and host arm entirely (the run is the device arm only).
- `--p50-max-us` / `--p99-max-us`: both required together; the per-rank verdict is
  `p50 <= p50_max && p99 <= p99_max` over the same sorted `total_us` samples the summary is computed
  from, emitted as `"gate":{"p50_us","p99_us","p50_max_us","p99_max_us","pass"}` in the `nccl` arm.
- `--histogram PATH`: writes the per-rank latency histogram as JSON:
  `{"arm":"nccl","bytes":10240,"samples":10000,"bin_width_us":1.0,"p50_us":…,"p99_us":…,"mean_us":…,"max_us":…,"bins":[{"lo_us":0.0,"count":…},…]}`
  (1 µs bins; the top bin absorbs the tail). With several `--bytes` payloads the file name gets a
  `-<bytes>B` suffix so the rows do not overwrite each other.
- Exit status: `0` measured (gate passed when thresholds were given), `2` usage, `3` any setup or
  measurement error (transport, library, id exchange, or a payload that mismatched the canonical
  reference by more than 2 ULP or was non-finite), **`6` the device-arm gate failed**
  (p50 or p99 over its ceiling). The runner treats non-zero as a failed gate; only 0/6 are "the
  measurement exists".

The binary cannot lock clocks and does not pretend to: its module docs say so, and the runner owns the
clock lock and the validity verdict.

### 4.3 The runner: `scripts/gpu-reduce-bench.sh gate`

New `gate` subcommand (existing `fetch|build|stage|preflight|run|collect|check` unchanged except the
usage line):

1. `preflight` (read-only): refuses a rank with a GPU process or the `:8000` listener unless
   `SPARK_WSB_ALLOW_BUSY=1`, because a latency measured next to the serving deployment is not a fabric
   number.
2. Lock clocks: read `clocks.max.graphics`, `nvidia-smi -lgc <max>` (passwordless sudo, else plain),
   record max per rank. **All four ranks must lock**: a partial lock is fatal (INVALID), not a warning,
   because one unlocked rank is enough to make the four-rank collective's tail incomparable with the
   ceilings; `SPARK_REDUCE_ALLOW_UNLOCKED=1` only lets such a window finish labelled invalid.
3. Start one local sampler per node (`nvidia-smi --query-gpu=clocks.sm,power.draw` at ~10 Hz into
   `clocks-rank<r>.csv`) and a stop flag file.
4. Launch the four ranks device-only:
   `gpu-reduce-bench --rank r --world 4 --control <ctl> --device-only --bytes 10240
   --samples 10000 --warmup 200 --mode sync --p50-max-us 30 --p99-max-us 60
   --histogram <remote>/histogram-rank<r>.json --json`, each rank's real exit status written to
   `bench-rank<r>.status`.
5. Wait for all four status files (no partial pass), stop the samplers, collect JSON/status/stderr/
   histogram/clock CSVs locally.
6. Verdict: check each rank's gate block, thresholds, mode, status, histogram presence and the 90 %
   clock rule, then write `target/gpu-reduce/<stamp>/gate-summary.json` and exit `0` PASS, `1` FAIL
   (the measurement is valid but over a ceiling), `2` INVALID (clocks, samples or a missing row).

The summary records: schema `sparkflash-reduce-gate/1`, commit, timestamp, payload/samples/warmup/mode,
thresholds, per-rank p50/p99/gate/min-mean clock/power/histogram path/status, `clock.valid`,
`valid`, `pass`, and separate `clock_problems`/`latency_problems` lists.

### 4.4 Exact commands (the `build` step was run here; every node step is UNRUN)

```bash
scripts/gpu-reduce-bench.sh fetch     # vendored libnccl.so.2 into target/deps/nccl
scripts/gpu-reduce-bench.sh build     # cross-build the gate binary for aarch64
scripts/gpu-reduce-bench.sh stage     # copy binary + lib to all four nodes
scripts/gpu-reduce-bench.sh gate      # preflight, lock clocks, 10k iterations, verdict
```

`build` succeeded on this machine (row in §5); `fetch`, `stage` and `gate` were **not** run — `fetch`
needs the NFC/deps source and all three node steps need the four-node ssh route.

Environment (defaults in brackets): `SPARK_GATE_BYTES` [10240], `SPARK_GATE_SAMPLES` [10000],
`SPARK_GATE_WARMUP` [200], `SPARK_GATE_P50_US` [30], `SPARK_GATE_P99_US` [60],
`SPARK_GATE_MIN_CLOCK_PCT` [90], `SPARK_REDUCE_ALLOW_UNLOCKED` [0], `SPARK_NODES`,
`SPARK_REDUCE_TIMEOUT` [600]. Artifacts land in `target/gpu-reduce/<stamp>/`:
`bench-rank<r>.json`, `bench-rank<r>.status`, `bench-rank<r>.err`, `histogram-rank<r>.json`,
`clocks-rank<r>.csv`, `clocks-max.txt`, `gate-summary.json`.

**This gate has not been run.** No clock was locked, no collective was enqueued, no latency exists.
The claim it will test is "p50 ≤ 30 µs, p99 ≤ 60 µs at 10 KiB, clock-locked"; the milestone's gate
result is whatever that run says, and only the Lead may put it in the result document.

## 5. What was run here

CPU only; every command below was run on a tree where `crates/sparkflash-comm` differs from `HEAD`
only by the P4 files listed in §5.1.

| Command | Result |
|---|---|
| `cargo test -p sparkflash-comm --lib` | 139 passed, 0 failed (F1 adds the two async-failure tests) |
| `cargo test -p sparkflash-comm --lib --features s1-host-bridge` | 145 passed, 0 failed (the 6 bridge tests are feature-gated) |
| `cargo clippy -p sparkflash-comm --lib -- -D warnings` | clean |
| `cargo clippy -p sparkflash-comm --lib --features s1-host-bridge -- -D warnings` | clean |
| `cargo clippy -p sparkflash-comm --all-targets -- -D warnings` (incl. the bench bin) | clean |
| `cargo fmt -p sparkflash-comm -- --check` | clean |
| `cargo check -p sparkflash-cli --lib` | clean (the new `NcclError::HostBridge` variant and stats fields do not break the CLI consumer) |
| `scripts/gpu-reduce-bench.sh` dispatch (before the cross-build existed) | `gate` with no staged binary refuses before any node contact; an unknown mode prints the new usage line including `gate` |
| `bash -n scripts/gpu-reduce-bench.sh` | clean |
| `scripts/gpu-reduce-bench.sh build` (the documented cross-build command; aarch64) | succeeded: `target/aarch64-unknown-linux-gnu/release/gpu-reduce-bench`, sha256 `f2b8b679…38cd` after the F1 library change (earlier build `00eeaeb0…9281a`); one pre-existing cross-linker warning (`ignoring deprecated linker optimization setting "1"`) |
| embedded `gate` verdict python: `compile()` | clean |
| gate verdict synthetic self-test (4 fabricated rank record sets: pass / latency fail / clock invalid / missing histogram) | exit 0 / 1 / 2 / 1 with the expected `gate-summary.json` verdict — the verdict logic is tested without any GPU; it is **not** a measurement |

### 5.1 Files changed by P4

| File | Change |
|---|---|
| `crates/sparkflash-comm/src/host_bridge.rs` | **new**: the declared bridge (§3) |
| `crates/sparkflash-comm/src/device_reduce.rs` | module-doc clause for the bridge; `ReduceArm::HostBridge` + `is_device`/`is_built`/`require_available`; bridge counters in `DeviceReduceStats` + `record_bridged`; tests; **F1**: `RunReduce`/`DeviceReduce::check_async_failure` + `async_failure_check_available` |
| `crates/sparkflash-comm/src/nccl.rs` | `NcclError::HostBridge` + Display text (phase, no-count rule) + test; **F1**: optional `ncclCommGetAsyncError` symbol, `NCCL_IN_PROGRESS`, `check_async_failure`, `NcclError::AsyncFailure`, tests |
| `crates/sparkflash-comm/Cargo.toml` | `s1-host-bridge = []` feature, off by default, commented |
| `crates/sparkflash-comm/src/lib.rs` | export the bridge module/items |
| `crates/sparkflash-comm/src/bin/gpu-reduce-bench.rs` | `--device-only`, `--histogram`, `--p50-max-us`, `--p99-max-us`, gate JSON, histogram writer, exit 6 |
| `scripts/gpu-reduce-bench.sh` | `gate` subcommand (§4.3), header, usage, verdict (python inside), clock lock/restore/samplers |
| `docs/archive/build/s1-p4-collectives.md` | this note |

Untouched: `crates/sparkflash-cli/**`, `crates/sparkflash-kernels/**`, `docs/build/speed-log.csv`,
`docs/build/s1-result.md`.

## 6. Runnable now vs node-only

Runnable on any machine with the workspace: the whole comm crate's tests (both feature states), the
clippy/fmt/check rows above, `bash -n`, and the bench binary's *building*. Anything that needs a GPU,
libnccl or the four nodes is **UNRUN**:

- the gate itself (clock lock, 10,000 iterations, p50/p99, histograms, `gate-summary.json`);
- any real NCCL collective (the layer's two per-layer reduces included);
- any real CUDA D2H/H2D through `BridgeIo` (the CPU tests use a `Vec`-backed device model: two fake
  device addresses, canonical host reduce over four in-process ranks, and a refusing transport);
- end-to-end layer-device runs: the producers `attention_output`/`moe_routed` still refuse (F4);
- CLI-side selection of the declared bridge (Lead's wiring, §3.4).

## 7. Residual risks

1. **No executed collective.** Every collective-correctness claim here is static. The first S1 device
   run is also the first test of the shapes. F1's API is CPU-tested against a fake symbol table, but
   its boundary call site, and F2/F3/F5/F7/F8, are unexercised paths.
2. **The gate depends on root/sudo for `-lgc`.** Without it the gate is INVALID by construction —
   which is deliberate, but it means the gate run needs a coordinated window with node privileges, not
   just ssh.
3. **Clock validity is a floor, not a lock proof.** A rank can lock to a lower max clock and still be
   internally consistent; the summary records per-rank max so a suspicious max is visible, but the
   gate cannot prove the number matches a reference clock.
4. **The bridge's staging is host memory and PCIe traffic by design.** If it is ever enabled for a
   large fraction of the decode, it will not meet 120 ms/step; it is an exception row, not a tuning
   lever, and the S1 result must label it.
5. **F4 is the milestone's critical path**: until `attention_output`/`moe_routed` land, the device
   arm's collectives cannot enqueue at all, and the collective gate above is the only early signal available.

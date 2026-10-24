# S3 capture audit (S3-0): the step is dominated by kernel time, so capture cannot reach 35 ms

**Status: complete as an audit; it re-orders S3 behind S4.** The audit does not produce a list of host
operations to remove and then a capture plan, because the arithmetic forbids the target: the step is
`139.00 ms` and the GPU kernel time alone is `186.40 ms/step` measured at `1500 MHz`, which bounds it
to `113-139 ms` at boost (`3.2-4.0x` the target; section 1). Graph capture replaces host submission; a step cannot become shorter than the work the GPU
still has to do, so capture cannot put the step under `35 ms` until kernel time is under `35 ms` -
and the bound below puts boost kernel time at `3.2-4.0x` the target, not `4x` flatly. That is S4's
job. `D58` records the re-ordering.

Work item: #1. Branch: `integration/s3-graph`. The S2-d row was measured at `a42b1aa1`; the code was
read at the branch tip; the S3 delta over `a2aa394b` is documentation only, and the intervening S2
review commits do not touch the decode step.

## 1. The argument, from measured numbers

| fact | value | source, and what it is |
| :-- | --: | :-- |
| S2-d step, median | **139.00 ms** | `target/s2-runs/s2-d-verdict.json` (`s2-d`, commit `a42b1aa1`) |
| S2-d step, mean | 146.03 ms | same artifact |
| S1 GPU kernel time, 40 layers + head, at `1500 MHz` | **186.397 ms/step** | `s1-result.md` section 8.2, `s1-c-probe` with `--forward-perf-probe-nosync` (deferred event harvest, no boundary drain) |
| S1 step at the same clock lock, median | 191.75 ms | `s1-result.md` section 8.2 key observation 2 |
| S1 host dispatch + collective wall, summed per block | 203.944 ms/step | `s1-result.md` section 8.2 - **not host-busy time**, see section 3 |
| collectives' share of an S2 step | `80 x 48.4 us = 3.87 ms`, `2.8%` | **estimate**, not a step measurement: `fabric.md` section 5.2 gives `p50 = 47.89-49.76 us` for the 10 KiB all-reduce in the `sync`-mode bench, and `s2-result.md` section 3.1 item 1 says so in as many words. **Not needed for the bound below** |

A step is at least as long as its GPU kernel work. At `1500 MHz` (`s1-c-probe`, rank 0 at 1482 MHz) the probe measures `186.397 ms/step` of GPU kernel
time over 40 layers + head - `5.3x` the `35 ms` target, but **not a like-for-like comparison** with a
boost-clock step. The same kernels cannot run faster than the full clock ratio (`1.62-1.65x`; taking the most favourable `1.649x = 2411/1462`),
so at boost their time is **at least `186.397 / 1.649 = 113 ms`, which is `3.2x` the target**, and at
most the `139.00 ms` step (`4.0x`). Either bound keeps `35 ms` out of reach until kernel time falls by
at least `3.2x` - which is S4's job, and the reason capture is sequenced after it. The `1500 MHz` row
is the S1 tree's kernel time and is **not re-measured**: no `1500 MHz` locked run is taken any more
(the hardware sits at 2405-2418 MHz and the protocol is boost clocks), so the re-tile needs the
per-stage table re-measured at boost; taking that row is what section 5 is about, and once its
documents are rendered (`scripts/tp4-perf-table.py`, section 5) that table, not this bound, becomes
the number the re-tile works from.

So the correct sequencing is: **shrink kernel time first (S4: `moe_routed` 38.6%, `attention_output`
18.8%, `moe_router` 7.7% of the 40-layer kernel total, `s1-result.md` section 8.2), then re-ask
whether the host has become the binding constraint.** It may well have: see section 3.

## 2. Capture-safety checklist (issue section 1.5), corrected

This is the checklist the issue asks for, answered against the device-arm step. Two of my earlier
answers were wrong and are corrected here. The record that matters is that **capture is not ready
today for reasons independent of any cost argument**, and that record already existed in S2's own
handoff: `s2-result.md` section 4.1 "To S3 (graph capture)" lists H1-H4, and its first item says a
capture today would miss both collectives.

| question | verdict | evidence |
| :-- | :-- | :-- |
| legacy default stream | **BLOCKER (H1)** - every launch and both NCCL enqueues use stream `0` | `tp4_forward.rs:12263-12266` (`reduce_row`: "stream 0 is the stream every launch of this run uses"), `layer_device.rs:62-63` ("the stages run in the order of `LayerStages::run_layer` on the default stream"); `s2-result.md` section 4.1 item 1 |
| kernel argument from per-step host state | **BLOCKER (H4)** - `n_pos`, `window_rows`, `selected_count`, `position` are host launch arguments | `tp4_forward.rs:15902` (`for layer in 0..n_layers`), `:15909` (`device.run_layer(layer, position)`); `s2-result.md` section 4.1 item 2 |
| graph topology varying across positions | **BLOCKER** - `reference_seam`'s compress branch and `ref_select`'s `n_pos == 0` skip are host branches that change the launch list per position | `s2-result.md` section 4.1 item 3; the per-position extents are host-computed (`layer_attention_stages.rs:17-18`: `window_rows = min(p + 1, 128)`, `selected_count = 0` on a window-only layer; `:68` `selected_count`), and `tp4_forward.rs:13946,14154` refuse a layer with no compress ratio - i.e. the branch is what the launch list depends on |
| host input/output inside the step | **BLOCKER (H2/H3)** - `begin_position` uploads the position's residual with synchronous `cuMemcpyHtoD`; the head's logits D2H ends the step | `tp4_forward.rs:13410,13415`; `s2-result.md` section 4.1 item 4 |
| `cuMemAlloc*` inside the step | **not instrumented** - no counter hook; the claim rests on `layer_device.rs` having no allocator call at all (grep for `cuMemAlloc`/`mem_alloc`/`alloc_device` returns nothing) and on every span being carved at open | `layer_device.rs` (`open` sizes the workspace table, `:42`); the recorders `record_allocation` / `record_sync_wait` are **never called in production** (see section 3) |
| synchronous `cuMemcpy` inside the step | **one known copy**, recorded at its call site | `tp4_forward.rs:16688` `step_counters.record_d2h_copy()`; the counter records, it does not intercept |
| `cuStreamSynchronize` / `cuEventSynchronize` / `cuCtxSynchronize` inside the step | **the step ends in one stream drain**, and `wall_ms` includes it | `tp4_forward.rs:16757`; `s1-result.md` section 8.2 head row ("incl. stream drain") |
| lazily allocating library call inside the step | **not a blocker** - communicator and registered buffers are created at open | every S2 gate summary records `registered_buffers: 2` |
| host reduce of the per-layer collectives | **absent** - the NCCL arm enqueues on the device | S2-d `run.log`: `forward reduce: requested="nccl" path=nccl ... this host slot makes no reduce call` |

`docs/build/s2-extent-args.md` is the entry list for the kernels that need the device-word variant
(section 4.1 item 2 names it). Nothing here is new work invented by this audit; the audit's
contribution is that the blockers exist **and** that clearing them would not reach the target while
kernel time is at least `113 ms` (`3.2x` the target; section 1).

## 3. What is not measured, and must not be presented as measured

Three numbers in the first draft of this document were wrong. They are retracted here, because a
decision register cannot rest on them:

1. **"The step counters are enforced, so the host cannot be faking its way out."** False. The step
   counters record `d2h_copies` at exactly one call site (`tp4_forward.rs:16688`), sited at the one
   copy the author knew about; `record_allocation` and `record_sync_wait` are never called in
   production (`device_counters.rs`), so `allocations = 0` and `sync_waits = 0` are **asserted
   constants**, and the check at `tp4_forward.rs:16718` cannot fail for those two. What the counters
   establish is the logits copy count and the absence of a host reduce, nothing more. Instrumenting
   `copy_htod` / `copy_dtoh` / the stream calls in `cuda_driver.rs` is a **prerequisite** for any
   claim that the step is host-free, and it is not done.
2. **"The collectives are 3.9 ms of the step."** An estimate, and it stays labelled one: it multiplies
   a `sync`-mode bench cost by 80. `s2-result.md` section 3.1 item 1 already said "an estimate from
   the sync-mode bench, not a step-level measurement", and this document does not upgrade it.
3. **"Capture's effect is 0 to -5 ms/step."** Retracted: nothing measures per-step host-busy time.
   The S1 table's "host dispatch / collective wall" column (`203.944 ms/step` summed) **conflates host
   dispatch with collective wall time** (`s1-result.md` section 8.2), so it cannot be read as
   host-busy time - and it exceeds the 191.75 ms step of the same row, which proves the column is not
   additive host cost. The honest statement of capture's ceiling is: *bounded above by the step's
   host-busy time, which is unmeasured*. What the audit can say without it is the section 1 argument -
   capture cannot beat kernel time, and kernel time is `3.2-4.0x` the target - which is enough to order S4
   first and not enough to quantify what capture will be worth afterwards.

The measurement that would settle section 3.3 is a `--forward-perf-probe-nosync` row read for
`host_ms = wall - wait` per block. The meter exists (`tp4_chain.rs:81-87`; `s1-c-probe` used it), **but
no committed four-rank launcher for the lean row exists in this repository** -
`scripts/tp4-window-bind.sh` has a token argv that is not the S2-d argv. Section 5 records what
happened when the row was attempted with a local launcher: it completes on some configurations and
not others, no factor is isolated, and the single next observation is named there. That observation,
not a launcher, is now S4's first task.

## 4. Handoff

| item | owner | why |
| :-- | :-- | :-- |
| **the one runnable single-factor row**: attempt 2's argv (12-token, `--decode-steps 9`) with `--forward-lean` dropped, device arm + NCCL + probe at boost | **S4, first task** | §5: every probe stall so far is device+NCCL under boost *with* lean, and both probe completions differ from them in more than one factor; dropping lean is the only single-factor variation the binary accepts without a code change |
| render the S1-style table from the row: per-stage `kernel ms/step` over the decode positions | **S4, renderer done** | §5: `scripts/tp4-perf-table.py` now renders it and reproduces `s1-c-probe`'s published table; the row it must render is the item above |
| `moe_routed` (38.6%), `attention_output` (18.8%), `moe_router` (7.7%) re-tiling | **S4** | the 40-layer kernel table says these are 65.1% of kernel time |
| instrument `copy_htod` / `copy_dtoh` / stream calls so the step's host interactions are counted, not asserted | **S4 (prerequisite for any host-free claim)** | section 3.1 |
| capture prerequisites H1 (one non-blocking stream), H4 (device position word), fixed topology, H2/H3 | **after S4, re-evaluated with a measured host time** | `s2-result.md` section 4.1; gated on kernel time dropping toward the target |
| full-step graph capture | **not scheduled** | it cannot reach the target before kernels do; re-open when kernel time is under `35 ms` or a measured host time shows the host is the binding constraint |

## 5. Attempted: the probe row, and the configuration it needs

`D58` leaves one number unmeasured: per-step host-busy time, which is what a capture would remove.
This section records the attempt to take that row with the in-tree no-sync probe
(`--forward-perf-probe-nosync`), what it establishes, and what it leaves open.

### What is known before the attempt: the probe has completed rows before

The instrument is not broken in general, and this section must not be read as claiming that it is.
Two rows on record completed with the no-sync probe:

- `s1-c-probe` (`docs/build/s1-result.md` §8.2; evidence row in
  `docs/build/s1-window-evidence/README.md`, which lists it as "device + `--forward-perf-probe-nosync`"
  under `nvidia-smi -lgc 1500,1500`; artifacts `target/s1-archive/s1-c-probe/`, four ranks,
  `run.exit` 0, 138 positions). Its per-layer collectives are NCCL (`s1-result.md` §8.2's table labels
  them "NCCL `bf16` sum", "NCCL stream 0"), which matters below.
- `wsd18-nosync-warm` (`docs/archive/build/ws-host-bound-night-20261022.md` §7, "two host-only probe
  rows on the final tree (`wsd18`, `7ecec4c2`)"). That text states the probe's wall mode and the tree;
  it does not state the rank count or the arm, and this section does not attribute either to it.

### What the attempts did

Six attempts on clean exclusive windows, all launched from the S2-d argv and all changing what the
line says (attempts 2-4 shorten the prompt, 5 runs with the probe **off** and `nsys profile
--trace=cuda` wrapping the same binary, 6 changes arm and reduce).
Clocks sampled `2502-2554 MHz` idle and `2405 MHz` under load on all four ranks. The last line is the
one that matters: it is the first attempt that finished, and it is what makes this section a
configuration record rather than an instrument verdict.

| attempt | configuration | outcome |
| :-- | :-- | :-- |
| 1 | boost, `--forward-lean`, **device arm + NCCL**, full 129-position prompt | killed at 38 min |
| 2 | same, 12-token prompt | killed at 20 min |
| 3 | same, 2-token prompt, `--decode-steps 1` | killed at 13 min |
| 4 | same, 2-token prompt, after two launcher fixes | killed at 14 min |
| 5 | same, **probe off**, `nsys profile --trace=cuda --sample=none --cpuctxsw=none` wrapping the rank command | killed at 14 min |
| **control** | the S2-d row (device arm, NCCL), no instrumentation, 129 positions | completed: **87 s** bind→exit; the separate 1,000-step soak row took **225 s** |
| **6** | boost, `--forward-lean`, **host arm + `SPARK_COMM_REDUCE=tcp`**, 12-token prompt, `--decode-steps 9` | **completed, four ranks `run.exit` 0**, each wrote `tp4-forward-perf-r{0..3}.json` |

Attempts 1-5 stop after the run header and the three `device 0` lines, before the forward loop's own
live prints (`forward reduce:`, then `forward loop: launch path async` — the line `s1-c-probe`'s log
carries). While they were stuck the GPU read `96%` at `2405 MHz` and host CPU `93-179%`; that is a
sample, it shows the GPU was busy, and it does not by itself distinguish work from a spin (see the
provenance note below). Only attempt 5's log survives to show this shape; attempts 1-4 were killed
before writing anything. Attempt 6 differed from attempt 2 in the layer arm and the reduce backend
together, and it finished.

### What the matrix does and does not separate

| row | arm | reduce | lean | clocks | probe | tree | outcome |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `s1-c-probe` | device | **NCCL** | not passed | 1500 MHz locked | on | S1 | **completed** (its `run.log:6` prints `forward loop: launch path async`; `s1-result.md` §8.2 labels the collectives "NCCL `bf16` sum ... NCCL stream 0") |
| attempts 1-4 | device | NCCL | on | boost | on | S2 | never reached the forward loop |
| attempt 5 | device | NCCL | on | boost | **off** (`nsys` instead) | S2 | same stall |
| attempt 6 | **host** | **tcp** | on | boost | on | S2 | **completed**, four ranks `run.exit` 0 |
| control `s2-d` | device | NCCL | on | boost | off | S2 | **completed** (but see the runner caveat below) |

The `s1-c-probe` "not passed" cell means exactly that: no `--forward-lean` argument is recorded for
that row anywhere, so its absence is an absence in the record rather than a documented choice.

The first thing to correct from earlier drafts: **NCCL is not implicated.** It appears in a completed
probe row (`s1-c-probe`) and in the completed control. The `run.log` line that reads
`forward reduce: requested="" path=tcp (reference exchange) ... tcp_calls=0` is the reference-exchange
slot, not the per-layer path, and reading it as "the S1 row used tcp" is wrong twice over: the S1
result table (`s1-result.md` §8.2) labels those collectives NCCL, and the row's per-layer `all_reduce`
block records no launches and `32 us` per call, which is an enqueue and not a `~0.51 ms` tcp reduce.

What the matrix does separate: **every stall is the S2 tree, device arm, NCCL, boost, with
`--forward-lean`** - attempts 1-4 with the probe, attempt 5 with the probe off and `nsys` in its
place. The probe alone is therefore not the whole story: a second, unrelated instrumentation failed
the same way on the same configuration, which is why this section does not blame the meter.

What it does not separate is any single factor, because the two **probe** completions each differ from
the stalls in more than one:
- `s1-c-probe` differs in tree (S1), clocks (1500, not boost) and lean (off);
- attempt 6 differs in arm (host) and reduce (tcp) — and the arm/reduce pair cannot be varied singly in
  the config at all: `ReducePath::require_for_arm` (`crates/sparkflash-cli/src/run_header.rs:80-91`)
  accepts only `(device, nccl)` and `(host, tcp|roce)`, refusing both mixed pairings by design.

So the honest statement is: **the probe has completed four-rank rows, and on the current tree under
boost with `--forward-lean` it has not reached the forward loop; the observations do not isolate
which of tree, clocks, lean, arm or reduce is responsible.**

### What is not established, and the cheapest runnable test

Cause is unknown, and the attempt-5 result above already rules out "the deferred meter specifically":
`nsys`'s CUDA tracing is a different mechanism and stalled identically. What the two instrumentations
share is that both observe every launch and both run alongside stream `0`, so the standing candidate
is host-side, per-launch observation of a launch path that NCCL is also enqueueing on - of which the
meter's deferred bracket (two fresh `cuEventCreate` handles per launch, recorded on the null
legacy-default stream, retained for a run-end harvest,
`crates/sparkflash-kernels/src/cuda_driver.rs:3652-3731`, `probe_meter_harvest` `:3744-3782`) is one
instance, not the thing itself. Section 2 notes every launch and both NCCL enqueues use stream `0`;
"the null stream collides with NCCL's internal streams" is a guess about NCCL internals, not something
this audit observed, and it is recorded as a guess.

The one single-factor variation the binary accepts is **lean**: dropping `--forward-lean` from the
attempts' own argv leaves device + NCCL + probe + boost, which is legal (attempts 2-4 exercised the
neighbouring combination, and the binary accepted them). If that row also stalls, lean is out and the
candidate set shrinks to the S2-tree/boost pairing - the S2 device-arm launch path and/or boost
clocks, since `s1-c-probe` differs from the stalls in both - which is a code question rather than a
row. If it completes, lean is the discriminator, and the next question is what lean does to the launch
path. Anything involving arm/reduce pairs is a code change, not a row.

### Provenance of this section's rows, stated plainly

- Attempts 1-4's row directories are empty (the runs were killed before writing documents); their kill
  times and the "stops after the three `device 0` lines" observation are node-side, not artifacts.
  Attempt 5's `run.log` survives and is what shows the stall's shape; its probe-off state is attested by
  the launcher's own stdout for that row - a node-side log, not in this repository (`== row
  20261024T0835Z: ... probe='off', wrap='nsys profile ...'`) - and **not** by its `row.json`, which lists `"probe": "nosync"`: the launcher wrote that key
  unconditionally (a duplicate-key bug that made every row claim the probe), so no `row.json` in
  `target/tp4-perf/` can attest the flag, and none is cited for it.
- Attempt 6's row (`20261024T0925Z`, artifacts `target/tp4-perf/20261024T0925Z/`, both untracked) is
  local output of an unmerged launcher at revision `8cb5cba5`, dirty. Its configuration is attested by
  the binary's own run log in `rank*/run.log` - `arm=host` and `SPARKFLASH_LAYER_DEVICE=0` at lines
  1-2, `run (9 positions)` at line 13, `lean=on` and `perf_probe=nosync` at lines 15-16 - **not** by
  that launcher's `row.json`, which records the launcher's own dirty checkout and a literal
  placeholder for the arm. The staged binary's sha256 was not recorded, so "the S2 tree" means "the
  staged `s2w01` build that the recorded attempt rows and the `s2-d` control used".
- Clocks are observations sampled around the runs (`2502-2554 MHz` idle, `2405 MHz` under load), not
  artifacts: no S2 or attempt row directory records its clocks, as `s2-d-verdict.json` itself notes.
  (The S1 archive does - each S1 row ships a per-rank `gpu.csv`, which is where §1's `1482 MHz` comes
  from - but nothing in this section's set of rows does.)
- The control row is the `s2-d` row produced by the S2 window's own runner, not by the local launcher
  the attempts used: it is a probe-off counterpart of attempt 1 by configuration (same `s2w01` staged
  build, same arm/reduce/lean/clocks) but **not by tooling**, so it is quoted for its timings and not
  as one half of a controlled pair. That is why the "differs in more than one factor" sentence in §4
  and above is scoped to the two probe completions rather than to every completion.
- The control's `87 s` (bind→exit) and the soak's `225 s` come from artifact timestamps on the nodes;
  the attempts' kill times (38/20/13/14/14 min) are launch→kill wall clocks from the same node-side
  view.
- "The GPU was pinned at `96%` / `2405 MHz` and host CPU at `93-179%`" is a `nvidia-smi`/`ps`
  observation during attempts 1-4 and 5. It shows the GPU was busy; it does **not** distinguish useful
  work from a spin inside a stalled collective, and this section does not claim it does.

### The consequence for S4

1. One row, single factor: **attempt 2's argv** (12-token prompt, `--decode-steps 9`, device arm, NCCL,
   boost) with `--forward-lean` dropped, so the probe and everything else stay as they were.
2. Then the table. No in-tree tool rendered the S1-style table: `scripts/tp4-probe-attribution.py` reads
   rank 0 only, sums the per-layer `wall_ms` and skips the `head`-flagged positions (which are the
   decode positions), so it is not the renderer for `s1-result.md` §8.2's `kernel ms/step` over decode
   positions. That gap is closed: `scripts/tp4-perf-table.py` renders §8.2's table from a probe row,
   and on `s1-c-probe` at `--steady 8` it reproduces every stage row that row records (the kernel
   column exactly; wall to the document's rounding convention, and the document's `all_gather` line
   has no row in the probe document at all), including the `2,029.5 / 179.216 / 4,480.4 / 161.184`
   subtotal and the `2,032.5 / 186.397 / 203.944` step total. `--expect` pins the kernel column. It
   always prints the position range and the row's probe mode beside the table, and the clock window
   when the row ships a `gpu.csv`. What it still needs is a completed device-arm row to render: the
   row is what is blocked, not the arithmetic.
3. If a later row shows the probe cannot be run on the production configuration, the finding is about
   the instrument plus the S2 launch path, and section 3 already lists the stream/copy instrumentation
   that is unbuilt; the same recording change would serve both.

Until a row completes on the production configuration, `D58`'s statement stands unchanged: **per-step
host-busy time is unmeasured**, and the kernel-time bound in §1 (`3.2-4.0x` the `35 ms` target) is
unaffected — it never needed the host time.

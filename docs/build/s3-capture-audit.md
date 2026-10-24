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
(the hardware sits at 2405-2418 MHz and the protocol is boost clocks), so S4's first task re-measures
the per-stage table at boost and that row, not this bound, becomes the number the re-tile works from.

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
`scripts/tp4-window-bind.sh` has a token argv that is not the S2-d argv - so the row was not taken.
Building that launcher and taking the table is S4's first task, and it is what makes this audit's
numbers step-level rather than borrowed.

## 4. Handoff

| item | owner | why |
| :-- | :-- | :-- |
| committed four-rank lean launcher + `--forward-perf-probe-nosync` per-stage table read for `host_ms = wall - wait` | **S4 (first task)** | the input that decides which stages to re-tile, and the only way to quantify what capture is worth later |
| `moe_routed` (38.6%), `attention_output` (18.8%), `moe_router` (7.7%) re-tiling | **S4** | the 40-layer kernel table says these are 65.1% of kernel time |
| instrument `copy_htod` / `copy_dtoh` / stream calls so the step's host interactions are counted, not asserted | **S4 (prerequisite for any host-free claim)** | section 3.1 |
| capture prerequisites H1 (one non-blocking stream), H4 (device position word), fixed topology, H2/H3 | **after S4, re-evaluated with a measured host time** | `s2-result.md` section 4.1; gated on kernel time dropping toward the target |
| full-step graph capture | **not scheduled** | it cannot reach the target before kernels do; re-open when kernel time is under `35 ms` or a measured host time shows the host is the binding constraint |

# WS-C CUDA-graph window (task-16), 2026-10-21

Status: **window measured 2026-10-21; delta ~0 (within noise), identity PASS on both rows.**
Code compiles, passes clippy and host tests on branch `task/ws-c-graph` (base `origin/main
44172974`, tree == `4a980913`); the seam hunk is applied to `run_layer_moe_routed` (device-lead's
compile-probed hunk, credited in the commit). The Lead ran the two window rows on one staged binary
(graph off and graph on); the result is recorded below, and `docs/build/ws-c-graph-window-20261021.md`
stays the recipe for a rerun.

## What is captured and why this stretch

The lean decode path is still host-driven per layer-position: pool allocation, H2D upload, a short
run of kernel launches, a D2H readback, pool release. A CUDA graph can cover only the launch-only
stretches, and only while the device addresses those launches read stay identical.

Chosen stretch, in `crates/sparkflash-cli/src/tp4_forward.rs` `run_layer_moe_routed`, current base
lines 4459-4494: the two routed gate/up launches and the weighted launch, three launches with no
CUDA call between them. They sit after the routed-weight `copy_dtod` staging loop and before the
first `copy_dtoh` (the stored-combine accumulator readback), so everything the stretch reads is
already on the device. The launch arguments are pool buffers and the rank, and the routed weight
slices are staged into those same pool buffers, so the launch list is expected to be identical on
every layer and position: one captured graph for the whole run. If the pool hands out different
pointers, each distinct list is its own graph (budget 32 per seam) and, past the budget, the run
executes the recorded launches on the direct path and says so.

Per step the stretch is 3 launches x ~573 layer-positions = ~1,719 launches replaced by ~573
`cuGraphLaunch` calls, minus the wrapper's per-call record+compare host work. This is not the
whole loop; most launches are in stretches that still allocate or read back on the host. The point
of the row is not only the delta: a refused capture names the host operation that keeps the next
stretch out of a graph.

## The switch

`SPARKFLASH_CUDA_GRAPH` is read once per process, on all four ranks:

| value | behavior |
| :-- | :-- |
| `0` / `off` / unset | the driver's graph state is never touched; all segments run the direct path. This is the control row. |
| `report` | capture once per distinct launch list to report coverage, then run the direct path for the rest of the run. No replay, no perf claim. |
| `1` / `on` / `capture` | capture and replay when the recorded launch list is identical to the captured one; execute the recorded launches on any mismatch. |

Unknown values are a start error, not a silent fallback.

## Rows (Lead's window; identical frozen lean argv and stage, only the variable differs)

Use the frozen `wsa2d-lean` / WS-A2 batch argv and stage. The only difference between rows is the
rank process environment; ask the Lead's `run-fourrank.sh` wrapper to export it per rank (the same
way the `SPARKFLASH_SYNC_LAUNCHES=1` and `SPARKFLASH_FORWARD_EVIDENCE=1` rows are staged).

| row | rank env | purpose | timing claim |
| :-- | :-- | :-- | :-- |
| R1 `ws-c-graph-off` | `SPARKFLASH_CUDA_GRAPH=off` | control, same binary, same argv | lean ms/step baseline |
| R2 `ws-c-graph-on` | `SPARKFLASH_CUDA_GRAPH=on` | capture + replay | treated lean ms/step |

Two rows, one staged binary, frozen `wsa2d-lean` argv, only this variable differs. The Lead spends a
third row in `report` mode only if the delta is ambiguous or a named refusal appears in the logs
(and that third row exists to read the refusal list and launch-list stability from the logs, never
a perf number).

## Evidence to collect from the run directory

Per rank, the wrapper writes `rank{0..3}/tp4-token-rN.json` (the step timing, tokens and logits
hash source) and its log. Grep the four logs for the wrapper's lines:

```sh
grep -H '^tp4-graph:' /home/user/ev-session-20261020/window/<run-name>/rank*/run.log
```

Expected first lines on R3 (`=1`):

```
tp4-graph: seam "moe-compute" captured graph GraphId(1) (3 launches) and replayed it; the launch list is a warm-up artifact from here on
tp4-graph: seam "moe-compute" mode=on calls=256 captured_lists=1 captured_launches=3 replays=255 fallbacks=0
...
```

If a stretch is refused, the line is the finding, for example:

```
tp4-graph: seam "moe-compute" disabled for the rest of the run: segment is not launch-only: a host or device copy cannot run inside a CUDA graph segment (capture records kernels only)
```

Also record, for R3: `captured_lists` (1 means the launch list is stable across every layer and
position; a number near 40 means per-layer weights make one list per layer; 32 means the budget was
hit), `replays` and `fallbacks`. If capture never happens because the driver refuses, the surfaced
refusal is what WS-A2's next stretch has to remove (allocation hoist, readback moved, sync removed).

## Identity gate (must pass on every row; stop and report on any mismatch)

```sh
scripts/identity-gate.sh --run /home/user/ev-session-20261020/window/ws-c-graph-off
scripts/identity-gate.sh --run /home/user/ev-session-20261020/window/ws-c-graph-report
scripts/identity-gate.sh --run /home/user/ev-session-20261020/window/ws-c-graph-on
```

The anchors are the frozen ones: tokens `[491, 92928, 201, 15, 4893, 16646, 15, 14265, 86]` and
logits bin sha256 `aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2`. The wrapper
replays only an exactly equal launch list, so a mismatch means the seam is wrong (not arithmetic
noise): stop, report the row, do not quote the delta.

## Delta

Compute the four-rank mean of `step_ms.mean_ms` for each run (the value `scripts/speed-log.sh
append` logs), then:

```text
delta_ms_per_step = mean_ms(R1 graph-off) - mean_ms(R2 graph-on)
delta_launches_per_step = R2 "replays" count / decode steps (one graph launch per replayed stretch)
```

Report the delta with the run-to-run window noise (previous lean rows move by ~1-2 ms/step), the
version tip, and the replay/fallback counts.

A delta of ~0 is a publishable result and must be written that way, not hunted: it means the host
launch path is not the wall at the current 543-600 ms/step, and the value of the captured path is
the <=150 ms endgame rather than today. The row still reports how many launches were replaced
(replays x launches per stretch) and whether the launch list was stable across layers and
positions (captured_lists == 1). A nonzero delta is reported with the same counts and the identity
gate, not as a headline in isolation.

## Code map

| piece | where |
| :-- | :-- |
| driver capture/record/launch API, segment guards, Drop cleanup | `crates/sparkflash-kernels/src/cuda_driver.rs` (`graph_capture_begin`, `graph_capture_end`, `graph_record_begin/end`, `graph_launch`, `graph_replay_records`, `CudaLaunchRecord`, `GraphId`) |
| wrapper, switch, plan, reporting | `crates/sparkflash-cli/src/tp4_graph.rs` (`segment`, `SPARKFLASH_CUDA_GRAPH`) |
| call site | `crates/sparkflash-cli/src/tp4_forward.rs` `run_layer_moe_routed`, the `segment(cuda, "moe-compute", ...)` bracket |

No kernel, cubin, D49 pin, ABI allow-list or arithmetic expression is touched: a replay is the same
launch list captured from the direct path, and a fallback executes the launches that were just
recorded. The driver refuses every non-launch operation inside a segment rather than silently
running it outside the graph.

## Window budget (Lead's call)

Each lean four-rank row is the usual arena-boot + decode shape (~7 min launch-to-evidence). The
window is two runs (R1 off, R2 on); a third `report` run is only spent if the two-row result is
ambiguous or a refusal needs the diagnostic path.

## Window result (2026-10-21, Lead-owned window)

Two rows on one staged aarch64 binary (stage tag `wsc`, git `18749398bd17403e9a0df588e9d5d6311f09234b`,
binary sha256 `d96be2a40c47d1241731d07d67f67fe940a30841f384c353c90844e9c9bdbc17` on all four
nodes), frozen `kind=token` `wsa2d-lean` argv, 9 positions (129..137), four ranks; only the rank
environment differed. Run dirs:

* `/home/user/ev-session-20261020/window/wsc-graphoff` (control, `SPARKFLASH_CUDA_GRAPH=off`)
* `/home/user/ev-session-20261020/window/wsc-graphon` (treated, `SPARKFLASH_CUDA_GRAPH=on`)

| row | rank env | four-rank mean step_ms | tokens | logits bin sha |
| :-- | :-- | :-- | :-- | :-- |
| graphoff | `SPARKFLASH_CUDA_GRAPH=off` | 596.00 | exact | `aed6054d...` |
| graphon | `SPARKFLASH_CUDA_GRAPH=on` | 595.94 | exact | `aed6054d...` |

`scripts/identity-gate.sh --run` passes on both dirs (4/4 ranks rc=0, frozen 9-token sequence,
`aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2` on all eight rank runs).
Per-rank means off / on: 595.44 vs 595.44, 596.78 vs 595.78, 595.44 vs 596.33, 596.33 vs 596.22.
Delta = off - on = **+0.056 ms/step** (graph-on faster by 0.009 %); per-rank deltas +1.0, 0.0,
-0.889, +0.111 ms against a 9-sample per-rank sd of 3.2-4.1 ms and the 1-2 ms run-to-run window
noise. The measured launch-overhead saving is **~0**: inside the noise.

What the treated row did, identically on all four ranks:

```text
tp4-graph: seam "moe-compute" captured graph GraphId(1) (3 launches) and replayed it; the launch list is a warm-up artifact from here on
tp4-graph: seam "moe-compute" captured graph GraphId(2) (3 launches) and replayed it; ...
tp4-graph: seam "moe-compute" mode=on calls=5376 captured_lists=2 captured_launches=3 replays=5376 fallbacks=0
```

* `captured_lists=2`, not 40: both variants are the same three-launch list with two different pool
  address layouts (consistent with ping-pong staging/double buffering; the wrapper treats any
  argument word change - a pointer included - as a different list, so it captures rather than
  patches). Both were captured inside the first 256 calls and no third list appeared over the next
  5,120 calls, so the launch list is **position- and layer-independent apart from those two
  variants**. `fallbacks=0` over 5,376 calls is the launch-list stability claim.
* 5,376 seam calls x 3 launches = **16,128 kernel launches replaced by 5,376 `cuGraphLaunch`
  calls** over the run; the run's 9 measured positions make that ~597 seam calls (~1,791 replaced
  launches) per measured step.
* No refusal was triggered: the named refusals for copy, synchronize, allocation, module load,
  release/free and repeated launch were built and are exercised in host tests, but the three-launch
  island is genuinely launch-only in the lean path (matching device-lead's reading), so **no hidden
  sync point, host read or allocation was exposed in this stretch**. The `report` row was not
  spent: capture succeeded and the delta was unambiguous.

**Verdict, publishable form.** Replacing ~1,791 host kernel launches per measured step with ~597
graph launches moved the step from 596.00 to 595.94 ms, which is inside the 3-4 ms sample noise.
At 596 ms/step the host launch path is not the wall; the captured path's value is the <=150 ms
endgame, where per-launch issue cost can matter, not this row. The net is also not an upper bound
on a trust-every-call fast path: the treated row includes the wrapper's per-call record+compare
host work, which offsets part of the saved launch issue. What this row does prove is that the
capture primitive works on the pinned CUDA 13 stack, the launch list is stable, replay is
identity-exact, and the stretch has no hidden host/device dependencies.

## Branch state

* Code tip: `18749398bd17403e9a0df588e9d5d6311f09234b` (seam in `run_layer_moe_routed`, hunk
  credited to device-lead; Lead approved).
* The window result section is a docs-only follow-up commit; the staged binary above is the code tip
  and is unchanged by it.
* Window done; task-16 deliverables (graph API, one replaying path, measured launch-overhead delta,
  identity) are complete.

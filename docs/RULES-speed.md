# Speed-track rules (in force 2026-10-21)

These rules govern the 90 tok/s track. They came from a direct user directive after a phase in which
the team produced measurement records, re-measurements and "no tok/s claim" disclaimers while the
number did not move. If a rule here conflicts with an older document's process, this file wins.

## 1. One published number

* The only speed metric is **lean decode ms/step, mean over the four ranks**, same frozen argv.
* It is **posted daily** by appending to [speed-log.csv](build/speed-log.csv)
  (`scripts/speed-log.sh append --run <window run dir>`); `scripts/speed-log.sh print` renders it.
* Every merge request states its **expected effect on ms/step before work starts**. A change with no
  expected effect on the metric is not speed work.

Current: **256.17 ms** (wsd5-lean, 2026-10-22, ws-d-gemv tip d6020787). Path: 1201.81 (start) ->
599.94 (WS-A2 batch, 4e4cea50) -> 469.81 (M=1 dense re-tile, 0d8797db) -> 356.39 (decode GEMV for
wq_a/wkv/wq_b/wo_b, 528b39fb) -> 350.39 (shared-expert wiring, 4f83eee5; partial - the device arm
was gated off, see 2f) -> 256.17 (shared expert's f32 reference arm on the device, d6020787,
bit-identical: 3 GEMV launches/layer at 171.1 GB/s kernel-side = 62.7 % of peak, host block
0.277 ms/layer against 2.623). Rows and dates live in [speed-log.csv](build/speed-log.csv); that
table is the record, not a document.

## 2. Milestone gates are the only progress

**<= 400 ms -> <= 150 -> <= 50 -> <= 20 -> <= 11.1**. The 400 gate is cleared (256.17); next gate is 150. Anything else (a doc, a record, an unwired
kernel, a re-run that confirms the same number) is not progress.

## 2b. Batch windows; one row must answer a gate decision

A row is a fresh four-rank process: it pays the full checkpoint upload before the first decode step
(~6-7 min on four GB10s) whether it runs 9 steps or 90. Nine rows of that is an hour of cluster time
for 81 measured steps. So:

- Batch slices into one window. A window is justified when the accumulated expected movement is
  above the ~50 ms noise floor, or when a numerics-affecting slice needs its byte-identity row.
  Never run one window per slice.
- Do not file a row whose answer is already known from CI (identity gate, counter assertions) or
  from construction. A row exists to decide something.
- The probe row is for picking the next slice, not for reporting a delta. Run it when the block
  table is stale, not once per merge.
- Every row lands as one line in [speed-log.csv](build/speed-log.csv). No per-row documents.
- Batch merge requests the same way: one branch, one pipeline, one merge for a group of slices.

## 2c. Estimates come from the hardware floor

When an estimate misses by ~4x, the next estimate for that block must be derived from the hardware
floor - weight bytes / peak bandwidth, MACs / peak MACs - and never scaled from the previous
kernel's measured rate. Kernel work is reported as GB/s achieved against peak next to ms, and the
bar for a single-token launch is **>= 60 % of peak memory bandwidth**. Beating our own previous
kernel is not a result.

## 2d. "Host path closed" means zero host arithmetic per token

The WS-A host path is closed only when the decode step reaches **zero
crates/sparkflash-engine reference arithmetic per token** (the linalg calls and the
forward/attention.rs / forward/engram.rs reference paths reached from tp4_forward.rs). Each
remaining call gets a device kernel or a deletion; the compressor and the Engram projections are
the first two. Host arithmetic is a floor that no kernel work can remove, and it blocks capturing
the step as a CUDA graph.

## 2e. Current workstreams (2026-10-21, after the WS-D un-park)

| workstream | gate | who |
| :-- | :-- | :-- |
| WS-D decode GEMV (wq_a, wkv, wq_b, wo_b, shared expert) | >= 60 % of peak bandwidth per M=1 launch; target <= 400 ms | device-lead |
| WS-C CUDA graph capture of the device-only segments | measured launch/sync win, byte-identical; next step to <= 150 ms | driver-fixer |
| WS-B interconnect admin (MTU 9000, nvidia-peermem) | applied on 4/4 nodes and verified, not ticketed | comm-lead + root admin |
| WS-A remainder: engine calls still reached per token | list, then kernel-or-delete per call | device-lead |

## 3. Rank work by the P0 table

The biggest rows win: the zero-kernel host blocks (`ffn_shared`, `moe_reduce`, `mhc_entry`), the
per-launch drains, and the `moe_gate_up` host residual - currently
`attention_projections 5.17`, `moe_gate_up 4.77`, `ffn_shared 2.74`, `attention_output 2.39` ms/layer
with **12 D2H observation points per layer-position**. If a task does not touch a top-five row, it
waits.

## 4. CI owns the identity check

`scripts/identity-gate.sh --run <window run dir>` is the one command that proves byte identity
(4/4 rank exit codes, the 9 reference tokens, the dumped logits bin sha
`aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2`, and `evidence_mode`). It runs
automatically at the end of every row and as the `identity-gate` CI job. **A result CI already proves
gets no document, decision number or window re-run.**

## 5. Measure less, batch more

Re-run a measurement only when the expected step change is larger than the run-to-run noise
(**~50 ms** at today's 843 ms step). Otherwise batch several changes and measure once.

## 6. Cap the process

* **Decision records**: only for hard-to-reverse choices, at most about one per week; status is one
  line, not a page of caveats.
* **Investigations**: two days, then ship a change or drop the thread.
* **Inventory**: a measured but unwired kernel is inventory - wire it or delete it.
* **No host/CPU twins** of device paths. The device path is the product.

## 7. Unblock, don't work around

* The RDMA admin items (MTU, `peermem`) are escalated to someone with root this week; they are the
  only thing between WS-B and its 25 us gate.
* Each person owns one workstream (WS-A..WS-G) with an explicit exit gate.

## 8. The review question

Ask only: **what did this move** (ms/step, TTFT, or open API gaps) and **what is the next biggest bar
in the table, and who is on it**? "Nothing, but it is confirmed" is rejected.

## 9. Risk is acceptable

A short-lived regression caught by CI and reverted fast is cheaper than slow proof. Byte identity is
checked by `identity-gate.sh` in one command, so a break is cheap to detect and cheap to revert.

## 10. Focus until the gate moves

For the next two weeks the speed team works **only on WS-A**: move the host blocks onto the device
and remove the per-launch drains and the 12 D2H observation points. WS-C (graphs), WS-D (kernel
efficiency), WS-E (MTP), WS-F (prefill/serving) and WS-G (parity) stay parked until the lean step is
below 400 ms.

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

Current: **599.94 ms** (wsa2d-lean, 2026-10-21, batch tip 4e4cea50). Before WS-A: 1201.81 ms. Rows and dates live in [speed-log.csv](build/speed-log.csv); that table is the record, not a document.

## 2. Milestone gates are the only progress

**<= 400 ms -> <= 150 -> <= 50 -> <= 20 -> <= 11.1**. Anything else (a doc, a record, an unwired
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

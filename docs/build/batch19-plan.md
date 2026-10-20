# Batch 19 plan (GPU device path) - skeleton for review

**Status: assembled on `batch/2026-10-20-19` as MR !141; landed as a single squash after CI green
on the Lead's final tree.** Batch 19 is assembled from four parts: (1) the device arm
`task/task34-arena` (four-rank forward loop; included as a labelled skeleton - no token yet, the
token gate moves to the batch that emits one); (2) the task-73 DSpark
acceptance-accounting chain; (3) the D49/D50 decision entries; (4) this page and
`docs/investigations/mtp-dspark-device-wiring.md`. Numbers stay placeholders until their owner
sends measured evidence; no device/host numbers share a table.

## 0. Acceptance target (D48 + addendum)

- target: **50-60 tokens/s vLLM parity with DSpark**; personal target **90 tokens/s**
  single-stream; step budget **<=42 ms/verify step at ~3.8 accepted tokens** (11.1 ms/accepted
  token, 13.9 ms weight-traffic roofline at 273 GB/s/rank, ~1.7 ms RDMA, ~2.2 ms draft);
- **tok/s gate baseline** is the current best **measured** decode row: **767.7 ms/step =
  1.3026 tokens/s** (batch-18 default sharded, bit-identical, binary `bbb20cf5` on tree
  `8ffaae6`), not the 939.7 ms/step batch-16 headline and not the 7.929 s/step batch-15 TP=4
  host A/B;
- sequence: **C2/C5 before C6** (DSpark wiring lands after batched kernels + CUDA graphs +
  residency); if C2 misses the step budget the tok/s gate is reported **unmet, not redefined**.

## 1. Device arm - `task/task34-arena`

Owner: engine-dev. This is the GPU device-chain branch (the performance path; the CPU host twin
is the exactness oracle and stopgap). It holds the four-rank forward loop. Batch 19 takes it as a
**labelled skeleton**: the loop boots, plans, allocates and runs the real-link collectives, but
it has **not emitted a token** and its own evidence says so. The token gate moves to the batch
that emits one; what lands here is measured boot/plan/arena/collective evidence, not per-step
decode time. Evidence:

- [x] four-rank plan and arena: 72.308 GiB/rank, 95,822 tensors, 161,736,798 ranges, plan hash
  `0x9a59035fc40776c8` identical on all four ranks (2026-10-20);
- [x] real-link collectives: 80 reduces/step, 819,200 B/step payload, canonical bf16 rule
  verified on hardware, 14-22 ms/step rank-0 transport-only;
- [x] boot: 416-521 s -> 332-351 s after the single-slab allocation and the whole-tensor read;
- [ ] first token, step time, exactness contract: **not reached** - a scope boundary of this
  batch, deferred to the batch that emits a token (not a placeholder);
- [ ] residency/zero-copy interaction and stage-2/3 sharding position: deferred with the token
  gate.

## 2. task-73 DSpark acceptance accounting (batch-19 input, not a separate MR)

Owner: serve-dev. The batch takes **six commits, head `2771019` (pending final confirmation)**, as one history, squash only:

- `5028832` task-73 accounting (the required base sha, unchanged);
- `9e66487` non-finite guard + honest synthetic test;
- `bd29b67` engine acceptance-evidence keys;
- `eba960e` drafts-only convention marker;
- `ccf33b1` invariant test (no production change);
- `2771019` draft-row ids for `accept_nonfinite_rows` (head; final confirmation pending).

**Apparent off-by-one investigated and refuted:** a report that the host replay compared the
anchor x_t against the same step's draft block D_t rather than D_{t-1} was checked against the
state machine and refuted - observe_token starts a block when no block is active and compares its
first draft only on the next target, so the comparison is correct. No production change was
needed; commit ccf33b1 adds an invariant regression test pinning that a newly started block
contributes zero rows seen at its anchor and is not compared until the next target. The
investigation was only possible by reading the state machine because neither synthetic model can
exercise it (all-NaN and all-zero weights). **Earlier synthetic acceptance counts are not
real-checkpoint evidence.**

**Honest-test finding:** the old synthetic served acceptance test was accepting against all-NaN
logits; the guard makes that path report zero, and the test-only `Weights::Zero` source is the
only finite synthetic non-zero acceptance test. **Real-checkpoint served evidence is pending a
node window.** The TP=4 arm is `0` labelled `no draft source in this slice`; the evidence line
convention is the **drafts-only** line.

## 3. Decision entries D49/D50

- **D49** - KV ring capacity 134 vs attention window 128: new ring symbol, slot = position %
  134, `window_rows = 128`, pin **18 -> 19** (one new `KernelId`, `FlashMlaCsa2Ring`); the three
  batched row symbols landed as extra entry points inside the existing `mega_mhc_sm121`,
  `cert_lm_head_sm121` and `lm_head_reduce_sm121` cubins, so those re-pin without new KernelIds;
  explicit compression ratio, NaN guard on the accept input; the frozen entry is not mutated.
- **D50** - drafter unification: the MTP drafter (`forward/mtp.rs`: 3 stages + Markov argmax
  chain) wins; `dspark.rs` is demoted to a scheduler-policy model; `accepted_prediction_tokens`
  comes from the real acceptance rule, never a synthetic zero.

## 4. DSpark device-wiring plan (analysis)

`docs/investigations/mtp-dspark-device-wiring.md` (527 lines) is the device-wiring analysis:
buffer inventory, launch counts, the acceptance/readback rule, staged commits C1-C6 with a gate
each, risks R1-R6 with checks, open questions, and no new measurement. It was reviewed against
the landed decisions: its open question 6 (residency budget) is answered by D45/D48 geometry
(285.504 GiB world=1, 71.376 GiB/rank world=4; the 87.6 GB/rank figure is the Engram-on-NVMe
variant plus replicated classes), question 1 (step baseline) by the D48 addendum, question 3
(pin change) by D49, question 4 (drafter) by D50, **Q2 (mHC row-batching) as IN SCOPE and already
done** - the batched `mega_mhc_fused_bf16_batched` (grid `(4, 6)`, per-row rendezvous, device-side
init kernel so the pair is CUDA-graph-capturable) and `lm_head_reduce_batched` landed by
kernel-dev (`68a1fa3`, fixes `baffdd5`/`abb23d9`; harnesses m23/m24) and were **measured on a
GB10 on 2026-10-20** (section 7); mHC batching is a **graph-capture enabler**, not
only a launch-count win, and the ~480 mHC launches/step figure drops accordingly (batched: 24 blocks in one launch) - and **Q7 (D42
dependency) as NO**: host-dense per-layer rings hold unverified rows for the commit/truncate path,
C1/C4 are not reordered, D49 is the real prerequisite, and until D42 lands **no paged-KV /
serving-cache claim may be attached to DSpark** and the commit path is labelled **host-dense rings**.
Its C2/C5-before-C6 ordering matches D48.

## 5. Gate note: D41 has not been computed

**No serving-parity claim may be made until the D41 floor JSON is computed and recorded.** D40
declares the protocol and provisional bars; D41 freezes the thresholds from the measured oracle
self-consistency floor (>= 3x the floor) and is recorded as a JSON bars file, and that file does
not exist yet. Batch-19 device runs are therefore labelled **diagnostic** until D41 lands; a bar
changed after a gate run is a new decision.

## 6. Evidence protocol

Each part lands only with its measured artefact path: device and host numbers never share a
table; per-change attribution (residency, occupancy, router/head, collectives, DSpark) is kept
separate, with no summed total win; every placeholder above is filled from the owner's message,
never from memory.

## 7. Kernel measurements on device (2026-10-20, node0 GB10)

All rows below are from the tree's own harnesses, cross-built aarch64, run against cubins rebuilt
with the recorded toolkit (`V13.0.88`, `cuda_13.0.r13.0/compiler.36424714_0`, sm_121a) whose
pins `check-kernel-baseline.sh` reproduced. Comparison classes are the harnesses' declared ones
(bytes-exact for outputs/ids; bars for the router's libm `expf`/`log1pf` score path).

| harness | verdict | measured |
| --- | --- | --- |
| `m9_router` | **PASS 7/7** | ids 48/48, determinism byte-identical; scores max abs 1.1920929e-7 / ULP 1 (bar 1e-6 / 8), weights 5.9604645e-8 / ULP 2 (bar 1e-6 / 16); 0.2279 ms per 8-token launch, 0.0285 ms/token |
| `m22_cert_batched` | **PASS 6/6** | 6/6 rows byte-identical (logits, certified, count, block_max, block_argmax); scalar one row p50 3.8689 ms vs batched six rows p50 11.0368 ms = 2.853x one launch and 0.465-0.477x six sequential (~2.1-2.15x faster than six launches); traffic roofline 1.2255 ms (11.1%) |
| `m23_mhc_batched` | **PASS** | per-row byte equality EXACT; six scalar launches+inits p50 0.3181 ms vs one batched init+main p50 0.0735 ms = 4.32x |
| `m24_lm_reduce_batched` | pass, no win | six scalar launches p50 2.4709 ms vs one batched p50 2.4535 ms = 1.01x |
| `m20_csa2_sequence --batched` | **PASS 6/6** | 6/6 rows byte-identical (out, block_scores, block_best); scalar p50 0.3214 ms, six sequential 1.9213 ms, batched six-row p50 1.1465 ms = 3.567x one row and 0.597x six sequential; reproducible on a second run (6/6, 1.1473 ms) |
| `m12_attn_projections` | **PASS 10/10 verdicts** | K2 wq_a, K3 wq_b, K4 wkv output-bit-identical to the in-crate reference at layer-0 dimensions; control within bar |

Non-claims: no token, no chain, no end-to-end step time; the router's 0.0285 ms/token (40 layers
~1.14 ms/token) is the serial ascending-k cost that keeps the score path exact and is counted in the
D48 step budget, not hidden. The first m9 attempt (K3-final tree) failed its own bars (scores ULP
131, weights ULP 21) and every K8 batched harness failed before the fixes in `baffdd5`/`abb23d9`;
the table above is the post-fix state only.

Logs: `/home/user/ev-window-20261020/k8/` on the dev box.

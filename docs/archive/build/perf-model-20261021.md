# WS-C token-path perf model (2026-10-21)

**Status: analysis only. No new device run, no node access, no new measurement.** This page
derives a per-position cost model from recorded rows on the shared filesystem and separates
what is measured from what is arithmetic, estimated, or a declared target. It is not a serving,
TTFT, tok/s, or speedup claim, and it does not replace any acceptance gate.

## 0. Reading rules

Every number below carries one label:

- **[M] measured** - quoted from a named run record/log; the path follows the value.
- **[A] arithmetic** - computed on this page from labelled inputs; the formula is given inline.
- **[E] estimate/planning** - roadmap hypothesis or planning extrapolation; not measured, not a
  prediction, not a claim that a stage will reach it.
- **[D] declared target/decision** - Lead declaration or decision entry (the 42 ms/11.1 ms budget,
  the D41 bars, D49/D50). A target is not an achieved value.

The unit convention is milliseconds unless stated. "Position" means one token position through
the 40-layer chain. "Four-rank loop" means the resident-arena tp4 forward loop.

Snapshot used: m1y8 (tag=m1w20, binary
ea6452b2c722187c5a29018c2cb111f22dff939ee483dafe20940aeffafc915f, run started
2026-10-21T05:19:45Z) and m1y7 (tag=m1w19, binary
3d7a55dcdc331d14187fba4518c946272d55cd89f24ce58db64d5226761c08a3) on peers
192.0.2.80/88/59/68; cubins manifest
dd9b7b0cbf065acba02db7c65e3c658b908274e2c9c7fbe8e310224500eacfac; checkpoint snapshot
1bae0e1088993b9b82de08f59de5be5a34e42170; driver 580.178.04 for every comparison in this
page. All comparisons stay inside that driver/build.

## 1. Measured inputs

| id | what | value and label | source |
| --- | --- | --- | --- |
| S1 | four-rank chain wall, one position | **total_wall_ms=2186 [M]**; 40 per-layer wall_ms entries **20-38 [M]**, **sum=1082 [A]**, mean **27.05 [A]** | window/m1y8/rank0/tp4-forward-progress-r0.json |
| S2 | shared-expert view prewarm | **wall_ms=1063 [M]**; 40 layers, **240 gather calls [M]**, 240 arena reads, **1,416,960,000 B [M]** (1.417 GB), reusable by every position of a prefill | window/m1y8/rank0/tp4-forward-embed-r0.json shared_expert_prewarm |
| S3 | prompt/prefill shape | **prefill_tokens=130 [M]**, **prefill_exchange_calls=130 [M]**, **prefill_exchange_bytes=5,324,800 [M]**, row_bytes=10,240; first_step_position=129; layers_executed=40; **token_claimed=false [M]**; next_missing = reference 128-row window body + selected-candidate construction, then the 130-position prefill sequence | window/m1y8/rank0/tp4-forward-embed-r0.json; token record window/m1y8/rank0/tp4-token-r0.json |
| S4 | arena residency | **uploaded/resident=77,640,198,984 B [M]** (72.308 GiB/rank), 95,818 tensors, **upload_ms=329,345 [M]**, peak host RSS 79,679,811,584 B | window/m1y8/rank0/tp4-loop-r0.json arena |
| S5 | collectives | **80 calls/step [M]**, **819,200 B/step [M]**, 11 recorded steps **33/19/20/25/22/20/18/20/18/16/16 ms [M]**, loopback **~0.51 ms/reduce [M]**; posted/async reduce ordering not implemented | window/m1y8/rank0/tp4-loop-r0.json collectives |
| S6 | per-layer launch structure | per layer: device projections (5 launches), CSA body (1), five device MoE launches (5) = **11 launches/layer [A from the run note]**; head = 4 launches, tail norm = 1 (single-stream D49 head/tail) [M from m27] -> **445 launches/rank/step [A: 40x11+4+1]** | window/m1y8/rank0/tp4-forward-embed-r0.json note; m27 artifact ADJUDICATION.md |
| S7 | m1y7 cross-run (no prewarm field) | **total_wall_ms=2154 [M]**; per-layer **42-75 [M]**, **sum=2115 [M]**, mean **52.875 [A]** | window/m1y7/rank0/tp4-forward-progress-r0.json |
| S8 | single-node staging chain ledger | **1,168 launches/position [M]**, **device_us=5,506,333 [M]** (5.506 s), staging_ms=12,793, **total_chain_ms=17,452 [M]**; 17 kernel classes, **12,873,976,109 B = 12.874 GB read/position [M/A]**, implied **47.2 ms/position roofline at 273 GB/s [A]** | docs/build/device-forward-chain.md:1443; docs/build/kernel-occupancy-inventory.md:60,85 |
| S9 | b68 steady-state position 2 | attention_staging_us=92,278; **attention_device_us=546,175 [M]**; moe_staging_us=4,964,229; **moe_device_us=4,106,543 [M]**; **total_device_us=4,652,718 [M]**; staging total 5,056,507 us; component counters do not add to the bucket totals exactly, so only the quoted totals are used | docs/build/kernel-occupancy-inventory.md:112-116, device-forward-chain.md:1483 |
| S10 | measured kernel/batch rows (GB10, 2026-10-20) | router **0.2279 ms/8-token launch, 0.0285 ms/token [M]**; cert batched **3.8689 ms one row vs 11.0368 ms six rows = 2.853x [M]**; mHC batched **0.3181 ms six scalar launches vs 0.0735 ms one batched = 4.32x [M]**; lm reduce **2.4709 vs 2.4535 ms = 1.01x, no win [M]**; CSA2 sequence **0.3214 ms scalar, 1.9213 ms six sequential, 1.1465 ms batched six = 3.567x [M]**; projections m12 exactness PASS | docs/build/batch19-plan.md section 7 |
| S11 | m27 head/tail stage | rmsnorm launch **0.1196 ms [M]**; four cert_lm_head_sm121 rank launches **3.8168 / 3.8529 / 3.8889 / 13.3395 ms [M]** (rank0 first-touch outlier); 597.5 ms was harness weight staging, not per-position work | window/m1y7/kernel-m27-adjudication/fixed-run/raw/m27-device.log |
| S12 | boot/collective reference rows | boot **332-351 s [M]** after single-slab allocation; 80 reduces/step, **14-22 ms/step rank-0 transport-only [M]** | docs/build/batch19-plan.md section 1 |
| S13 | targets (not measurements) | **42 ms/verify step [D]**, **3.8 accepted tokens/step [D]**, **11.1 ms/accepted token [D]** (=42/3.8, i.e. ~90 tok/s [A]); weight roofline **13.9 ms/step/rank [D/E]**, RDMA **~1.7 ms/step [D/E]**, draft **~2.2 ms/step [D/E]**; CPU stopgap **767.7 ms/step = 1.3026 tok/s [M, host-twin path, different contract]** | docs/build/gpu-device-roadmap.md; batch19-plan.md |
| S14 | D41 declared quality bars | positions_min 9; top1_agreement_min 0.80889423; top20_overlap_mean_min 0.23333333; top100_overlap_mean_min 0.40; top1000_overlap_mean_min 0.37; max_abs_delta_max 15.10289097; mean_abs_delta_max 1.48408739; kl_mean_max 0.50954607; kl_p95_max 3.18471709; js_mean_max 0.10852956 **[D]** | docs/build/d41-bars-declaration.md |

## 2. What m1y8 actually ran: prefill vs decode separation

The record is explicit, and the separation matters more than the wall number:

- **130 prompt-token embeddings were gathered on rank 0** in 130 exchange calls carrying 5,324,800 B
  [M, S3]. This is a host gather of the prompt embedding rows. It is **not** a prefill pass: no
  reference attention body on 128-row windows, no selected-candidate construction, no layer walk
  over the 130 positions.
- **One position (position 129) walked the 40-layer chain** and wrote all 40 per-layer entries
  [M, S1]; the head/tail stage then ran ("loop-complete, tail-complete, and head-complete",
  evidence written) but the forward ends in **refusal** (chain-r0.rc=2; the token record says
  forward=refused, positions_run=0, no ids/argmax/logits dump, wall_ms=[], reduce_calls_per_step=0)
  [M].
- The head block records token 5367, value 14.866787910461426, certified=false, claimable=true,
  token_claimed=false, twin null [M]. As in m27, claimable=true is a head-stage statement; the
  run claims no token.
- **next_missing** is the reference 128-row window attention body and selected-candidate
  construction, then the 130-position prefill sequence [M, S3].

Consequence: m1y8 gives **one measured single-position structural chain plus one measured
one-time prewarm plus one measured 130-row embedding gather**. It gives **no measured prefill
sequence, no decode step, no TTFT and no tok/s**. Every "130 + 9" number in section 5 is
therefore arithmetic over that measured chain, not a run.

## 3. Per-position cost model (single position, measured decomposition)

The 2,186 ms single-position run decomposes, using only measured rows and one subtraction:

| component | value | label | share of 2,186 ms |
| --- | --- | --- | --- |
| shared-expert view prewarm (one-time per prefill in this binary) | **1,063 ms** | [M] S2 | 48.6% [A] |
| 40-layer per-position chain (sum of the 40 layer wall_ms) | **1,082 ms** | [M] S1; sum [A] | 49.5% [A] |
| residual not covered by either counter (130-call embedding gather, head/tail, glue) | **41 ms** | [A: 2186-1063-1082=41] | 1.9% [A] |
| **total chain wall** | **2,186 ms** | [M] S1 | 100% |

Cross-run check (this is why the prewarm interpretation is credible, not just plausible):

- m1y7 has no prewarm field and its per-layer counters sum **2,115 ms** [M, S7]; m1y8's sum is
  **1,082 ms** [M]. Difference = **1,033 ms [A]**, against the measured prewarm **1,063 ms [M]**
  (the 30 ms gap is run-to-run/position difference; there is no repeat data, so do not over-read
  it).
- m1y7 total **2,154 ms [M]** vs m1y8 **2,186 ms [M]**: the totals differ by 32 ms (1.5% [A])
  while the layer-counter sums differ by 2x. That is consistent with m1w20 moving the
  per-position shared-expert view gather out of the layer counter into a one-time prewarm
  [A inference from the two records, not a code diff]. The layer-counter scope is not comparable
  across the two binaries; the **totals** are.

So, for planning:

- **first position after a fresh prefill load**: 2,186 ms measured, including the 1,063 ms prewarm.
- **steady-state position after the prewarm is warm**: **1,082 ms [M]** layer chain, plus an
  un-isolated per-position glue term; **1,082-1,123 ms [A]** is the planning range (1,123 uses
  the one-position residual 41 ms as a per-position upper bound; the residual is not proven to
  repeat per position).
- **per-layer average**: 2,186/40 = **54.65 ms [A]**, of which the attributed layer counter
  averages **27.05 ms [A]**. One layer at the derived 54.65 ms average already exceeds the entire
  **42 ms** verify-step budget by **1.30x [A]**; at 27.05 ms it uses **2.44 token budgets [A]**.
- **launch arithmetic**: 11 launches/layer x 40 = 440, +4 head launches +1 tail norm =
  **445 launches/rank/step [A]** (the single-node staging ledger has 1,168 launches/position
  [M, S8] because it runs the four rank-slice launches serially on one node; do not compare the
  two counts directly).
- **collectives**: 80 calls/step = 2/layer, 819,200 B/step [M, S5]; the measured transport-only
  skeleton is **16-25 ms/step steady [M]** and the loopback reference is **0.51 ms/reduce [M]**.
  The per-layer sequence note lists the attention and MoE all-reduces inside each layer, so the
  layer walls are expected to include those reduces already [M note]; the transport-only skeleton
  is a separate measurement and is an amplifier/overlap note, not an addend to 1,082 ms.

## 4. Where the 11.1 ms budget is spent today

The budget is declared, not achieved [D, S13]. Its derived composition (arithmetic only) is:

| budget component | ms/step | ms/accepted token | label |
| --- | ---: | ---: | --- |
| weight-traffic roofline | 13.9 | 3.66 | [A: 13.9/3.8] |
| RDMA collectives | 1.7 | 0.45 | [A: 1.7/3.8] |
| DSpark draft | 2.2 | 0.58 | [A: 2.2/3.8] |
| remainder for all compute/memory work | 24.2 | 6.37 | [A: 42-13.9-1.7-2.2] |
| **verify step / accepted token** | **42** | **11.1** | [D] |

Current measured/anchor spend, expressed in those same units:

| row | ms | token budgets (11.1 ms) | step budgets (42 ms) | label |
| --- | ---: | ---: | ---: | --- |
| one-position chain wall (includes one prewarm) | 2,186 | 196.9x | 52.0x | [A over M] |
| steady-state position, layer chain only | 1,082 | 97.5x | 25.8x | [A over M] |
| steady-state position, layer chain + residual bound | 1,123 | 101.2x | 26.7x | [A over M] |
| one prewarm | 1,063 | 95.8x | 25.3x | [A over M] |
| residual (embedding gather + head/tail/glue) | 41 | 3.7x | 0.98x | [A] |
| a single layer at the derived average | 54.65 | 4.9x | 1.30x | [A] |
| DSpark 6-row verify step, rows sequential | 6,492 | - | 154.6x | [A: 6x1082] |
| per accepted token at 3.8/step, rows sequential | 1,708 | 153.9x | 40.7x | [A: 6492/3.8] |
| measured CPU stopgap, host-twin path (context only) | 767.7 | 69.2x | 18.3x | [M, different path] |

The measured CPU stopgap is **767.7 ms/step = 1.3026 tok/s [M]** on the host twin; it is a
different path and contract and is shown only so nobody mistakes the absence of a device row for
a device regression.

**DSpark acceptance accounting (status, not a rate).** The 3.8 accepted tokens/step is a budget
assumption [D] used to derive the 11.1 ms target; it is not a measured acceptance rate for this
arm. Batch-19 records that real-checkpoint served acceptance evidence is still pending, that the
earlier synthetic acceptance counts are not real-checkpoint evidence, and that the TP=4 arm
reports 0 ("no draft source in this slice") [D/M context, batch19-plan.md section 2]. The D41
bars below gate the device-vs-oracle quality leg; m1y8 did not run the oracle leg (no full-vocab
dump), so no D41 row is claimed here.

**D41 declared bars [D, S14]:** positions_min 9; top1 agreement >= 0.80889423; top20 overlap
mean >= 0.23333333; top100 >= 0.40; top1000 >= 0.37; max abs delta <= 15.10289097; mean abs
delta <= 1.48408739; KL mean <= 0.50954607; KL p95 <= 3.18471709; JS mean <= 0.10852956. These
are quality bars for a full-vocabulary device-vs-oracle dump; they do not replace the 42 ms/3.8-
token throughput budget and they do not turn m1y8's refusal into a pass.

## 5. Current structural chain: 130-position prefill + 9 decode steps

This scenario is arithmetic over the measured steady-state position, with explicit assumptions:

- the reference prefill body is **not** wired (section 2), so the only way the current chain can
  process a prompt position is the single-position structural chain;
- the prewarm is paid once per prefill and reused by every position [M, S2];
- per position after the prewarm, use **1,082 ms [M]** layer chain, and treat the 41 ms
  one-position residual as an upper bound on per-position glue (1,123 ms [A]);
- rows are processed sequentially (there is no row-batch path wired).

| scenario | prefill 130 positions | decode 9 positions | warm total | TTFT warm | TTFT cold (+329,345 ms upload [M]) | label |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current structural chain, layer chain only | 1,063 + 130x1,082 = **141,723 ms = 141.72 s** (+41 ms glue = 141.76 s) | 9x1,082 = **9,738 ms = 9.74 s** | **151.50 s** | **141.76 s** | **471.11 s** | [A over M] |
| current structural chain, per-position glue bound | 141.76 s | 9x1,123 = **10,107 ms = 10.11 s** | **151.87 s** | 141.76 s | 471.11 s | [A over M] |
| current chain + DSpark 6-row verify, rows sequential | 141.76 s | 9x6x1,082 = **58,428 ms = 58.43 s** | **200.19 s** | 141.76 s | 471.11 s | [A; 6-row chain not measured] |

Notes on the table:

- The 141.72 s prefill arithmetic is 1,063 ms prewarm + 130 x 1,082 ms. The layer-chain share is
  **140.66 s = 99.2% [A]** of it; the prewarm is 0.75% [A].
- Under the DSpark 6-row variant, 34.2 accepted tokens [A: 9x3.8] in 58.43 s of row-sequential
  decode gives **0.585 tok/s decode [A]** and **1,708 ms/accepted token = 153.9x the 11.1 ms
  budget [A]**. The measured 6-row batched kernels (2.853x/3.567x/4.32x, S10) are not a wired
  6-row chain; this row is arithmetic, not a prediction.
- Collectives in the sequential scenario: 130 x 80 = **10,400 calls [A]**; at the measured
  skeleton 16-25 ms/step that would be 2.1-3.3 s, and at the 0.51 ms loopback reference 5.3 s
  [A]. Those are inside the per-layer counters above and are listed only as an amplification
  note; do not add them to 141.72 s a second time.
- A cold process pays the measured **329,345 ms [M]** arena upload before any of this. For a warm
  (already resident) model the upload is one-time and is not part of per-request cost; for a
  per-run cold start it dominates TTFT.

## 6. Top-3 reducible costs, with magnitudes

1. **The per-position 40-layer chain itself: 1,082 ms [M sum] = 49.5% of the one-position wall
   [A].** This is 97.5 token budgets and 25.8 step budgets [A]; it is 445 launches/rank/step
   [A]. The strongest measured evidence that this is serialization rather than irreducible kernel
   time is the single-node split: **17,452 ms total vs 5,506 ms device = 11,946 ms (68.5%) not
   device [M/A, S8]**, and the occupancy audit's launch-count note calls graph replay the largest
   single launch-count reduction available. Reducible moves: graph replay of the whole position
   (1,168 -> one graph launch in the single-node shape), device-resident router/cert/mHC readbacks
   (today read back per layer), multistream on the independent projections. Measured kernel
   batches help only where a row axis exists (DSpark 6 rows): 2.853x cert, 3.567x CSA body,
   4.32x mHC [M]; there is no measured one-row graph row, so the removable fraction inside
   1,082 ms is **not** claimed beyond the measured-wall ceiling.
2. **Shared-expert view gather/prewarm: 1,063 ms [M] per prefill = 48.6% of the single-position
   wall [A]; 240 gather calls and 1.417 GB [M].** If paid per position (m1y7 shape) it is
   **25.9 ms/layer [A: (2115-1082)/40 = 25.825, against 26.575 from the measured 1,063]**, i.e.
   roughly one full per-layer counter; amortized over 130 prefill positions it is **8.18 ms/
   position [A]** (0.75% of the current prefill). Reducible by moving the gather to model load /
   residency (one-time, not per request) or by gathering the views device-side once. For a
   single decode position with a cold prewarm it is roughly half the first-position wall
   (48.6%); for the 130-prefill scenario it is small.
3. **The routed-MoE kernel/launch class: 680 of 1,168 launches (58.2% [A]) and 4.869 of 12.874 GB
   read per position (37.8% [A]).** In the measured b68 row the MoE bucket's device time
   (moe_device_us=4,106,543 [M]) is **88.3% [A: 4106543/4652718]** of total_device_us=4,652,718
   [M] - note the occupancy page states 78%; that text is not reproducible from its own quoted
   counters, while the counters themselves are consistent under 88.3%. (b68 predates the
   scalar-serial-K rewrite described in the same page, so its 4.1 s is the old schedule's
   number, not today's MoE time.) The measured code-derived 47.2 ms/position roofline at
   273 GB/s [A, S8] and the exact m26 5-launch/rank staged MoE proof [M] bracket the target:
   fuse w1+w3 and fold the weighted SwiGLU/quantise into the epilogue (12 -> 4 launches/layer,
   480 -> 160 per position single-node shape) and collapse the 4 serial combine launches to 1
   [A from occupancy audit section 5]. Caveat: the single-node ledger and the four-rank loop do
   not have a one-to-one launch mapping; use the class magnitude, not a direct subtraction.

Not in the top three, recorded so they are not re-litigated: attention projections are 5.071 GB
(39.4% of bytes) but b68 attributes only 546 ms (11.7%) of device time to the attention
bucket; the certified head
is 1.333 GB (10.4% of bytes) but only 5 launches/position single-node (4 in the single-stream
chain) and the m27 measured head stage is 0.1196 ms norm + four rank-shard launches of
3.82-13.34 ms [M] (13.34 is rank0's first-touch outlier, steady shards ~3.85 ms); mHC is 82
launches but 0.171 GB (1.3% of bytes); collectives are 80 calls/step but a measured 16-25 ms/step
skeleton.

## 7. After realistic reductions: planning envelope, not measurement

**7.1 What already exists as measured ingredients.** Residency is real in m1y8: the loop
holds 77.64 GB/rank resident, one-time 329.3 s upload, and its layer note runs the device work
from the live arena rather than the b53 per-position staging shape [M/A]. The 6-row batched
entry points are measured and exact: cert 2.853x, CSA body 3.567x, mHC 4.32x, lm-reduce 1.01x
(no win) [M]. The exact m26 staged MoE proof exists at tokens=1 [M]. The m27 head/tail stage is
measured [M]. What does **not** exist is a measured 6-row or 130-row *chain*, or a measured
post-rewrite per-stage decomposition of the four-rank position.

**7.2 Roadmap hypothesis envelope.** The roadmap's stage hypotheses are explicit *estimates*,
not measurements [E]: residency 9,100 -> ~2,500 ms, kernel occupancy rewrite -> ~295 ms,
router/head -> ~24 ms, RDMA one-shot + CUDA graphs -> ~15.8 ms per step. The same page's rule is
that a stage that has not run keeps its estimate label; none of these has a measured four-rank
device row yet. m1y8's 2,186 ms measured wall already sits *below* the roadmap's pre-residency
9,100 ms estimate, which is consistent with residency being real but says nothing about the
post-rewrite step estimate.

**7.3 Scenario envelopes** (all [E] unless labelled otherwise):

| scenario | prefill 130 | decode 9 | warm total | TTFT warm | TTFT cold | label |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current structural chain (section 5) | 141.76 s | 9.74-10.11 s | 151.50-151.87 s | 141.76 s | 471.11 s | [A over M] |
| batched prefill only (new change), sequential decode unchanged | memory-traffic floor 0.51-0.82 s/rank + unmeasured activation work; planning range **~1-5 s** | 9.74 s | **~10.7-14.7 s** | **~1-5 s** | ~330.3-334.3 s | [E planning; floor [A]] |
| full roadmap hypothesis (batched kernels + graphs + seams) | 2.05-3.12 s if 15.8-24 ms per sequential position; weight floor 0.28 s/rank once batching exists | 0.14-0.22 s | **2.2-4.4 s** (incl. 1.06 s prewarm if not moved to load) | 2.05-4.4 s | ~331.5-333.6 s | [E hypothesis] |

The batched-prefill memory floor is worth spelling out because it is arithmetic on measured
bytes, not a guess about kernel speed: a full per-rank resident pass is 77,640,198,984 B /
273 GB/s = **284.4 ms [A]**; the routed-expert reads for 130 rows are 4.512 GB/4 = 1.128 GB/rank
per position, x130 = **146.64 GB/rank = 537.1 ms [A]** if every row's six experts are read for
every row, or **62.72 GB/rank = 229.8 ms [A/E: uniform top-6 independence gives ~333.6 unique
experts/layer = 86.9%]** if expert reads are shared across rows. That puts the weight/expert
traffic floor at **0.51-0.82 s/rank**; activation compute for 130 rows is unmeasured and is the
reason the planning range stops at "seconds, not tens of seconds". The cold-start column remains
dominated by the measured 329.3 s upload unless model load is explicitly amortized or overlapped.

The 90 tok/s target and the 42 ms step are **targets [D]**; nothing in this table is an achieved
tok/s, TTFT, or step time. The only measured decode-row in the tree remains the host-twin CPU
stopgap at 767.7 ms/step (1.3026 tok/s) [M, different path].

## 8. Recommendation: the single next-window kernel/launch change

**Make the per-layer launch stage batch-shaped: one batched pass over the prompt rows per layer,
not 130 single-position chain executions.** Concretely: for each of the 40 layers, launch each
kernel class once over the prompt-row batch (up to a 128-row window, with a second tail window for
130) and issue the phase reduce once per layer per batch, instead of running the whole 40-layer
chain once per position.

Why this one, with numbers:

- **140.66 of the 141.72 s current prefill [A, 99.2%] is 130 copies of the same 1,082 ms
  per-position layer chain** [M + A]. Collapsing the position loop into a layer/row-batched pass
  removes 129/130 of the host launch/collective sequences: 5,200 layer executions -> 40-80
  (65-130x fewer [A]) and 10,400 collective calls -> 80-160 [A].
- **The memory-traffic floor is low enough to matter:** 0.51-0.82 s/rank of weight/expert traffic
  for the whole 130-row prefill [A, section 7.3], versus 140.66 s of per-position layer chain.
  That is a >150x gap between the current serialization and the traffic floor; even a fraction of
  it recovered is the largest absolute win available.
- **The measured exactness path already points this way:** the 6-row batched entry points are
  measured and exact (2.853x cert, 3.567x CSA body, 4.32x mHC, 1.01x lm-reduce) [M]; the
  reference prefill body is a 128-row window by contract; the m26 staged MoE is exact at tokens=1
  [M]. The DSpark 6-row verify step needs the same row-batch machinery, so this change is shared
  infrastructure rather than a prefill-only detour.

Why not graph capture first (it is the runner-up, not the recommendation): graph replay removes
launches but does not remove the 130x per-position work in the scenario; its measured-wall
ceiling is at most 1,082 ms per position, i.e. **<=70.3 s across 130 positions [A]** even if all
of the attributed layer wall were removed, versus 140.66 s of per-position chain that batched
prefill targets. Graph capture remains the right **second** move for the 42 ms decode budget and
should follow once the row-batch path exists.

What the change needs (not built now, no claim it is cheap): row-batched b12x dense projection
paths (the dense reference folds K element-by-element, so each row's fold order is unchanged when
rows are independent - exactness per row is the requirement, and m12's exactness shows the current
per-row arithmetic; a row-batched form is not yet measured); the reference 128-row window body
(device-lead's next_missing); row-indexed routed-MoE gather/grouped GEMM (the m26 stored combine
is the exactness anchor, but it was measured at tokens=1 only); batched mHC and head entry points
already measured; and moving the shared-expert prewarm to model load so it is not paid per
prefill. Risk: no 130-row kernel exists and exactness must hold row-by-row; the first gate for any
implementation is a row-batched exactness harness, not a timing row. This page claims no speedup
for the change, only the measured magnitude of the cost it targets and a traffic floor.

## 9. Non-claims and open measurement gaps

Non-claims:

- No new device measurement is reported here; every [M] row is quoted from an existing record.
- No token was emitted by m1y8 or m1y7 (token_claimed=false, forward refused); this page makes
  no token, TTFT, tok/s, step-time, or serving-parity claim, and does not compare across host and
  device paths.
- The 42 ms/step, 3.8 accepted tokens/step, 11.1 ms/token and 90 tok/s values are declared targets;
  they are not achieved values. The CPU stopgap row is a different (host-twin) path.
- The D41 bars are declared quality bars; the oracle leg is unmeasured for m1y8, so no D41 row is
  claimed. Synthetic acceptance counts are not real-checkpoint evidence, and the TP=4 DSpark arm
  reports no draft source in this slice.
- The per-position decomposition uses one run; per-layer wall_ms is a counter whose scope
  changed between m1w19 and m1w20, so it is used only through its sum and cross-checked against
  m1y7; no variance claim is possible from one sample per layer.
- b68's component counters are not additive to the bucket totals; only bucket totals are used
  (the 78% MoE-share sentence in the occupancy page is not reproducible; 88.3% is arithmetic
  from the page's own quoted counters).
- The batched-prefill floor is memory traffic only; activation compute is unmeasured, and no
  measured 6-row or 130-row chain exists. The 0.51-0.82 s/rank figure is a floor, not a target
  or a prediction.

Open gaps, in priority order for the next measurement (no node needed for the analysis):
1. a per-stage timing row for the four-rank resident position (the chain_timing pattern:
   host staging vs launch vs readback vs kernel time) to attribute the 1,082 ms layer wall and
   the 2,186-1,063-1,082 = 41 ms residual;
2. a repeat run of m1y8 for variance (the 30 ms cross-run difference currently has no error bar);
3. a 6-row verify chain row and a 130-row prefill row, both of which do not exist;
4. the b68 counter-scope question (routed+shared device vs moe_device), only if that row is used
   again.

## 10. Provenance and how to reproduce the arithmetic

Paths (all on the shared filesystem, no node access used for this page; window/... means
/home/user/ev-session-20261020/window/...):

- m1y8 records: window/m1y8/run.meta, rank0/tp4-forward-progress-r0.json,
  rank0/tp4-forward-embed-r0.json, rank0/tp4-loop-r0.json, rank0/tp4-token-r0.json,
  rank0/chain-r0.log, rank0/chain-r0.rc (run started 2026-10-21T05:19:45Z; tag m1w20;
  binary ea6452b2...; cubins manifest dd9b7b0c...; checkpoint 1bae0e10...).
- m1y7 records: window/m1y7/run.meta, rank0/tp4-forward-progress-r0.json (tag m1w19;
  binary 3d7a55dc...).
- m27 head/tail: window/m1y7/kernel-m27-adjudication/fixed-run/raw/m27-device.log (norm 0.1196 ms,
  head launches 3.8168/3.8529/3.8889/13.3395 ms; accepted artifact sha256sums.txt).
- Ledger/rows: docs/build/device-forward-chain.md:1443, docs/build/kernel-occupancy-inventory.md
  sections 2-5, docs/build/batch19-plan.md sections 1, 2, 7, docs/build/gpu-device-roadmap.md,
  docs/build/d41-bars-declaration.md.

The arithmetic in this page is deterministic; for example:

- 2186 - 1063 - 1082 = 41;
- 130*1082 + 1063 + 41 = 141764 ms = 141.76 s;
- 9*1082 = 9738 ms; 9*(1082+41) = 10107 ms;
- 6*1082 = 6492 ms; 6492/3.8 = 1708.4 ms; 42/3.8 = 11.05 ~ 11.1 ms;
- 17452 - 5506.333 = 11945.667 ms (68.45% of 17452);
- 77640198984 / 273e9 = 0.284396 s;
- 1.128e9*130/273e9 = 0.537143 s;
- 384*(1-exp(-780/384)) = 333.6 unique experts per layer under uniform independent top-6.

No code, no cubin, no checkpoint, no pin and no node state is changed by this page.

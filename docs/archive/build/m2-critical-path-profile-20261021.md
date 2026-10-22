# M2 critical-path profile from retained evidence (2026-10-21) - measured decomposition, no new window

Scope: where the 2.77-2.89 s per-token step actually goes, read from the already-retained four-rank evidence
of six windows (m1w30 default/combine/mhc/both, m1y32 combine acceptance, m1y33 identity and
combine,shared). No new measurement, no claim about any block that the evidence does not separately time.
This is the starting point for the M2 milestone (>=15 tok/s = 66.7 ms/token; current 2.77-2.89 s/token =
~0.35-0.36 tok/s, so the measured gap is ~41-43x).

## 1. Measured decomposition

Per step (= per position; each window ran 9 positions 129..137 after 2 warmup steps, 36-45 samples):

| window | step mean ms | sum of per-layer wall ms | non-layer residual ms | engram layer 1 ms | engram layer 14 ms | mean other-layer ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| m1w30 default | 2774.9 | 1946 | 828.9 | 64 | 75 | 47.6 |
| m1w30 combine | 2789.1 | 1972 | 817.1 | 66 | 78 | 48.1 |
| m1w30 mhc | 2776.4 | 1963 | 813.4 | 66 | 75 | 47.9 |
| m1w30 both | 2791.9 | 1974 | 817.9 | 65 | 77 | 48.2 |
| m1y32 combine (accepted arm) | 2785.4 | 1965 | 820.4 | 66 | 77 | 47.9 |
| m1y33 identity (no flag) | 2773.2 | 1948 | 825.2 | 66 | 76 | 47.5 |
| m1y33 combine,shared | 2886.9 | 2054 | 832.9 | 70 | 83 | 50.0 |

Measured facts:

- The 40 layer bodies account for **1946-2054 ms of the 2773-2887 ms step (70-71%)**; the residual
  **813-833 ms (29-30%) is spent outside the per-layer measured region** and is not attributed by retained
  evidence (it includes the per-position head/tail path - rmsnorm, cert_lm_head over 40 chunks, the exact f32
  logit all-gather of 129,280 values and the argmax - plus any per-position host-seam/mhc work the loop does
  outside the layer bodies). Instrumenting that region is the first M2 task; no claim is made about its split.
- Mean non-engram layer body is **47.5-48.2 ms** across arms (min 40 ms, max 54 ms for ordinary layers).
- The two Engram layers (1 and 14) cost **64-70 ms and 75-83 ms**, i.e. **+16-35 ms per engram layer** over the
  ordinary layer mean (~+40-55 ms/step total). At M2 scale that is 60-80% of the whole per-token budget for
  two of forty layers - Engram residency/lookup is a first-order M2 target, not a rounding detail.
- The 80-call transport walk costs **19-28 ms/step** (measured in the collectives block of the same evidence,
  819,200 B/step payload) - about 1% of the step, so transport-walk latency is not the M2 bottleneck at this
  scale; the M3 RDMA work is about capacity/overlap, not this 20 ms.
- Arithmetic-arm deltas are small against the step: native combine +14 ms (+0.5%), native mhc -10 ms, identity
  (default) -12 ms vs combine, native shared +102 ms (+3.7%). The switches change host-side arithmetic only;
  they do not measure seam round-trip cost, so seam wall time remains unmeasured (see section 2).

## 2. What this means for M2/M3 (hypotheses, to be measured before any claim)

1. **The step is dominated by device work volume, not by the transport walk or arithmetic-arm choice.** 70% of
   the step is inside 40 layer bodies. Reaching 66.7 ms/token needs a ~30x cut inside the layer bodies and a
   ~500x cut of the 820 ms outside them (or removal of that region from the per-token path, e.g. hoisting the
   head out of the position loop if it is per-position duplicated work - to be established by instrumentation).
2. **Layer body composition is the primary unknown.** ~48 ms per layer at 40 layers for: attention projections
   (native dense launches), flash_mla_csa2_seq_sm121 attention body, wo_a/wo_b and the one attention reduce,
   routed MoE (5 launches) + MOE_REDUCE_SEQ, shared expert, FFN epilogue seams, mhc entry. There is no per-block
   timing in the retained evidence - the next instrumentation pass must timestamp each block per layer
   (one representative layer set or all 40) to separate kernel time from host seam time from launch overhead.
3. **CUDA-graph capture of the layer body** is the obvious M3 direction (remove per-launch CPU overhead and
   per-position host seams), but it is only justified after (2): if a single kernel (e.g. attention body or a
   MoE launch) already eats most of the 48 ms, graph capture will not help enough and the kernel itself must
   change first.
4. **Engram layers and the head/tail region are the two structural outliers** and should be attacked before
   micro-tuning ordinary layers: 2 engram layers ~+45 ms/step is 1.5x the whole M2 per-token budget for 2/40
   layers, and 820 ms/step outside the layer measured region is 12x the whole M2 budget.
5. **The native-rounding arms do not move the wall materially** (±0.5-3.7%), so the fidelity work and the
   performance work are orthogonal at this scale; no measured conflict with the D41 arms exists.

## 3. Next M2 actions (proposed, no claim yet)

- A. Instrument: per-block wall timestamps inside one ordinary layer, one Engram layer, and the head/tail
  region; on rank 0 only, default-off flags to keep the default path byte-identical; report a measured
  breakdown. This is a new kit + one four-rank row (or a rank-0 forward run if the instrumentation is
  host-side only).
- B. Profile the head/tail region first (820 ms candidate): establish whether the 40-chunk cert_lm_head + f32
  logit all-gather + argmax is per-position or once-only, and its measured cost.
- C. Then decide the ordered attack: head/tail hoisting, Engram residency/lookup, layer-body graph capture,
  kernel-level attention/MoE occupancy work. Every step keeps byte-identical-default proofs and re-pinned
  baselines per the standing kernel rules.

## 4. Non-claims

No tok/s, TTFT or serving claim is made here: the measured numbers are per-token step means from the four-rank
D41 windows and the 0.35-0.36 tok/s figure is arithmetic on those means, not a serving measurement. No M2
requirement is met. The 820 ms residual attribution, the head/tail cost, and the per-block split inside a layer
are explicitly unmeasured and must not be quoted as measured until the instrumentation pass lands.

## 5. Addendum - measured per-position split (decode step vs prefill position), from the same retained evidence

The forward-embed record of each window carries `wall_ms` for all 138 positions (`head_position` true for the
last 9, false for the 129 prefill positions), and the loop requests the bf16 twin only at head positions,
positions < 4 and positions % 8 == 0 (`tp4_forward.rs`, `twin_requested`). That splits the retained wall times
into measured groups without any new window:

| window | head positions n=9 mean ms | non-head, twin n=20 mean ms | non-head, no twin n=109 mean ms | twin overhead ms | head position - lean position ms |
| :--- | ---: | ---: | ---: | ---: | ---: |
| m1y32 combine | 2785.4 | 1638.7 | 1367.4 | 271.3 | 1418.0 |
| m1y33 identity (default) | 2773.2 | 1623.8 | 1354.4 | 269.4 | 1418.8 |
| m1y33 combine,shared | 2886.9 | 1661.2 | 1389.9 | 271.2 | 1497.0 |

Measured consequences:

- **A decode (head) position costs 2773-2887 ms; a lean prefill position costs 1354-1390 ms.** The head-position
  extra of **1418-1497 ms** is the head/tail path plus the head-position-only per-layer progress writes and
  head-position evidence work (the loop writes the growing progress JSON after every layer of a head position,
  and every head position requests the twin). How that extra splits three ways - twin ~270 ms measured,
  head-only progress writes, head tail - is NOT yet measured; it needs the instrumentation pass.
- **The twin costs 269-271 ms per position measured** (identical across three kits). Every D41 step mean in
  this record (2773-2887 ms) is a head-position wall that includes the twin and the head-only writes, so it is
  not a lean decode-step cost. M2 performance work must measure with those diagnostic paths off (lean mode),
  otherwise the reported step mean carries ~0.27 s + unknown-write overhead that a serving path would not pay.
- **The true present-day per-token compute floor is the lean position: ~1354-1390 ms** (40 layers + Engram,
  minimal evidence writes) => ~0.72-0.74 tok/s if a decode step could be a lean position; still ~20x from the
  M2 target of 66.7 ms. The gap is not in transport (~20 ms measured) or in the rounding arms (+-0.5-3.7%);
  it is in the layer blocks, the Engram layers and the head/tail path.
- The evidence-writing path is itself a measurable optimization target: the loop serializes a growing progress
  JSON after every layer of every head position (O(layers^2) work per token); deferring/summarising it is a
  cheap, low-risk M2 win that does not change arithmetic at all.

This addendum supersedes nothing in sections 1-4; it adds the measured per-position split that section 2.2 said
was missing. The attribution of the 1418-1497 ms remains unmeasured until the instrumentation pass.

## 6. Correction to section 5 - the twin costs ~0.55-0.59 s at the decode window, not 0.27 s

Section 5's "twin overhead 269-271 ms" was computed as the difference of two small-window group means and is
mixture-biased: the twin cost grows with the attention window (more rows staged), and the twin prefill positions
are mostly early positions with small windows while the plain group is dominated by late positions. This was
caught in kernel-lead's M2 plan and independently reproduced here by regressing non-head positions:

lean wall = a + b*(position+1) fitted on the 109 plain non-head positions, then twin residual = wall - lean fit
fitted against position+1. Measured fits (ms):

| window | lean fit a | lean fit b ms/row | twin fit intercept ms | twin slope ms/row | twin at 128 rows ms |
| :--- | ---: | ---: | ---: | ---: | ---: |
| m1w30 default | 1407.7 | -0.197 | 97.0 | 3.523 | 548.0 |
| m1y32 combine | 1361.5 | 0.088 | 40.8 | 4.155 | 572.7 |
| m1y33 identity | 1337.8 | 0.249 | 30.7 | 4.333 | 585.4 |

(Cross-check: the head position's own 40-layer wall sum 1946/1965/1948 ms is reproduced by lean layer work
~1392/1392/1362 ms + twin@128 549/573/586 ms within 25 ms.)

Corrected decode-step decomposition (m1y32, mean 2785.4 ms):

| component | ms | status |
| :--- | ---: | :--- |
| 40 layer bodies, twin excluded (lean layer work) | ~1392 | measured via regression + layer-sum cross-check |
| attention twin at the decode window (128 rows) | ~573 | measured via regression |
| head residual (head position wall minus its 40 layer walls) | ~781-789 | measured directly |
| unaccounted | ~30 | residual |

So the earlier "head-position extra over lean = 1418-1497 ms" (section 5) is correct as a total, but its split is
**~573 ms twin + ~789 ms head residual**, not 270 + 1400. The head residual is the head/tail path plus
head-position-only per-layer progress writes, and the twin is a diagnostic-only cost present in every published
D41 step mean. Both must be excluded (lean mode) before any M2 performance number is quoted, and both are
exactly what the Phase-1 probe is designed to attribute. Section 5's group-mean table stays as raw data; its
"twin overhead" column is superseded by this regression.

Also recorded from kernel-lead's plan: `TailStage::open` runs per HEAD position (tp4_forward.rs:5022) and is not
cheap - it reads the rank's head shard from the device (330,956,800 B D2H), derives the block maxima with a
165.5M-iteration host loop, copies 2.47 MB back and loads two cubins, then closes after each position
(tp4_tail.rs:270-347). That is position-independent work, so opening once per run is the cheapest measured
lever after the probe. All per-block costs remain hypotheses until the probe row lands.

## 7. E record - M2 lean+hoist kit (m1w34): measured 2.78 s -> 1.41 s per token, byte-identical

Kit: branch task/m2-lean-hoist tip 4e4be4e2 (kernel-lead), merged into win/m1w25 as e88559a0; staged binary
sha256 e3470a141a886bdc3413c12073fd0a16d37e445bbc9b39aca4bb70ce452643eb on all four nodes; cubins 20/20
unchanged; gates green (fmt, workspace clippy -D warnings, full workspace tests, scripts tests, cuda check).

What changed (no arithmetic, no launch, no pin, no default behaviour change):

1. **Phase 2A - run-scope tail.** One `TailStage::open`/`close` and one decoded final-norm weight read per run
   (after the bind, before the position loop); every head position borrows them. The per-head-position 331 MB
   head-shard D2H, 165.5M-iteration host block-maxima derivation, 331 MB `head_shard_sha256`, 8 alloc/free pairs
   and 2 module loads are now paid once per run instead of nine times.
2. **Phase 2B - `--forward-lean` (default off).** Twin forced off; per-layer progress writes on the non-head
   cadence (40 -> 5 at a head position, removing the O(layers^2) growing-JSON rewrite per token); per-position
   head seam f32 files skipped except the last-position alias; shas and head-record shape unchanged. Default
   path takes the existing branches byte-for-byte. Evidence delta: one `forward_perf: {lean: bool}` field on
   every row plus the lean evidence-shape deltas (twin_skipped values, progress cadence, seam files).

Measured rows (four ranks rc=0, 9 positions 129..137, same frozen inputs; wall = per-head-position step mean):

| row | flags | logits bin sha256 | tokens | step mean | head extra over lean |
| :--- | :--- | :--- | :--- | ---: | ---: |
| pre-kit m1y33/m1y32 | none | aed6054d... | [491,92928,201,...] | 2773.2-2785.4 ms | ~1418-1497 ms |
| **m1y34-identity (A)** | none (hoisted) | **aed6054d... (exact)** | identical | **2030.67 ms** | **645.8 ms** |
| **m1y34-lean (B)** | `--forward-lean` | **aed6054d... (exact)** | identical | **1407.89 ms** | **27.1 ms** |

Acceptance:

- A (default path, hoisted): logits bin sha `aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2`
  EXACT vs the pre-kit rows; tokens identical; head-record shas equal; all 10 head-input files byte-identical
  (verified file-by-file); so the hoist is default-path byte-identical including evidence.
- B (lean): logits bin sha and tokens identical; only diagnostic evidence differs (head-input holds the last
  alias only, 5 progress writes per head position, twin skipped).
- Per-head-position walls are now tightly clustered in both rows (2022-2037 ms A, 1397-1421 ms B), i.e. the
  position-varying tail_open cost is gone.

Measured M2 consequence: the decode step fell **2773-2785 ms -> 1407.9 ms (-49.2%, 1.97x)**; the default
(evidence-bearing, D41-comparable) step fell **-26.7% to 2030.7 ms**; the head-position extra over a lean prefill
position fell **1418 ms -> 27.1 ms**. The hypothesis chain recorded in sections 5-6 is now measured: per-position
`TailStage::open` ~0.77 s, twin ~0.55-0.62 s, progress writes in the remainder, and the head residual is now
~27 ms. What remains in the 1.41 s lean step is the 40 layer bodies + Engram (lean prefill position measured
1.354-1.390 s pre-kit, consistent). M2 target 66.7 ms/token is still ~21x away and is layer/Engram/kernel work,
not tail work; no tok/s claim is made.

Deviations: two pre-existing clippy `collapsible_if` sites in the boundary-replay block suppressed with
targeted `#[allow]` attributes (behaviour unchanged) so `-D warnings` stays green. Probe (Phase 1
instrumentation) stays on unmerged task/m2-probe WIP 685f3dcf and is reassigned to a fresh work block.

Evidence: window/m1y34-identity/, window/m1y34-lean/ (four ranks each), kit m1w34; QA not asked to freeze -
neither row is a D41 gate row and no bar/oracle/comparator was used.

### 7.1 Independent A/B verification (device-lead, read-only)

Bundle: device-lead/m2-ab-verify-20261021/ (AB-VERIFY-NOTE.md 0ad7d65c..., verify-ab.py 825cd678...,
verify-report.json c77a507d..., SHA256SUMS checked). Four ranks.

- Row A identity PASS on all ranks: head-input 10 files/204,800 B per rank identical names/sizes/shas; head
  record equal after dropping the run path; logits bin sha aed6054d... exact (4,654,080 B); logits jsonl and
  token body equal; bind manifests byte-identical. Only run paths, wall/timing values and the new
  `forward_perf {"lean": false}` field differ.
- Lean delta: head-input 1 alias file/rank; twin entries replaced by the existing skipped form/`twin_skipped`;
  progress in-loop rewrites 40 -> 5 per head position; head records, logits and tokens exact.
- Reconciliation rank 0: A-PRE head walls -742.55 ms (hoist), head extra -773.03, step -742.55; total_wall -768
  (non-head +30.5 ms/row warm-up shape on the default row, plus +1,480 ms one-time run-scope open now outside
  the position walls). B-A step -622.78 ms (ranks -623.67/-623.00/-624.00), head extra 645.80 -> 27.06 ms,
  non-head non-twin -4.0 ms, 20 twin-cadence rows -290.40 ms/row, total_wall -11,683 ms. Implied twin
  578.7-598.7 ms (regression estimate 573) and progress 20-40 ms (35 fewer rewrites = 2,065,778 B less pretty
  JSON per head position, 18.6 MB/run). Nothing exceeds the 50 ms reconciliation flag; ranks agree within
  1.2 ms on the head deltas.
- Open evidence nit (non-numeric, lean-only): the lean head record's `input.file` still names the per-position
  `pre_head_hidden-pos137.f32`, which lean does not write (only the alias is written). Scheduled fix: lean-only
  evidence delta whitelisted in the probe/2C branch. No gate/token claim in this bundle.

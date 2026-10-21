# D41 gate validity on the b69 fixture: the f32 column fails, the twin leg is structurally unsatisfiable, and the m1y16 arm was non-native (Lead record, 2026-10-21)

Status: **recorded findings, no bar change, no re-pin, no verdict change.**
Owner: Lead. Evidence paths and sha256s are at the end. m1y16 stays MEASURED FAIL; the m1w28 kit and the
declared bars (d41-bars-declared-20261021T010800Z.json, sha256 e53500d0...) are untouched. The reference
columns themselves were found to be stale: the phase2 fixture and the twin artifact predate the Engram
write at layers 1/14 and the cross-layer hc pre-mix, so all fixture-based comparisons recorded before
2026-10-21 are annotated, not re-judged (F4-F5). The current-reference re-render of the f32 column is in
flight (F1 completion) and will be appended as a dated addendum.

This record exists because the M1 four-rank device token run (m1y16) cannot be read correctly without it:
the acceptance runner composes a twin-token gate and the D41 device-vs-oracle numeric gate, and on this
fixture no implementation can pass both, while a faithful f32 implementation of the model cannot pass the
numeric gate at all. Separately, the m1y16 arm turned out to run a precision-first (non-native) projection
family, so its D41 row is a diagnostic on a non-native arm. All four findings are about the references,
the protocol composition and the arm wiring - none changes a measured number.

## 1. What was measured

1. **The f32 column was re-rendered with full logits and scored by the frozen comparator.** The golden
   reference binary sequence_fixture from main f72738b (sha256 ed3838df..., the exact binary that produced
   the phase2 fixture and the b69 130-token golden) was run on node rank0 with --logits all on the b69
   prompt (130 tokens, 8 decode steps, 9 compared positions 129..137). Result: 9/9 steps carry full
   129280-wide f32 logits; the prefill row is **bit-identical** to the stored phase2 fixture.
2. **The f32 column was scored against the pinned native oracle with the unchanged comparator and the
   unchanged declared bars** (compare_native_vs_vllm.py, --bars d41-bars-declared...): verdict **FAIL**,
   3 of 10 bars. Like-for-like is proven (prefix mismatches 0).
3. **The device column (m1y16, 7/10 bars) was compared against the same oracle rows** and against the f32
   column directly, full-vocabulary, at all 9 positions.

## 2. Finding F1 - the declared D41 bars are not passable by the faithful f32 simulation column

| bar | threshold | f32 column | f32 pass | device (m1y16) | device pass |
| :--- | ---: | ---: | :--- | ---: | :--- |
| top1_agreement_min | 0.80889423 | 0.00000000 | FAIL (0/9) | 0.33333333 | FAIL |
| kl_mean_max | 0.50954607 | 2.51492555 | FAIL | 1.06274511 | FAIL |
| js_mean_max | 0.10852956 | 0.37667555 | FAIL | 0.19237731 | FAIL |
| mean_abs_delta_max | 1.48408739 | 2.87796403 | FAIL | 1.40867295 | PASS |
| max_abs_delta_max | 15.10289097 | 14.84676964 | PASS | 11.14933266 | PASS |
| kl_p95_max | 3.18471709 | 4.21957087 | FAIL | 3.01375316 | PASS |
| top20_overlap_mean_min | 0.23333333 | 0.28888889 | PASS | 0.57222222 | PASS |
| top100_overlap_mean_min | 0.40000000 | 0.35222222 | FAIL | 0.57000000 | PASS |
| top1000_overlap_mean_min | 0.37000000 | 0.33155556 | FAIL | 0.56288889 | PASS |
| positions_min | 9 | 9 | PASS | 9 | PASS |

Per-position KL(oracle||column), f32 vs device: 129 3.1827/1.7453; 130 4.2196/3.0138; 131 1.8914/0.3506;
132 1.6878/0.2558; 133 2.1970/0.4072; 134 3.8663/1.6933; 135 0.8286/0.1617; 136 3.8513/1.8467;
137 0.9096/0.0902. The device column is closer to the native oracle than the f32 column **at every
position and on every bar metric**. Full-vocabulary f32-vs-device: top1 1/9, top20 0.378, top100 0.396,
JS mean 0.309, KL(f32||dev) mean 1.864; only the prefill row (129) is close (JS 0.0537, KL 0.221, top20
17/20) - the decode rows 130..137 are 1.1-3.6 KL apart (see section 6, open question).

**F1 status: partial.** The table above is the f32 column rendered by the **stale golden binary** (f72738b,
pre-Engram-write semantics). It is kept as a diagnostic, and its exact numbers must not be quoted as "the
faithful f32 column" any more. The same-context re-render from the **current reference build** (git
8125fb9e) is in flight: rows 129 and 130 are in hand (row 130 is bit-identical between the reference's
prefill and decode paths), rows 131..137 are being produced as same-context prefill rows (the current
reference's own greedy stream differs from the stale fixture's continuation, so its decode rows are not
the oracle's teacher-forced contexts). The F1 verdict - can a faithful f32 simulation of this checkpoint
meet a native-calibrated bar - is deferred to that column; the arithmetic-family argument below is
unchanged and does not depend on it.

Reading: the declared bars were derived from **vLLM-vs-vLLM repeatability** (D41 declaration,
docs/build/d41-bars-declaration.md): two independent native captures of the same rows agree top1 9/9 with
KL(released||suite) mean 0.1407, median 0.0852, max 0.4790. That floor is a native-arithmetic floor. The
f32 simulation is a different arithmetic family (D37/D39 record exactly this), so it cannot meet a 3x
native-repeat bar, and its failure is **not** evidence that the f32 chain is unfaithful. Consequence: the
f32 column is a **diagnostic**, not a numeric acceptance reference for a native arm, and a twin-token
match against it cannot be a device-parity criterion (F2). The D41 device-vs-oracle row remains the
meaningful instrument.

## 3. Finding F2 - the twin token gate and the D41 top1 bar are mutually unsatisfiable on b69

The acceptance runner's overall PASS requires **both** twin PASS and oracle PASS
(scripts/window_acceptance.py:438-449). twin PASS is zero mismatches over the compared positions
(scripts/accept_device_token.py:593-648). Over the same 9 positions the two reference streams are:

| position | 129 | 130 | 131 | 132 | 133 | 134 | 135 | 136 | 137 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| twin token (f32 reference, b69-twin/twin-r0.json) | 491 | 367 | 201 | 80 | 3302 | 201 | 54 | 201 | 4341 |
| oracle argmax (pinned native oracle) | 367 | 84 | 223 | 15 | 100262 | 223 | 15 | 6438 | 54 |
| agree | no | no | no | no | no | no | no | no | no |

They agree at **0 of 9** positions. The device token is the logits argmax by the head rule (ties to the
lowest id), so a device that passes the twin gate has argmax exactly equal to the twin stream, and its D41
top1 agreement is therefore **0/9 < 0.80889423** -> FAIL. Conversely a device that passes the D41 top1 bar
has at least 8/9 positions equal to the oracle argmax, so it mismatches the twin at >=8 positions -> twin
FAIL. **No logits can satisfy both legs on this fixture.** The composed M1 acceptance therefore could not
PASS for any implementation, including a bit-perfect one. This is structural (two disjoint reference
streams), independent of any device measurement.

**Updated with the current reference (2026-10-21):** the stale fixture's greedy stream is
[491,367,201,80,3302,201,54,201,4341]; the **current reference's** greedy stream for the same b69 prompt is
[491,92928,201,15,223,4389,6525,6015,5998] (git 8125fb9e, artifact sha256 6e3f1602...; see F5). Against
the pinned oracle argmax [367,84,223,15,100262,223,15,6438,54] it agrees at **1 of 9** positions (132
only) - so a twin PASS against the current reference would still cap D41 top1 at 1/9, far below
0.80889423. The structural conflict is therefore not an artifact of the stale twin; it survives the
re-render. What does change is that a twin gate must not compare contexts: the current reference's tokens
evolve the context greedily, while the D41 window is teacher-forced on the oracle's ids, so a twin row
scored in that window compares different contexts from position 131 onward. The honest protocol keeps the
twin as a **reference-only diagnostic** and compares it only in a window teacher-forced on the twin's own
stream (a separate diagnostic run, not the acceptance window).

Note also that the twin leg is token equality against the f32 reference, which D39 rule 2 already retired as
a fidelity criterion for the device arithmetic class; the runner's exit-0 condition kept it anyway. That is
a **protocol defect** to fix in the runner/protocol, not a bar to relax: the honest repair is to retire the
f32 twin as a pass/fail criterion (keep it as a diagnostic row) and let the D41 device-vs-oracle row be the
numeric gate, and/or to use an in-distribution fixture where the two reference streams agree (suite-wide
native-vs-f32 top1 agreement is 0.9363, vs 0/9 on this OOD random prompt).

## 4. Finding F3 - the m1y16 arm ran the D37 precision-first bf16a dense family (non-native), so its D41 row is a non-native diagnostic

Three independent sources (device-lead, verified against the frozen evidence and pinned source):

1. The m1y16 evidence labels it: window/m1y16/rank0/tp4-forward-embed-r0.json names
   b12x_bf16a_mxfp8w_dense_sm121 for wq_a/wkv/wq_b and b12x_bf16a_mxfp8w_wo_a_sm121 for wo_a AND wo_b,
   with the 'D37 bf16a family: BF16 activations, FP8 E4M3 B12x weights, ue8m0 scale grids' exact string.
2. Code: crates/sparkflash-cli/src/tp4_forward.rs:301-328 selects exactly that family (wiring commits
   fb641c2e/e363fe5d/406dfdfd, 2026-10-20 19:12-19:39, i.e. after D39 was recorded).
3. Policy: D39 rule 1 (docs/DECISIONS.md:2257-2271) makes native E4M3 activations with one ue8m0 scale
   per 32 contiguous values for the five attention projections and the shared expert, rejects f32/BF16-class
   activations as computing a different model, and the task-29 note allows the bf16a form for wo_a **only**;
   native-kernel-arithmetic-audit.md Appendix A says the dense-bf16a variants stay unwired. The single-node
   native chain (attention_device.rs DENSE_KERNEL = b12x_mxfp8_dense_sm121 + host quantize_activation_fp8)
   is correct on this point and the four-rank loop is not.

This reconciles every measurement in this record: at p129 the arm's distribution is close to the f32 column
(JS 0.052, flat p(367)=0.0955) and far from native (oracle p(367)=0.9214, device 0.0955); a bf16-class arm
cannot pass native-calibrated bars, and the device sits between f32 and native because it is a hybrid
(native body/MoE/head, reference-leaning dense projections and host seams). It also explains why the f32
column fails worse: both are in the same bf16/f32 class for the projections, but the f32 column has no
native body/MoE/head at all.

ACTION (Lead decision, option B) - **IMPLEMENTED and MERGED (2026-10-21 04:55Z):** device-lead branch
task/m1-four-rank-token (wiring commit a8f1004e, later message-amended to af61cd3c with identical code),
merged as 6683f88c on win/m1w25. It wires the native dense family
into the four-rank loop and re-measures D41 in one window (m1w29). Host twin test on the captured
m1y16 rank0 pre-head hidden row pos129 shows mxfp8_ref::dense_gemm bit-equal to linalg::matmul_f8_scaled
(ScaleGrid::Block32) after the kernel's final bf16 round, 0/1280 mismatches, no tolerance. No cubin or pin
change: b12x_mxfp8_dense_sm121 (sha256 06dc726f...) is already in the m1w28 staged set and the tree
baseline; only the sparkflash binary restages. Wiring note: device-lead/gate-validity-20261021T0816Z/
native-dense-wiring-20261021.md sha256 5cb9b243... (revised: includes the wo_b label correction; the
m1y16 census table labelled wo_b _wo_a while the launched kernel was dense - the wiring commit names the
launched kernel for all five operands and a host test asserts the evidence blocks). Device-side equality and the bar movement are m1w29
measurements, not claimed here. Original scope: Scope: wq_a/wkv/wq_b AND wo_b -> b12x_mxfp8_dense_sm121 with host E4M3/ue8m0 activation
quantisation mirroring attention_device.rs (DENSE_KERNEL line 892, quantize_activation_fp8, 11-arg ABI
a_fp8+a_scales); wo_a stays b12x_bf16a_mxfp8w_wo_a_sm121 (native per task-29). No engine/reference changes,
no new cubins (native dense cubin already in the m1w28 staged set), no pin mutation. Required before
staging: host twin test (native launch == engine native coded_matmul on a captured row), evidence-field
changes, a wiring note with sha256, the exact new launch list, and a labelled prediction. wo_b is included
now because the audit lists it in the non-native dense-bf16a group.

## 5. What this record does not do

- It does **not** change the declared bars, the derivation, the m1w28 contract, or the m1y16 verdict
  (MEASURED FAIL, frozen; qa/window/m1y16-acceptance-20261021T075813Z/).
- It does **not** claim M1 numeric acceptance. D41 stays FAIL at 7/10 bars on the non-native arm, and the
  three failing bars are recorded as a genuine gap for the native-arm re-measurement. No token, tok/s,
  TTFT or serving-parity claim is made here.
- It does **not** re-open the bars as a response to a failing measurement. F1/F2 are findings about the
  reference columns and the protocol composition; F3 is an arm-wiring fact. Any future bar or protocol
  change needs a separate Lead declaration that says explicitly it is not a response to a failing device
  measurement. The protocol correction requested from QA (retire the twin leg as pass/fail for m1w29+,
  keep it diagnostic, keep D41 as the numeric gate, no threshold change) is exactly that: a D39-consistency
  and structural-impossibility correction, not a threshold movement.
- It does **not** retire the attention-body / native-parity work. Kernel-lead's measured verdict (recorded
  in LEAD-NOTES.md): row-0 attention_out 1.98e-2 vs the f32 capture is the designed bf16-projection ->
  native FP8-window-KV class (body twin 0/8192 bf16 mismatches on the exact staged operands; body expected
  floor 1.66e-3), and the bounded f32-wkv variant is deferred because with the arm moving native the target
  is native parity, not the f32 capture.

## 6. Finding F4 - the reference engine's decode path is exactly consistent with its prefill path, and F5 - the golden fixture and the twin artifact are stale

Two controlled runs on node rank0 with the current reference build (git 8125fb9e, cross-built example
sha256 75cc2383...): (A) 130-token b69 prompt + 1 decode step, (B) 131-token b69 prompt + 1 decode step.
Both stored full 129280-wide logits for all steps (--logits all, --provenance node).

- **F4 (prefill/decode consistency):** B's prefill row at position 130 (context ids[0..130]) is
  **bit-identical** to A's decode row at position 130 (same context, produced after a real KV-cache
  prefill + decode step): KL 0.0000, maxabs 0.0000, top1 both 92928. The reference engine computes the
  same distribution for the same context regardless of prefill/decode bookkeeping, so the fixture's
  decode-row mechanism is sound and there is no reference prefill/decode ambiguity.
- **F5 (stale fixture):** the phase2 golden fixture sequence-130-released.json (sha256 e6df8290..., main
  f72738b, 2026-10-18) predates the Engram write at layers 1/14 (43c210b9) and the cross-layer hc pre-mix
  (bebdfa35). On the same context its row 130 differs from the current reference by KL 2.7052, maxabs
  11.8254, and its top1 is 367 where the current reference says 92928. The b69 twin artifact
  (qa/window/b69-twin/twin-r0.json, sha256 305ab775...) is derived from that fixture, so its 8/9
  mismatch against m1y16 is **not** evidence about the current reference (which is why m1y16's twin row is
  annotated, verdict unchanged). The current-reference artifacts supersede it for diagnostics:
  ref-head-130x8.json sha256 25870a3a... (prefill row 129 + decode rows 130..137, all logits) and
  twin-current-reference-b69.json sha256 6e3f1602... (current-reference greedy stream
  [491,92928,201,15,223,4389,6525,6015,5998]).
- **Consequence for the device-vs-f32 decode-row gap** recorded in the previous section: it was a
  stale-reference artifact at the rows that can be compared same-context. Device m1y16 vs current
  reference: row 129 KL 0.2782 / JS 0.0668 (top1 differs 367 vs 491), row 130 KL 0.1247 / JS 0.0285
  (top1 same 92928), while the same device rows vs the stale fixture were KL 0.221 and KL 2.93. The
  device tracks the current reference; for rows 131..137 a same-context current-reference row requires the
  reference prefill at the oracle's teacher-forced prefixes (the reference's greedy continuation diverges
  from the stale ids at position 131), and those runs are in flight. This removes the "possible
  reference decode-vs-prefill difference" hypothesis entirely.
- **Consequence for the twin gate:** the twin artifact must be re-rendered from a pinned current
  reference build before any twin comparison is honest, and because the twin stream evolves the context
  greedily, a twin row may only be scored in a window teacher-forced on the twin's own stream. The
  acceptance window's D41 leg teacher-forces the oracle's ids, so the amended protocol (QA contract
  m1w29 amendment, sha256 4f615ee7...) makes the twin a reference-only diagnostic/omitted row.
## 7. Verdict

- **F1 CONFIRMED (measured, complete):** the f32 column rendered by the stale golden binary fails the
  declared D41 bars 3/10 (top1 0/9, KL mean 2.515, JS mean 0.377), and the device column is closer to the
  native oracle at all 9 positions. The faithful current-reference same-context column (section 12) scores
  **7/10** - it still fails kl_mean (1.1557 vs 0.5095), js_mean (0.2088 vs 0.1085) and top1 (2/9 vs 8/9),
  while passing max_abs, mean_abs, kl_p95 (3.1789 vs 3.1847), the top-k overlaps and positions. So the
  declared D41 aggregate bars are not passable by a faithful f32 simulation of the released model either:
  the KL/JS/top1 floor is intrinsic to the cross-implementation comparison at these rows, not a device
  defect and not an artifact of the stale binary. The arithmetic-family argument (D37/D39) is unchanged.
- **F2 CONFIRMED (structural, both twins):** twin PASS and D41 top1 PASS are mutually unsatisfiable on
  b69 - 0/9 shared argmax with the stale twin stream, 1/9 with the current-reference stream - so the
  composed M1 acceptance could never PASS on this fixture, and the twin row must not gate.
- **F3 CONFIRMED (measured/wiring):** m1y16 ran the D37 precision-first bf16a dense family for wq_a/wkv/wq_b/
  wo_b contrary to D39 rule 1 and the audit, so its D41 row is a non-native diagnostic. Native dense wiring
  is implemented and merged (6683f88c); the native-arm D41 measurement is window m1w29.
- **F4 CONFIRMED (measured):** the reference engine's prefill and decode paths agree bit-for-bit at the same
  context (KL 0.0, maxabs 0.0), so there is no reference prefill/decode ambiguity and no device decode-path
  defect to infer from the earlier fixture comparison.
- **F5 CONFIRMED (measured):** the phase2 golden fixture and the b69 twin artifact are stale (pre-Engram-write
  semantics), so fixture-based comparisons recorded before today are annotated, not re-judged. Current
  reference artifacts supersede them for diagnostics.
- **Protocol:** QA (contract owner) concurs with the amendment (m1w29 protocol amendment sha256 4f615ee7...):
  twin becomes a reference-only diagnostic/omitted row, D41 device-vs-oracle stays the numeric gate, bind +
  token emission with rank consensus + D41 form the PASS. No bar, oracle, comparator or threshold change.
- **M1 status:** four-rank device token measured, functional path complete; numeric acceptance **NOT MET**
  (D41 7/10 on a non-native arm, frozen FAIL). Next: native-arm D41 window m1w29. The program proceeds to
  the M2 performance milestone in parallel; the numeric gap and the protocol defect stay on the record for
  the final report.
## 8. Evidence

Bundle: /home/user/ev-session-20261020/lead-diag/gate-validity-20261021T0820Z/ (MANIFEST.sha256 inside)
- f32-column-dump.jsonl + .bin - the exact 9-row dump scored below (f32 logits, frozen comparator input)
- device-vs-f32-column.json - full-vocabulary device(m1y16)-vs-f32 table, per position and aggregate
- device-vs-f32-bars.json - the per-bar table above with both columns
- twin-vs-oracle-conflict.json - the two reference streams and the implication
- sibling lead-diag/hostsim-d41-f32-column-20261021T081319Z/hostsim-vs-oracle.json/.md - the frozen
  comparator's own output including the bar table and per-position rows

Reference-consistency and stale-fixture evidence (F4/F5, all on node rank0, host-only, no GPU):
- /home/user/phase2-evidence/refconsistency-20261021/: ref-130p.json (130-token prompt + 1 decode step,
  full logits), ref-131p.json (131-token prompt + 1 decode step), ref-head-130x8.json sha256 25870a3aefc8698f736e094efb5282cc3c93fd458d7d45ae15a324f80605d23d
  (130-token prompt + 8 decode steps, all logits: the current-reference column for rows 129..137), the
  cross-built example sha256 75cc2383557fff4d5bcc3cb3403b939ffb974d973aa169aaad136cdedbfb8a52, and
  ref-prefill-131..137.json (same-context current-reference prefill rows for the oracle's teacher-forced
  prefixes, in flight at the time of writing).
- /home/user/ev-session-20261020/lead-diag/current-ref-twin-20261021T084500Z/twin-current-reference-b69.json
  sha256 6e3f1602dadfbb1a4a53e7503101044fa3029c73ef78c004cf79ccf27f94bff2 (current-reference greedy
  stream, supersedes qa/window/b69-twin/twin-r0.json sha256 305ab775...; diagnostic only).
- /home/user/ev-session-20261020/lead-diag/current-ref-column-20261021/: the assembled current-reference
  dump for the frozen comparator (jsonl + .bin), produced when the re-render completes.

Inputs and reference verdicts:
- f32 column fixture: /home/user/phase2-evidence/sequence-130-released-alllogits.json
  sha256 ce028a54d8b3b8f41395b800fc81835edfb134a0d1bb2637746eef9603dcfe05 (9/9 full logits; prefill row
  bit-identical to sequence-130-released.json sha256 e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308)
- reference binary: sha256 ed3838df5ca28722a19d6dda689d7b90c4dd67c8faf205cafcc3469dcb459ffa, main f72738b
- oracle: /home/user/ev-session-20261020/qa/artifacts/native-oracle-suite.json
  sha256 efce16c2637aeab70475a43b8dc0c5cbe1f066b6bb9cd671480c7d7e03c4572a (prompt b69-fixture-130)
- bars: /home/user/ev-session-20261020/qa/artifacts/d41-bars-declared-20261021T010800Z.json
  sha256 e53500d0... (unchanged)
- comparator: /home/user/wt/qa-accept/scripts/compare_native_vs_vllm.py (unchanged)
- twin stream: /home/user/ev-session-20261020/qa/window/b69-twin/twin-r0.json
- device column: qa/window/m1y16-acceptance-20261021T075813Z/accept-device-token.json sha256 8d9c1c35... (frozen)
- runner composition: scripts/window_acceptance.py:438-449 and scripts/accept_device_token.py:593-648 as pinned
  by the m1w28 contract
- D39 rule 1 + task-29 note: docs/DECISIONS.md:2250-2286; audit: docs/build/native-kernel-arithmetic-audit.md
- m1y16 arm kernel labels: window/m1y16/rank0/tp4-forward-embed-r0.json; wiring: crates/sparkflash-cli/src/tp4_forward.rs:301-328
- QA contract-owner statements: qa/window/m1y16-acceptance-20261021T075813Z-annotation/GATE-VALIDITY-STATEMENT.md
  sha256 3d7734e14618ad0e5d316720daacaa7566f1107dec9afa159c1b5649b319a815 and
  PER-POSITION-BAR-DECOMPOSITION.md sha256 999f3afe...
- device-lead independent evidence: device-lead/gate-validity-20261021T0816Z/ (f32col-vs-oracle.json/.md,
  sim9.jsonl+bin, alllogits.json, oracle-repeat-consistency.json, position-sharpness.json,
  position-129-distribution.json, logits2-device-vs-f32.txt, logits2-m1y14-vs-f32.txt, SHA256SUMS.txt)
- kernel-lead attention analysis and device-lead native-repeat measurement: recorded in LEAD-NOTES.md
  (2026-10-21 sections 03:40-04:40Z)

## 9. Window m1w29 - first native-arm D41 measurement (2026-10-21 08:39-08:49Z)

Kit m1w29 = merge 6683f88c (native dense projections for wq_a/wkv/wq_b/wo_b, wo_a on the native bf16a
variant) on win/m1w25 at b9d43bc4; staged binary sha256
cdb90d8a9b57a6f83fa1e3ace7266184c445a75bd192075efcc50f9e9af2c8d4, cubin set 20/20 pins match, cubin
manifest dd9b7b0c... unchanged from m1w28 (binary-only restage). Invocation identical in shape to m1y16:
four-rank token, the 138 ids of b69-fixture-138-ids.txt (sha256 82f80902f430f1ae8f62ed88134dca03c6b3e49e7dda23b1c69dcd93c9ff6220),
9 compared positions 129..137, teacher-forced on the oracle's reference ids, --forward-reference-ids
491,367,201,80,3302,201,54,201,4341. Four ranks rc=0, joined after 542 s, all ranks bound and walked;
rank0 mean step 2772.778 ms (worst 3157 ms), window row 2773.528 ms/step.

Device tokens: [491, 92928, 201, 15, 4893, 16646, 15, 14265, 86] at positions 129..137. Logits dump sha256
935290a6... (jsonl) / aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2 (bin,
4,654,080 B). Evidence confirms the native kernel labels (7 dense occurrences across the three stages) and
the activation_quant census blocks (5120/160, 1280/40, 2048/64).

D41 device-vs-oracle (frozen comparator, declared bars, oracle and contract unchanged): FAIL, 6/10 bars.

| bar | threshold | m1w29 native arm | m1y16 bf16a arm | m1w29 |
| :--- | ---: | ---: | ---: | :--- |
| max_abs_delta_max | 15.10289097 | 11.92137480 | 11.14933266 | PASS |
| mean_abs_delta_max | 1.48408739 | 1.34467748 | 1.40867295 | PASS |
| kl_mean_max | 0.50954607 | 1.18058155 | 1.06274511 | FAIL |
| kl_p95_max | 3.18471709 | 3.71258261 | 3.01375316 | FAIL (m1y16 passed) |
| js_mean_max | 0.10852956 | 0.19558285 | 0.19237731 | FAIL |
| top20_overlap_mean_min | 0.23333333 | 0.60000000 | 0.57222222 | PASS |
| top100_overlap_mean_min | 0.40000000 | 0.60333333 | 0.57000000 | PASS |
| top1000_overlap_mean_min | 0.37000000 | 0.57755556 | 0.56288889 | PASS |
| top1_agreement_min | 0.80889423 | 0.22222222 | 0.33333333 | FAIL (132, 135 only) |
| positions_min | 9 | 9 | 9 | PASS |

Frozen record: raw acceptance /home/user/ev-session-20261020/qa/window/m1y29-acceptance-20261021T084921Z/
(window-acceptance.json eaa22062..., accept-device-token.json 74f285bf..., d48-step-gate.json 276e5421...)
and QA freeze/annotation ...-annotation/ (MANIFEST.sha256 475fb319..., verdict-summary.json 1ddcc4cb...,
KNOWN-CAUSE.md 7c813b2c..., NATIVE-ARM-D41-ANNOTATION.md b70ab631...). Overall FAIL; component exits
d48-step-gate=3, accept-device-token=1; bind/walk PASS; token MEASURED; twin NOT_ATTEMPTED exactly
"not-attempted: context-incompatible" (amended protocol, not gated); oracle comparison FAIL; step time
MEASURED target-only; accepted tokens/step and tok/s NOT-ATTEMPTED.

Measured reading:

- The swap changed the arithmetic (native vs bf16a distributions differ by KL 0.0388-0.5973, JS 0.0096-0.1255
  at the same contexts; largest rows 129 and 130), and at the same-context rows the native arm is farther
  from the current f32 reference than the bf16a arm (row 129 KL 0.708 vs 0.278; row 130 0.323 vs 0.125) -
  consistent with native and f32 being different arithmetic classes (D39).
- It did NOT move the aggregate oracle metrics toward parity: kl_mean 1.063 -> 1.181, kl_p95 3.014 -> 3.713
  (a pass became a fail), js_mean 0.1924 -> 0.1956, top1 3/9 -> 2/9; mean_abs 1.409 -> 1.345 and the top-k
  overlaps improved slightly. The labelled prediction that native wiring would reach ~8/10 bars is refuted
  by the measurement and is recorded as a contradicted prediction, not attributed.
- device-lead's per-position addendum on the frozen JSONs: the +1.061 sum-KL regression vs m1y16 is dominated
  by p129 alone (+1.967; the other eight rows net -0.906, with four rows improving). p129 is the known
  serving-stack anomaly row (task-32/F1: the f32 column also fails hardest there); device p(oracle top1=367)
  at p129 went 0.0955 -> 0.0115 (oracle 0.9277) and the native arm's p129 argmax returns to 491 (f32-like).
  On non-anomalous rows the swap moves toward the oracle on top-token mass in several rows but nowhere near
  it yet.
- Conclusion recorded: the residual against the oracle is not (only) the dense projection family. The next
  measured candidates are the recorded non-native terms the kernel audit lists (M2 MoE combine rounding, M4
  mHC BF16 stream), the host-f32 shared-expert path (M3 loop path), and the p129 anomaly's effect on the
  aggregate. Window m1w30 (kit 67a12f85, kernel branch 3c4f4430) runs the M2/M4 ablations
  (--native-rounding none|combine|mhc|both) and is scored per-position and aggregate with p129 separated.
- Non-claims: m1y29 is a measured FAIL and a diagnostic of arm arithmetic relative to the pinned oracle; it
  is not native-parity achievement, not a pass, not a device-fidelity certificate. No token/tok-s/TTFT/
  serving claim. The D41 bars, oracle, comparator and contract are unchanged; m1y16 stays frozen.

## 10. Window m1w30 - native-rounding A/B (M2 MoE combine, M4 mHC BF16 stream)

Kit m1w30 = merge 67a12f85 (win/m1w25 b9d43bc4 + kernel branch 3c4f4430, --native-rounding
<none|combine|mhc|both>) staged as binary sha256
e463c0e0e56939932263b7a4fc5f0d32ea8fcdd1af0f86775cbb4d1eccaf5c5a, cubin set 20/20 pins match
(unchanged; binary-only restage). Four windows, each a four-rank token run on the same b69 fixture (138 ids,
9 positions 129..137, teacher-forced on the reference ids); all rows four ranks rc=0, bind/walk PASS, joins
540-556 s, mean step 2774-2792 ms. Evidence window/m1y30-{default,combine,mhc,both}/; scored copies
lead-diag/ab-m1w30/{row}/bars.json and ab-summary.json.

Default-path byte identity is empirically confirmed: the no-flag row's logits dump bin sha256 is exactly
aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2, identical to m1y29's dump (same tokens,
same D41 bars), so the switch plumbing is arithmetic-neutral by default end to end.

Wiring engagement measured per row: launch_census.native_rounding = none/combine/mhc/both as invoked;
forward-embed evidence combine_arith = rank_bf16_reduce with 10,240 B payload (default) vs
native_f32_rank_gather with 122,880 B payload (combine).

D41 diagnostic on the frozen oracle/comparator/bars (positions 129..137; no bar, threshold or oracle change):

| row | bars | kl_mean | kl_p95 | js_mean | top1 | step ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| default / none | 6/10 | 1.180582 | 3.712583 FAIL | 0.195583 | 0.222 | 2775 |
| combine (M2) | 7/10 | 1.029503 | 2.493012 PASS | 0.190749 | 0.333 | 2789 |
| mhc (M4) | 6/10 | 1.334410 | 4.560921 FAIL | 0.203051 | 0.222 | 2776 |
| both | 7/10 | 1.170425 | 3.095213 FAIL | 0.206613 | 0.111 | 2792 |

Combine row tokens: [491, 92928, 223, 15, 4893, 16646, 15, 14265, 86] (position 131 becomes 223 = oracle
argmax, so top1 3/9). Per-position KL default -> combine: p129 3.7126 -> 2.4079, p130 3.0657 -> 2.4930,
p131 0.2579 -> 0.3100, p132 0.1410 -> 0.2794, p133 0.4679 -> 0.3114, p134 1.4705 -> 1.9622, p135 0.1728 ->
0.1744, p136 1.2196 -> 1.2030, p137 0.1171 -> 0.1242.

Recorded reading:

- M2 native combine is a measured POSITIVE on the frozen oracle: +1 bar, kl_p95 PASS recovered
  (3.7126 -> 2.4930), kl_mean -12.8%, top1 +1 row. Measured wall cost +14 ms/step (~0.5%); the earlier
  4.5-9 ms model range is retired in favour of the measured number.
- M4 mHC alone is a measured NEGATIVE (kl_p95 4.5609, p129/p130 worse) and both is worse than combine
  alone (top1 1/9). The mHC native boundaries are faithful to our reference's boundaries (host twins on
  captured activations), but that does not establish that the oracle's serving path executes those
  boundaries; mhc is therefore NOT adopted and stays a documented measured-negative switch.
- Hypothesis (flagged as hypothesis, not conclusion): the p129 anomaly was at least partly M2 combine
  arithmetic rather than purely a serving-stack artifact - p129 KL drops 35% under combine - because the
  oracle is a different TP4 execution whose reduce order need not equal our release-order epilogue.
- Best D41 arm measured so far: native dense + M2 native combine = 7/10 bars (kl_mean 1.0295 vs bar
  0.5095, js_mean 0.1907 vs 0.1085, top1 3/9 vs 8/9). Still NOT acceptance and not a bar change; these are
  diagnostic A/B rows, not a new kit acceptance contract.

## 11. Rank-0 native dense launch parity - the projection launch is exonerated (m1w31)

Kit m1w31 = merge bea605ef (win/m1w25 54095b71 + device-lead branch ca2b5f47, default-off
--forward-native-dense-parity [LAYER[:POSITION]]), staged binary sha256
c796de64d31e5a156702bda5320a5a21f0ef3e8c54100b089aac19bc7cd1f04e, cubin set unchanged (20/20).
Three rank-0 diagnostic rows, each a full four-rank token window on the same fixture with the default
arithmetic arm (no --native-rounding), all four ranks rc=0:

| row | capture point | internal checks | external tie to retained evidence |
| :--- | :--- | :--- | :--- |
| m1y31-parity | layer 0 / position 129 (bare flag = first decode row) | all zero, all four operands | INTERNAL-ALL-CLEAR (no retained stage capture at this position) |
| m1y31-parity-pos0 | layer 0 / position 0 (explicit 0:0) | all zero, all four operands | INTERNAL-ALL-CLEAR (no retained stage capture at this position) |
| m1y31-parity-pos137 | layer 0 / position 137 (explicit 0:137) | all zero, all four operands | RETAINED-EVIDENCE-TIE - reproduces every retained m1y29/m1w30-default digest exactly |

The per-operand check (wq_a, wkv, wq_b, wo_b) runs, at the captured launch: (1) the single-node chain's
activation quantiser (attention_device::quantise_native_activation = bf16_round then
quantize_activation_fp8) on the captured f32 values against the loop's codes/scales; (2) the chain's own
dense launch construction (Mxfp8DenseParams::new + mxfp8_dense_launch::launch, ring 0, step_seq 0) on the
loop's codes/scales and the arena's exact B12x weight bytes, compared bf16-word by bf16-word with the loop's
production output; (3) sparkflash_kernels::mxfp8_ref::dense_gemm on the same bytes; (4) geometry derived from
the operand bytes vs claimed vs plan. Results are zero on all three rows:

| operand | quantiser code/scale mismatches | chain launch mismatches / slots / max_abs | twin mismatches / slots / max_abs | out sha == twin sha == production |
| :--- | ---: | ---: | ---: | :--- |
| wq_a | 0 / 0 | 0 / 1280 / 0.0 | 0 / 1280 / 0.0 | yes |
| wkv | 0 / 0 | 0 / 512 / 0.0 | 0 / 512 / 0.0 | yes |
| wq_b | 0 / 0 | 0 / 8192 / 0.0 | 0 / 8192 / 0.0 | yes |
| wo_b | 0 / 0 | 0 / 5120 / 0.0 | 0 / 5120 / 0.0 | yes |

Geometry (claimed == plan == byte-derived) and payload/scale bytes: wq_a (1280, 5120) 6,553,600/6,400;
wkv (512, 5120) 2,621,440/2,560; wq_b (8192, 1280) 10,485,760/10,240; wo_b (5120, 2048)
10,485,760/10,240. Dense cubin sha256 d751f2b65b076659ff0c4cc37542f8ae4899ec61b7f40c3d8223ba3181f9d350
(the staged m1w29/m1w30 dense cubin). One-shot elapsed 11-33 ms per operand; default-off, additive evidence
file only.

Retained-evidence tie at position 137 (m1y31-parity-pos137): activation f32 sha256
871b0517db06588c90d5faa4db12d8a57521d6fa323818b78509b74639274bba (wq_a, wkv); codes/scales
a6ecf084000944f9.../76a900d3c34989a4... (wq_a, wkv), 01f2f76e.../a748d488... (wq_b), 7e1d1c32.../1835d8f9...
(wo_b); production outputs 1587cbf9..., b3d4ea41..., 96b2e309..., 46b6e4cc... - all identical to the
staged m1y29 / m1w30-default tp4-forward-embed rank-0 evidence. This is the strongest bounded claim
available: on the exact retained bytes, the loop's native dense launch is bit-identical to the chain's launch
construction and to the kernel reference twin.

Scope limits recorded: wq_a/wkv are replicated at TP4 so their comparison is the literal loop-vs-chain
launch; wq_b is a row shard and wo_b an inner shard, so those compare the rank's shard launch against the
chain construction on the same shard bytes, and no world-1 full-inner bit-equality is claimed for wo_b
(the f32 cross-rank reduce order is a designed TP4 non-associativity). The capture is rank-0, one layer, one
position, so identity is conditional on the captured input bytes - which the 0:137 tie now anchors to the
retained evidence.

Recorded consequence for attribution (attribution inference conditioned on the captured bytes, not a
measurement over every layer/position): the measured bit-identity covers the four projection shapes at
layer 0 / position 137 and is reproduced at two further capture points; the launch code path is
shape-parameterised and position/layer-independent, so the dense projection launch family is not a supported
source of the D41 residual. ("Exonerated family-wide" is the inference label; the measurement is the zero
mismatch table above.) The residual candidates that remain are the M2 MoE combine term (measured to improve the
oracle metrics: 6/10 -> 7/10, kl_p95 PASS), the M4 mHC term (measured negative, not adopted), the host-f32
shared-expert path (recorded by the kernel audit, unmeasured), the attention body/head arithmetic, and the
p129-class anomaly rows. Capture-point note: the bare parity flag arms at the first decode row (129) while
the retained stage evidence is written at the loop's last position (137), which is why the 0:137 rerun was
needed; that is a capture-point difference, not a parity failure. The first reader run's CONTEXT-DIFF label
for the 129/0 captures was a pre-registration mis-attribution and is superseded: those rows are
INTERNAL-ALL-CLEAR (no retained capture at those positions).

Row labels: 0:129 INTERNAL-ALL-CLEAR (no retained stage capture at that position); 0:0
INTERNAL-ALL-CLEAR (no retained stage capture); 0:137 RETAINED-EVIDENCE-TIE.

Evidence: /home/user/ev-session-20261020/window/{m1y31-parity,m1y31-parity-pos0,m1y31-parity-pos137}/
(rank0/tp4-native-dense-parity-r0.json, rank0/tp4-token-r0.json, all four ranks); device-lead design/read
bundle /home/user/ev-session-20261020/device-lead/native-dense-parity-20261021T0915Z/
(parity-mode-design-20261021.md, parity-reading-checklist-20261021.md, read-parity-json.py,
m1w30-ab-four-arm-per-position-20261021.md, SHA256SUMS.txt).

## 12. F1 re-render complete - the faithful current-reference f32 column also fails the declared bars

The F1 re-render promised in section 2 is complete: a same-context column from the **current reference
build** (win/m1w25 lineage; host reference engine only, no device) over the same rows and teacher-forced
contexts as the oracle, assembled from prefill runs at the oracle's contexts (rows 129/130 from
ref-130p/ref-131p; rows 131..137 from current-reference prefill rows with the teacher-forced golden
context ids). Comparator: the same frozen oracle (efce16c2...), comparator (d717a1a2...) and declared bars
(e53500d0...); positions 129..137; prefix_mismatches 0.

Result: **7/10 bars** (the stale-binary column had scored 3/10).

| bar | threshold | current-reference f32 column | stale 2026-10-19 binary column | verdict |
| :--- | ---: | ---: | ---: | :--- |
| max_abs_delta_max | 15.10289097 | 10.87602518 | 14.84676964 | PASS (both) |
| mean_abs_delta_max | 1.48408739 | 1.45886207 | 2.87796403 | current PASS, stale FAIL |
| kl_mean_max | 0.50954607 | 1.15568181 | 2.51492555 | FAIL (both) |
| kl_p95_max | 3.18471709 | 3.17888676 | 4.21957087 | current PASS, stale FAIL |
| js_mean_max | 0.10852956 | 0.20876699 | 0.37667555 | FAIL (both) |
| top20_overlap_mean_min | 0.23333333 | 0.56111111 | 0.28888889 | PASS (both) |
| top100_overlap_mean_min | 0.40000000 | 0.57444444 | 0.35222222 | current PASS, stale FAIL |
| top1000_overlap_mean_min | 0.37000000 | 0.55366667 | 0.33155556 | current PASS, stale FAIL |
| top1_agreement_min | 0.80889423 | 0.22222222 | 0.00000000 | FAIL (both) |
| positions_min | 9 | 9 | 9 | PASS (both) |

The stale-binary column (f72738b, 3/10) is quoted from section 2's table as the diagnostic record; the
current-reference column is the F1 re-render (7/10). The column header order is: bar | threshold |
current reference | stale binary | verdict.

Per-position KL(oracle||column, current reference): 129 2.1198, 130 3.1789, 131 0.5312, 132 0.2405,
133 0.6470, 134 1.9825, 135 0.1498, 136 1.4203, 137 0.1310; top1 agreement only at 132 and 135.

Recorded conclusion: a faithful f32 simulation of the released model's reference arithmetic cannot pass the
declared D41 aggregate bars at these rows either. The failures are the intrinsic cross-implementation
failures (kl_mean, js_mean, top1), not the stale binary's broad numeric error (the re-render passes
max_abs/mean_abs/kl_p95/top-k). The declared bars were derived from vLLM-vs-vLLM repeatability; they are a
native-repeatability floor, not a cross-implementation acceptance criterion. This is the F1 verdict and it
does not change any bar, oracle, comparator or contract - it is a gate-validity finding.

Cross-arm context measured in the same period: the device best arm (native dense + M2 combine) scores
**7/10** with kl_mean 1.0295, js_mean 0.1907, kl_p95 2.4930, top1 3/9 - i.e. the device arm is at least as
close to the vLLM oracle as the f32 reference simulation on the aggregate metrics. That is consistent with
vLLM being a native/quantised implementation (the oracle) while both the f32 simulation and the device chain
are different arithmetic classes; it is recorded as context, not as acceptance.

Artifacts (host; sha256): column dump jsonl e98cc5cc2bcfbaf1fd0969e2919e2b01d757cb4b258b47dca5cef386d0da09d8,
column logits bin d54dd1271bf89c8fb792a29fa3852460fd86459788801154824b0a980677401d (4,654,080 B),
comparator out d0ef781991d76d6cd1ca1c3e253468f0fb85dcf224b049b61dd404fb423a7167 / .md
32f2c15dfde60002f5f86ce4c5245cfe0d3c1cb0aef124f6165c109febc9ac22 (under
/home/user/ev-session-20261020/lead-diag/). Source prefill rows on node .80
/home/user/phase2-evidence/refconsistency-20261021/ref-prefill-131..137.json (copies on the dev host).
Non-claims: this column is an auxiliary diagnostic, not a serving-parity or token-acceptance artifact, and
QA records it as annotation only.

## 13. p129/p130 boundary-row provenance - confirmed capture-regime mismatch, but it does not explain the anomaly

Read-only analysis by device-lead on retained artifacts (no vLLM restart): bundle
/home/user/ev-session-20261020/device-lead/p129-provenance-20261021/ - ANALYSIS.md sha256
9db2c631..., p129-provenance.json 23c3d331..., gate-annotation-numbers.json 4cbb47b9..., analyzer
7b6e9473..., SHA256SUMS af338159... (verifies). It resolves two questions and corrects three statements in
this record; no bar, oracle, comparator, contract or verdict changes.

**Mapping is like-for-like; the measurement regime is not.** The oracle row p is prompt_logprobs[p+1]
(distribution after p+1 prompt tokens) from one ~140-token request (prompt_logprobs 1000, max_tokens 1, temp 0,
top_p 1, sentinel token); the device row p is a teacher-forced decode step after a 130-token prefill. Same-row
shape probe: with the same 130-token prefix and target 491, the oracle's p(367) at p129 is 0.9214 in the
D41 140-token capture but 0.4482 in a matched minimal 131-token shape - KL 0.880 over the shared band
(690 tokens), top100 overlap 0.69, mean |dlogprob| 2.02 nats; identical-shape repeats are bit-identical
(3/3, KL 0). The capture shape alone moves this row by more than the kl_mean bar, so the D41 oracle row was
never a decode-vs-decode number. No decode-regime capture (same prefill, generate with logprobs) is retained
anywhere, and it cannot be taken now (vLLM restart forbidden): "does vLLM's own prefill prompt_logprobs equal
its decode row at this context" is UNMEASURED.

**But the regime mismatch is not the cause of the anomaly.** At matched minimal shapes, f32-vs-oracle is
KL 0.082 at p128 with identical top-1, but KL 1.726 at p129 (f32 argmax 491 vs oracle 367) - far outside the
measured length envelope (KL <= 0.235). The oracle argmax 367 is invariant across every retained shape/load
(0.448 idle 131, 0.598-0.882 busy 131-133, 0.820 at 139, 0.921 at 140, 0.690-0.908 batch 4); bf16a device
picks 367 (flat, p 0.0955) while native/f32 device picks 491. p129 is therefore a boundary-row
cross-implementation arithmetic-class divergence (task32: at p129 neither column is truth), not a device
detect and not removable by shape matching.

**Corrections to statements in this record (measured, no verdict change):**

- "p129 dominates every D41 aggregate" is measured-inaccurate: p130 is the largest-KL row in 5 of 7 measured
  columns; the p129+p130 boundary pair is what dominates - default 34.9%+28.9% = 63.8% of the KL sum, combine
  52.9%, mhc 66.6%, both 53.7%, faithful f32 current-ref 50.9%, stale f32 32.7%, m1y16 49.8%.
- "the faithful f32 column fails hardest at p129" is measured-inaccurate: current-reference f32 worst is p130
  (3.1789 vs p129 2.1198); stale f32 worst p130 (4.2196 vs 3.1827); m1y16 worst p130 (3.0138 vs 1.7453).
- "non-anomaly-row lens (positions 130..137)" should be 131..137: p130 is itself boundary-dominant. The
  corrected lens still fails the declared bars: 131..137 kl_mean 0.5496 (default), 0.6235 (combine), 0.7289
  (faithful f32), 2.1760 (stale f32); top1 <= 3/7 for every column. Boundary exclusion cannot rescue the gate,
  which is consistent with F1 and the gate-design proposal.
- Every p129-inclusive aggregate carries the capture-regime qualifier above; the same row at matched shape is
  p(367) 0.4482, KL 0.880 from the D41 capture - it is not a decode-vs-decode number, and the idle/busy
  capture state (0.448 -> 0.882 at 131 tokens) cannot be corrected post hoc because batch/load/cache state was
  not recorded with the oracle rows.
- Missing evidence stated explicitly: no vLLM decode-regime capture and no matched-shape oracle rows for
  p130..p137 (retained oracle rows: 7, 128, 129 only). The anomaly is nevertheless evidenced independently of
  the regime question by the matched-shape prefill-vs-prefill split.

Consequence for the gate-design proposal (docs/build/d41-gate-proposal-20261021.md): Option D is now
answered as far as the retained evidence allows - the boundary pair is a capture-regime plus
arithmetic-class issue, not a device defect; a future gate should use decode-regime oracle rows and
same-shape captures, which is a protocol change for a future acceptance round, not a bar edit now.

## 14. Window m1y32 - acceptance record of the best measured arm (native dense + M2 combine)

Kit m1w32 = git tip 3c81c45c (win/m1w25; parity merge bea605ef = 67a12f85 + ca2b5f47, parity code
default-off), staged binary sha256 c796de64d31e5a156702bda5320a5a21f0ef3e8c54100b089aac19bc7cd1f04e on
all four nodes (same sha as m1w31 - code identical to bea605ef, docs-only delta), cubins 20/20 unchanged.
QA contract final: qa/window/m1w32-acceptance-contract-20261021T103200Z/ (CONTRACT f027a6e1..., launcher
423668ad..., MANIFEST 8381b196...); the only change vs m1y29 is the kit/binary and the pinned
--native-rounding combine flag (not a default-arithmetic change; no bar/oracle/comparator change).

Four ranks rc=0, joined 546 s at 2026-10-21T10:40:18Z; bind/walk PASS 208/208; mean step 2785.22 ms;
tokens [491, 92928, 223, 15, 4893, 16646, 15, 14265, 86]; logits dump bin sha256
227a0234cd6fbaaf3dc573cb2aad0289b3ae46131b1826a5dd14f4e98db35b3e - identical to the m1w30-combine dump
under a different binary (e463c0e0 -> c796de64), i.e. the combine arm is deterministic across kits. Wiring
engaged: run.meta records the flag, argv 4/4, launch_census.native_rounding.active = combine
(combine=true, mhc=false), 102/102 evidence files verified.

QA acceptance verdict: **overall FAIL** (window exit 1; component exits d48-step-gate=3,
accept-device-token=1); token MEASURED (9 positions, rank-consistent); twin NOT_ATTEMPTED
(not-attempted: context-incompatible, gated=false); oracle comparison FAIL 7/10 declared bars; step time
MEASURED target-only; accepted tokens/step and tok/s NOT-ATTEMPTED; standalone d48-step-gate
NOT-ATTEMPTED/STEP-TIME-ONLY (same nuance as m1y16/m1y29, not INVALID).

D41 7/10: PASS max_abs 11.6700 (<15.1029), mean_abs 1.3319 (<1.4841), kl_p95 2.4930 (<3.1847), top20
0.5778, top100 0.5911, top1000 0.5703, positions 9. FAIL kl_mean 1.0295 (bar 0.5095), js_mean 0.1907 (bar
0.1085), top1 0.3333 (bar 0.8089; agreement at 131, 132, 135). Prefix mismatches 0. This is the best measured
D41 arm record: a measured FAIL with improved numbers relative to m1y29 (6/10) and m1y16 (7/10 non-native
arm), not a pass and not native-parity.

Frozen record: qa/window/m1y32-acceptance-20261021T103500Z/ (window-acceptance.json 17654d5e...,
accept-device-token.json b0d6eba3..., d48-step-gate.json 3f2d53be..., 10/10 raw files pinned) and annotation
...-annotation/ (MANIFEST 5e1fdc84..., verdict-summary 54e37e72..., m1y32-ANNOTATION.md 91df83d7...,
KNOWN-CAUSE 4e86a93a..., FREEZE f188aaa9..., FROZEN-FILES f01aacdc...). Per-position annex (QA, explanatory
only, p129 OPEN-CAVEAT pending provenance resolution, included in aggregate, no second gate):
D41-PER-POSITION-ANNEX.md 814272b0..., .json 9e6362ef... .

Note: p129/p130 provenance analysis (section 13) landed after this freeze; its findings do not change this
record's verdict or bars - they qualify every p129-inclusive aggregate as prefill-prompt_logprobs vs decode
regime and correct the anomaly unit to the p129+p130 boundary pair.

## 15. Window m1y33 - native shared-expert arm (M3 term) measured negative; default-off capability kept

Kit m1w33 = merge 8c35c5f3 (win/m1w25 + task/native-shared e2524fdd; kernel-lead phases A-C green:
forward/moe.rs shared_expert_rows , native_linear (released F8-E4M3 32x32 Linear: BF16 in, FP8-E4M3 activation
quantisation with ue8m0 block scales, f32 accumulate, BF16 out), add_shared_expert_native (released shared
Expert: released swiglu_limit clamps, one BF16 round before the down projection, BF16 row out); CLI
NativeRounding.shared + per-layer evidence shared_arith/shared_sha256/shared_l2 and layer-0
shared_f32_sha256/shared_native_vs_f32_rel_l2; committed m26 captured MoE-input fixture pins the activation
quantiser codes/scales exactly as the loop's host-seam quantiser output (provenance corrected: not a device
capture)). Staged binary sha256 fbfd3a0cdc4fa48e7b741b7506f4e2692e71cfb11b865bdce2b35571abb1b77b on all
four nodes; cubin pins 20/20 unchanged; fmt/clippy/workspace tests clean (engine 209, CLI 322).

Row 1 - default byte-identity gate (diagnostic): m1y33-identity, no --native-rounding flag. All four ranks
rc=0, active=none, argv flag count 0; rank-0 logits bin sha256
aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2 - EXACTLY the m1y29/m1w32-default seed
value. The new evidence code and the additive shared seam leave default arithmetic bit-identical. QA verified
the same independently (JSONL sha 472ea54f...). Not scored, diagnostic only.

Row 2 - M3 arm (diagnostic): m1y33-shared, --native-rounding combine,shared. All four ranks rc=0, 9 positions,
active=combine,shared, 40/40 layers shared_arith native_fp8_block32_bf16 with non-zero shared_l2, layer-0
shared_f32_sha256 non-null with shared_native_vs_f32_rel_l2 = 0.020136 (wiring engaged), tokens [491, 92928,
201, 15, 4893, 16646, 15, 14265, 86], mean step 2886.89 ms (vs 2785.22 ms combine-only, +3.7%), logits bin
sha a4996363... . Scored with the same frozen oracle/comparator/bars: **6/10 bars**, kl_mean 1.3159 (bar
0.5095 FAIL), kl_p95 4.1851 (bar 3.1847 FAIL), js_mean 0.2036 (bar 0.1085 FAIL), top1 0.2222 (FAIL); PASS
max_abs 11.3623, mean_abs 1.3501, top20 0.6111, top100 0.6044, top1000 0.5780, positions 9. So the native
shared expert **does not improve the declared aggregate gate**: 6/10 vs 7/10 for combine-only (m1y32/m1w30).

Per-position delta vs the combine-only row (delta KL, positive = worse): 129 +1.7772, 130 +1.2773, 131
-0.0809, 132 -0.1625, 133 +0.0103, 134 -0.2510, 135 -0.0478, 136 +0.0848, 137 -0.0300. The native shared
term improves six of the seven non-boundary rows (non-boundary 131..137 kl_mean 0.5554 vs 0.6235 combine,
-11%) but regresses the p129/p130 boundary pair severely (+1.78/+1.28), and the boundary pair dominates the
aggregate. Measured consequence: M3 shared-expert term is NOT adopted as a candidate arm; the default path is
unchanged; the default-off capability and its evidence/tests are kept on the branch for future
composition/debugging. No bar/oracle/comparator/contract change; diagnostic only (no contract, no freeze).

Boundary-pair observation (no claim): the boundary rows are exactly the rows whose oracle values came from the
prefill prompt_logprobs capture regime (section 13); the arm that most closely reproduces released arithmetic
regresses them furthest while improving interior rows. This is consistent with the boundary rows measuring a
regime/arithmetic-class difference rather than decode arithmetic; it is recorded as an observation for the
future protocol fix (decode-regime oracle rows), not as an excuse for the aggregate FAIL.

Evidence: window/m1y33-identity, window/m1y33-shared; comparator output lead-diag/ab-m1w33/
(m1y33-shared-bars.json/md, m1y33-shared-summary.json). QA diagnostic scoring requested independently.

QA independent diagnostic record for the M3 arm (m1y33): qa/window/m1y33-diagnostic-ab-20261021T111300Z/ -
diagnostic-summary.json 75ab7c2d..., .md 143b7579..., wiring-check.json 3917e1ee..., identity-guard.json
192609b2..., comparator.json ab11d9a7..., source-hashes.json c7b158fe..., MANIFEST c2c81a8b... (10/10 verify).
QA reproduced the local diagnostic exactly (6/10; same aggregates/per-position/tokens), confirmed wiring PASS
(active combine,shared all ranks; 40/40 layers shared_arith native_fp8_block32_bf16; layer-0 rel L2 0.020136;
combine_arith native_f32_rank_gather and 80 reduce calls/step unchanged, both read from
rank*/tp4-forward-embed-r*.json under moe_routed), and recorded the arm as measured NEGATIVE - no freeze, no
contract, no promotion, frozen m1y32 record verified unchanged.

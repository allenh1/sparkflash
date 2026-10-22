# D41 gate-design proposal (2026-10-21) - proposal only, NOT adopted, no bar change

Status: **proposal for a program decision**, written after the F1 re-render (gate-validity record
`docs/build/d41-gate-validity-20261021.md` sections 2 and 12), the m1w30 A/B (section 10) and the parity
diagnostics (section 11). Nothing in this document changes the declared D41 bars
(`d41-bars-declared-20261021T010800Z.json`, sha256 `e53500d02d71f33bfc85dbb0debc4f32b336452d8b5a7edf0d90e1f6742148ae`),
the oracle (`native-oracle-suite.json`, `efce16c2...`), the comparator (`d717a1a2...`) or any QA contract.
Adopting anything here would require a separate, explicit Lead declaration under D39 rule 2 - explicitly
not a response to a failing measurement - and QA would have to re-derive any threshold from a measured floor.

## 1. The measured problem

The declared bars were derived from **vLLM-vs-vLLM repeatability** (top1 9/9, top20 0.7611, KL mean 0.1698,
KL p95 1.0616, max abs 5.0343, JS mean 0.03618; rule ceil = max(provisional, 3x reference floor)). They are
a native-repeatability floor. Measurements of faithful *different-arithmetic* implementations at the b69 rows:

| column | D41 bars passed | kl_mean | js_mean | kl_p95 | top1 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| faithful current-reference f32 simulation (host) | 7/10 | 1.1557 | 0.2088 | 3.1789 | 0.222 |
| device native dense + M2 combine (best measured arm) | 7/10 | 1.0295 | 0.1907 | 2.4930 | 0.333 |
| device native dense, f32 rounding (m1w30 default) | 6/10 | 1.1806 | 0.1956 | 3.7126 | 0.222 |
| device native dense + mhc (measured negative) | 6/10 | 1.3344 | 0.2031 | 4.5609 | 0.222 |

The faithful reference simulation and the device chain fail the same three intrinsic bars (kl_mean, js_mean,
top1); the device best arm is at least as close to the oracle as the f32 reference simulation. The residuals
are therefore dominated by cross-implementation arithmetic differences and by the p129-class anomaly row, not
by one device wiring defect (the dense launch family was measured bit-exact against the chain construction and
the kernel twin; section 11).

Non-anomaly-row lens (positions 130..137, p129 excluded; measured from the same frozen comparator outputs):

| column | n | kl_mean | js_mean | top1 | top20 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| faithful f32 column (current ref) | 8 | 1.0352 | 0.1788 | 0.250 | 0.569 |
| device m1w30 default/none | 8 | 0.8641 | 0.1518 | 0.250 | 0.600 |
| device m1w30 combine | 8 | 0.8572 | 0.1556 | 0.375 | 0.562 |
| device m1w30 mhc | 8 | 0.9311 | 0.1573 | 0.250 | 0.600 |
| device m1w30 both | 8 | 0.9972 | 0.1714 | 0.125 | 0.550 |

Even on non-anomaly rows the declared thresholds are not met by either faithful implementation, so excluding
p129 alone would not make the gate pass; the KL/JS/top1 thresholds themselves reflect native-vs-native
repeatability, not cross-implementation agreement.

## 2. Options (for the program owner / QA)

- **Option A - keep the gate as declared (recommended default for the record).** M1 numeric acceptance
  remains measured NOT MET: best 7/10 bars. Honest, requires no declaration; the evidence trail is complete
  (m1y16 frozen FAIL non-native arm, m1y29 frozen FAIL native arm, m1w29 contract + amendment, m1w30
  diagnostic A/B, parity diagnostics).
- **Option B - re-derive D41 from a cross-implementation repeat floor.** Measure vLLM vs the faithful
  reference column repeatability on the same rows (the reference column already exists:
  `lead-diag/current-ref-column-20261021/`), then declare new floors as max(provisional, 3x
  cross-implementation floor) with the same rule shape. Requires a formal Lead declaration ("not a response
  to a failing measurement") and QA re-derivation plus a new bars artifact; bars would change by declaration.
- **Option C - keep the declared gate and add a secondary, non-gating attribution metric** (non-anomaly-row
  kl_mean/js_mean/top1 and a per-position table) to separate "cross-implementation floor" from "regression".
  No thresholds move; acceptance verdict stays as declared. Cheapest and lowest-risk.
- **Option D - p129 investigation.** The anomaly row is the single largest aggregate contributor (default
  3.7126 of kl_mean 1.1806; combine improves it 35%). Worth one focused investigation (oracle serving-stack
  behaviour at that row: task-32) before any threshold decision, because it dominates every aggregate.

## 3. Recommendation

Keep the declared gate (Option A) as the acceptance criterion for any near-term claim, adopt Option C as a
secondary diagnostic in the next acceptance record, and only consider Option B after a focused p129
investigation (Option D) and a measured cross-implementation repeat floor. Do not change bars now.

## 4. Non-claims

This proposal is not an acceptance result, not a bar change, and not a claim that M1 passed. M1 numeric
acceptance remains NOT MET (best measured arm 7/10). All numbers above are from the frozen comparator
outputs at the paths cited in `docs/build/d41-gate-validity-20261021.md` sections 10-12.

## 5. Amendment (2026-10-21, after the p129/p130 provenance analysis)

Measured findings in docs/build/d41-gate-validity-20261021.md section 13 (device-lead bundle
p129-provenance-20261021, SHA256SUMS af338159...) correct three statements in sections 1-2 above, with no
bar/oracle/comparator/contract change:

1. The anomalous unit is the **p129+p130 boundary pair**, not p129 alone: p130 is the largest-KL row in 5 of 7
   measured columns, and the pair carries 51-67% of the KL sum in the six 9-row columns (32.7% in the stale f32
   column).
2. The faithful f32 column fails hardest at **p130**, not p129 (3.1789 vs 2.1198 current-reference; 4.2196 vs
   3.1827 stale; m1y16 3.0138 vs 1.7453).
3. The non-boundary lens is **positions 131..137**, not 130..137. Corrected lens (annotation only, positions_min
   bar requires 9 rows): kl_mean default 0.5496, combine 0.6235, faithful f32 0.7289, stale f32 2.1760; top1
   <= 3/7 for every column - still above the declared kl_mean bar 0.5095, so boundary exclusion cannot rescue
   the gate.

Added provenance finding: the D41 oracle leg is a **prefill prompt_logprobs** capture (one ~140-token request)
while the device leg is **decode** rows; the same row at a matched 131-token shape has p(367) 0.4482 vs 0.9214
in the D41 capture (KL 0.880), so p129-inclusive aggregates are not decode-vs-decode numbers. The mismatch does
not explain the anomaly (matched-shape f32-vs-oracle p128 KL 0.082 vs p129 1.726; oracle argmax 367 invariant
across shapes) - it is a boundary-row arithmetic-class divergence. Missing evidence: no vLLM decode-regime
capture and no matched-shape oracle rows for p130..p137; a future acceptance round should capture
decode-regime oracle rows and same-shape inputs. This is a protocol recommendation, not a bar edit; Option A
(keep the declared gate) remains the recommendation for any near-term claim, with Option C (non-gating
per-position annex) already applied by QA.

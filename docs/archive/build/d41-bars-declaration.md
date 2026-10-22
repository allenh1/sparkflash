# D41 bars declared (Lead, 2026-10-21)

**Status: DECLARED GATE** for the device-vs-oracle leg (D40 re-measurement / M1 token window).

- Gate file: `/home/user/ev-session-20261020/qa/artifacts/d41-bars-declared-20261021T010800Z.json`
- sha256: `e53500d02d71f33bfc85dbb0debc4f32b336452d8b5a7edf0d90e1f6742148ae`
- Proposal it declares: `qa/d41-bars-proposal-20261021T004132Z/d41-bars-proposal.json`, sha256
  `c0f6361d51bc70d5361f5f5679798a93e16c083a1addda21c5bec5a338e85439` (qa-lead task-12,
  commit 50f4219; rationale in `docs/build/d41-bars-proposal-20261020.md`).
- `scripts/compare_native_vs_vllm.py:check_gate_bars` accepts the declared file (status ok).

## Why these values, and why now

The bars are derived only from measurements that predate any device token: two independent released-snapshot
vLLM oracle runs of the same suite plus a third same-context capture, at positions 129..137 of prompt
`b69-fixture-130`. The measured reference repeat at these positions is wider than the D40 provisional bars
(KL mean 0.1698 > 0.02, KL p95 1.0616 > 0.10, mean abs 0.4594 > 0.10, max abs 5.0343 > 1.0, top20 0.7611 <
0.90), so freezing the provisional values would fail a correct chain that merely reruns the reference in a
new session. The declared bars are the D41 rule output (ceilings = max(D40 provisional, 3x worst measured
reference floor); floors = min(D40 provisional, 1-3(1-worst floor)); top1 = min(0.95, 1-3x suite-wide rank-1
flip rate)), unchanged from the proposal.

Declared values: positions_min 9; top1_agreement_min 0.80889423; top20_overlap_mean_min 0.23333333;
top100_overlap_mean_min 0.40; top1000_overlap_mean_min 0.37; max_abs_delta_max 15.10289097;
mean_abs_delta_max 1.48408739; kl_mean_max 0.50954607; kl_p95_max 3.18471709; js_mean_max 0.10852956.

Discrimination is retained: the PRE-FIX b70-vs-oracle dump fails all nine gated bars; both measured reference
repeats pass. The top-k bars are weak at these positions by evidence (reference top-20 overlap 0.744); they
are kept because dropping them or inventing tighter numbers after seeing results is the failure mode this
pre-registration prevents.

## Rules

- Any later change to these numbers needs a Lead declaration that explicitly says it is not a response to a
  failing measurement.
- The oracle leg stays UNMEASURED unless the device window carries an explicit full-vocabulary dump
  (`--device-dump` + `--device-bin`) and this declared file (`--oracle-bars`); the window acceptance runner
  forwards both.
- The bars never turn a refusal into a pass: dry-run proof in
  `qa/d41-bars-dryrun-20261021T004425Z/DRYRUN.md` (14/14 checks) and qa-lead's rerun of the m1w refused tree
  with and without bars stays exit 3 / NOT_ATTEMPTED.

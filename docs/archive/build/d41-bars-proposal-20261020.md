# D41 bars proposal (2026-10-20): evidence-derived floors before the token window

**Status: PROPOSAL, not a gate and not installed.** The file is `sparkflash-d41-bars/1` with `status: "proposal"`; `compare.check_gate_bars` refuses any
status other than `ok`, so this document cannot become the gate without the Lead reviewing and
declaring a status-ok file (task-12; owner qa-lead). Nothing here is post-hoc: the derivation inputs
are two vLLM oracle runs and one D41 capture that all predate the device token.

## Why a proposal now

A strict D41 capture floor is not available: the 2026-10-20 capture has no `busy_distractor` block and the idle
class carries the unresolved rank-1 flips recorded in `docs/build/d41-floor-verdict-20261020.md`. Waiting for a
perfect capture would mean writing bars after seeing the device token, which is exactly the failure mode
this proposal prevents. Instead the proposal uses the strongest reference-side evidence that exists now:

1. **vLLM-vs-vLLM repeatability.** The same released-snapshot oracle suite (`b69-fixture-130` prompt, positions
   129..137) exists as two independent full runs, plus a third run through the D41 capture at the same
   contexts. Same-request repeats inside the capture are bit-identical (three pairs: KL, JS and abs delta
   exactly 0, top-k overlap exactly 1.0), so the measurement path is deterministic.
2. **Oracle-suite internal stability.** Both runs contain the same 8 prompts, same prompt lengths, same
   position tables and the same target tokens; the suite-wide repeat (1664 positions) is quantified below,
   including its own limits.
3. **D40 provisional diagnostics spread.** The D40 provisional values are evaluated against that reference
   repeat and against the PRE-FIX b70-vs-oracle dump (`qa/o4/compare-native-vs-vllm.json`).

## Inputs (all pre-token)

| input | path | sha256 |
| :-- | :-- | :-- |
| canonical oracle suite | `/home/user/ev-session-20261020/qa/artifacts/native-oracle-suite.json` | `efce16c2637aeab7...` |
| second full oracle run | `/home/user/phase2-evidence/window-2026-10-18/native-oracle-130.json` | `f35d2bc26457e31a...` |
| D41 capture (id `d41-b70`) | `/home/user/ev-session-20261020/qa/artifacts/d41-capture-20261020T012706Z.json` | `0de8d3173860d3de...` |
| PRE-FIX b70 comparison (D40 spread) | `/home/user/ev-session-20261020/qa/o4/compare-native-vs-vllm.json` | `fc91dd5360ba127e...` |

## Comparison set

- Prompt `b69-fixture-130`; canonical 138 fed ids; positions **129..137** (9 positions); position p means
  the next-token distribution after consuming p.
- Oracle support: the stored top-1000 logprobs. KL/JS are renormalised over the compared oracle support;
  abs deltas are raw oracle-vs-device logprobs. This is the same convention as D39.
- Metrics: top-1 agreement, top-k overlap for k = 20/100/1000 (top-5 dropped as in D41), max/mean abs
  logprob delta, KL(oracle||device) mean and p95, JS mean.
- `--flat-prob 0.05` is the D41 flat-position threshold; it is never widened.

## Measured vLLM-vs-vLLM repeatability at positions 129..137

| metric | two oracle runs (A vs B) | capture len-139 vs canonical oracle |
| :-- | ---: | ---: |
| positions | 9 | 9 |
| top1 agreement | 1.0000 | 1.0000 |
| top20 overlap mean | 0.7611 | 0.7444 |
| top100 overlap mean | 0.8056 | 0.8000 |
| top1000 overlap mean | 0.7903 | 0.7900 |
| max abs delta | 5.0343 | 4.1955 |
| mean abs delta | 0.4594 | 0.4947 |
| KL mean | 0.1698 | 0.1468 |
| KL p95 | 1.0616 | 0.4727 |
| JS mean | 0.03618 | 0.03358 |

Suite-wide (both runs, all 8 prompts, 1664 positions): strict top-1 agreement 0.9363, 106 rank-1 flips
above the flat threshold (rate 0.0637). Per prompt, top-1 agreement ranges 0.8600-0.9699 and KL mean
0.0103-0.0846; every prompt has identical lengths/positions/target tokens in both runs. The suite is
internally consistent; it is not statistically tight at the rank-1 level.

### Oracle-suite limits at these positions

Top-1000 retained mass per position is
`[0.9932, 0.8390, 0.1739, 0.3942, 0.8314, 0.4831, 0.3442, 0.9705, 0.5197]` (mean 0.5804 over the whole suite,
minimum 0.1360). The comparison is therefore a top-1000-band comparison, not a full-vocabulary claim.
Argmax top-1 probabilities at these positions are as low as 0.0110 (position 131), 0.0162 (135) and
0.0266 (137); those three positions are flat by the D41 rule (top-1 prob < 0.05). This is why the top-1
bar is derived from the suite-wide flip rate rather than from a 9/9-only pair.

## Proposed bars (would-be gate file, status proposal)

Rule: ceilings = max(D40 provisional, 3 x worst measured reference floor); floors = min(D40 provisional,
1 - 3 x (1 - worst measured reference floor)); top-1 = min(0.95, 1 - 3 x suite-wide flip rate). "Worst"
means the worse of the two measured reference pairs above.

| bar | proposed | D40 provisional | measured floor | flag |
| :-- | ---: | ---: | ---: | :-- |
| positions_min | 9 | 9 | pinned set | - |
| top1_agreement_min | 0.80889 | 0.95 | 1.0000 (9/9); suite-wide flip rate 0.0637 | looser |
| top20_overlap_mean_min | 0.23333 | 0.90 | 0.7444 | looser |
| top100_overlap_mean_min | 0.40000 | 0.80 | 0.8000 | looser |
| top1000_overlap_mean_min | 0.37000 | 0.60 | 0.7900 | looser |
| max_abs_delta_max | 15.1029 | 1.0 | 5.0343 | looser |
| mean_abs_delta_max | 1.4841 | 0.10 | 0.4947 | looser |
| kl_mean_max | 0.50955 | 0.02 | 0.1698 | looser |
| kl_p95_max | 3.18472 | 0.10 | 1.0616 | looser |
| js_mean_max | 0.10853 | 0.05 | 0.03618 | looser |

Nine of ten proposed bars are looser than the D40 provisional. That is a finding, not a convenience:
the D40 provisional bars **fail the reference's own cross-session repeat** at these positions (KL mean
0.1698 > 0.02, KL p95 1.0616 > 0.10, mean abs 0.4594 > 0.10, max abs 5.0343 > 1.0, top20 0.7611 < 0.90).
Freezing the provisional values would fail a correct chain that merely reruns the reference in a new
session; the 3x rule is the D41-compatible way to keep the gate above measured reference noise.

Discrimination is retained: the proposed bars evaluated on the PRE-FIX b70-vs-oracle dump fail **all
nine gated bars** (top1 0.0, KL mean 6.4278, KL p95 14.3411, mean abs 8.1617, max abs 26.0218, JS mean
0.5393, top20 0.0778, top100 0.1056, top1000 0.1364), while both measured reference pairs pass. The top-k bars are
weak at these positions (reference top-20 overlap itself is only 0.744); they are included because the
evidence is what it is and the Lead can drop them more safely than a numeric value can be invented.

## Dry-run proof (mechanics only)

Evidence: `/home/user/ev-session-20261020/qa/d41-bars-dryrun-20261021T004425Z/` (generated by
`scripts/d41_bars_dryrun.py`; all 14 checks pass, exit 0):

| scenario | result |
| :-- | :-- |
| existing m1w refused tree, no bars | exit 3, overall NOT_ATTEMPTED, token NOT-ATTEMPTED, oracle UNMEASURED |
| existing m1w refused tree, dry-run bars | exit 3, same overall/token verdicts; bars cannot turn a refusal into a pass |
| proposal file passed as `--oracle-bars` | exit 2, refused: `status 'proposal'` (not installable) |
| synthetic ideal device, no bars | exit 0, overall PASS, oracle UNBARED |
| synthetic ideal device, dry-run bars | exit 0, overall PASS, oracle **BARED PASS** |
| synthetic ideal device, impossible bars | exit 1, overall FAIL, oracle BARED FAIL (bars do gate) |
| synthetic wrong dump, no bars | exit 0, overall PASS, oracle UNBARED |
| synthetic wrong dump, dry-run bars | exit 1, overall FAIL, oracle BARED FAIL; shape/token/twin unchanged |
| window command on m1w refused tree, with and without bars | exit 3 both; composed verdict unchanged |

The synthetic device is a full-vocab distribution constructed from the canonical oracle (labelled
DRY-RUN); it is not device evidence. The dry run proves the bars file only moves the oracle leg between
UNBARED / BARED-PASS / BARED-FAIL and never causes a pass when no token was emitted.

## Non-claims

- No device token was measured; there is no device-vs-oracle numeric claim in this proposal.
- This is not a D41 capture floor: no capture has all three captured classes; the derivation class is
  reference-repeat (vLLM-vs-vLLM) and the Lead decides whether that is acceptable.
- The comparison is restricted to the oracle top-1000 support; the retained-mass limits above bound it.
- The top-1 bar allows one flip in nine positions; the suite-wide measured flip rate (6.37%) is the basis.

## Declared gate (Lead review complete)

The Lead accepted the ten proposed bars unchanged on 2026-10-21T00:59Z. The gate file is
`/home/user/ev-session-20261020/qa/artifacts/d41-bars-declared-20261021T010800Z.json`, sha256
`e53500d02d71f33bfc85dbb0debc4f32b336452d8b5a7edf0d90e1f6742148ae`, status `ok`, `declared_by: "lead"`;
its declaration block preserves the proposal sha `c0f6361d…` and the "any future change is not a response
to a failing measurement" rule. `compare.check_gate_bars` accepts it and its bars are identical to the
proposal (no top-k drop). `window_acceptance.py` records the declared sha in `inputs.oracle_bars_sha256`
whenever `--oracle-bars` is passed, including pre-token runs whose oracle leg is UNMEASURED.

## Token-window runbook

The exact command for the m26 token attempt (declared bars + explicit dump + twin, `--require-token`):

```
cd /home/user/wt/qa-accept
OUT=/home/user/ev-session-20261020/qa/window/m26-token-$(date -u +%Y%m%dT%H%M%SZ)
python3 scripts/window_acceptance.py --device-dir <m26 device rank tree> --twin <host-twin evidence dir or file> --oracle /home/user/ev-session-20261020/qa/artifacts/native-oracle-suite.json --oracle-prompt b69-fixture-130 --oracle-bars /home/user/ev-session-20261020/qa/artifacts/d41-bars-declared-20261021T010800Z.json --device-dump <device full-vocab dump.jsonl> --device-bin <device full-vocab dump.jsonl.bin> --dump-source "m26 device-arm D41 token dump" --positions 129,130,131,132,133,134,135,136,137 --expected-steps 3 --expected-root 1bae0e1088993b9b82de08f59de5be5a34e42170 --expect-vocab 129280 --expect-attention-bound 208 --expect-moe-bounds routed_experts_gate_up=61440,routed_experts_down=30720,shared_experts=240,router=120 --expect-manifest-lines 185481 --require-token --out-dir "$OUT"
```

Expected: exit 0 only with twin PASS + oracle BARED PASS; exit 1 if the oracle leg fails the declared
bars or the dump shape fails; exit 3 if token/twin evidence is absent (NOT_ATTEMPTED, not a pass);
exit 2 on refusal/error. The dump must have 9 rows at positions 129..137, vocab 129280, a tiled
`<dump>.bin` sidecar, and reference ids `[491, 367, 201, 80, 3302, 201, 54, 201, 4341]`, passed
explicitly. A pre-token window still carries `--oracle-bars` and records the declared sha, but its
oracle row is UNMEASURED because there is no device dump.

Failure classification: a tree with bind/plan evidence but no token envelope (for example a run whose
collective walk dies with rc=2 before any forward) is INVALID / no data. Do not run
`window_acceptance.py` on it as a verdict and do not report it as PRE-TOKEN-PASS; record it separately from
acceptance. The m1x walk-failure record is at
`/home/user/ev-session-20261020/qa/window/m1x-INVALID-collective-20261021T013647Z/`
(`invalid-run.json` sha256 `73116beb906a2fe7511d82f2494496899124fcc89f75b6e9a7b98d1aa65acc0e`, label INVALID), and the
declared gate is untouched by it.

## What the next-window verdict still needs

1. **The declared gate is available** (path/sha above); keep it with the window evidence and never modify
   it in response to a measurement. Without a dump the oracle leg stays UNBARED/UNMEASURED and the verdict
   is an M1-only token PASS, not a numeric gate.

2. **The device full-vocab dump:** 9 rows for positions 129..137 plus the `.jsonl.bin` sidecar, written by
   the device arm, passed explicitly as `--device-dump <file> --device-bin <file.bin> --dump-source <label>`.
   `window_acceptance.py` now forwards all three (and `--oracle-bars`); a dump that is not passed is
   never guessed into a distributional oracle leg.
3. **Twin evidence:** a host twin PASS (or accepted host-reference fixture) is still required for the
   window's overall PASS; the oracle leg alone is separate.
4. **Canonical oracle + prompt:** the frozen `native-oracle-suite.json` with `b69-fixture-130` and the released
   root `1bae0e10...`.
5. **Token contract:** 138 fed ids, 9 device token ids/positions 129..137, reference ids
   `[491, 367, 201, 80, 3302, 201, 54, 201, 4341]` at p129..138.
6. **Complete bind/walk evidence** for the window tree (the m1w tree shape is the reference), and the
   window command run with `--oracle-bars <declared file>` so the verdict can be `PASS (twin + D41-gated oracle)`.
7. **If the dump or the bars file is missing:** state the verdict as M1 token PASS with oracle
   UNBARED/UNMEASURED; do not describe it as a D41 acceptance.

Reproduce:

```
python3 scripts/d41_bars_proposal.py --out-dir /home/user/ev-session-20261020/qa/d41-bars-proposal-20261021T004132Z
python3 scripts/d41_bars_dryrun.py --proposal-dir /home/user/ev-session-20261020/qa/d41-bars-proposal-20261021T004132Z \
    --out-dir /home/user/ev-session-20261020/qa/d41-bars-dryrun-20261021T004425Z
```

Committed fixtures: `scripts/fixtures/d41-bars-proposal-20261020.json` (status proposal),
`scripts/fixtures/d41-bars-dryrun-ok-20261020.json` (status ok, mechanics only) and
`scripts/fixtures/d41-bars-dryrun-impossible-20261020.json`.

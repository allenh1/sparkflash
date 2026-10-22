# WS-G: vLLM recapture, CI perf gate and the path to parity (2026-10-21)

Owner: qa-lead. Status: perf baseline MEASURED on the live deployment; CI gate MECHANISM ready and
honestly SKIPPING; released-root parity NOT STARTED because its preconditions are not met.

This document is the executable plan from the measured deployment baseline to token-accuracy,
classification/KL and evaluation parity. It separates what is automated now from what needs a GPU
window (the vLLM instance currently owns the four nodes' cards and must not be restarted without
Lead authorisation). Every claim below is either measured and cited, or explicitly NOT-MEASURED.

## 1. What exists now

| item | status | evidence |
| :-- | :-- | :-- |
| Live vLLM deployment baseline (TTFT, decode tok/s, tokens, logprobs, provenance) | MEASURED, UNDER-LOAD, heretic root | docs/build/d38-vllm-baseline-20261021.json + docs/build/d38-vllm-recapture-20261021.md |
| Prompt logprobs on the random-token fixture and on natural-language prompts | MEASURED; natural-language self-prediction NORMAL, so the fixture row is EXPECTED behaviour, not degeneracy | same JSON: prompt_logprobs, prompt_logprobs_natural, natural_language_self_consistency |
| Co-tenant attribution | MEASURED-LOCAL: the long-lived streams are the DSH harness's own model backend | same JSON: co_tenant_identification |
| CI perf-regression gate with PASS/REGRESSION/INVALID/SKIP exit codes | AUTOMATED; unit-tested; prints exact evidence path and reason on SKIP | scripts/perf_regression_gate.py, scripts/tests/test_perf_regression_gate.py, .gitlab-ci.yml perf-regression |
| scripts/tests in CI | AUTOMATED (previously nothing ran them) | .gitlab-ci.yml scripts-tests |
| Released-root oracle capture (D38/D41) | BLOCKED: needs the released snapshot served | D38 precondition, see section 2 |
| Device logits dump + D39 distribution comparison | BLOCKED: needs the GPU cards and the released root | scripts/compare_native_vs_vllm.py |
| Token accuracy / token emission | BLOCKED: the tp4 chain does not emit tokens yet (D48 collector reports NOT-ATTEMPTED) | scripts/d48_step_gate.py docstring |
| Evaluation parity (GSM8K / HumanEval / MMLU) | NOT STARTED: no eval harness exists in this tree | section 6 |

The deployment is NOT a parity oracle. serving-readiness.md section 5.1 already records that the
deployed vLLM serves the heretic snapshot 12d58ad0... and not the released snapshot 1bae0e10...;
this work re-measured it rather than assuming it. The D52 interim rule is respected: the baseline
rows are PERF rows, not gate rows, and no token-accuracy, tok/s, TTFT or gate-validity claim is made
from them.

## 2. Precondition 0 - the released root (NOT MET)

| item | value |
| :-- | :-- |
| released D38 root | 1bae0e1088993b9b82de08f59de5be5a34e42170 |
| served root on 192.0.2.80:8000 | .../models--pqhaz--DeepSeek-V4.1-Flash-1017-heretic/snapshots/12d58ad0878b5008e80d720251adb969a0069412 |
| D38 root precondition | NOT MET |
| gate_eligible | false (root mismatch, plus UNDER-LOAD) |

Consequences, stated once and applied everywhere below:

1. This baseline cannot be joined to the frozen D41/D38 released-root rows. It is the deployment's
   own reference, useful for TTFT/decode-tok/s regression while the deployment stays unmodified.
2. Any claim about the RELEASED checkpoint needs the released snapshot served (a restart, i.e. a
   GPU window with Lead authorisation) or an equivalent released-root oracle capture from another
   instance.
3. The heretic snapshot's weights are not the released weights, so a parity comparison against it
   would measure the snapshot difference, not SparkFlash. No such comparison is planned.

Check it before any release-critical run (read-only):

    curl -sS http://192.0.2.80:8000/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["root"])'

The D41/D38 runners refuse the wrong root themselves: scripts/d41_recapture.py --expect-root
1bae0e1088993b9b82de08f59de5be5a34e42170 and scripts/oracle_native.py --expect-root (same default)
exit before sending any completion request. The perf runner (scripts/d38_vllm_recapture.py) records
the mismatch as gate_eligible=false instead of refusing, because a deployment baseline is legitimate
work.

## 3. Stage 1 - quiet deployment baseline (runnable, no restart)

The committed baseline was captured with 2-4 concurrent harness streams running, so its latency is
UNDER-LOAD and is not a single-stream number. The runner supports a bounded idle wait rather than a
restart:

    python3 scripts/d38_vllm_recapture.py \
      --wait-idle-s 20 --idle-budget-s 300 \
      --co-tenant-note "quiet-window attempt by qa-lead" \
      --out-json docs/build/d38-vllm-baseline-$(date -u +%Y%m%dT%H%M%SZ).json \
      --out-md   docs/build/d38-vllm-recapture-$(date -u +%Y%m%dT%H%M%SZ).md

Acceptance for treat-as-quiet: load_conditions.verdict == QUIET in the new JSON (num_requests_running
== 0 before every sample) and at least three ok samples per row. The co-tenant is this host's own DSH
harness (the agent team's model backend, identified by socket owners in the baseline JSON), so a
quiet window needs the team to be quiescent; it cannot be created by stopping the streams inside the
task.

When such a file exists, the CI gate compares it to the committed baseline automatically: same
served root, QUIET verdict, at least three samples, same prompt/max_tokens/recipe, newer than the
baseline and no older than --max-age-days. Until then the gate exits 3 with the exact path it
searched. That is the honest state, and the perf-regression CI job prints it as a SKIP line.

## 4. Stage 2 - distribution parity (KL, top-1, top-k) once the root is released

The machinery already exists; it needs the released root and the device cards.

Step 2a: oracle capture from the released-root vLLM (GPU window; the instance must be serving the
released snapshot):

    python3 scripts/oracle_native.py \
      --base-url http://192.0.2.80:8000 \
      --suite scripts/oracle_prompt_suite.json \
      --fixture /home/user/phase2-evidence/sequence-130-released.json \
      --out /home/user/phase2-evidence/qa/artifacts/native-oracle-suite-RELEASED.json \
      --prompt-logprobs 1000

The suite must keep the b69-fixture-130 prompt (130 token ids, seed 7) as the joined prompt; the
fixture supplies the nine continuation token ids, and the oracle file keys logprobs by device
position (positions 129..137 for the nine continuation steps).

Step 2b: device full-vocabulary logits dump (needs the four cards; today the GPU slot is held by the
vLLM instance, ~111 of 121 GB per node):

    /home/user/ev-session-20261020/run-fourrank.sh <stage-tag> <run-name> step

with --dump-logits <path>.jsonl (the hybrid_sequence stage is the only stage where --dump-logits is
wired; see crates/sparkflash-cli/src/device_chain.rs). The run writes rank evidence and the logits
sidecar; the D48 collector records step time separately:

    python3 scripts/d48_step_gate.py \
      --device-dir /home/user/ev-session-20261020/window/<run-name> \
      --out-json .../d48-step-gate.json --out-md .../d48-step-gate.md

Step 2c: D41 floors and bars (the comparator refuses status != ok):

    python3 scripts/d41_recapture.py --plan scripts/d41_capture_plan.json --out <capture>.json
    python3 scripts/d41_floors.py --capture <capture>.json --out-json <floors>.json \
      --out-md <floors>.md --out-bars <bars>.json
    python3 scripts/d41_bars_proposal.py   # stays status=proposal; only the Lead declares a bar

Step 2d: the comparison itself - top-1 agreement, top-k overlap, mean/max |logprob delta|, KL(oracle
|| device) and JS divergence, per position and aggregate:

    python3 scripts/compare_native_vs_vllm.py \
      --dump <dump>.jsonl --bin <dump>.jsonl.bin \
      --oracle <oracle>.json --oracle-prompt b69-fixture-130 \
      --bars <bars>.json --expect-root 1bae0e1088993b9b82de08f59de5be5a34e42170 \
      --expect-vocab 129280 \
      --out-json <compare>.json --out-md <compare>.md

Without --bars the run is UNBARED: measured evidence, no verdict. D39 forbids turning token equality
into a pass; the token-level metric here is top-1 agreement of the full vocabulary, not string
equality.

## 5. Stage 3 - token accuracy and greedy agreement

Definitions used in this work, so they cannot drift:

* top-1 agreement: fraction of compared positions where the device's argmax token id equals the
  oracle's top token id. Measured by Step 2d. Diagnostic only (D39).
* greedy agreement: fraction of the nine continuation positions where the device's greedy token
  sequence equals the reference fixture's steps[].next_token. This is what the D38 recapture records
  as vs_b69_nine_step_reference; on the deployment capture it is 0/9 and that row is a DIAGNOSTIC of
  the heretic snapshot, not a SparkFlash result.
* token accuracy as a gate: NOT AVAILABLE. The four-rank path does not emit tokens yet, so the D48
  collector reports NOT-ATTEMPTED (exit 3) rather than a number. Do not report a token-accuracy
  percentage until a stage writes tokens and the collector records them.

Acceptance for Stage 3, when it unblocks: top-1 agreement and KL inside the Lead-declared D41 bars
on the same positions, at least three repeat pairs bit-identical (the zero-noise control), and the
greedy continuation matching the fixture only as a diagnostic - never as the gate.

## 6. Stage 4 - evaluation parity within +-1 point

There is no evaluation harness in this tree. Verified by searching scripts/ and docs/build/ for
gsm8k/humaneval/mmlu: the only mention is reference/MODEL_CARD.md, which reports the checkpoint's
own numbers, not a runnable harness. So Stage 4 is a build task, not a rerun.

Protocol (identical for both sides - this is the only way a delta means anything):

1. Same checkpoint: the released snapshot 1bae0e10..., served once by vLLM and once through the
   SparkFlash path (device build). The heretic deployment is excluded by the root precondition.
2. Same prompts: the official item text verbatim, same few-shot examples, same order, same stop
   strings. Zero-shot vs few-shot changes the score by more than 1 point, so it is fixed once and
   recorded.
3. Same sampler: greedy (temperature 0, top_p 1) for reproducibility. State plainly that this is NOT
   the model card protocol (temperature 1.0, top_p 0.95, reasoning_effort=100), so these rows are a
   parity DELTA between two implementations of the same weights, never a reproduction of card
   numbers.
4. Same extraction/scoring code, run over both outputs. Exact-match graders for GSM8K/MMLU, the
   official pass@1 harness for HumanEval. A grader difference is a bug, not a finding.
5. Datasets (needs network on the dev host; CI has none): GSM8K test 1319 items, HumanEval 164
   items, MMLU-Pro test 12032 items (or an explicitly named MMLU subset; the card reports MMLU-Pro
   5-shot, so do not silently substitute plain MMLU). Store the downloaded JSONL under
   /home/user/phase2-evidence/evals/ (not in this repository: size/licence), and record the
   dataset file sha256 in the evidence JSON.

Statistics that decide whether +-1 point is even measurable:

* Unpaired worst case at p=0.5, 95% confidence, +-1 point: n = 1.96^2 * 0.25 / 0.01^2 = 9604 items.
  Below that, a 1-point difference is inside sampling noise and must not be reported as a regression.
* GSM8K test 1319 items: one item = 0.076 points, so +-1 point is about 13 items - a real tolerance,
  but the right test is paired (same items, McNemar on discordant items), not two independent CIs.
* HumanEval 164 items: one item = 0.61 points. A +-1 point tolerance is 1.6 items, so HumanEval
  cannot establish +-1 point parity on its own. Report exact pass counts, the discordant-item count
  and a paired test; do not convert 1-2 item differences into a percentage claim.
* MMLU-Pro 12032 items: one item = 0.008 points; paired McNemar is still preferred because it
  controls item difficulty.

Practical cost: 1319 + 164 + 12032 greedy generations per side, at the measured deployment decode
rate (~8-20 tok/s under load) and with long reasoning outputs, is many GPU-hours per side. That is
why Stage 4 is gated on a GPU window and should start with a 200-item smoke run per benchmark before
any full run.

Acceptance for Stage 4: the full paired run shows no benchmark delta beyond the declared tolerance
(+1 point for GSM8K/MMLU-Pro; exact pass-count parity statement for HumanEval), with the harness,
prompt set, sampler, grader and dataset hashes recorded in the evidence JSON. Anything less is
reported as PARTIAL / SMOKE, never as parity.

## 7. Automation matrix

| layer | automated now | needs GPU window |
| :-- | :-- | :-- |
| Evidence JSON/MD generation | scripts/d38_vllm_recapture.py, scripts/serve_output_probe.py | - |
| Perf regression on committed rows | scripts/perf_regression_gate.py + CI job (PASS/FAIL/SKIP) | producing the fresh QUIET capture |
| Python tooling correctness | CI scripts-tests (unittest discover) | - |
| Distribution comparison | comparator + floors scripts are ready | oracle capture, device logits dump, D41 capture |
| Token accuracy | not available (no token emission) | yes, plus the token path |
| Evaluation parity | nothing yet | yes, plus harness build and datasets |

## 8. Non-claims

* The committed baseline is not a released-root row and does not measure the released checkpoint.
* Its latency is UNDER-LOAD; it is not a single-stream TTFT or tok/s baseline.
* The random-token fixture's low prompt logprobs are expected of a correct model; only the
  natural-language rows test self-prediction, and they came out normal.
* "The deployment's greedy completions are degenerate" (serving-readiness 5.1) is a claim about
  generated output, and prompt_logprobs on given text does not refute it; a Stage 4 run on the
  released root is what would.
* No SparkFlash number appears in the deployment baseline: the cards are held by the vLLM
  deployment, so every SparkFlash row there is NOT-MEASURED.
* Nothing here declares a D41 bar. D41 bars stay status=proposal until the Lead freezes them.

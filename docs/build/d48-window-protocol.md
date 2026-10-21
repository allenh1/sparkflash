# D48 window measurement protocol: one four-rank loop run, one honest step row

**Status: revised 2026-10-20 (v3, adds the device-lead bind evidence contract) by qa-lead
(task-4) for the Lead's single GPU window.** v2 corrected the v1 draft's nonexistent stage/argv; v3
records the final per-rank evidence contract (tp4-bind-rR.json is primary; the step mode will not
land tonight) and the defect classes that must fail loudly. Verified against device-lead's branch-tip
evidence contract of 2026-10-20 and the integrated tree (crates/sparkflash-cli/src/tp4_chain.rs
run_loop, tp4_resident.rs, resident_weights.rs). It does not run anything: the Lead owns the window,
the vLLM stop and the restore.

Decision references (not reopened): D48 target/budget (docs/DECISIONS.md), D40/D41 bars (no D41
bars file exists, so no serving-parity pass/fail), D38 (released snapshot root in every artefact).
Measurement rules: only measured numbers, device and host numbers never share a table, estimates
labelled, a gate reported **unmet, not redefined** when it misses.

## 0. What tonight can and cannot produce

| row | tonight | evidence | label |
| :--- | :--- | :--- | :--- |
| four-rank bind coverage + collective walk | **yes** | 4 x tp4-bind-rR.json (primary) + 4 x tp4-loop-rR.json + names manifests | measured |
| D48 step row (42 ms / step, 3.8 accepted tok/step, 90 tok/s) | **target only, not gate-grade** | d48-step-gate.json/md | STEP-TIME-ONLY / NOT-ATTEMPTED |
| M1 first token (four-rank) | **not reachable tonight** | accept-device-token.json/md | NOT_ATTEMPTED (never FAIL for an absent token) |

| host twin reference | optional, only if >= 6 min remain | separate host-twin document | HOST-REFERENCE, separate table |
| kernel harness rows | yes, separate track | kernel evidence | measured |

The four-rank step mode will **not** land tonight: routed_experts_down binds but cannot launch from
the arena until the stored-layout combine cubin exists (kernel-lead's moe-stored-layout-variant plan,
next session). So the M1 token is NOT_ATTEMPTED tonight and the deliverable is the bind + walk row;
this is a wiring/layout fact, not a measurement failure.

Tonight's GPU work is device (~8 min: ~6 min arena boot + walk + evidence) then kernel harnesses
(~12 min) = **<= 20 min**, then stop + restore (~13 min). The "first token + 10 counted steps" rows
are **not** reachable in this budget; they are the next-window section (§8).

## 1. Verified window interface (Lead correction, 2026-10-20)

- There is **no** --run-stage tp4-device. The four-rank arm stage is **--run-stage tp4-chain**.
- Tonight's runnable mode is the **loop path**: it binds the device-resident names, performs the
  collective walk (3 steps: 2 warm-up + 1 steady, 80 calls/step = 40 layers x 2 phases) and writes
  two JSON documents per rank: **tp4-bind-rR.json** (schema sparkflash-tp4-bind/1, the PRIMARY
  coverage evidence, written progressively *before* the walk) and tp4-loop-rR.json (schema
  sparkflash-tp4-loop/1, the walk/canonical-rule document, which embeds the same bind object), plus
  tp4-bind-names-rR.txt (one name per line, 185,481 logical lines when complete, never parsed as JSON). It
  does not run a forward pass and does not emit a token.
- The bind document is the file to read for coverage: attention tensors_expected=208,
  tensors_bound=208 (tensors_table_entries=215; 7 host seams excluded), MoE class bound counts
  {routed_experts_gate_up 61440, routed_experts_down 30720, shared_experts 240, router 120}, tail 3,
  census.resident_tensors=95818 with unclassified=[], summary{device_bound_names 92729,
  per_token_weight_h2d_bytes 0, complete true}. If the MoE bind fails the file still exists with
  attention complete, error set and moe.performed=false: that is a FAILED bind, not an absent one.
- Known bound-but-not-launch-compatible class (reported, never hidden): moe.launch_compatibility
  routed_experts_down is launch-incompatible until the stored-layout combine cubin lands; bind
  completeness does not make the forward runnable.
- A step-mode argv will come from device-lead when a step is wired; **it is not specified here in
  advance.** No flag guessing.
- tp4-chain flags that exist: --tp-rank --tp-world-size --checkpoint|--tensor-index --plan-only
  --upload --loop --evidence-dir --comm-peers --comm-base-port --comm-smoke --decode-steps
  --warmup-steps --cuda-graphs --prompt-tokens --cubin-dir. There are **no**
  --comm-timeout/--comm-connect-timeout flags on tp4-chain (the transport hardcodes 180 s frame /
  60 s connect), **no** --bench-steps, and its --prompt-tokens is a **COUNT** used only by the
  world=1 hybrid_sequence path - it is not a prompt-id list.
- The launch wrapper is the Lead's: /home/user/ev-session-20261020/run-fourrank.sh
  <stage-tag> <run-name> loop|raw. KIND=loop produces exactly:

```
device-chain --run-stage tp4-chain --checkpoint CKPT \
  --tp-rank R --tp-world-size 4 --comm-peers PEERS --comm-base-port 39000 \
  --upload --loop --decode-steps 8 --warmup-steps 2 --evidence-dir EV
```

  The exact flags are the Lead-owned wrapper's business: do not hand-build ssh loops and do not
  assume --cubin-dir. device-lead's four-rank argv carries no --cubin-dir, so the world=1
  hybrid-sequence call is not invoked at all and records {executed:false, note:no --cubin-dir: the
  hybrid-sequence runner was not invoked}; with --cubin-dir it records
  {executed:false, error:hybrid_sequence needs --fixture FILE}. Treat executed!==true as
  NOT_ATTEMPTED either way. The wrapper preflights binary/cubin sha256 + MemAvailable + ports on all
  four nodes and collects, into /home/user/ev-session-20261020/window/<run-name>/rank{0..3}/:
  tp4-bind-rR.json, tp4-bind-names-rR.txt, tp4-loop-rR.json, run.log, run.exit, preflight/timeline
  logs.
- **Consequence: the four-rank loop writes no token, no logits dump and no hybrid-timing.json.**
  Do not go looking for hybrid-dump-logits.jsonl in this run; a dump, if anyone ever stages one, is
  explicit-only evidence (never auto-promoted from a directory).
## 2. Wall-time budget (GPU work <= 20 min, corrected)

| phase | budget | notes |
| :--- | ---: | :--- |
| stop vLLM + verify GPU free | 3 min | Lead |
| stage + preflight on nodes | 3 min | wrapper preflights sha/MemAvailable/ports |
| launch 4 ranks, arena boot + bind coverage | 6 min | measured 332-351 s/rank |
| collective walk + evidence write | 2 min | 40 layers x 2 phases x (2 warm-up + 8 decode-steps-plan) |
| kernel harnesses (separate tracks) | ~12 min | kernel-lead brings its own harness rows |
| teardown + process/listener verification + evidence copy | 4 min | 0 processes, 0 listeners |
| host twin (optional, only if >= 6 min remain) | 6 min | separate kind; skip -> twin leg UNMEASURED |
| **stop + restore vLLM + health check** | **~13 min** | not counted in the 20 min GPU work |

If the window runs behind: cut the optional host twin first, then kernel work; **never** cut the
evidence copy, the teardown verification or the restore health check.

## 3. Device command (loop path, tonight)

The Lead invokes the wrapper; do not hand-build the ssh loops:

```
/home/user/ev-session-20261020/run-fourrank.sh b69ebb2 m1-bind bind
# frozen Lead argv: COMM_BASE_PORT=29601 DECODE_STEPS=1 WARMUP_STEPS=2 SKIP_PLAN=1
#   EXTRA_ARGS flag forwarding is the wrapper's business (the evidence contract below is what matters)
# writes per rank EXACTLY: tp4-bind-rR.json (primary, sparkflash-tp4-bind/1),
#   tp4-bind-names-rR.txt (185,481 logical lines), tp4-loop-rR.json (embeds bind + plan +
#   3-step 80-call walk), chain-rR.log (rank stdout+stderr), chain-rR.rc (exit code, expect 0)
# no run.log / run.exit: both collectors accept chain-rR.log/.rc as the equivalents
# decoded walk: warmup 2 + decode 1 = 3 steps x 80 calls = 240 reduces
```

Traps (BRIEF, keep them): ssh -n or nohup setsid ... </dev/null &; never probe a listening rank with
/dev/tcp (protocol violation, loses the run); no cargo on the nodes; launch order is the wrapper's;
a retry is normal. Transport flags: the frozen tag does expose --comm-timeout and
--comm-connect-timeout on tp4-chain (ChainArgs.comm_timeout_secs / comm_connect_timeout_secs,
defaults 180 s / 60 s, wired into TcpConfig by fae3b9a); the frozen EXTRA_ARGS values are forwarded
to the ranks as intended, so do not strip them. An older tree before fae3b9a would reject them,
which is why the preflight verifies the assembled argv.

## 3b. Expected values tonight (frozen b69ebb2, binary sha b25eb1b3...)

Per rank, the clean shape the checks are written for: attention tensors_expected/bound 208/208
(tensors_table_entries 215, coverage.complete true), MoE class bound counts routed_experts_gate_up
61,440 / routed_experts_down 30,720 / shared_experts 240 / router 120 (all complete), tail 3,
census resident_tensors 95,818 with by_class special_token 3 and unclassified [], summary
device_bound_names 92,731, summary.complete true, error null, per_token_weight_h2d_bytes 0, plan
hash 0x9a59035fc40776c8 with 95,818 resident / 4 disk shards / 72.308 GiB resident, names manifest
185,481 logical lines, walk 3 steps x 80 calls = 240 reduces all calls_per_step_ok, forward not-run,
tokens 0, hybrid_sequence.executed false (no --cubin-dir), chain-rR.rc 0. NOT_ATTEMPTED is the
correct step-time result: no forward pass, no token, no measurement.

Accepted alternative shape (only if the 348caff binary is run instead): census.unclassified ==
exactly ["image_end","image_newline","image_start"] and summary.complete false while every device
set is complete. Both tools record that as an info-only census caveat plus the clean bind-set
result; they never call it a bind failure. Any other unclassified name or unexplained
complete=false is a loud failure.

Freeze note: the wrapper command above is the Lead's; this doc's job is the evidence contract, so if
the Lead's assembled argv differs (e.g. EXTRA_ARGS names forwarded to tp4-chain), the preflight is
the authority.

## 4. D48 step collector (this repo): scripts/d48_step_gate.py

The collector reads the per-rank evidence and applies the budget arithmetic with every number
attributed to its file. It writes exactly one kind per document (device or host-twin), never a
summed row, and never fills a missing measurement with a placeholder.

```
python3 scripts/d48_step_gate.py \
  --device-dir /home/user/ev-session-20261020/window/<run-name> \
  --kind device --expected-steps 3 \
  --out-json /home/user/ev-session-20261020/window/<run-name>/d48-step-gate.json \
  --out-md   /home/user/ev-session-20261020/window/<run-name>/d48-step-gate.md
```

Exit codes: 0 STEP-TIME-PASS, 1 STEP-TIME-FAIL / STEP-TIME-REFUSED / **BIND-FAIL**, 3 NOT-ATTEMPTED,
2 bad input. For tonight's loop run the expected (and correct) result is exit 3, status
NOT-ATTEMPTED, with the bind coverage and the 3 x 80-call walk reported per rank from the same
files. A run that emits loop evidence but no tp4-bind-rR.json (and is not plan-only) is BIND-FAIL
exit 1: the bind file is written before the walk by design, so its absence is a failed stage.

Tolerated non-acceptance files: the resident stage document (integer schema version) and the rank-0
tp4-chain-summary.json may share the directory; both are reported as info-only auxiliary-evidence
rows and never become device rows. A JSON with no string schema and no token block is also treated
as auxiliary; a real untrusted document would still trip the runner shape checks.

Schema sparkflash-d48-step-gate/1 (fields the collector actually writes):

```
{
  "schema": "sparkflash-d48-step-gate/1",
  "generated_utc": "...",
  "kind": "device | host-twin (exactly one per document)",
  "provenance": {
    "tree": "...", "binary_sha256": "...", "cubins_manifest_sha256": "...",
    "checkpoint_root": ".../1bae0e10...", "world_size": 4, "expected_steps": 8,
    "evidence_files": [{"path": "...", "sha256": "...", "rank": 0, "schema": "...", "stage": "tp4-chain"}],
    "ignored_files": [{"path": "...jsonl", "reason": "dumps are consumed by accept_device_token.py"}],
    "clock": "host wall clock per rank; no GPU timestamps unless the rank recorded them"
  },
  "per_rank": [{
    "rank": 0, "rc": 0, "schema": "sparkflash-tp4-loop/1", "stage": "tp4-chain",
    "path": "...", "sha256": "...", "boot_s": null, "first_token_ms": null,
    "tokens": [], "token_count": 0, "steps_run": null,
    "step_ms": {"samples": [], "mean": null, "median": null, "min": null, "max": null, "p95": null},
    "layers_run": 40, "reduces_posted": 240, "reduces_expected": 240,
    "weight_bytes_per_token": null, "attention_staged_bytes": null, "moe_staged_bytes": null,
    "accepted_tokens_per_step": null,
    "per_layer_count": 0, "per_layer_components_us_sum": null,
    "attention_bound": 208, "attention_expected": 208, "walk_performed": true,
    "hybrid_sequence": {"executed": false, "note": "no --cubin-dir: the hybrid-sequence runner was not invoked"},
    "bind": {"performed": true, "error": null, "attention": {"expected": 208, "bound": 208, "complete": true},
             "moe": {"performed": true, "classes": [{"class": "...", "bound": 61440, "complete": true}],
                     "complete": true, "weight_h2d_bytes_per_token": 0, "launch_incompatible": ["routed_experts_down"]},
             "tail": {"complete": true}, "census": {"unclassified": []},
             "summary": {"device_bound_names": 92729, "per_token_weight_h2d_bytes": 0, "complete": true}},
    "bind_path": ".../tp4-bind-rR.json", "bind_sha256": "...",
    "bind_names_manifest": {"path": ".../tp4-bind-names-rR.txt", "present": true},
    "notes": ["..."]
  }],
  "aggregate": {
    "ranks": [0,1,2,3], "ranks_with_step_samples": [],
    "step_ms_mean_of_ranks": null, "step_ms_worst_rank": null, "step_ms_best_rank": null,
    "step_ms_median_all_ranks": null, "steps_counted_min": null, "steps_counted_max": null,
    "tokens_identical_all_ranks": null,
    "per_layer_time_sum_us": null, "step_wall_us": null, "unattributed_us": null
  },
  "checks": [{"name": "...", "status": "pass|fail|info", "detail": "..."}],
  "budget": {
    "step_budget_ms": 42.0, "accepted_tokens_per_step": null, "accepted_tokens_source": null,
    "tok_s_target": 90.0, "parity_band_tok_s": [50.0, 60.0],
    "derived": {"ms_per_accepted_token_if_measured": null,
                "accepted_tokens_per_second_if_measured": null,
                "single_token_tokens_per_second": null}
  },
  "verdict": {
    "step_time_budget": "PASS | FAIL | UNMEASURED",
    "accepted_tokens_per_step": "MEASURED (source: ...) | NOT-WIRED (DSpark/MTP verify step not on the device path)",
    "tok_s_gate": "UNMEASURED | MET | UNMET",
    "label": "GATE-GRADE | STEP-TIME-ONLY | STEP-TIME-FAIL | STEP-TIME-REFUSED | HOST-REFERENCE"
  },
  "status": "STEP-TIME-PASS | STEP-TIME-FAIL | NOT-ATTEMPTED | BIND-FAIL | HOST-TWIN-ROW",
  "labels": ["..."],
  "failures": []
}
```

### Pass/fail arithmetic applied by the collector (D48, not redefined)

- **step budget:** step_ms_worst_rank = mean of the counted steps per rank, worst rank decides;
  PASS iff <= 42.0 ms. A single counted step above 42 ms is reported by the "every counted step
  within budget" check. Warm-up steps are excluded by the arm, not by the collector.
- **accepted tokens/s:** only measured when the arm's real DSpark/MTP verify step recorded
  accepted_tokens_per_step (D50 rule; no synthetic placeholders). The CLI flag requires
  --accepted-tokens-source naming the arm field/log line; without a measured value the field is
  null, tok_s_gate is UNMEASURED and the row is labelled STEP-TIME-ONLY.
- **single-token lower bound:** 1000/step_ms at 1 token/step, recorded as a lower bound only; it is
  never presented as the verify-step metric or compared with 90 tok/s.
- **refusing cross-checks:** all ranks rc==0; reduces_posted == 2*layers_run*steps;
  weight_bytes_per_token == 0 (a leaking run's step time is refused); tokens identical across ranks;
  per-layer breakdown present on every rank (device kind). A step-budget PASS with a failed
  cross-check is STEP-TIME-REFUSED, not PASS.
- **partial row:** if fewer counted steps were recorded than planned, the row says so in labels and
  status stays what the measured numbers allow (a 3-step row is honest, not a failure).
- **bind evidence (loud failures, device kind only):** tp4-bind-rR.json present on every rank (or an
  explicit plan-only info row); bind errors empty; attention coverage complete (208/208); every MoE
  class complete; tail coverage complete with non-null pointers; census unclassified empty; per-token
  weight H2D == 0; summary.complete true. The launch-compatibility row for routed_experts_down
  (bound but incompatible) is reported as info, never silently folded into a pass. Missing names
  manifests are reported; the JSON bind object is the evidence of record.

## 5. M1 acceptance runner (this repo): scripts/accept_device_token.py

Run after the window, offline, on the copied evidence. Separate verdicts, never a blended pass.

```
python3 scripts/accept_device_token.py \
  --device /home/user/ev-session-20261020/window/m1-bind \
  --expect-attention-bound 208 \
  --expect-moe-bounds routed_experts_gate_up=61440,routed_experts_down=30720,shared_experts=240,router=120 \
  --expect-manifest-lines 185481 \
  --twin /home/user/ev-session-20261020/window/<twin-run-name>   # optional
  --out-json .../accept-device-token.json --out-md .../accept-device-token.md
```

What it reports tonight (honest expectation):

- shape: PASS when every rank has tp4-bind-rR.json beside tp4-loop-rR.json, attention bound 208,
  the four MoE class bound counts above, the names manifest line count, a clean run.log (no
  error/mismatch/fatal/panic lines) and run.exit 0. A missing or incomplete primary bind document is
  a loud shape/pre_token FAIL, not an absent measurement. No dump present -> info. For a labelled
  negative window (forward refused) rc=2 is accepted only when that rank's tp4-token-rR.json has
  forward=="refused", positions_run==0 and a named error; any other non-zero rc still fails.
- pre_token: PASS when the bind document is complete on every rank (attention/MoE/tail coverage,
  census fully classified, summary.complete, per-token weight H2D 0) and the walk reconciles
  2*layers*steps with calls_per_step_ok true and the canonical bf16 rule; the routed_experts_down
  launch incompatibility is reported as a caution, not silently called OK.
- twin_match: NOT_ATTEMPTED when the device evidence has no token ids (never FAIL for absence).
- oracle_comparison: UNMEASURED without a device dump; UNBARED when a dump exists but no D41 gate
  file was passed (no bars file exists), with the D39 metrics recorded. --oracle-bars takes only a
  real sparkflash-d41-bars/1 gate file; the D40 published provisional values can only be printed as
  D40-PROVISIONAL-NOT-A-GATE diagnostics (--provisional-d40).
- overall: NOT_ATTEMPTED for tonight's loop run; the M1 gate is typed as positions_run=<n> partial
  or not reached, never silently passed.

Rules the runner enforces: a host/other dump is never device evidence; a dump is only consumed when
passed explicitly with --device-dump (+ --dump-source label); token ids are only joined to oracle
positions when the evidence names positions (no invented indices); M1 evidence paths are the
four-rank arm's own rank files from /home/user/ev-session-20261020/window/<run>/rank<R>/.

### Defect classes that must fail loudly (device-lead, 2026-10-20)

These are greppable and are failures, not tolerances: bind JSON missing; bind coverage missing or
unexpected names non-empty; summary.complete false; bind attention bound != 208; MoE class bound
counts != {61440, 30720, 240, 120}; names manifest line count != 185,481; collectives
calls_per_step_ok false or calls != 80/step; any reduce-verification error in run.log; run.exit != 0.
The runner/collector above enforce each of these (the manifest line count and the run.log scan are
runner checks; pass the values with --expect-manifest-lines and the auto-detected run.log). The
Lead's tools/adjudicate-bind-window.py is an independent structural check; the qa collectors remain
the evidence of record.

Counting and log details from device-lead (2026-10-20): the names manifest has no header, no trailing
blank line and no terminating newline, so the logical line count is 185,481 while wc -l prints
185,480; the runner counts logical lines (the 185,481 form). Real window composition (device-lead/Lead measured): attention expected 208 + attention bound 208 + host_seam 19 + 2x61,440 + 2x30,720 + 2x240 + 2x120 + 2x3 = 185,481; the earlier predicted 185,462 omitted the host-seam rows and was corrected from the artefact. The success-path run log (chain-rR.log from the Lead's wrapper,
or run.log from our own runs) contains only progress lines starting with the rank prefix; the runner
fails case-sensitively on a line starting
error/panicked/fatal, a Rust panic header, or the word mismatch, and reports any broader
error/fatal/panic wording as info only (run.exit and the JSON checks decide). hybrid_sequence
executed !== true is NOT_ATTEMPTED, never a failure. The loop JSON plan object carries the plan hash
(0x9a59035fc40776c8 for the frozen stage), resident_tensors 95,818 and resident_bytes; the runner
reports a per-rank plan-identity info row and joins it by rank (the bind file is about binding only).

### Refusal-shaped negatives (m1w adjudication, 2026-10-20)

The m1w window is the labelled negative: the four-rank attention launches ran (wq_a, wkv, wq_b and
the CSA union body wiring smoke), then the forward refused with forward="refused", positions_run=0,
no dump and rc=2 on every rank. Three harness gaps, not device failures, were fixed before the
verdict:

1. rank tp4-forward-embed-rR.json is now a recognised pre-token layout (stage embedding-gathered):
   reported as a "forward-embed evidence" shape row (token_claimed false, prompt_len, decode
   positions, reference id count) and never consumed as a token id source;
2. the run's world identity comes from acceptance evidence only: a plan-only preflight resident
   document (tp4-resident, world 1, e.g. rank0/plan-world1/) is auxiliary and no longer makes
   "world size" fail;
3. rc=2 is accepted only for the exact labelled refusal envelope above (accept runner and D48
   collector), with the refusal reported as info; rc=2 without that envelope, or with a different
   forward label, stays a loud failure.

Nothing else was relaxed: attention 208, the four MoE class counts, the 185,481-line manifest, the
walk reconciliation, the bind-coverage gates and the oracle rules are unchanged. The verdict is
PRE-TOKEN-PASS (exit 3): bind/walk PASS, token NOT-ATTEMPTED, twin NOT-ATTEMPTED, oracle
UNMEASURED (no positions to join; nothing is guessed).

## 6. Host twin (optional; only if >= 6 min remain after device + kernel)

--run-stage tp4-host exists with: --bench-steps N, --prompt-tokens 3,7,11 (id list here),
--fixture DIR|--checkpoint DIR, --resident-cache, --profile, --evidence-dir. It is a **host**
measurement: collect it with the same collector under its own kind and paths.

```
python3 scripts/d48_step_gate.py \
  --device-dir /home/user/ev-session-20261020/window/<twin-run-name> \
  --kind host-twin --out-json .../d48-step-gate-twin.json --out-md .../d48-step-gate-twin.md
```

The twin document has kind host-twin, status HOST-TWIN-ROW, step budget NOT-APPLICABLE (42 ms is a
device target), and its own table. It is never mixed with the device table and never summed into a
win. The twin is also the only twin-match input for the runner, and only when it ran the same prompt
and step count as the device arm. The twin evidence tree, the two producers (node host twin vs the
dev-host host-reference fixture via scripts/twin_evidence.py) and the never-self-twin rule are
pinned in docs/build/d38-vllm-ab-protocol.md, section “The twin leg”.

## 7. Per-layer breakdown requirement (for the first real step row, next window)

No profiler is installed on a node during a window. The breakdown is the arm's own measured timers
and counters, per rank and per layer, for every counted step: kernel launches per named kernel
(counts; unknown = null, not 0), host staging us + H2D calls/bytes issued, device/sync-gated us +
readbacks, reduce seam us (posted/wait split when async), layer total, and a reconciliation
sum(measured layer components) - step wall = residual reported as unattributed_us. The residual is
never divided across layers and never attributed to a kernel without its own measurement. Optional
and separate: one nsys run as corroboration only; the gate row must not depend on it.

## 8. Next window (to make the step row gate-grade and reach the first token)

1. The step mode did not land in the 2026-10-20 window: the routed down-projection cannot launch
   from the arena yet (stored-layout combine cubin pending), so M1 token is NOT_ATTEMPTED tonight.
   Next window: device-lead supplies the step-mode argv (real forward step, per-step ms, per-layer
   timers); this protocol does not pre-specify flags.
2. Same wrapper/evidence paths; the collector then reports STEP-TIME-PASS/FAIL against 42 ms with
   per-layer reconciliation; accepted tokens/step stays null until the DSpark/MTP verify step is
   wired (then D50 acceptance rule, tok_s = accepted/step * 1000/step_ms vs 90 target).
3. First token + 10 counted steps: real prompt state (if the full 130-token prefill is not on the
   arm, report a 1-position partial row with positions_run=1, never a fabricated 9-position row).
4. Host twin with the same prompt/step count for the twin-match leg.
5. Only then can M1 be graded against the pinned b69 fixture at p129..p137 (BARED requires a D41
   gate file; UNBARED otherwise) and a D48 gate row be claimed.

## 9. Restore and teardown (Lead owns)

Stop evidence processes; verify 0 processes and 0 listeners on the arm ports; copy evidence + sha256
back to the dev host; restart the recorded vLLM recipe; wait for health; verify GET /v1/models is
200 and the served root equals the root recorded before the stop. No window claim is reported
before the restore is verified.

## 9b. Result of the 2026-10-20 m1-bind window (measured, recorded)

The frozen b69ebb2 four-rank run started 2026-10-20T21:14:38Z (17:14:38 EDT), all four ranks
rc=0, launch-to-last-exit 469 s; evidence copied to
/home/user/ev-session-20261020/window/m1-bind/rank{0..3}/ (five files per rank plus argv.txt,
run.meta and the Lead's bind-window-adjudication.json). Collectors run offline after the copy:

- scripts/d48_step_gate.py --device-dir <evidence> --kind device --expected-steps 3
  -> exit 3, status NOT-ATTEMPTED, zero failures. Every bind/walk check pass on all four ranks:
  primary tp4-bind-rR.json present, bind errors none, attention coverage complete, MoE four classes
  complete, tail complete, census classified (unclassified [] with special_token 3), per-token
  weight H2D 0, summary.complete true, names manifest 185,481 logical lines, chain-rR.rc 0, bind
  coverage bound==expected, reduces 240/240 reconciled. Info only: routed_experts_down launch
  incompatible (A8 pre-padded 640-K row layout vs checkpoint stored layout - the known M1 blocker),
  plan hash 0x9a59035fc40776c8 / 95,818 resident shards / 77,640,198,984 resident bytes, walk
  3 x 80 calls with payload 819,200 B/step and steady wall_ms 20 (warmup 41, 22), hybrid_sequence
  executed false.
- scripts/accept_device_token.py --device <evidence> --expect-attention-bound 208
  --expect-moe-bounds ... --expect-manifest-lines 185481
  -> exit 3, shape PASS (34 checks), pre_token PASS (44 checks: attention 208/208 per rank, MoE
  61,440/30,720/240/120, per-token H2D 0, plan identity, 240 = 40 layers x 3 steps x 2 reduces,
  canonical bf16 rule, chain-rR.log clean, chain-rR.rc 0), twin match NOT_ATTEMPTED, oracle
  UNMEASURED (no oracle/dump supplied; no token-equality claim), step time UNMEASURED, overall
  NOT_ATTEMPTED. The Lead's bind-window-adjudication.json is a derived document and is reported as
  info-only auxiliary evidence, never as an input.
- Artefacts and shas: /home/user/ev-session-20261020/qa/window/m1-bind/ (d48-step-gate.json
  8e5aa305..., d48-step-gate.md 6fbe1383..., accept-device-token.json 287d0e4f...,
  accept-device-token.md 4ff9f2e2..., evidence-sha256.txt over the 23 evidence files,
  artefacts-sha256.txt, dry-run-synthetic/ with the pre-window dry runs).

Honest reading: the bind and the 3-step/80-call four-rank walk are measured and clean; no forward
pass ran and no token was emitted, so the M1 token gate and the D48 step-time gate are
NOT_ATTEMPTED, not passed and not failed. Nothing here is a tok/s claim.

## 10. Artefacts to hand to the Lead by 19:45 EDT

- d48-step-gate.json + .md (loop run: NOT-ATTEMPTED + per-rank bind/walk facts; BIND-FAIL if
  the primary bind document is missing or incomplete; or step row if a step lands)
- accept-device-token.json + .md (shape / pre_token / twin / oracle verdicts, BARED or UNBARED;
  expected tonight: shape PASS, pre_token PASS, twin NOT_ATTEMPTED, oracle UNMEASURED, overall
  NOT_ATTEMPTED)
- the wrapper's preflight/timeline logs, per-rank run.log + run.exit, and a sha256 manifest over
  tp4-bind-rR.json, tp4-bind-names-rR.txt and tp4-loop-rR.json
- a short note stating exactly which rows are device, which are host, and which are unmeasured

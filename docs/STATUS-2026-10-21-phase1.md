# STATUS 2026-10-21 - Phase 1 (WS-A) measured on device, reviewed adversarially, and merged

Branch `task/phase1-async-device`. Every number below is labelled **MEASURED** (a run that happened)
or **ESTIMATE** (arithmetic on a measurement). Nothing here is a projection presented as a result.

## 0 Headline

The four-rank lean decode step went from **1201.81 ms to 842.69 ms MEASURED (-29.9 %, -359.12 ms)**,
with byte-identical tokens `[491,92928,201,15,4893,16646,15,14265,86]` and byte-identical dumped
logits sha `aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2` on all four ranks.
That is **1.187 tok/s** from 0.832 tok/s. The Phase-1 exit gate (lean step <= 400 ms) is **NOT met**:
842.69 ms is 2.11x the gate.

The evidence-ON identity row the first window was missing has now been run: at 925.19 ms it
reproduces the pre-M2 layer chain **digest for digest** (40 layers, identical; the only differing
field is the volatile `wall_ms`), with `evidence_digest_calls = 271,194` against **0** in the lean
row. The async launch lever is measured at **+1.9 ms/step, i.e. no benefit on this path**; the gain
is the evidence gate (C10) plus the cubin/module cache and buffer pool (C1/C2). An adversarial
review found two real soundness defects in the new driver primitives (a pointer-key cache hit that
did not compare bytes; a pool without ownership state that could double-free / double-hand-out), and
both are fixed on the branch with tests that fail against the old code.

## 1 What landed (branch, 7 commits)

| workstream | state | evidence |
| :-- | :-- | :-- |
| WS-A host path (C1/C2/C3/C5/C10) | **MEASURED on device twice, byte-identical**; two soundness fixes from review | sections 2-3, `docs/build/ws-a-host-path-20261021.md` |
| WS-B NCCL/RDMA | code + tickets + bench; fabric unmeasured | `docs/build/ws-b-infra-tickets-20261021.md`, `docs/build/ws-b-collectives-20261021.md` |
| WS-F OpenAI surface | implemented + tested | `docs/build/ws-f-api-20261021.md` |
| WS-F scheduler / WS-E MTP policy | implemented + tested host-side; default **not** flipped | `docs/build/ws-f-scheduler-20261021.md`, `docs/build/ws-e-mtp-device-20261021.md` |
| WS-G live baseline + CI gate + parity plan | measured live rows (under load); CI gate green with a quiet-node SKIP | `docs/build/d38-vllm-baseline-20261021.json`, `docs/build/d38-vllm-recapture-20261021.md`, `docs/build/ws-g-vllm-parity-plan-20261021.md` |
| review follow-ups | B1/B2 code fixes, B3 identity row + evidence_mode, B4 attribution script, nits 1-9 | section 8 |

## 2 The measured windows

Two Lead-owned windows, each with new vLLM stopped first and restored by the same job:

* **m2p1** (staged from `c8bb4883`, binary sha256 `4358146c...`, 20/20 cubin pins): rows lean, sync, probe.
* **m2p1b** (staged from `f5191109`, binary sha256 `aec7f58b163b64c9393ec99a31ffc6d46c8d0f89ca85a83bb866080466eb5854` on all four nodes, provenance `f5191109`, 20/20 cubin pins match the tree baseline): rows evidence, lean.

Frozen argv (both windows): 130 token ids, 9 reference ids, `--warmup-steps 2 --decode-steps 9
--forward --forward-lean`, staged `--kernel-cubin-dir`, released checkpoint `1bae0e10...`, comm base
port 39000. Each row is four ranks, one process per GB10 node; all rc=0. The m2p1b `argv.txt` files
hold exactly four lines each (one per rank, `--decode-steps 9`); the m2p1 `argv.txt` files still hold
stale lines from the aborted preflight attempts (see section 8, N1).

| row | run | env | step mean MEASURED (4 ranks) | tokens | logits bin sha |
| :-- | :-- | :-- | --: | :-- | :-- |
| baseline | m1w37-lean | old path, evidence ON | 1201.81 ms | identical | `aed6054d...` |
| B | m2p1-lean | async, evidence OFF | 845.86 ms | identical | `aed6054d...` |
| D | m2p1-sync | `SPARKFLASH_SYNC_LAUNCHES=1`, evidence OFF | 843.94 ms | identical | `aed6054d...` |
| C | m2p1-probe | probe armed (drained path), evidence OFF | 852.06 ms | identical | `aed6054d...` |
| A | m2p1b-evidence | async, **evidence ON** | **925.19 ms** | identical | `aed6054d...` |
| **B2** | **m2p1b-lean** | **async, evidence OFF (clean argv)** | **842.69 ms** (842.78/842.56/842.78/842.67) | identical | `aed6054d...` |

Steps are the nine per-step samples. Row A is 925.19 ms versus the 1201.81 ms baseline: even with the
chain recording, C1/C2/C10 remove 276.6 ms. Row B2 reproduces row B within run noise (842.69 vs
845.86, -0.37 %), so the fix commit did not change the gate row.

### 2.1 What the 359 ms is (MEASURED, probe row)

Regenerate with `python3 scripts/tp4-probe-attribution.py <new run> <baseline run> --markdown`
(committed, tested by `scripts/tests/test_tp4_probe_attribution.py`). Three totals exist in the
artifact and they are not interchangeable: **seam-layers** (the per-layer `total_ms` mean over
ordinary layers; enigma layers 1 and 14 excluded - the attribution basis below), **all-layers** (same
mean over every layer) and **probe-scope** (`run.block_summary[layer_total].wall_ms / count`, which
includes the probe's per-block bracketing overhead). Rank 0:

| total | baseline | WS-A | delta |
| :-- | --: | --: | --: |
| seam-layers | 25.722 | 19.295 | -6.427 |
| all-layers | 27.341 | 20.782 | -6.559 |
| probe-scope | 27.366 | 20.803 | -6.563 |

Per-block, seam-layers basis (ms/layer, new <- baseline): attention_projections 5.172 (5.691, -0.519),
moe_gate_up 4.767 (5.095, -0.327), ffn_shared 2.744 (3.026, -0.282), attention_output 2.393 (3.242,
-0.849), moe_reduce 0.969 (1.070), query_path 0.868 (1.419, -0.551), ffn_seams 0.516 (0.606),
ref_seam 0.420 (0.409), mhc_entry 0.322 (0.551, -0.228), moe_router 0.261 (0.304), moe_stored_combine
0.220 (0.219), attention_body 0.188 (0.600, -0.412), stage_attention 0.165 (0.164), attention_residual
0.069 (0.376, -0.307), kv_path 0.056 (0.067), ffn_hc_post 0.054, moe_weighted 0.012, ffn_combine 0.006,
entry_json 0.000 (0.201, -0.201).

Run counters (`run.loop_total`, mapped by `legend.block_fields`; these are NOT in `run.counters`):

| counter | baseline | WS-A |
| :-- | --: | --: |
| sha256 calls / bytes / ms | 315,378 / 9,516,944,072 / 16,817.87 | **30 / 334,420,672 / 584.49** |
| alloc_count / free_count | 187,680 / 187,680 | **28 / 0** |
| module_load_count | 49,680 | **7** |
| launches / syncs (probe) | 66,258 / 66,258 | 66,258 / 66,258 |
| d2h_count / bytes | 88,374 / 1,094,818,020 | 88,374 / 1,094,818,020 |
| h2d_count / bytes | 126,978 / 2,011,396,188 | 126,370 / 1,988,245,596 |
| loop wall | 152,936.5 ms | 116,321.7 ms |

Device-to-host observations are **12 per layer-position**, unchanged (5160 layer-positions, 7 blocks:
moe_router 3, attention_output 2, attention_projections 2, query_path 2, attention_body 1, kv_path 1,
moe_stored_combine 1). The 14 grep hits on the layer path include two conditional observations that do
not fire in these rows.

### 2.2 The identity row (MEASURED)

`m2p1b-evidence` versus the pre-M2 `m1w37-lean` embed JSON, rank 0 (all four ranks were checked the
same way):

* `layer_chain`: 40 entries in both; **identical after stripping `wall_ms`** - i.e. every layer digest
  and every data-dependent field is byte-identical. `wall_ms` is the only differing key per entry.
* Top level: only four non-timing differences and all three are expected: `evidence_mode` and
  `evidence_digest_calls` are the new in-band fields (recording / 271,194), and `combine_preflight.cubin`
  plus `head.input.file` are staging-path strings that differ because the window tag differs.
* `head` and `position_records`: identical except timing and those paths.
* `tp4-forward-embed-r0.json` for the lean row carries `"evidence_mode": "off"` and
  `"evidence_digest_calls": 0` (and an empty `layer_chain`), so an off run can no longer be mistaken
  for a chain-present run.

## 3 Attribution, stated carefully

* The evidence gate (C10) is the dominant measured lever: the sha counters fall from 16.8 s to 0.58 s
  over the probe run, and the seam-heavy host blocks (attention_output -0.85, query_path -0.55,
  attention_projections -0.52, attention_body -0.41, attention_residual -0.31, mhc_entry -0.23,
  entry_json -0.20 ms/layer) account for most of the -6.43 ms/layer.
* The module cache and buffer pool (C1/C2) are visible in the counters (module loads 49,680 -> 7;
  alloc/free 187,680/187,680 -> 28/0). They remove 34 driver calls per layer-position; their wall-time
  share is smaller than C10's.
* **C3 (async launch) is worth nothing on this path**: async 845.86 vs sync 843.94 ms on the m2p1 pair
  (a +1.92 ms difference in the wrong direction, inside run noise), and the 12 D2H observation points
  per layer-position are unchanged. Each observation still waits for the queued kernels, so the host
  cannot run ahead for long. The async switch is correct (byte identity above) and stays available,
  but it is **not** a lever toward the gate.
* **C5 is unmeasured in isolation**: the H2D per layer-position fell only 23.00 -> 22.89 (-0.5 %) in
  the probe row (attention_body 7.00 -> 6.89), so the zero-store skip fires on few layer-positions in a
  prefill-dominated probe and no row isolates its decode-step effect. It is not claimed as a win.

## 4 Where the remaining 843 ms is, and what 90 tok/s would actually need

MEASURED split (19.295 ms/layer seam basis): attention_projections 5.17 (26.8 %), moe_gate_up 4.77
(24.7 %), ffn_shared 2.74 (14.2 %), attention_output 2.39 (12.4 %), moe_reduce 0.97, query_path 0.87,
everything else < 0.6. The kernels are ~7.5 ms/layer of that, so the host is still the bottleneck,
and the 12 D2H observation points per layer-position are the structural reason.

The 90 tok/s target with the measured MTP acceptance length (1.571 accepted/verify step, host-twin
measurement) needs a **17.46 ms** verify step (`1000 * 1.571 / 90`). Today's step is 842.69 ms: the
required reduction is **48.3x**, and MTP on today's step would give 1.86 tok/s, not 90. Honest
conclusion: 90 tok/s is not reachable by tuning this host path; it needs the decode step to become a
single (or few) CUDA-graph launches with **no per-layer host arithmetic and no D2H observation
points**, i.e. the device-resident step that WS-A only started. The Phase-1 gate (<= 400 ms) needs
4.3x off today's 19.295 ms/layer (6.8 ms/layer budget).

## 5 Gates (MEASURED on this branch, run `python3 ci/host-gates.py`)

| gate | result |
| :-- | :-- |
| fmt, check, clippy, rustdoc | ok |
| m4-abi-freeze | ok |
| m4-abi-split | **ok after the allow-list refresh in this branch** (was red on main; see 8/m4) |
| test | 2297 passed, 0 failed (108 binaries) |
| trace-no-default | 30 passed |
| test-examples | 397 passed |
| features-self-test, features-sweep | 2141 passed, 0 failed (81 binaries) |

Also green: `python3 -m unittest discover -s scripts/tests` (411 tests, including the new
`test_tp4_probe_attribution.py`), `bash -n` on the shell tools, `.gitlab-ci.yml` parses.

## 6 Measured vs not measured

**MEASURED**: the step means in section 2; byte-identical tokens and logits sha (6 rows x 4 ranks); the
evidence-chain identity in 2.2; the probe block table and run counters; the live vLLM deployment
baseline (under load); the host gates above; the python suites.

**NOT MEASURED / design-only**: WS-B fabric all-reduce (the 25 us gate is not-yet-measured; the
36.0-37.1 us 10 KiB row is the "before" row only); NCCL data path on device; the device seam of
`TpCollective`; CUDA graphs and fusion (WS-C); device batched prefill and continuous batching
(host-side bit-identical tests only); MTP draft/verify execution on device (policy harness only); the
endpoint TTFT effect of any prefill change; released-root vLLM parity. C5's isolated effect is
unmeasured (section 3).

## 7 Deployment state around the windows

Both windows stopped the live TP4 deployment (`docker stop vllm_node` on all four nodes, MemAvailable
back to 118 GiB, port 8000 closed) and restored it with the documented recipe
(`run-recipe.py deepseek-v4.1-flash-1017-heretic` on 192.0.2.80). Restore health was verified by
the job each time: `/v1/models` 200 and served root `.../12d58ad0878b5008e80d720251adb969a0069412`
(the heretic snapshot, exactly what was being served before). The deployment was under load during
its baseline capture (long-lived sockets from this host's harness process; every baseline row is
labelled UNDER-LOAD, and the canonical prompt is 130 random token ids, so its low prompt logprobs are
EXPECTED, not degeneracy). **Root verdict unchanged: the served root is not the released
`1bae0e10...`, so `gate_eligible=false` and the D38 precondition is NOT met; no D41/D38 join is
claimed.** Released-root parity needs a window that serves the released snapshot in vLLM, which is a
separate, explicit deployment decision.

## 8 Adversarial review outcome

The review asked for four blocking fixes; all four are addressed on the branch.

* **B1 (module-cache pointer hit was unsound).** `ModuleCache::intern` returned the cached id on a
  `(pointer, length)` hit without comparing bytes, so a freed cubin buffer reused by another
  same-length image could resolve to the old module. Fixed: the pointer hit is byte-verified (one
  memcmp, no allocation) and a reused address with different bytes interns a new id. New deterministic
  test `a_pointer_key_reused_by_different_bytes_is_a_different_image` pins the dangerous direction
  (it fails against the old pointer-only path).
* **B2 (pool bookkeeping could double-free / double-hand-out).** `BufferPool` had no ownership state
  and `Cuda::drop` freed every recorded pointer. Fixed: the pool tracks checked-in state, `release`
  refuses unknown/double releases, `free`/`free_buffers` forget pooled pointers instead of freeing
  them (reclaimed at primary-context release), and `Drop` frees only the authoritative free lists. The
  `issued`/`completed` fence is unchanged. New tests: double release refused; pooled pointer freed
  through `free` is forgotten and Drop frees nothing.
* **B3 (identity row never run; default emits a chain-free evidence file).** Both parts fixed: the row
  exists (2.2) and the artifact now carries `evidence_mode` + `evidence_digest_calls`.
* **B4 (attribution table not reproducible).** Fixed: the table is emitted by a committed, tested
  script that covers all three totals and states the aggregation; section 2.1 uses it. The probe
  counter rows are re-derived from `run.loop_total` and labelled as such.

Nits: **N1** clean `argv.txt` in the m2p1b rows (the m2p1 stale lines are noted in 2); **N2** the async
switch is process-global (an `AtomicBool`), unlike the thread-local evidence switch - documented in
the API docs; **N3** the per-layer engram record is now cleared even when evidence is off; **N4** the
evidence mode is now explicit in the artifact; **N5** CORS defaults to any-origin and auth is opt-in
(matches vLLM), recorded as an operator note here; **N6** the CI perf gate maps a quiet-node SKIP to
exit 0 and its committed reference band is an UNDER-LOAD row, so it cannot catch a regression yet -
recorded as follow-up work; **N7** the verify-step formula is now `1000 * 1.571 / 90`; **N9** the
test-count claims cite the actual suites (411 python tests; 2297 workspace tests).

**m4-abi-split**: red on main because main's kernel batch rewrote the m4_abi assert counts/messages and
the verifier's allow-list was never updated. The branch refreshes the allow-list with the exact
post-split edits (b12x_mxfp4_moe 10 -> 12, checked 78 -> 98 -> 100, the assert message, and the two
MoE test mutation targets) and touches no kernel file or pin; the verifier now reports all three
files byte-identical to the regeneration, family 13926 lines, pin matches, 64 tests.

## 9 Next steps, in payoff order

1. **Kill the D2H observation points** (12/layer-position, unchanged; section 2.1). This is the only
   lever with a measured bound: the async path cannot help until the host stops reading device values
   mid-layer. Move router/gate-up/slot accumulation and the projection reference arithmetic to device
   kernels and keep one D2H at the step boundary; then re-run this exact window. The seam blocks that
   remain (attention_output 2.39, attention_projections 5.17, moe_gate_up 4.77) are where the budget
   goes; no number is claimed until the row exists.
2. **CUDA graphs / fusion (WS-C)** after 1, to make the per-step host work < 1 ms.
3. **Device-side MoE reduce / WS-B**: `moe_reduce` is 0.97 ms/layer on the host today; the fabric path
   is unmeasured; the 25 us gate still needs a free-window run of `scripts/ws-b-allreduce-bench.sh all`.
4. **WS-E MTP on device** once the verify step is small: at AL 1.571 the target needs 17.46 ms/step.
5. **WS-F prefill/scheduler**: build the device batched-prefill executor that honours explicit ranges,
   wire `ContinuousBatcher` into the TP4 serve loop, then flip `SHIPPED_PREFILL_MODE` only after an
   endpoint row beats the token-loop 39.13 s (the bench row 43.62 s is why the default is still
   TokenLoop).
6. **WS-G**: serve the released snapshot in a dedicated window for parity (precondition 0); also make
   the perf gate require a QUIET reference band (N6).

## 10 Commits on `task/phase1-async-device`

```text
f5191109 WS-A: fix the two driver soundness defects and the evidence/attribution gaps found in review
e79bc8fb Phase-1 status: WS-A measured on device (1201.81 -> 845.86 ms, byte-identical), gate not met
3e489f5e WS-G: live vLLM deployment baseline, CI perf-regression gate, parity plan
12c3bc1b WS-E/WS-F: testable batching, continuous batching and MTP policy; default not flipped
bd84f245 WS-F: close OpenAI surface gaps (CORS, tool policy, streaming usage, tokenize/detokenize)
d837bb76 WS-B: NCCL dispatch policy, dlopen FFI, 4-rank collective microbench and infra tickets
c8bb4883 WS-A: async launch path, per-run cubin/module cache, pooled buffers, evidence-gated digests
```

Reproduce: stage the branch with `stage-window.sh <worktree> <tag>`, stop vLLM with
`window-control.sh stop`, run `run-fourrank.sh <tag> <row> token` with the frozen argv and
`SPARKFLASH_STAGE_TIMING=0` (`SPARKFLASH_FORWARD_EVIDENCE=1` for row A,
`SPARKFLASH_SYNC_LAUNCHES=1` for the sync control), then `window-control.sh start` + `health`.
Evidence: `/home/user/ev-session-20261020/window/{m2p1-lean,m2p1-sync,m2p1-probe,m2p1b-evidence,m2p1b-lean}`.

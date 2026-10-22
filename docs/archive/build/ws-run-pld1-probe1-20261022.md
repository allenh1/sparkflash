# Device probe row, window pld1 (2026-10-22): the device arm is selected, its open refuses at layer 0 for a missing weight class

Status: **frontier probe, not a measurement.** No ms/step, no numerical-gate arm, no
published number. The published host-arm number (179.81 ms/step) is untouched: no device stage
ran, and the two host-control rows that did run are in a window whose GPUs were idling at 208 MHz
of 3003, so their step_ms are recorded as evidence that the arms execute and never as ms/step.

This record was corrected after an adversarial read of every claim against the kept evidence.
Where a statement is an inference from code order rather than a logged observation, it now says
so; where the earlier draft claimed more than the evidence, it is weakened. The corrections
that matter: the arm did not "open" (the open was entered and refused), the four refusal lines
are identical apart from the rank id (not byte for byte), the 19 host-seam names are enumerated
from the census rather than a brace pattern, "no collective traffic" was wrong (the setup C3
walk ran and verified - no *device* collective ran), and the byte/MiB units are fixed.

## What was staged

Stage directory `/home/user/ev-session-window/pld1/stage`, copied to all four nodes
(`192.0.2.80/.88/.59/.68`) and sha256-verified per node before launch (identical binary sha,
27 cubins, manifest sha on all four):

| artifact | identity |
| --- | --- |
| binary `sparkflash-pld1` | `b957731e592cb8e6a2ca171099d197b5c9f43288246982f8ea8ed1674b9b2943` (aarch64 ELF, release, `--features=cuda`) |
| cubins | **27** pinned kernels; manifest (sorted `sha256sum *.cubin` list) `0ec18e1293ed08fd7139e124e3227da1b1b00b7d5f9a6ef7af5a01815cd43cbb`. Every one matched `baseline/kernel_baseline.txt` byte-for-byte and by sha256 after regeneration with the toolkit the baseline was recorded under (`nvcc V13.0.88`). This is the same 27-pin identity evidence the branch gate checks. |
| `lib/libnccl.so.2` | `aeda7a87ab4c3707216d428c6f4dee0f178ed87a6ed76f5bd6be9639e737a51a`, 39,507,352 B, aarch64, SONAME `libnccl.so.2`; reported as NCCL 2.31.2 by the bench`s `--check-nccl` on a node the same evening |

The binary is the batch tip of `port/layer-device-batch` (MR !168) at the time of staging
(code tip `83a6606b`; the run-record commit `037e6034` came after).

**Where libnccl came from.** The nodes have no libnccl discoverable by the loader (they ran it
from inside the vLLM container), and the earlier draft called `scripts/fetch-nccl.sh` defective
because its pin did not match the downloaded wheel. That diagnosis was wrong and is corrected
here: the pinned sha `c77f5ced...` is the **extracted `libnccl.so.2`** (346,523,448 B), while the
**wheel** is 305,107,282 B at `a5bee92b...` - the script hashed the wheel against the extracted
library`s pin, so a good download failed. comm-lead fixed the script to pin both and re-verified
a real fetch (`task/bench-scratch-nccl-pin`, commit `4c6f283b`). For this row the library was
extracted from `vllm-node-ds4:latest` at `/usr/lib/aarch64-linux-gnu/libnccl.so.2` - the NCCL that
actually ran beside vLLM on these nodes - and staged with its sha recorded, because a dlopen-ed
collective library is part of the run identity.

## Run argv (frozen shape, shortened step budget)

```
--tp-rank <r> --tp-world-size 4 --comm-peers 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68
--comm-base-port 39000 --comm-timeout 120 --comm-connect-timeout 120 --upload --loop
--warmup-steps 1 --decode-steps 1 --forward --forward-preflight
--kernel-cubin-dir /home/user/ev-session-window/pld1/stage/cubins
--token-ids 52479,56659,101226,82590,3235,110894,99524,9807,75
```

Env passed to all four ranks by the harness: `LD_LIBRARY_PATH=<stage>/lib`,
`SPARKFLASH_LAYER_DEVICE=1`, `SPARKFLASH_DEVICE_TRACE=1`, `SPARKFLASH_STAGE_TIMING=1`;
`SPARKFLASH_FORWARD_EVIDENCE` unset. The pre-launch capture showed `SPARKFLASH_LAYER_DEVICE` and
`SPARKFLASH_FORWARD_EVIDENCE` unset on all four nodes, and the harness passes the same extra env
to every rank; **identical per-rank launch env is inferred from that plus the four-rank outcome,
not captured per rank**. Preflight (read-only, per node): 116 GiB MemAvailable, 0 GPU compute
apps, port 8000 free, comm ports 39000-39003 free, checkpoint present, no stale process.

## Attempt 1: the pre-open status refusal, exactly as designed

With no libnccl reachable, each rank exited 2. Excerpt (rank id abbreviated; the omitted prefix
is `tp4-chain: forward reduce bootstrap r<r>: NCCL bootstrap failed before any communicator was
opened:`):

```
rank(s) [0, 1, 2, 3] could not load the NCCL library before the bootstrap gather; refusing on every
rank before any communicator is opened (a peer that entered NCCL init here would wait for a rank
that never joins); aborting this rank before any layer - a bootstrap failure must not reach a layer
```

What this proves: the load outcome rode the pre-gather status word and was **assembled on all four
ranks** - a local check can only know its own status, so each rank printing `[0, 1, 2, 3]` means
the gather executed and the agreed refusal was produced everywhere. It is evidence for the
**pre-open half** of the bootstrap protocol only: no id handoff, no communicator open, no warm-up,
no setup barrier, and **not** the mixed case that M1(a) is actually about (a rank with a good
library refusing on a peer`s word) - all four lacked it, so the mixed path was not exercised. That
path exists in `device_reduce.rs:802-813` and is guarded only by a short-frame unit test; a
one-rank-missing attempt or a flag-word table test is the cheap way to close it.

`no rank reached a layer` is provable (the refusal precedes `run_with_arena`); `no hang` is
provable from exit 2 rather than a signal; `no outer timeout` is **not** in the kept record - exit
2 is the only evidence, so the earlier claim is dropped.

Evidence: `/home/user/ev-session-20261020/window/pld1-probe1`.

## Attempt 2: the device arm is selected and its open refuses at layer 0

With libnccl staged, the harness printed `joined after 312 s` (its own figure; the upload
dominates). Each rank exited 2; the four lines are **identical apart from the rank identifier**:

```
tp4-chain: forward refused on rank <r>: layer device seams: layer 0: layer device weight:
no hc_attn_fn for layer 0 in the resolved set
```

What ran, in code order - an **inference from position, not a logged NCCL observation** (the kept
logs contain no NCCL line; the `path=tcp (reference exchange)` lines are the host reduce path):
the refusal is emitted from `DeviceLayer::open` inside `run_with_arena`; the chain reaches
`run_with_arena` only after `open_run_reduce` returned Ok; every bootstrap failure (post-gather
library/init/arm refusal, warm-up, setup sync, barrier) returns earlier with its own distinct
message; and all four ranks show the downstream refusal, so the setup barrier completed on all
four. The arm was selected from the request (`SPARKFLASH_LAYER_DEVICE=1`; only that arm reaches
`DeviceLayer::open`), the open was **entered** and refused while resolving the mHC weight class,
and no kernel of layer 0 ran. One bootstrap-success log line (rank, world, nccl version, barrier
ok) would turn that inference into an observation; it is the first evidence improvement to ask for.

Observed in the same run, all four ranks: plan hash `0x9a59035fc40776c8`, 72.308 GiB arena,
**95,818** resident tensors, attention bind **208/208**, `host_seams_excluded` **19**,
`device_bound_names` 92,731, and `grep -c hc_ tp4-bind-names-r0.txt` = **0**.

There is **no** `tp4-layer-device-r*.json` and there are **no** stage counters in this record: the
refusal is at the open, so no device row and no stage counter exist. The next stage that would be
reached is `mhc_entry`.

Evidence: `/home/user/ev-session-20261020/window/pld1-probe2` (`chain-r*.log`, `tp4-bind-r*.json`,
`tp4-bind-names-r*.txt`, `tp4-loop-r*.json`, `tp4-token-r*.json`).

## The finding (F1): the ten per-layer seam tensors are in no bind class

`LayerWeights::resolve` scans the bind for names ending in the `WEIGHT_NAMES` suffixes. The census
(`tp4-bind-r0.json`) says: attention class **208 expected, 208 bound**, `coverage.complete = true`,
`host_seam_names = 19`; the census class counts are resident_tensors **95,818** with
`attention_projection 208`, `attention_host_seam 19`, `attention_aux_host_read 416`, `mhc 240`,
`routed_expert 92,160`, `shared_expert 240`, `router 120`, `tail 3`, `special_token 3`, `engram 8`,
`mtp 2,401`.

The 19 host-seam names, cited from `attention.coverage.host_seam_names` (the brace pattern an
earlier draft used implied 24 and was wrong): compressor `wgate`+`wkv` for layers 2, 8 and 14, and
**`wkv` only** for layer 20 (7 names); indexer `weights_proj`+`wk` for layers 2, 8, 14 and 20, and
**`weights_proj` only** for layers 24, 28, 32 and 36 (12 names). They are the producers W2 already
declares zero-filled or absent.

There is no `hc_` name and no per-layer `attn_norm`/`ffn_norm`/`q_norm`/`kv_norm` in the bind at
all (two `norm.weight` lines, both the final `tail` norm). The device arm`s first body needs them
resident: the six `hc_*`, the two 5120-wide block norms, and the two norms the attention body and
KV path read. Release geometry is **ten stored tensors per layer, 3,956,440 B/layer, 400 tensors,
158,257,600 B (150.92 MiB) over 40 layers** - 0.2% of the 72 GiB arena.

The plan **already enumerates and sizes all ten** (`mhc 240`, `attention_aux_host_read 416`); the
streaming upload already reads every resident shard. So the fix is **bind-only**: a second pointer
family plus its coverage counters, no plan row, no byte, no read, no copy, no allocation. Recorded
as F1 and assigned: device-lead owns the class (`task/mhc-seam-class`, cut from `037e6034`),
gemv-f32-impl owns the single `LayerWeights::resolve` merge, and both halves gate together.

## What this window proves, and what it does not

Proven on hardware, four ranks:
1. A missing collective library fails as an **agreed, named refusal on every rank before any
   communicator is opened** - assembled from all four via the pre-gather status word - rather
   than as a hang; each rank exited 2.
2. The 27-pin kernel identity holds on the staged artifacts (each cubin sha-equal to the baseline).
3. The device arm is selected by the opt-in switch and the evidence path does not force it;
   `DeviceLayer::open` was entered and refused at layer 0 for the missing resident class,
   **identically on all four ranks apart from the rank id** (identity across ranks is measured;
   determinism across runs is a code claim, not measured because there was one run per arm).
4. The setup`s four-rank TCP reference-exchange walk ran and verified: `tp4-loop-r*.json` records
   `collectives.performed = true`, `calls_per_step_ok = true`, 80 calls per step, 819,200 B
   payload, phase collective-skeleton. **This is a transport skeleton, not a model step**: its
   `wall_ms` are not ms/step and are not comparable to 179.81.

Not proven, and not claimed: any stage body executed, any device collective or any collective
inside a model step, any model-step timing (`tp4-token` `step_ms` is empty, `positions_run` 0), any
computational-numerical gate arm, any frontier stage beyond the open. The frontier moved from a
stage-order statement to a **measured refusal at the open**; it did not reach a stage.

## Verdict (qa tool, after it learned the class)

`scripts/device-frontier.py --row-dir .../pld1-probe2` exits **3**, not 2:

```
verdict: OPEN REFUSAL (exit 3) - the arm refused at DeviceLayer::open, blocker named, NO frontier exists.
  this is the correct cheap outcome, NOT a frontier stage, NOT a ms/step and NOT missing evidence
  blocker: layer device weight: no hc_attn_fn for layer 0 in the resolved set
  stage being opened: mhc_entry (layer 0)
  rank logs: 4 of 4 carry the same refusal line; evidence: bind=4/4 loop=4/4 token=4/4 run_meta argv
```

The distinguishing rule (recorded as `docs/RULES-speed.md` 2b): a row that ends at open is a result,
not a number, and never "missing evidence". The tool refuses to invent a frontier document and
refuses to call a bare absence a refusal.

## F1 landed after the row (next row unblocked)

`task/mhc-seam-class` (`bbce699e` class + gemv`s resolve, docs fix `e3e07127`, merged into the batch)
adds the resident `mhc_seam` bind class: ten stored tensors per layer, bound by `stored_bytes` from
the already-uploaded arena (bind-only - no plan row, no byte, no read, no copy), its own coverage
block in `tp4-bind-r*.json` (400 expected / 400 bound), 400 names in the names manifest, and the
absent-class refusal that names the class instead of a tensor. `LayerWeights::resolve` merges
`mhc_seam.pointers` with `attention.pointers` into the one `WEIGHT_NAMES` index space, so
`SeamLayerWeights::resolve` is unchanged. Host arm untouched (the class is read only inside
`DeviceLayer::open` on the device arm). Expected next-row evidence: `mhc_seam 400/400`, attention
still 208/208, `host_seams_excluded` 19, `grep -c hc_` = 240, `norm.weight` = 162,
`device_bound_names` 93,131 - and the layer-0 refusal gone, so the arm proceeds past `mhc_entry`
and the next gap, if any, is named honestly.

## Host-control rows in the same window: the window is not timing-comparable

Two host-arm rows ran with the frozen argv shape (2 warmup, 9 decode, 9 prompt ids, no device arm):

| run | binary | positions | `step_ms` samples | mean |
| --- | --- | --- | --- | --- |
| `pld1-hostctl` | batch tip `b957731e` | 9, rc=0 on 4/4 | 569, 433, 437, 449, 452, 449, 453, 467, 462 | **463.4** |
| `pld1-hostctl-old` | staged `wsd29` `6c479395` (23 cubins) | 9, rc=0 on 4/4 | 481, 440, 442, 446, 438, 442, 431, 426, 319 | **429.4** |

The `wsd29` binary measured **180.1 ms/step** in its own window on this same argv; tonight the same
binary measures 429.4. So the 2.4x is **the window, not the batch**:
`nvidia-smi` at run time read SM clock **208 MHz of a 3003 MHz max**, 4.53 W, 44 C, 0% utilisation -
the GPUs were idling and clocked down, and nine short positions do not carry a ramp. The published
179.81 ms/step rows were taken with the part boosted.

Consequences, stated rather than smoothed over:
- **No regression is claimed and none is exonerated by the batch**: the comparison that would show
  one has to be run at comparable clocks (sustained load or a clock pin), and that is not this
  window. The published number stands untouched and untested tonight.
- **No number from this window is publishable** - neither 463 nor 429, and neither is a D54 row:
  no tolerance arm was declared before either run, and no frozen reference exists for these
  fixtures (`fcfb75c0...` for the 9-id host rows), so `scripts/numerical-gate.sh` correctly says
  "no frozen reference", which is not a failure of the run.
- The host arm does execute the full forward on four ranks at rc=0, which is the code claim the
  batch needs from a control row; the timing claim is left for a clocked window.

## Fabric bench in the same window (comm-lead, transport only)

Four ranks, NCCL from the vLLM image (`aeda7a87...`, 2.31.2), bench `6f2be253...`, 200 samples,
20 warmup, correctness check 0 differing / 0 ULP against the canonical f32 rank-order reduce on
both payloads:

| payload | amortised-40 per call p50 (all four ranks) | enqueue+sync per call p50 | host arm p50 |
| --- | --- | --- | --- |
| 10,240 B/rank | 43.05-43.11 us (mean 46.10, p99 59.76-59.77) | 59.68-61.07 us (p99 582-585 tail) | 173-191 us |
| 20,480 B/rank | 46.49-46.51 us (mean 46.59) | 53.79-53.98 us (p99 <= 104) | 165-174 us |

These are per-collective microseconds, never a step time: 20,480 B costs ~3.4 us more than 10,240 B,
so the fabric is not the decode floor at these sizes. A bench defect was found and fixed before any
number was quoted (the warm-up buffer aliased the reduce input, making the first check read
apparent mismatches); the invalid run is archived and never quoted. Evidence:
`/home/user/gpu-reduce/20261022T192742Z/{batch40,sync}` and `target/gpu-reduce/20261022T19274{9,8}Z`.

## Close-out: what this window produced

1. MR !168 reviewed and green locally, batch tip now carries F1, the bench fix and the qa
   open-refusal classification.
2. Two device probes: a named, four-rank bootstrap refusal (no library) and a named, four-rank
   open refusal (missing resident class) - the refusal invariant holds on hardware.
3. F1: the missing resident class, measured, sized (400 tensors, 158,257,600 B, already in the
   arena) and implemented with its coverage counters; the next row tests it.
4. A transport-only fabric number for the per-layer collective size, and a bench defect fixed.
5. A window that cannot produce timing: clocks at 208 MHz, same old binary 2.4x slower than its
   own window. Recorded so nobody quotes it as a regression or a result.
## Second device row (pld2, after F1 landed): the coverage check fires on MTP lanes (F1b)

Rebuilt from the batch tip after the `mhc_seam` class merged (`f286c847...`, git `3736fd9f`, same 27
cubins, same NCCL), staged as `pld2` and run with the same argv as probe 2. All four ranks exit 2
with the same new line:

```
mhc_seam bind coverage: 400 expected, 430 bound, 0 host seams excluded;
expected-but-not-bound: [];
bound-but-not-expected: ["mtp.0.attn.kv_norm.weight", ... "mtp.2.hc_ffn_scale"]
```

So the class works for the backbone: every one of the 400 expected names bound (nothing missing).
The bind pass scans resident shards by suffix, and the **30 MTP-lane tensors** (`mtp.0/1/2`, ten
each) carry the same suffixes, so they were bound as well. The coverage equality then refused -
identically on all four ranks, at open, before any stage - which is the invariant working rather
than a silent over-bind.

**F1b, next iteration first item, a decision not just a patch:** either the pass binds from the
expected name set (backbone `layers.` only - preferred, since the device layer is the 40 backbone
layers and the MTP lanes are not part of it) or the expected set gains the MTP lanes explicitly
(only if a later stage consumes them). Whichever is chosen, `bound == expected` stays the assertion
and `bound-but-not-expected` keeps refusing. No kernel, ABI or plan change; this is the bind pass
and its expected-set derivation.

Evidence: `/home/user/ev-session-20261020/window/pld2-probe1`.

What the two device rows in this window establish: the arm is selected, the bootstrap opens, the
bind exposes the ten-key class, and every coverage mismatch - absent class (probe 2) or over-bind
(pld2) - refuses at open, with the counter pair that names the cause, identically on four ranks.
The next false step will be equally loud; the frontier is still the open, and no stage body has run
yet.
# Task-7: the HSI indexer is one warp per block (Phase 3, roadmap 6.3 item 1)

Status: **docs-only note. No code changed here.** The indexer change is already in `main`: it landed
as `826f4f0` (!35), "kernels: one warp per HSI block, stage 2 once - 36.2x on the measured launch,
indexer self-checked", whose tree is byte-identical to the `task41/indexer-device` branch commit
`1bc6093`. The task-7 branch therefore has no kernel, ABI or harness delta. It records what was
measured, of what class, and what is still owed.

**Update 2026-10-18: the re-measure window ran.** Runs A, B and C succeeded and their numbers are in
section 2 below, quoted from the logs. Run D (the stale stage-1 printed-class log) failed
immediately on a cubin file name - the example loads
`mega_mhc_fused_bf16_sm121.cubin` while the baseline manifest keys the kernel as
`mega_mhc_sm121`; section 3 explains the two names and section 4.1 builds the file under the name
the example uses. **A short follow-up window at 17:47:23Z staged that renamed copy and re-ran stage
1; it passed 4 of 4 tokens at every boundary** (numbers and log paths in section 5). Both halves of
the stale printed-class logs are therefore regenerated: stage 2 by A/B/C, stage 1 by the retry.

## 1. What landed, and where it can be cited from

- **Kernel** (`kernels/flash_mla_csa2_sm121.cu`): the slot dot is a five-step
  `__shfl_down_sync` reduction (`spark_mla_warp_sum`, line 111) with four values per lane, so one
  warp scores one HSI block and the block's eight warps walk eight blocks. The index-head pair that
  shares a query splits the pool - the even head takes the even HSI blocks, the odd head the odd
  ones - and `blockIdx.y` strides in units of eight blocks per index head. Stage 2's attention is
  guarded to `blockIdx.y == 0`; the trace enter/exit pair is balanced on the same slice guard.
- **Harness** (`crates/sparkflash-kernels/examples/m7_layer0/main.rs`, `stage2.rs` since the !69
  split): `--grid-y` (default 64, range 1..=256), the indexer self-check (integer-valued bf16
  index data, both heads of index head 0 compared against `mla_ref::index_block_best`; a block the
  kernel does not write keeps its `u32::MAX` sentinel, so partial coverage fails instead of
  passing), and the stage-2 timing line that `--repeat` enables.
- **Pins** (`crates/sparkflash-kernels/src/m4_abi.rs`): the landed change moved the `flash_mla`
  pins to 13 declared asserts in a 74-assert scan; the later CSA-union format 2 work raised those to
  **17 asserts / 78 checked** and added the `index_query + ...` dispatch pin. The branch's
  `1bc6093` numbers are superseded: do not cherry-pick it, it predates format 2 and would revert
  the current pins.
- **Baseline** (`crates/sparkflash-kernels/baseline/kernel_baseline.txt`): the current main cubin
  is `flash_mla_csa2_sm121` **60,792 B, sha256 `64fc2944d510885490ef1caeb680600f30b1d552c6ec31b0f6ed840ad0f09d2b`**
  under the pinned `nvcc V13.0.88` (CUDA 13.0.3), `arch=sm_121a`.
- **Tree identity** (measured with git, not asserted): `1bc6093` and `826f4f0` have the same
  parent `6574390` (!30) and the same 5-file/220-28 stat, and
  `git diff 1bc6093 826f4f0 -- <those five files>` is empty. `826f4f0` is an ancestor of
  `2245a4e` (main), so main contains the change under a different SHA than the branch the task
  was written against.

## 2. The measurement (window 2026-10-18, runs A/B/C)

**Provenance (measured, from the window log and the logs' own headers):** window start
2026-10-18T17:16:17Z on **node0**, GPU free at 17:16:51Z after vLLM was stopped, a vLLM
restart was initiated on exit (the log records the initiation, not a completed health check).
Harness binary `m7_layer0_task7_aarch64`, sha256
`43603e4442829b73590bbc35a5ab433ae67cc50e94b8741822a9d0d959513cdc`. Stage-2 cubin
`flash_mla_csa2_sm121.cubin`, sha256 `5bfee4c7...` (60,792 B, the current main pin). Checkpoint
snapshot `/home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170`.
Window log: `/home/user/phase2-evidence/window-2026-10-18/logs/window-20261018T171617Z.log`;
per-run logs in `/home/user/phase2-evidence/window-2026-10-18/runs/` - that directory also holds
the window's m8 sequence run, which is not part of this note; the four sha256s below are the task-7
runs (sha256: A
`45b65e258fd617568ffa39a815645671a4bbcb3e7d8f8a9c30dfafe429103168`, B
`9b9109dce7ce7c57eda444a8a23eb80b9dceb9f692fe14f98dc68bb2804edd12`, C
`f7b93ce68cfa1abef167b17deb754a67ffab863db5f55530ade0a4269b0442df`, D
`495923bb1f87e8a8f03a43e20867780054f281875b63b448d3bf72446116ddde`). Class: **device, measured
on the current format-2 cubin** - this replaces the carried !35 row for these shapes; nothing below
is extrapolated from !35.

| Run | Shape | Verdict | Correctness | Timing, 200 launches (ms) |
| :--- | :--- | :--- | :--- | :--- |
| A | `--stage 2 --tokens 130 --stage2-kv-format engine --grid-y 64 --repeat 200` | **PASS, 10/10 printed checks** | engine format, 5 positions: 163,834/163,840 exact (99.9963%), max abs delta 9.765625e-4, max ULP 1; determinism: two device runs per position, byte-identical; indexer block_scores 2,048/2,048 exact and block_best 2,048/2,048 exact | min 0.2461, **p50 0.2492**, mean 0.2505, max 0.3166; batched 0.2534 |
| B | `--stage 2 --tokens 1 --stage2-kv-format engine --grid-y 64 --repeat 200` | **PASS, 2/2** | engine format, 1 position: 32,768/32,768 exact, max abs delta 0, max ULP 0; determinism byte-identical | min 0.0913, **p50 0.0930**, mean 0.0930, max 0.1003; batched 0.0943 |
| C | `--stage 2 --tokens 130 --stage2-kv-format fp4 --grid-y 64 --repeat 200` | **PASS, 5/5** | FP4 historical format, 5 positions: 163,838/163,840 exact (99.9988%), max abs delta 2.4414063e-4, max ULP 1; `merged-chain FP4 equality: PASS` (got exactly the expected 163838/163840, 2.4414063e-4, 1); determinism byte-identical | min 0.2443, **p50 0.2471**, mean 0.2503, max 0.3411; batched 0.2523 |
| D | `--stage 1 --tokens 4 --layer 0` | **FAILED before any GPU work** | none - see section 5 | none |
| D (retry, 17:47:23Z) | `--stage 1 --tokens 4 --layer 0` | **PASS, 4 of 4 tokens at every boundary** | per token: `hc_coeffs` B 0/16 exact (max abs 3.95e-7), A 1/4 (2.74e-6), C 0/4 (7.15e-7); `xhat_fp8` 5,120/5,120 exact and 160/160 ue8m0 scales; residual `X_l` 20,480/20,480 bf16 exact; `1/rms(X_hat)` max abs 3.43e-5 (bar 3e-4); block-input f32 capture 5,120/5,120 exact | none (correctness run) |

The three stage-2 runs print `launch grid: (64, 64, 1) blocks of (256, 1, 1) threads; each warp
walks 2 HSI blocks per head`, so the geometry is the one this note documents.

**The indexer/attention split is a proxy, and it flipped against the carried row.** Both A and B
score the full 2,048-block HSI pool at every candidate count (that is the indexer's work), while the
128-candidate attention cost scales with the candidates. So:

- indexer proxy = run B p50 = **0.0930 ms** (one candidate's attention is negligible at position 0);
- attention-proper proxy = A p50 - B p50 = 0.2492 - 0.0930 = **0.1562 ms** at 128 candidates;
- ratio: the indexer is now about **0.60x** the attention proper (carried !35 said about 1.4x).

**Why the ratio flipped.** The movement is in the attention-proper term, not in the indexer. The
carried !35 proxy had the indexer at 0.0937 ms and attention proper at 0.0681 ms (ratio ~1.38x);
this window has the indexer proxy flat at 0.0930 ms against attention proper 0.1562 ms. The
attention term is about 2.3x the carried one while the indexer is unchanged, so a flat indexer
against a larger attention crosses below 1.0x. The same movement explains the FP4 row: the replaced
carried 0.1618 ms is 1.53x below this window's 0.2471 ms at the same shape, and the flat indexer
proxy cannot account for that gap, so the delta is in the attention/back-end part of the launch, not
in the indexer. This says *which term moved*, not why the attention term is larger; attributing that
would need a phase-isolated run or a per-change A/B, neither of which this docs-only note contains.

This is the same same-buffers subtraction method !35 used, and it is still a proxy: it assumes one
candidate's attention is negligible in B and that A and B differ only in that attention cost. It is
**not** a phase-isolated measurement - the harness launches one kernel and times the whole launch.
A phase-isolated split would need a harness or kernel change, which this docs-only task does not
contain. The honest statement is therefore: with the window's numbers the roadmap 6.3 item 1
comparison is now **measurable** and, under the proxy, the indexer (0.0930 ms) is below the implied
attention proper (0.1562 ms), i.e. the "at or below" target would hold - but that is a proxy
conclusion, not an established phase-isolated one, and both remain far below the item's 1.5 ms
ceiling.

## 3. Cubins to stage (and the two names of the Mega-mHC cubin)

Stage 2 loads exactly one cubin: `<cubin-dir>/flash_mla_csa2_sm121.cubin`. **Stage 1 loads exactly
one cubin under the example's name, `<cubin-dir>/mega_mhc_fused_bf16_sm121.cubin`** - not
`mega_mhc_sm121.cubin`. The two names are the same kernel:

- `kernels/mega_mhc_sm121.cu` is the source, and `baseline/kernel_baseline.txt` keys the pinned
  cubin as `mega_mhc_sm121` (70,312 B, sha256 `e21293e3...`), because the baseline names kernels by
  source.
- `examples/m7_layer0/main.rs` maps the name `mega_mhc_fused_bf16_sm121` to
  `KernelId::MegaMhcBf16` (`examples/m7_layer0/main.rs:199`) and builds the path
  `<cubin-dir>/<kernel>.cubin` from that name. The example's own module docs
  (`examples/m7_layer0/main.rs:132`) show the matching nvcc line with
  `-o out/mega_mhc_fused_bf16_sm121.cubin`.

A rename or copy does not change a cubin's bytes: stage the pinned file under the example's name. If
either `sha256sum` differs from the table below, the cubin was not built from current main by the
pinned toolkit; stop and rebuild rather than running the window.

| Cubin as staged | Pinned bytes | Pinned sha256 (V13.0.88, sm_121a) |
| :--- | :--- | :--- |
| `flash_mla_csa2_sm121.cubin` | 60,792 | `64fc2944d510885490ef1caeb680600f30b1d552c6ec31b0f6ed840ad0f09d2b` |
| `mega_mhc_fused_bf16_sm121.cubin` (copied/renamed from `mega_mhc_sm121.cubin`) | 70,312 | `e21293e3f6288dbe413a2b998efce6c43c8a56dd993eca5cd1bfc6138cdca618` |

The full seven-entry pinned set is in
[crates/sparkflash-kernels/baseline/kernel_baseline.txt](../../../crates/sparkflash-kernels/baseline/kernel_baseline.txt).

## 4. Re-measure protocol (the Lead runs this; the exact commands)

### 4.1 Build the aarch64 harness and the two cubins on the dev box

```sh
# from the merge commit of this branch (aarch64 harness; the node has no cargo)
./scripts/build-aarch64.sh --example=m7_layer0
sha256sum target/aarch64-unknown-linux-gnu/release/examples/m7_layer0   # record it in the log header

# cubins from the pinned CUDA 13.0.3 devel image (nvcc V13.0.88), from the same commit
mkdir -p out
docker run --rm -v "$PWD:/w" -w /w nvidia/cuda:13.0.3-devel-ubuntu22.04 sh -c '
  nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/flash_mla_csa2_sm121.cu -o out/flash_mla_csa2_sm121.cubin &&
  nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/mega_mhc_sm121.cu -o out/mega_mhc_fused_bf16_sm121.cubin'
sha256sum out/flash_mla_csa2_sm121.cubin out/mega_mhc_fused_bf16_sm121.cubin   # must equal section 3
```

The second `-o` name is deliberate, not a typo: it is the name the stage-1 example loads (section
3). If you prefer to compile under the source name, normalise afterwards with a rename:

```sh
mv out/mega_mhc_sm121.cubin out/mega_mhc_fused_bf16_sm121.cubin
```

Stage the binary and the two cubins on node0 (the Lead's existing `scp` path, e.g.
`~/hostrun/m7_layer0-task7` and `~/hostrun/task7-cubins`); the node is offline to the kernels
agents and only the Lead executes there.

### 4.2 The runs (node, inside the window)

```sh
BIN=/home/user/hostrun/m7_layer0-task7
CUBINS=/home/user/hostrun/task7-cubins
CKPT=/home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
OUT=/home/user/phase3-evidence
mkdir -p "$OUT"

# A - indexer + attention at 128 candidates (last query position 129); engine format
"$BIN" --stage 2 --tokens 130 --stage2-kv-format engine --grid-y 64 --repeat 200 \
  --checkpoint "$CKPT" --cubin-dir "$CUBINS" 2>&1 | tee "$OUT/phase3-indexer-stage2-engine-130-$(date -u +%Y%m%dT%H%M%SZ).log"

# B - indexer-dominated proxy (position 0, one candidate)
"$BIN" --stage 2 --tokens 1 --stage2-kv-format engine --grid-y 64 --repeat 200 \
  --checkpoint "$CKPT" --cubin-dir "$CUBINS" 2>&1 | tee "$OUT/phase3-indexer-stage2-engine-1-$(date -u +%Y%m%dT%H%M%SZ).log"

# C - keeps the merged-chain FP4 claim fresh and checks the same timing point on the default format
"$BIN" --stage 2 --tokens 130 --stage2-kv-format fp4 --grid-y 64 --repeat 200 \
  --checkpoint "$CKPT" --cubin-dir "$CUBINS" 2>&1 | tee "$OUT/phase3-indexer-stage2-fp4-130-$(date -u +%Y%m%dT%H%M%SZ).log"

# D - regenerate the stale stage-1 printed-class log (roadmap 12.1 footer).
#     Needs <CUBINS>/mega_mhc_fused_bf16_sm121.cubin (section 3); the first window's attempt failed
#     2026-10-18 with Error: "<CUBINS>/mega_mhc_fused_bf16_sm121.cubin: No such file or directory
#     (os error 2) (build the cubin first; the module docs give the nvcc line)"
#     (verbatim except the elided <CUBINS> prefix). The 17:47:23Z retry staged the renamed cubin and
#     printed stage 1: PASS (4 of 4 tokens; section 5).
"$BIN" --stage 1 --tokens 4 --layer 0 \
  --checkpoint "$CKPT" --cubin-dir "$CUBINS" 2>&1 | tee "$OUT/phase3-indexer-stage1-$(date -u +%Y%m%dT%H%M%SZ).log"
```

Notes on the flags. **`--grid-y` is not optional for a self-describing log but needs no sweep**:
64 is the already-measured optimum (!35: 0.1613 ms at 64 against 0.1813 ms at 8 on the same
128-candidate point) and it is the harness default; passing it explicitly pins the geometry in the
log header. A `grid.y` sweep is roadmap 6.3 item 2 and would be a separate window. **No external
timing harness is needed**: `--repeat 200` makes the harness time 200 launches of the last query
position on the same buffers with CUDA events and print min/p50/mean/max. The indexer-vs-attention
split uses !35's own method - run A minus run B - because the indexer scores the full 2,048-block
pool at every candidate count while the attention cost scales with the 128 candidates of position
129; run B is therefore an indexer proxy (one candidate's attention is neglected), not an
instrument that isolates kernel phases. Splitting them without that assumption would need a
harness or kernel change, which this docs-only task does not contain.

### 4.3 What to quote back (the exact lines)

From each log, copy these lines verbatim (they are the harness's own prints, not a summary). The
values the 2026-10-18 window actually printed are shown after each line:

1. `launch grid: (64, 64, 1) blocks of (256, 1, 1) threads; each warp walks 2 HSI blocks per head`
   - the proof the geometry and `--grid-y` are the ones recorded here (all of A/B/C, identical).
2. `  indexer block_scores (head pair 0/1, the full pool): 2048 values, exact 2048/2048 (100.0000%), max abs delta 0e0 at index 0`
3. `  indexer block_best: 2048/2048 exact (an unwritten block keeps u32::MAX)`
4. `  indexer self-check: PASS at grid.y 64 (both parities of the pool, one head each)`
5. `timing: 200 launches at the last query position on the same buffers; per-launch ms min ... p50 ... mean ... max ...; batched ms/launch ...`
   - A: `min 0.2461 p50 0.2492 mean 0.2505 max 0.3166; batched ms/launch 0.2534`
   - B: `min 0.0913 p50 0.0930 mean 0.0930 max 0.1003; batched ms/launch 0.0943`
   - C: `min 0.2443 p50 0.2471 mean 0.2503 max 0.3411; batched ms/launch 0.2523`
6. `format engine (released FP8 window format, bf16 storage): totals across 5 positions: 163834/163840 exact (99.9963%), max abs delta 9.765625e-4, max ULP 1; determinism: two device runs per position, byte-identical`
   (A; B prints the same line with `across 1 positions: 32768/32768 exact ... 0e0 ... 0`).
7. `format fp4 (historical kernel format): totals across 5 positions: 163838/163840 exact (99.9988%), max abs delta 2.4414063e-4, max ULP 1; determinism: two device runs per position, byte-identical`
   plus `merged-chain FP4 equality: PASS (... got 163838/163840, 2.4414063e-4, 1)` (C).
8. `stage 2: PASS (10 of 10 printed checks ...)` (A); `stage 2: PASS (2 of 2 ...)` (B);
   `stage 2: PASS (5 of 5 ...)` (C). The stage-1 retry printed
   `stage 1: PASS (4 of 4 tokens at every boundary)`; its per-token lines are quoted in section 5.
9. The harness's own header lines from each per-run log: checkpoint path, memory-floor lines and
   the run's geometry. The binary and cubin sha256s are NOT repeated in the per-run logs; they are
   the window log's provenance lines, quoted verbatim as
   `43603e4442829b73590bbc35a5ab433ae67cc50e94b8741822a9d0d959513cdc  .../m7_layer0_task7_aarch64` and
   `64fc2944d510885490ef1caeb680600f30b1d552c6ec31b0f6ed840ad0f09d2b  .../cubins/flash_mla_csa2_sm121.cubin`
   (`/home/user/phase2-evidence/window-2026-10-18/logs/window-20261018T171617Z.log`). The
   window's UTC start/end is recorded in that same log, not in the per-run logs. The stage-1 retry's
   provenance lines are in its own window log
   `/home/user/phase2-evidence/window-2026-10-18/window-stage1-20261018T174723Z.log`: the same
   binary sha256 `43603e44...` and
   `.../cubins/mega_mhc_fused_bf16_sm121.cubin` sha256 `e21293e3...` (the renamed copy of the
   baseline's `mega_mhc_sm121`, 70,312 B).

### 4.4 Pass bar (declared before the numbers)

The harness prints its bar before it judges anything: for the engine format at least 99.5% of the
32,768 attention outputs bit-identical to the bf16 rounding of the f32 reference and max abs delta
<= 1.6e-2, with determinism (two device runs per position byte-identical) required; the FP4 run
must reproduce the merged-chain equality line exactly (163,838/163,840 exact, max abs delta
2.4414063e-4, max ULP 1). The indexer self-check is exact by construction: 2,048/2,048 block scores
and 2,048/2,048 block_best. A run that misses any of these exits non-zero and the log is a finding,
not a pass. The 2026-10-18 window met the bar on A, B and C; D failed before reaching its bar and
the 17:47:23Z retry (section 5) met it on 4 of 4 tokens.

## 5. The stale stage-1/stage-2 printed-class logs

Roadmap 12.1 records that the stage-1 and stage-2 logs cited there predate the M2 fix of the
harness's printed comparison-class strings and are "pending regeneration in the next device
window". **Both halves are now regenerated in the 2026-10-18 windows.**

- **Stage 2**: runs A and B (engine format) and C (FP4) print the current comparison-class strings
  and are the replacement logs for that class (section 2).
- **Stage 1**: run D failed before any GPU work because the example loads
  `mega_mhc_fused_bf16_sm121.cubin` while the baseline manifest keys the kernel `mega_mhc_sm121`
  (section 3). A short follow-up window at **2026-10-18T17:47:23Z** staged the renamed copy - cubin
  `mega_mhc_fused_bf16_sm121.cubin`, 70,312 B, sha256
  `e21293e3f6288dbe413a2b998efce6c43c8a56dd993eca5cd1bfc6138cdca618`, the same bytes the baseline
  records for `mega_mhc_sm121`; the rename does not change the sha - and re-ran argv
  `m7_layer0 --stage 1 --tokens 4 --layer 0 --checkpoint <snapshot> --cubin-dir <cubins>`.
  Result: **`stage 1: PASS (4 of 4 tokens at every boundary)`**, exit 0. Logs:
  `/home/user/phase2-evidence/window-2026-10-18/window-stage1-20261018T174723Z.log` and
  `.../runs/d-stage1-layer0-retry.log`; binary `m7_layer0_task7_aarch64` sha256
  `43603e4442829b73590bbc35a5ab433ae67cc50e94b8741822a9d0d959513cdc`. Per token the harness printed:
  `hc_coeffs` B 0/16 exact (max abs 3.95e-7), A 1/4 (2.74e-6), C 0/4 (7.15e-7); `xhat_fp8`
  5,120/5,120 exact with 160/160 ue8m0 scales; the residual `X_l` 20,480/20,480 bf16 exact;
  `1/rms(X_hat)` max abs 3.43e-5 against its 3e-4 bar; and the block-input f32 capture 5,120/5,120
  exact. No pass-bar miss, no non-zero exit.

Roadmap 12.1's HSI indexer row and its stale-log note now cite the regenerated logs; the numbers
there come from the harness's printed lines in those logs, not from a stale printed class line. The
class is unchanged: stage 1 is internal consistency against the in-crate f32 reference on released
layer-0 bytes, not model fidelity.

## 6. What this note does not establish

- Stage 1 was re-run on the device in the 17:47:23Z retry window and passed 4 of 4 tokens at every
  boundary (section 5); that run is internal consistency against the in-crate reference on released
  layer-0 bytes, not model fidelity, and it makes no timing claim. Stage 2's numbers are measured
  (section 2). The stale printed-class note is retired: both halves are regenerated.
- It does not claim roadmap 6.3 item 1's "at or below the attention proper" target is *established*.
  Under the A-minus-B proxy the indexer (0.0930 ms) is below the implied attention proper
  (0.1562 ms), which is the first measured window where the comparison favours the indexer, but the
  split is a proxy, not a phase-isolated measurement.
- It does not re-optimise `--grid-y` and does not time the indexer independently of the
  one-candidate subtraction assumption.
- It does not touch formats 0/1/2 semantics; format 2's own device claims are in
  [task14-union-attention-case.md](task14-union-attention-case.md) and roadmap 12.1.

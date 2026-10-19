# Phase 3 m14: the attention stage (K5 flash_mla_csa2, format 2 - the CSA union) in isolation

**Status: window b12 PASSED both shapes with the corrected bf16-vs-bf16 measurement and is the standing record.**
b11 (the first device execution of format 2) is superseded and kept below as history, with its old
numbers.
This is the m14 deliverable: the
attention kernel the chain calls at K5, at the real layer-0 geometry, in the m10-m13 window-harness
pattern. The subject is **format 2**, the released CSA union of the 128-row engine-format window and
the compressed candidate list with the per-head attention sink, softmaxed together. Window b11 was
the first device execution of format 2 (the kernel header records formats 0 FP4 and 1 engine FP8 as
the earlier measured columns, and that format 2 had no device execution before b11); before that
window the m4_abi guards and mla_ref union tests were the only coverage.

The harness is [m14_flash_mla.rs](../../crates/sparkflash-kernels/examples/m14_flash_mla.rs). The next
chain stage after attention is the output-projection pair (K6 wo_a, K7 wo_b), which m13 covers in
isolation; the chain wiring itself is M3's and is not in this branch.

## Subject and launch shape

| item | value |
| :--- | :--- |
| kernel | `flash_mla_csa2_sm121` |
| pin | 60,792 B, sha256 `5bfee4c7a0ece7c921e591fc6b8bd9d98c39b95d735d7c1d699ec58edaa51ea3`, 0 PTX membar lines |
| format argument | `SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION = 2` |
| grid / block | (64, 1, 1) blocks of (256, 1, 1) threads: one query head per block |
| arguments | the pinned 19-word list through `cuda_harness::flash_mla_params` (q, kv_fp4, kv_scales, window_codes, window_scales, window_kv_format, window_rows, selected, index_key, index_query, sink, block_scores, block_best, out, position, selected_count, trace_ring, layer_id, step_seq) |
| geometry | 64 query heads x 512-wide latent, 128-token sliding window, index_topk 512 (union capacity 640) |
| launches | one launch group per query position |

Format-2 semantics, from the kernel and `mla_ref::attention_with_sink_union`: the supplied window
rows are all attended, oldest first; then the selected compressed candidates are attended when
`candidate <= position` (the reference's causal reachability mask - a candidate inside the window is
not excluded, duplicates are attended once per entry); one f32 softmax runs over that union with the
per-head sink in the denominator after the maximum. Out-of-window selected ids therefore contribute,
which the window-only formats cannot express.

## Provenance and the declared seams

Every weight, norm and sink byte is read verbatim from the released checkpoint
`layers.0.attn.{wq_a,wq_b,wkv}.{weight,scale}`, `layers.0.attn.q_norm.weight`,
`layers.0.attn.kv_norm.weight` and `layers.0.attn.attn_sink` through the snapshot reader, which
checks each header's dtype and shape first. The FP8 weights are packed with the declared
value-preserving `pack_b12x` permutation; the scale tensors are uploaded as stored (the dense GEMM's
own grid), with the byte counts asserted against the B12x rules.

The checkpoint stores no activation for this stage, so the operands are host-built CPU twins over
those released bytes, identical for the device run and the reference:

| operand | declared construction |
| :--- | :--- |
| attention block input | the released `embed.weight` row t directly, D8-quantised - a declared stand-in for K1's mHC output; no mHC pass runs here |
| q | wq_a twin -> `q_norm` (NORM_EPS 1e-20) -> D8 -> wq_b twin -> rope tail (last 64 of 512, theta 10000) at position t -> bf16, the dtype the kernel consumes |
| main-KV pool | wkv twin on the same input -> `kv_norm` -> rope at its own position -> sparkflash-kv's FP4 main-KV quantiser (E2M1 codes, one E4M3 scale per 16 channels, low nibble first); format-2 candidate ids index this per-position view |
| window rows | the same kv latent rounded to bf16 (the engine's storage dtype) and quantised with the engine's per-32-channel E4M3 x ue8m0 window format (`m4_quant::act_quant_fp8_row`, which M7 stage 2 asserted byte-identical to the engine's quantiser); supplied oldest first |
| sink | `layers.0.attn.attn_sink` F32 [64], consumed on both sides: a [64] kernel argument added to the denominator after the maximum, and the same per-head logit in the reference |
| selected list | a DECLARED host list: the compressed absolute positions strictly older than the query's window, `(max(0, p+1-128-512))..(p+1-128)`, i.e. every position behind the window capped at index_topk - NOT the indexer's top-512 |

Window construction for position p: window rows for positions `max(0, p+1-128)..=p`, oldest first.

The index side channel (index_key, index_query, block_scores, block_best) is uploaded as zeros plus
sentinels and its writes are ignored, never compared: the HSI indexer scoring stage is **not**
claimed.

## Comparison class (declared before any number)

The kernel's union attention is compared to `mla_ref::attention_with_sink_union` - the reference
union path - on the same q bytes, the same window rows, the same compressed pool bytes, the same
candidate list, position and sink.

This is deliberately **not** a bit-identity claim. Both sides are three-pass union paths, but they
still differ in expf against libm exp, the tree reduction of the normalising sum against the serial
one, the cached against recomputed weight handling, and the bf16 output write - the degrees of
freedom the M7 stage-2 record names. The declared bars for format 2 are that stage-2 tolerance class,
whose two measured members are separated here rather than merged:

* format 0 (FP4), the merged chain that contains D8: 163,838/163,840 exact (99.9988%),
  2.4414063e-4 max abs delta, one bf16 ULP (M7 stage 2 and stage 7);
* format 1 (engine FP8 window format, task-3): 163,834/163,840 exact (99.9963%),
  9.765625e-4 max abs delta, one bf16 ULP (M7 stage 2, engine follow-up).

The bars themselves:

* **exact fraction** >= 99.5% of the bf16 output elements equal to the bf16 rounding of the
  reference (the stage-2 EXACT_BAR), applied to the aggregate **and** to **every single position's**
  own fraction - a sub-bar position fails the run, it is not averaged away;
* **max abs delta** <= 1.6e-2 (the stage-2 DELTA_BAR), computed bf16 device output against the
  **bf16 rounding** of the f32 reference (bf16-vs-bf16, the same quantity the stage-2 record bars),
  with the measured max ULP distance reported.

Determinism: every position's launch is repeated and all output bytes must match. Separate control:
the engine window format's quantisation gap - the **bf16-stored row** against its decoded E4M3 x
ue8m0 values, so the storage round is not counted as quantisation error - bounded at 10% of the
operand's maximum absolute value. The control is not part of the exactness class.

## NOT claimed

* the HSI indexer scoring stage: its side channel is zeros plus sentinels and its writes are ignored;
* the CSA candidate selection: the selected list is the declared host list above, not the indexer's
  top-512 merge;
* the K1-K4 device stages: q and the KV rows are CPU twins with the declared seams above (the K1 mHC
  pass does not run, and wq_a/wq_b/wkv are not launched);
* RoPE as a claimed step: the kernel contract is that the caller supplies rope'd q and k, and the
  harness applies the engine reference's rope as a declared host seam;
* the attention output's inverse rope, wo_a, wo_b, the residual add/hc_post, any chain ordering or
  seam ledger, any token/text/end-to-end output, routing/SwiGLU/MoE/shared expert/combine, weight
  staging, device-budget admission, engine integration, and any host-time saving;
* the historical formats 0 (FP4) and 1 (engine FP8): format 0 is measured in M7 stage 2 and stage 7
  (163,838/163,840, 2.4414063e-4) and format 1 in M7 stage 2's engine follow-up (163,834/163,840,
  9.765625e-4); neither is re-run here;
* long candidate lists: at the fixture lengths the union is at most 128 window rows plus the
  positions older than the window (0 at 8 positions; 1 at 129 positions, where position 128 is the
  first with a candidate behind the window; 2 at 130), so the 512-candidate / 640-slot regime is
  **not** device-exercised by this harness - it is covered by mla_ref's union tests and the m4_abi
  guards, not here;
* any per-decode-token timing derived from the measured launch times.

## Cubin: build and pin

Built with the pinned toolkit (CUDA 13.0.3, nvcc V13.0.88) under
`nvidia/cuda:13.0.3-devel-ubuntu22.04`:

```
nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/flash_mla_csa2_sm121.cu -o out/flash_mla_csa2_sm121.cubin
```

The harness reads the cubin, refuses anything that is not the pinned size and sha256 from
`baseline/kernel_baseline.txt` (and refuses if the manifest's own pin moved), and only then loads the
kernel and allocates device memory.

## Building and running the harness

```
cargo build --example m14_flash_mla                               # host check
scripts/build-aarch64.sh --example=m14_flash_mla                 # node binary
```

Usage: `m14_flash_mla --checkpoint <snapshot dir> --cubin-dir <dir with the cubin> [--tokens 8]
[--repeat 100] [--floor-gib 32]`. `--tokens` is the number of independent query positions
(positions 0..N-1), not a batch or sequence dimension. Candidate lists behind the window start at
position 128 (so 129 positions already has one), and 130 is the fixture prefill length. `--layer` must
be 0 (the default). The harness checks
`MemAvailable` against `--floor-gib` at start and end, so it refuses to run under the serving stack.

## Window commands (for the Lead window)

The b12 binary is staged on the node at
`~/phase2-evidence/window-2026-10-18/m14_flash_mla_157350f4_aarch64` (4,027,896 B, sha256
`157350f4449ba6aa7cd9d86242291704550e591ee6bc96ff530601bcc3173587`) with the cubin unchanged in
`~/phase2-evidence/window-2026-10-18/cubins/flash_mla_csa2_sm121.cubin`. The exact commands, with
vLLM stopped on all four nodes first:

```
~/phase2-evidence/window-2026-10-18/m14_flash_mla_157350f4_aarch64 \
  --checkpoint /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins --tokens 8 --repeat 100
```

and the same line with `--tokens 130`. (The b11 binary `5efb7976`, 3,995,840 B, sha256
`5efb797600af22d14e91188bd012a22e869ba04fe9c4a4207d5cf8ff88782ba5`, is still on the node; it is the
pre-correction harness and must not be used for b12.)

## Pass bars (the window record must show)

| check | bar |
| :--- | :--- |
| cubin pin | 60,792 B / `5bfee4c7...` verified before any device allocation |
| output exact fraction | >= 99.5% of the bf16 elements equal to the bf16 rounding of `mla_ref::attention_with_sink_union`, for the aggregate **and for every position individually**, at both position counts |
| output max abs delta | <= 1.6e-2 computed bf16 device vs bf16-rounded reference, with the max ULP distance reported |
| determinism | every position's repeat launch byte-identical |
| window control | max window-format quantisation gap (decoded value against the bf16-stored row) <= 10% of the operand maximum |
| exit code | 0 at both position counts |
| summary | every verdict PASS and "m14_flash_mla: PASS" |

## Timings: what was measured and what is not

The harness prints per-launch event timing (min / p50 / mean / max over `--repeat` launches of one
query position's launch group, plus the batched ms/launch) and the position count it ran with. These
are whole-launch attention measurements on one node with vLLM stopped. **No per-decode-token cost is
claimed or derived from them**; a per-token figure would be a hypothesis, would have to state which
position count it assumes, and would still not be a decode-step measurement.

## Window records: b11 (superseded) and b12 (standing, PASS)

b11 was the kernel's **first device execution of CSA format 2**, and it PASSed both position counts
under the metric this branch shipped at the time. That record is **superseded, not withdrawn**. Review
found three defects in the harness's judging, none in the kernel run: the exact verdict used only the
aggregate fraction, so one sub-bar position could hide; the max abs delta was computed against the f32
reference instead of the bf16 rounding (bf16-vs-bf16 is the quantity the stage-2 bar means); and the
window control compared the decoded window against the unrounded f32 latent instead of the bf16-stored
row. The harness now judges every position, reports the bf16-vs-bf16 delta and measures the control
against the stored row, so the measured deltas change and **window b12 must re-run it**. The b11
numbers below are kept only as history; they must not be read as the current record.

b11 historical record (superseded): window b11 start 2026-10-18T23:43:20Z (host node0), vLLM
stopped on all four nodes from 23:43:20Z, GPU compute apps 0 at 23:44:33Z, vLLM restarted 23:44:40Z.
Evidence is node-only: `window-b11-20261018T234320Z.log`, `runs11/m14-tokens8.log`
(23:44:33Z-23:44:34Z, exit 0) and `runs11/m14-tokens130.log` (23:44:34Z-23:44:40Z, exit 0) under
`/home/user/phase2-evidence/window-2026-10-18/`. Binary
`m14_flash_mla_5efb7976_aarch64`, 3,995,840 B, sha256
`5efb797600af22d14e91188bd012a22e869ba04fe9c4a4207d5cf8ff88782ba5`; cubin `flash_mla_csa2_sm121`,
60,792 B, sha256 `5bfee4c7a0ece7c921e591fc6b8bd9d98c39b95d735d7c1d699ec58edaa51ea3`; MemAvailable
117.7 GiB to 117.2 GiB (8 positions) and 117.6 GiB to 117.1 GiB (130 positions), against the 32-GiB
floor. Old-metric numbers, retained for comparison only: 8 positions 262,136/262,144 exact
(99.9969%), old f32-reference max abs delta 7.8060627e-3, max ULP 1, control 4.1607%; 130 positions
4,259,627/4,259,840 exact (99.9950%), old delta 7.807255e-3, max ULP 7, control 4.9418%; determinism
byte-identical at both shapes. Worst single positions were 99.9939% (position 1) and 99.9817%
(position 93) - retained because those are the fractions the corrected verdict now checks. The b11
"single-ULP" phrasing was wrong for the 130-position shape (max ULP 7): the mismatches are ULP-scale
rounding differences, not a single-ULP class and not a systematic offset.

Provenance note on the b12 binary: b11 ran the pre-correction binary `5efb7976`, which also carried
an example-local `engine_window_row` helper that the m4_abi pin forbids. The branch now calls the
shared `sparkflash_kernels::cuda_harness::engine_window_row` (same bf16-round-then-engine-quantise
rule, asserted byte-identical to `m4_quant::act_quant_fp8_row`), so the window bytes for a given kv
latent are identical and that change is not measurement-bearing. The b12 binary is
`m14_flash_mla_157350f4_aarch64` (4,027,896 B, sha256
`157350f4449ba6aa7cd9d86242291704550e591ee6bc96ff530601bcc3173587`), staged on the node; the cubin
pin is unchanged (a rebuilt cubin from the updated header has the same bytes, 60,792 B /
`5bfee4c7...`).

### Window b12 record: PASS, the standing window

Ran in a Lead window on 2026-10-19 with the corrected harness (bf16-vs-bf16 delta, aggregate AND
per-position exact verdict, control against the bf16-stored row). Window b12 start 00:15:36Z (host
node0), vLLM stopped on all four nodes from 00:15:36Z, GPU compute apps 0 at 00:16:50Z, vLLM
restarted after the runs at 00:16:57Z (the Lead verifies the served endpoint separately). Evidence is
node-only: `/home/user/phase2-evidence/window-2026-10-18/window-b12-20261019T001536Z.log`,
`runs12/m14-tokens8.log` (00:16:50Z-00:16:51Z, exit 0) and `runs12/m14-tokens130.log`
(00:16:51Z-00:16:57Z, exit 0) under the same directory.

Provenance: binary `m14_flash_mla_157350f4_aarch64`, 4,027,896 B, sha256
`157350f4449ba6aa7cd9d86242291704550e591ee6bc96ff530601bcc3173587`; cubin `flash_mla_csa2_sm121`,
60,792 B, sha256 `5bfee4c7a0ece7c921e591fc6b8bd9d98c39b95d735d7c1d699ec58edaa51ea3`, verified by the
harness by size and sha256 before any device allocation; MemAvailable 117.8 GiB at start and 117.2 GiB
at end for the 8-position run, 117.7 GiB to 117.1 GiB for the 130-position run, against the 32-GiB
floor.

The b12 binary predates one judging/reporting parity change made after the window: the delta loop now
carries stage2.rs's non-finite -> INFINITY guard (a non-finite delta becomes `INFINITY` instead of
being dropped by a plain `max`) and reports the element where the maximum delta occurs. Neither can
change a finite measurement, and b12 had no non-finite output, so **the b12 numbers above stand and no
re-window is needed**; a rebuilt binary simply has a different sha256 from `157350f4...`.

Result: **PASS** at both position counts - 5 of 5 verdicts each (cubin pin, output exact fraction
with the per-position bar, output max abs delta bf16-vs-bf16, determinism, window control), 0 failed.

Measured at 8 positions (positions 0..7; window rows 1..8; no out-of-window candidate exists, so the
union is window-only, max 8 slots):

| check | measured |
| :--- | :--- |
| cubin pin | 60,792 B, sha256 `5bfee4c7...`, 0 PTX membar lines |
| output exact fraction | 262,136 / 262,144 bf16 elements exact (99.9969% overall); worst position 1 at 99.9939%; bar 99.5% on both |
| output max abs delta | 9.765625e-4 bf16-vs-bf16, bar 1.6e-2; max ULP 1 |
| determinism | every position's repeat launch byte-identical |
| window control | 4.0936% (gap 1.09375e-1 of operand max 2.671875), bar 10% |
| timing | 100 launches of one position: min 0.1918 p50 0.1935 mean 0.1961 max 0.2515 ms; batched 0.1962 ms/launch |

Measured at 130 positions (positions 0..129; window rows 1..128; out-of-window candidate lists at 2
positions, max union 130 slots):

| check | measured |
| :--- | :--- |
| cubin pin | 60,792 B, sha256 `5bfee4c7...`, 0 PTX membar lines |
| output exact fraction | 4,259,627 / 4,259,840 bf16 elements exact (99.9950% overall); worst position 93 at 99.9817%; bar 99.5% on both |
| output max abs delta | 3.90625e-3 bf16-vs-bf16, bar 1.6e-2; max ULP 7 |
| determinism | every position's repeat launch byte-identical |
| window control | 4.6784% (gap 1.25e-1 of operand max 2.671875), bar 10% |
| timing | 100 launches of one position (position 129, the 128-row window): min 0.3235 p50 0.3260 mean 0.3288 max 0.4219 ms; batched 0.3276 ms/launch |

Measuring the delta the stage-2 way changed it, and the record states this explicitly: b12's
bf16-vs-bf16 max abs delta is **9.765625e-4 at 8 positions and 3.90625e-3 at 130**, smaller than
b11's 7.8060627e-3 / 7.807255e-3 **because b11 subtracted the f32 reference** instead of its bf16
rounding; part of b11's delta was the reference's own f32-to-bf16 write, which the bar never meant to
count. The exact fractions and the worst positions are unchanged by the metric fix (the comparison
bits are the same): 99.9969%/99.9939% and 99.9950%/99.9817%. The mismatches are ULP-scale rounding
differences of the declared class, max ULP 1 at 8 positions and 7 at 130, reported per shape and not
averaged or selected.

Union coverage: at these fixture lengths the union tops out at 128 window rows plus 2 older positions
(out-of-window lists at 2 of the 130 positions); the 512-candidate / 640-slot regime is **not**
device-exercised here and remains covered by `mla_ref`'s union tests and the `m4_abi` guards only.

The class statement is unchanged and honest: this is the **stage-2 tolerance class, not bit-identity**;
the HSI indexer scoring and the CSA candidate selection (the selected list is a declared host list, not
the indexer's top-512) are **not** claimed; formats 0 (FP4, 163,838/163,840 at 2.4414063e-4) and 1
(engine FP8, 163,834/163,840 at 9.765625e-4) are separate M7 columns and are not re-run here; the chain
and any token/text output are not claimed; and no per-decode-token cost is derived from the timings.

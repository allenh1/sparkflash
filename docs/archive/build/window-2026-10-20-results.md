# GPU window results, 2026-10-20 (Lead record, batch 2026-10-20-20)

**One window, 21:12:33-21:35:17Z (17:12:33-17:35:17 EDT), total vLLM outage 22 min 44 s.** Order:
stop vLLM on all four nodes -> four-rank device bind run -> kernel base-vs-new slot on node0 ->
restore vLLM and verify. This page records only what was measured in that window and what was not.

## Service restoration (the non-negotiable success condition)

vLLM was stopped only after the pre-window health check (200, served root heretic 12d58ad0...) and
restored with the same recipe (`run-recipe.py deepseek-v4.1-flash-1017-heretic`). Restored at
21:35:17Z: `GET /v1/models` 200, `GET /health` 200, served root
`.../snapshots/12d58ad0878b5008e80d720251adb969a0069412` = the same heretic snapshot as before the
window, max_model_len 1,048,576. No leftover sparkflash process on any node after the window. The
stop verification had to be rewritten for GB10 before the window (`nvidia-smi memory.used` is
`[N/A]` on unified memory); the working check is container exited + MemAvailable >= 95 GiB +
port 8000 closed + no vllm process, per node.

## Leg 1 - four-rank device bind coverage (M1 pre-token): PASS 4/4

Package `b69ebb2` (branch task/m1-four-rank-token), aarch64 CLI sha256
`b25eb1b3c8fc96b5082ca7c4adf5d6a0133a7675ba1f43bdbd22eecbc4553f4f`, 19/19 pinned cubins,
one process per node, rc=0 and `error: null` on every rank, launch-to-last-exit 469 s (worst-rank
arena boot ~380 s). argv: `--run-stage tp4-chain --tp-rank R --tp-world-size 4 --upload --loop
--comm-base-port 29601 --comm-timeout 120 --comm-connect-timeout 120 --warmup-steps 2
--decode-steps 1`, no `--cubin-dir`.

Per rank, from `tp4-bind-r<R>.json` (schema `sparkflash-tp4-bind/1`):

| class | expected == bound | missing | unexpected | complete |
| :--- | :--- | :--- | :--- | :--- |
| attention (table 215, 19 host seams excluded) | 208 / 208 | [] | [] | true |
| routed_experts_gate_up | 61,440 | [] | [] | true |
| routed_experts_down | 30,720 | [] | [] | true |
| shared_experts | 240 | [] | [] | true |
| router | 120 | [] | [] | true |
| tail | 3 / 3 (pointers non-zero) | [] | [] | true |

`summary.complete` true, `device_bound_names` 92,731, `summary.per_token_weight_h2d_bytes` 0,
`moe.weight_h2d_bytes_per_token` 0, `forward: not-run`, `tokens: 0`. Plan identity identical on
all four ranks: 95,822 planned shards (95,818 resident / 4 disk), 77,640,198,984 B = 72.308 GiB,
161,736,798 ranges, plan hash `0x9a59035fc40776c8`; arena stored == uploaded == 77,640,198,984 B.

Census over the real plan totals 95,818 with no unclassified name: attention_projection 208,
attention_host_seam 19, attention_aux_host_read 416, routed_expert 92,160, shared_expert 240, router
120, tail 3, special_token 3, mhc 240, engram 8, mtp 2,401. The `special_token` class (committed as
`b69ebb2` during window prep) claims the released checkpoint's image_start/image_end/image_newline
embeddings by name; without it the census would report three unclassified resident tensors and
`summary.complete` would be false for a non-bind reason.

Collective walk (transport only, no cubins): tcp, 3 steps x 80 calls = 240 reduces, every step
`calls_per_step_ok` true, payload 819,200 B/step, wall_ms 41 / 22 / 20. `hybrid_sequence.executed`
false.

Launch compatibility, recorded in the bind JSON: routed gate/up true, routed down **false** (the
arena stores the checkpoint's 576-K rows while `b12x_mxfp4_moe_sm121` reads the A8 pre-padded 640-K
row layout; bound and byte-verified, but a launch would misindex - the stored-layout kernel is the
next-session blocker), shared true, router true. That single false row is why the run is a
**pre-token PASS**: no forward pass ran and no token, position, step time or tok/s is claimed.

Names manifest per rank: 185,481 logical lines = 208 attention expected + 208 attention bound +
19 attention host seams + 2 x 92,520 MoE + 2 x 3 tail, byte-identical across ranks (sha256
`6176d0e9060dc69973c15b83aa065e6fc8ba29ef409e88a4777096f545c1a567`); no trailing newline, so
python counts 185,481 and `wc -l` counts 185,480. (The pre-window estimate of 185,462 omitted the
19 host-seam lines; corrected on evidence and in the kit/QA collector.)

Adjudication: Lead structural check `tools/adjudicate-bind-window.py` -> BIND-OK-PRE-TOKEN, exit 0;
QA D48 collector -> NOT-ATTEMPTED, exit 3, zero failures (every bind/walk check pass); QA acceptance
runner -> shape PASS (34 checks), pre_token PASS (44 checks), twin/oracle/step time NOT_ATTEMPTED,
overall NOT-ATTEMPTED, exit 3. Twin equality, token equality and step time are explicitly NOT
measured - no forward pass ran.

Evidence (dev host, copied before restore):
`/home/user/ev-session-20261020/window/m1-bind/` - rank{0..3}/ with the five files per rank,
`argv.txt`, `run.meta`, `sha256-manifest.txt` (22 files, manifest sha256
`1d084f74be4bae8ad17ad85c4f6f7dcbcbe6bfd0064ab1af290f13cc16412ba7`),
`bind-window-adjudication.json`; QA verdicts under
`/home/user/ev-session-20261020/qa/window/m1-bind/` (`d48-step-gate.json`
`8e5aa30516fd1b1eca0fbb62b170ac1590b987a8245f83130203524b48c731d7`, `accept-device-token.json`
`287d0e4fc5c28937372b518679a5c1bd1c6fe923e7ae70c175e5f243596e41b5`).

Two defects the window itself found: the bind console line summed the two routed rows as "gate/up"
and printed their difference as "down" (JSON was always correct; fixed in `847856b`), and the census
completeness rule initially asked the text-path bind to explain multi-modal tokens (fixed by
`b69ebb2`, before the window). Neither affects the stored evidence.

## Leg 2 - kernel base-vs-new slot (WS-C): 5 rows green, 2 measured wins, 1 row uncertified, branch NOT merged

Rank 0 of node0 only; base stage c62fb0c vs new stage 32888c4 (new script eda1d1e); 12
invocations, 24 s total; no pin was regenerated. Evidence:
`/home/user/ev-session-20261020/window/kernels-32888c4/` (summary.txt, provenance.txt, 12 logs).
All timings are per-launch kernel measurements, not a per-position extrapolation.

| row | verdict | base p50 | new p50 | speedup | note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| routed_moe_gate_up | BIT-IDENTICAL | 0.9398 / 0.9443 ms | 0.3805 / 0.3845 ms | 2.47x / 2.46x | 12.1% of node bandwidth; both stages byte-identical to the same CPU twin (gate 3840/3840, up 3840/3840), pinned-cubin checks PASS |
| routed_moe_down | uncertified (row invalid, not kernel) | 0.6552 ms | 0.4183 ms | 1.57x | corrected roofline 9.3% (the summary's 97.7% is an instrumentation over-count: it used the 64-expert staging arena instead of the 6 routed experts); no value-neutrality result |
| router | PASS | 0.2309 ms | 0.2279 ms | 1.01x | unchanged cubin, no regression |
| lm_reduce_batched | PASS | 2.4819 ms | 2.4897 ms | 1.00x | unchanged cubin; "byte equality: EXACT" |
| attn_projections | PASS | 11.7505 / 6.3180 / 12.0106 ms | 11.7488 / 6.3363 / 12.0096 ms | 1.00x | unchanged cubin |
| cert_lm_head | PASS | 3.9864 / 11.0936 ms | 3.9779 / 11.1161 ms | 1.00x | unchanged cubin; self-reported roofline 11.0% both stages |

Why routed_moe_down is uncertified: `moe_slice` zeroes the f32 slot accumulator once and the kernel
adds into it, so `--repeat 200` compares 201 accumulating launches against a one-launch reference;
every element differs by construction and the failure is identical on the incumbent c62fb0c cubin
(0/30,720 f32 partials, 1 bf16 match, max abs delta 6.382656e2 in both logs). The identical
statistics are consistent with value-neutrality but are not proof, so no pin was changed and
`task/kernel-occupancy` was **not merged**: pins stay at c62fb0c. Next-session certification is
ready and cheap: rebuild `moe_slice` from tip 20d0df8 (sha256
`c09d36ec439e2199c096e7850b69b4245fe1f5a30c99c794e8cc5b17a397d11e`), run the one-launch
correctness invocation at both stages (expected `rank accumulator: 30720/30720`,
`exact bf16 matches: 30720 (100.0000%)`, max abs delta 0e0) plus m10 both stages (~25 s), then merge
with exactly two pins (`b12x_mxfp4_moe_sm121` 11,704 B `ccb9d127...`, `routed_gate_up_sm121`
15,672 B `12ddc097...`).

## What this window moved and what it did not

Moved (measured, evidence above): the four-rank arena boots and binds **every** device name on real
GB10 hardware with zero per-token weight H2D (attention 208/208, routed/shared/router 92,520, tail
3), the plan is identity-stable across all four ranks and matches the earlier window's hash, the
census is total over the real 95,818 resident tensors, and the 80-reduce/step walk still holds on the
frozen binary. The previous window's "bound 208 of 196" denominator bug is closed on real evidence.
Kernel-side, the split-K gate/up kernel is certified byte-identical and 2.46-2.47x faster per
launch; the split-K combine is 1.57x faster per launch but waits on a one-launch correctness run.

Not moved: **no token, no position, no step time, no tok/s, no twin match, no D41/D48 bar** - the
device arm has no forward loop, no head/logit emission, and the routed down-projection cannot launch
from the arena. Kernel pins did not change. RDMA is still unavailable (driver 580.173.02 <
580.178.04), so the TCP transport cost is unchanged.

Next critical path, in order: (1) stored-layout combine kernel + `m25_moe_stored` harness green on
the released stored layout (design committed: `docs/build/moe-stored-layout-variant-plan.md`),
(2) wire the stored cubin in `moe_device.rs` and re-run the 2 s kernel certification, (3) four-rank
forward loop wiring (attention from arena -> attention reduce -> MoE from arena -> MoE reduce ->
final norm, head, argmax) with head/logit emission on the rank arena, (4) 8-9 position oracle
comparison. The first token (M1) and every throughput claim depend on all four.

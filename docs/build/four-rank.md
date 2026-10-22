# Four-rank (TP=4) host path

Current-state handbook for the four-rank path. Numbers are labelled **MEASURED** (recorded run),
**DESIGN** (declared target/arithmetic, not yet measured) or **ESTIMATE** (projection). Numbers
without a run id come from the newest record under Archived sources. Operational process rules live
in `docs/RULES.md` and are not restated here.

## 1. What runs today

A **host (CPU) forward over the released checkpoint bytes**, one process per node, routed experts
sharded and everything else replicated. It is *not* vLLM's TP=4 layout: attention, indexer,
compressor, embedding and LM head are full tensors on every rank.

| component | state |
| --- | --- |
| process model | 4 ranks, one process/node, TCP collectives; no MPI, torchrun or Ray |
| sharded class | language-layer routed experts `w1`/`w2`/`w3` + scales only |
| residence | one in-process `ResidentCache` per rank; Engram tables excluded, read per token |
| attention/KV | replicated per rank, each rank owns its `SequenceState` and KV copy |
| device driver | not opened; `device_execution: false` written unconditionally |
| sampling / concurrency | greedy only; one sequence, one request in flight, one queued |

## 2. Topology, ports, launch

| rank | node | address |
| --- | --- | --- |
| 0 | node0 | 192.0.2.80 (coordinator, HTTP, evidence owner) |
| 1 | node1 | 192.0.2.88 |
| 2 | node2 | 192.0.2.68 |
| 3 | node3 | 192.0.2.59 |

Rank `r` binds `base_port + r` for the engine collective (**29601** in serving runs); rank 0 also
holds serve control on `base_port + 100` (**29701**) plus HTTP 8000, tokenizer and sampling; ranks
1-3 never detokenize. Lower ranks dial upward; the hello carries the rank and the first all-reduce
is the implicit barrier. Frames have deadlines (`--comm-*timeout`, 120 s at four nodes); a dead peer
is `PeerClosed`, a silent one `Timeout`, and either aborts the run with no single-rank fallback.
Commands come from `scripts/tp4-window-bind.sh --print-argv[-token] <R>` and `scripts/serve-four-node.sh`; builds are aarch64 and staged before a window.

## 3. Sharding contract (Phase-1)

A static `ShardingPlan` is verified against shard headers before any tensor byte is read; plan hash `0xcd8adc09d2c71d55`.

| class | rule |
| --- | --- |
| routed experts `w1`/`w3` | column: rows / world; local `[inter/world, dim]` |
| routed experts `w2` | row: input columns / world |
| attention (wq_a, wq_b, wkv, wo_a, wo_b), indexer, compressor | replicated full |
| embed, LM head | replicated full; embedding row read by byte range |
| shared expert, router, mHC, norms, MTP heads | replicated full |
| MTP expert blocks (`mtp.0/1/2.ffn.experts.*`) | replicated **full**, outside the `layers.` plan |
| Engram tables | disk, paged by row; design shards 6 of 24 hash columns/rank |

The released index has **96,085 tensors**; this plan shards **92,160** (40 x 384 x 6). The full
TP=4 design splits attention and head too (wq_b 8,192 rows/rank, wo_b 2,048 in/out, vocab 32,320
rows/rank) — **DESIGN**.

Collective arithmetic is fixed and is part of acceptance: bf16 payloads, f32 accumulation in
ascending rank order from `0.0`, exactly **one** bf16 round-to-nearest-even, then the routed
epilogue. Any other rule invalidates the single-rank oracle. The host path issues **40
all-reduces/token** (one per layer); the full device design needs **123 all-reduces + 2
all-gathers per position** — DESIGN. Measured host wire cost is 8.6 ms/step (40 x 0.216 ms), so a
one-shot RDMA replacement must beat 8.6 ms, not the 92.2 ms "collective" bucket (rank-arrival skew).

## 4. Memory (two layouts, do not mix)

| item | value | label |
| --- | --- | --- |
| plan slices (language routed experts) | 67.236 GiB/rank | MEASURED (cache 90,944,617,800 B) |
| replicated non-expert classes | 10.738 GiB/rank | MEASURED |
| replicated MTP expert blocks (full) | 6.724 GiB/rank | MEASURED |
| **non-Engram working set, Phase-1 host** | **84.699 GiB/rank**, 96,081 tensors | MEASURED |
| Engram tables | 188.833 GiB, disk-resident | MEASURED |
| full TP=4 design resident weights | 72.31 GiB/rank | DESIGN |
| full TP=4 design all-in | ~90.4 GiB/rank vs 121.69 GiB node | DESIGN |
| reference vLLM resident + non-torch; KV | 87.07 GiB; 14.99 GiB = 3,682,262 tokens | MEASURED (reference) |

**Conflict.** 72.31 GiB/rank assumes attention/head sharded and MTP experts quartered; 84.699
GiB/rank is the Phase-1 layout with everything but routed experts replicated and MTP experts full.
Both are correct for their layout; name the layout in every budget statement. Resident fill is
233-258 s cold, RSS 85.835 GiB, VmHWM 90.302 GiB.

## 5. Host twin (window-free correctness carrier)

`--write-fixture DIR` writes a deterministic 40-layer tiny checkpoint (compression 0, Engram at
layers 1 and 14 with 1,024 rows, `moe_inter_dim = 128`): 2,193 tensors, 21,878,732 bytes, hash
`0xb656f193fb1b27f2`. Two ranks over loopback TCP, 3 prompt + 2 decode steps:

| check | value | label |
| --- | --- | --- |
| tokens, all 5 steps | `[6, 6, 6]`, equal to single-rank reference | MEASURED |
| max logits relative L2 vs reference | 1.28e-7 (bar 2.0e-2) | MEASURED |
| max routed-seam relative L2 | 2.75e-3 (bar 5.0e-3) | MEASURED |
| seam comparisons / all-reduces | 200 / 200 (12,800 elements); wall 6.4 s | MEASURED |

`cargo test -p sparkflash-cli tp4_host::` runs 8 tests at world 2/4 (~7 s); the bars are recorded, not gated, at four nodes.

## 6. Baselines

**Offline host bench**: `--run-stage tp4-host --bench-steps 10 --resident-cache --profile`,
`SPARKFLASH_REFERENCE_THREADS=20`, four ranks, prompt tokens
`11111,201,8197,14,270,3417,344,28`. Metric is decode-only tokens/s; load and prefill are excluded
by construction and stated in `bench.measure`.

| run (tree) | mean step | median | decode tok/s | label |
| --- | ---: | ---: | ---: | --- |
| RUN 1 baseline (`a9f8e07`) | 5,958.4 ms | 5,953.0 ms | 0.1678 | MEASURED |
| RUN 4b fused decode (`f18d139`+`4256616`) | 939.7 ms | 936.5 ms | 1.0642 | MEASURED |
| fixed tree, token loop | 935.2 ms | — | 1.0693 | MEASURED |
| fixed tree, chunked prefill | 926.9 ms | — | 1.0787 | MEASURED |

All runs share the plan hash, 720 reduces per 10 steps and identical tokens per rank. After the CSA
shared-channel publication fix, RUN 4b is comparable on corrected arithmetic; the pre-fix sequence
`[344, 436, 3338, ...]` is **void**. Corrected sequence:
`[1192, 1623, 2734, 1918, 270, 6857, 1893, 305, 270, 10317, 1893]`.

**Serving endpoint** (4b/17 tree, 37-token prompt, `max_tokens=64`):

| metric | value | label |
| --- | --- | --- |
| TTFT, default token loop | 38.964 s | MEASURED |
| TTFT, chunked prefill opt-in | 43.625 s (slower, hence opt-in) | MEASURED |
| decode | 0.9402 tok/s | MEASURED |
| wall | 105.99 s | MEASURED |

Earlier first-online baseline (`fd09d5c`, pre-fusion): 37+8 tokens, TTFT 519.48 s, decode
0.0710 tok/s, wall 618.64 s — the same path before the fused kernels, not a competing number.

Chunked and token-loop prefill are bit-identical on the real checkpoint at 2/4/8/16/37 prompt
tokens (0 differing logit bits); bench prefill is 24.6 s vs 33.1 s at 37 tokens (1.34x, same-run
bisect). The chunk is opt-in via `SPARKFLASH_TP4_PREFILL_MODE=batched` (legacy
`SPARKFLASH_TP4_PREFILL_CHUNK=1`); default is the token loop. The batch-15 allocator pair moved an
older end-to-end harness 13.827 s -> 7.929 s/step — not the bench metric.

## 7. Token evidence format (`sparkflash-tp4-token/1`)

| file | writer | contents |
| --- | --- | --- |
| `tp4-token-r<rank>.json` | `tp4_token.rs` | the envelope |
| `tp4-logits-r<rank>.jsonl` | `logits_dump.rs` (every rank since `893ef1a3`) | one object/position, 11 frozen fields |
| `tp4-logits-r<rank>.jsonl.bin` | same | full-vocab little-endian f32, `129,280 * 4` B/position |

Envelope: `schema`, `stage`, `loop_step`, `rank`, `world`, `forward: run|refused`,
`tokens{ids, positions, start_position, prompt_len, argmax_logits, top1_margins, logits_sha256}`,
`positions_run`, `step_ms{samples,mean_ms,worst_ms}`, `wall_ms[]`, `reduce_calls_per_step`,
`head{rule,vocab,tie_break}`, `logits_dump{path,bin,sha256,bin_sha256,dtype,vocab,positions,rows}`,
`error`, `launch_census`.

Writer-enforced rules: on `run`, ids non-empty, `positions_run == len(ids) == len(positions)`,
positions strictly increasing from `start_position < prompt_len`, one argmax and top-1 margin per id,
dump `bin == path + ".bin"` with 64-hex hashes and matching rows, `reduce_calls_per_step == 80`
(2 x 40 layers), `routed_experts_down == true`, `error == null`, `len(wall_ms) ==
len(step_ms.samples)`. On `refused`: no ids/positions, `positions_run == 0`, null start/len and
dump, non-empty error, `routed_experts_down == false`. A refusal is never a partial claim.

The 11 frozen dump fields: `position, device_argmax, logits_argmax, reference_id, hybrid_id,
fixture_id, exact_token_equal, vocab, logits_offset, logits_bin, top20`. The token stage sets
`device_argmax = hybrid_id = consensus token`, `reference_id/fixture_id = null`, and tiles
`logits_offset` by 129,280 f32/position; rank 0 is primary, replicas compare by `bin_sha256`.

Canonical per-rank argv: `--run-stage tp4-chain --checkpoint <released> --tp-rank R
--tp-world-size 4 --evidence-dir ... --upload --loop --comm-peers <ip0..ip3> --comm-base-port
29601 --comm-timeout 120 --comm-connect-timeout 120 --warmup-steps 2 --decode-steps 9 --forward
--forward-preflight --kernel-cubin-dir <cubins> --token-ids <prompt ids>`; 129 ids + 9 decode
steps covers positions 129..137. `--kernel-cubin-dir` is not the older `--cubin-dir` (world=1
hybrid diagnostic, never at world 4). `--forward-reference-ids` carries the teacher-forced nine
ids `[491,367,201,80,3302,201,54,201,4341]`, one per dumped row.

## 8. Prompt-logprobs rank contract

All-position prompt logprobs are scored in one bounded-span pass, not one forward per position.
Each rank screens its contiguous shard of `head.weight` (`local_vocab` rows at
`offset = rank * local_vocab`); three all-gathers per span, fixed stride:

| phase | block per (rank, position) | width |
| --- | --- | --- |
| 1 | own value, offset, local_vocab, local_max, omitted_exact_max, certified_count, certified_max_bound, then `(token_id, value)` pairs | `k*2 + 7` |
| 2 | f64 log-sum-exp partial as exact `(hi, lo)` f32 pair, scaled by `exp(local_max - global_max)` | 2 |
| 3 | count of local values whose f32 logprob is strictly greater than the owning token's | 1 |

Merge: `logprob = ((value - global_max) as f64 - ln(sum)).ln() as f32`, `sum` joined in rank
order; owning-token rank = `1 + Σ greater_count` (exact integer); alternatives are the union of
all ranks' k candidates plus the owning token, sorted by f32 logprob desc then token id asc.

Bounds: `n` alternatives with `0 <= n <= 63`; candidate budget `k = clamp(n + 1, 16, 64)`; HTTP
`prompt_logprobs` cap 20. Span `TP4_PROMPT_LOGPROBS_SPAN = 64` prompt tokens; the last span asks
for no row at the final position. Per rank a span is `64 * 32,320 * 4 B ≈ 8.3 MiB` on the
released vocab, independent of prompt length. A replicated head is the correctness fallback: rank 0
sends the full shard, others an empty block at `offset = vocab`.

A position is reported only when its f32 order is **proven**: `boundary > omitted` strictly,
`boundary - omitted > margin`, and `next` absent or `boundary - next > margin`, with
`margin = 4 * f32::EPSILON * max(1, |boundary|)`. Otherwise the request is refused with
`UnprovenOrder` plus the uncertified values. Other typed refusals: `AllRequiresFullVocabulary`
(`-1`), `RequestTooWide`, `CertifiedEntries`, `NonFiniteExact`/`OwnTokenNonFinite`,
`NoDistribution`, `OwnTokenMissing`, `ShardLayout`/`RankLayout`. HTTP maps a refusal to
`tp4_prompt_logprobs_refused` (503); every rank decides identically, so the group drains it and
resynchronises on the next request, while any other rank error stays group-fatal.

Evidence is host-only on the 40-layer tiny twin (vocab 128): merge equality at worlds 1/2/4 and
selections none/1/3/5, values within 1 ULP (measured 0), payload bit-identical to the serial path.

## 9. Known limits and open items

- **Not vLLM's TP=4.** Attention/head are replicated; the real layout needs device collectives
  (tasks 34/35/36), so Gate B (serving parity against vLLM TP=4) is unmet.
- **No device execution on this path.** No driver, no CUDA context, no kernel. The four-rank
  device token loop exists but refuses at unwired stages (window construction, format-2 body,
  remaining layers, certified head); no token is claimable from it.
- **Speed and KV.** Host decode is ~1.07 bench tok/s and ~0.94 endpoint tok/s; device is next. KV
  is 4x replicated (MLA latent `num_kv_heads=1`): no KV collective, no paged KV.
- **Serving gaps:** greedy only; no stop strings (`run_tp4_request` ignores `Job.stop`); `n > 1`
  refused; prompt logprobs bounded; load (229-261 s/rank) reported separately from request wall.
- **Serving/parity evidence has no D41 bars** (the D41 programme is parked); it is not on the
  D54 speed gate either. The rank-directory file set is part of the contract. Plans verify by
  name/shape, not checksum: record the root.

## Archived sources

tp4-design.md, tp4-host-twin.md, tp4-online-baseline.md, tp4-output-baseline.md,
tp4-tokens-per-second.md, tp4-token-evidence.md, tp4-prompt-logprobs-rank-contract.md,
prompt-logprobs-one-prefill.md

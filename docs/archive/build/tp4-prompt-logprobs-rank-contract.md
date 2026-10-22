# TP=4 all-position prompt-logprobs rank contract

Host path, synthetic tiny plan, UNBARED. This is a contract document for the task-18 increment: how
the Phase-1 TP=4 host path scores every prompt position without the full-vocabulary gather it used to
refuse, when it reports a value, and when it refuses instead of approximating.

## 1. What changed and why

Task-15 made the single-process reference path score a whole prompt in one pass: the engine returns
the final-normed head inputs of every requested prompt position
(`InferenceEngine::prefill_with_position_logits`, engine `forward/mod.rs`) in bounded spans
(`PROMPT_LOGPROBS_ROW_BUDGET`), and the serial driver turns each row into the vLLM payload
(`prompt_logprobs_pass` in `serve.rs`). The Phase-1 TP=4 host path could not use that seam: its rank
prefill loop ran one decode-shaped forward per prompt token and kept only the last position's
logits, so it refused prompt logprobs outright rather than serve a wrong intermediate position.

Task-18 adds the all-position rank contract in three pieces and changes the refusal into a
criterion:

1. **Engine seam.** `run_hybrid_sequence_chunk` accepts selected chunk positions
   (`position_logits`) and returns their final-normed head inputs in
   `HybridChunkOutput::position_head_inputs`. `InferenceEngine::head_rank_logits` computes one
   rank's local LM-head rows without the exact gather and reports which vocabulary slice they are
   (`RankHeadLogits { logits, local_vocab, rank, world, offset }`).
2. **Rank contract.** `sparkflash-engine/src/prompt_ranks.rs` defines the candidate blocks, the
   three exchange phases and the deterministic host merge; `sparkflash-cli/src/tp4_host.rs`
   implements `Tp4Rank::prefill_prompt_logprobs`, which feeds the prompt in bounded spans, asks the
   chunk forward for each span's scored head inputs, runs the exchange over the same
   `TpCollective::all_gather_f32` the forward uses, and merges.
3. **Serve seam.** The control frame `Request` carries an optional `prompt_logprobs` top count;
   every worker runs the same rank method (and discards the merge); rank 0 builds the OpenAI
   payload with the serial assembly rules and delivers it once after prefill, before the first
   generated-token delta.

No new kernel, no device code, no vLLM process. The TP=4 rank path still uses the host f32 engine.

## 2. The wire contract (per bounded span, per selected position)

Every rank screens only its own contiguous shard of `head.weight` (`local_vocab` rows starting at
`offset = rank * local_vocab`; a replicated head is the special case below). For each selected
position the rank keeps, from its local row:

- the **owning token's exact value** — the request names the token, so it is always known locally
  on exactly one rank;
- the **top k local candidates** by value (value desc, token id asc; `k = clamp(n + 1, 16, 64)`
  where `n` is the requested alternatives count);
- the **local maximum**, the **maximum exact value outside the k candidates**
  (`omitted_exact_max`), and — from a certified head — the **certified count** and the
  **certified upper bound** — with m16/m27 semantics: a `certified == 0` entry holds an exact f32
  logit, a `certified == 1` entry holds an upper certificate bound that is only **>= the exact
  value**, never the exact value;
- after phase 2, the f64 partial log-sum-exp of its shard.

Three all-gathers carry the contract (the same `all_gather_f32` the forward already uses; the
blocks are fixed-stride so every rank addresses the same offsets):

| phase | block per (rank, position) | width |
| :--- | :--- | :--- |
| 1 | own value, offset, local_vocab, local_max, omitted_exact_max, certified_count, certified_max_bound, then candidate `(token_id, value)` pairs | `k*2 + 7` |
| 2 | the f64 log-sum-exp partial as an exact `(hi, lo)` f32 pair, scaled by `exp(local_max - global_max)` | 2 |
| 3 | the rank's count of local values whose f32 logprob is strictly greater than the owning token's f32 logprob | 1 |

The merge then computes the same values the serial formula computes, from the distributed sum:

- `logprob(value) = ((value - global_max) as f64 - ln(sum)).ln() as f32` per entry, with
  `sum = Σ_ranks join(hi, lo)` in rank order (each local sum is an f64 accumulation, so the
  regrouped total is the serial total up to f64 regrouping; measured bit-identical on the synthetic
  plan and pinned at 1 ULP in the tests);
- the owning token's **rank = 1 + Σ_ranks greater_count** — an exact integer, not an f32 comparison;
- the **alternative order** from the union of every rank's k candidates plus the owning token,
  sorted by reported f32 logprob desc then token id asc — the same order `top_k_indices` gives.

## 3. When a position is reported, and when it refuses

The request's top condition is a statement about the serial path's f32 logprobs. Before a position
is reported, the merge proves the requested order at f32 precision:

- `boundary` = the reported logprob of the last requested union entry; `next` = the logprob of the
  first omitted union entry (if any); `omitted` = the largest exact value outside the union
  candidates, converted with the same formula;
- the order is **proven** when
  `boundary > omitted` (strictly, at f32) **and**
  `boundary - omitted > margin` **and**
  `next` is absent or `boundary - next > margin`, where
  `margin = 4 * f32::EPSILON * max(1, |boundary|)` — a representability bound: at that separation
  no f32 round-off could swap two entries, so the reported order is the serial order;
- if the union itself is shorter than requested, or the boundary cannot be separated, the position
  is **not reported**: the request is refused with `UnprovenOrder` and the logprob values that
  could not be certified. No prefix of the requested list is reported as if it were the list.

Refusals are typed (`prompt_ranks::MergeRefusal`) and travel as a rank error whose message begins
with `TP=4 prompt logprobs refused:`:

| refusal | criterion |
| :--- | :--- |
| `AllRequiresFullVocabulary` | the `-1` whole-vocabulary selection: bounded blocks cannot carry it |
| `RequestTooWide { requested, budget }` | top count above the bounded budget (`n > 63`; the HTTP layer's cap is 20, so this is reachable through the rank/control seam) |
| `CertifiedEntries { position, rank, certified_count }` | a rank entry is an upper certificate bound, not an exact logit: the distribution and every value would be approximations |
| `NonFiniteExact { rank, token_id }` / `OwnTokenNonFinite` | an exact candidate or the owning token is non-finite |
| `NoDistribution { position }` | no finite global maximum or a non-positive/non-finite distributed sum |
| `OwnTokenMissing { position, token_id }` | the owning token is on no rank's shard (a broken partition, not a value question) |
| `ShardLayout` / `RankLayout` | the shard blocks are not one contiguous partition, a candidate is outside its rank's range, duplicate token ids, wrong rank labels, or two ranks claim the owning token |
| `UnprovenOrder { position, requested, boundary_logprob, omitted_logprob }` | the f32 margin proof above fails |

The serve layer maps a refusal to a terminal `EngineEvent::Error` with code
`tp4_prompt_logprobs_refused` (HTTP 503). Because every rank makes the same decision from the same
gathered blocks, the control channel treats this code as request-scoped: rank 0 broadcasts the typed
`Error` frame and every worker drains it and resynchronises on the next `Request` instead of
exiting. Any other rank error remains group-fatal.

The whole-vocabulary `-1` selection and a top count at or above the bounded maximum are also
refused in `Tp4Driver::generate` before a job is admitted, so those requests never touch the group.

## 4. Bounds and shapes

- **Selection.** `n` alternatives, `0 <= n <= 63` (the candidate budget is `clamp(n + 1, 16, 64)`;
  the HTTP layer caps a request's `prompt_logprobs` at `DEFAULT_MAX_LOGPROBS = 20`).
  `Some(0)` is the owning-token-only contract, the same payload as asking for no alternatives.
- **Span.** `TP4_PROMPT_LOGPROBS_SPAN = 64` prompt tokens per chunk forward, and the last span asks
  for no row at the final position (its distribution is the generation distribution, not a prompt
  position). Per rank, one span holds `span * local_vocab` f32 head rows: on the released
  vocabulary that is 64 * 32,320 * 4 B ≈ 8.3 MiB per rank, independent of the prompt length.
- **Replicated head fallback.** When the source serves all vocabulary rows in one rank's shard (the
  pre-batch-18 hatch `local_vocab == vocab`), rank 0 sends the full-shard candidates and every
  other rank sends an empty block at `offset = vocab`; the merge still sees one contiguous
  partition and computes the same payload. This is the correctness fallback, not a measured speed
  path.

## 5. Payload equality

The TP=4 path reports the serial host path's payload, not merely similar numbers:

- position 0 is `null` (no conditioning token), each later position scores the prompt token
  itself, own entry first with the competition rank, then the alternatives with positional ranks,
  collapsed by token id exactly as vLLM's logprob dict collapses duplicates;
- token strings (own and alternatives) are decoded with the same preceding-context sequencing the
  serial path uses (`LogprobDecoder`);
- the tokens array is the prompt's token ids/strings in order.

Evidence (host path, synthetic 40-layer tiny twin fixture, vocab 128, UNBARED):

- `crates/sparkflash-engine/tests/prompt_ranks.rs` — merge exactness against the serial
  `log_softmax`/rank formula on synthetic shards for worlds 1/2/4 and selections
  none/1/3/5, wire round-trips, every refusal case, budget/layout bounds.
- `crates/sparkflash-cli/src/tp4_host.rs` — `the_four_rank_prompt_logprob_contract_matches_the_serial_host_path`:
  four real loopback ranks over the twin fixture, two identical rounds, every rank's merged payload
  bit-identical, exact top order against the serial host path, logprob values within the pinned
  1 ULP tolerance (measured 0 on this plan), plus
  `the_four_rank_contract_refuses_an_over_wide_request_before_any_collective`: every rank refuses
  with the same typed message before any collective and the group serves the next request.
- `crates/sparkflash-cli/src/serve.rs` — the payload-assembly equality test against serial
  `token_logprob`, and a control-channel e2e with a stub rank that pins
  `Request.prompt_logprobs = Some(3)`, the leading `PromptLogprobs` reply before any delta, the
  field shape and the `Done` terminal reply.
- Census: `scripts/tp4-prompt-logprobs-census.py` writes
  `tp4-prompt-logprobs-census-<utc>.{json,md,sha256}` under
  `/home/user/ev-session-20261020/serve/`: rounds scored, serial order equality, rank
  bit-equality, max |logprob delta|, refusal accounting and recovery, round wall times, probe binary
  sha256, fixture hash and source-manifest sha256s.

## 6. Non-claims and limits

- Host path only: no CUDA kernel, no device, no GPU, no vLLM process, no throughput or speedup
  claim. The census wall times are repeatability of one synthetic plan on one host.
- The fixture is the deterministic 40-layer tiny twin (vocab 128, 2,193 tensors), not the released
  checkpoint. The released vocabulary path is the same code, exercised for shape by the engine
  unit tests, not measured here.
- The full four-process `sparkflash serve --tp4-*` path is not exercised end to end on this host:
  the serve CLI opens the released checkpoint, while the twin fixture is only reachable through the
  `tp4-host` fixture source mode. The HTTP/control seam is therefore covered with the control
  channel plus a stub rank, and the rank contract with four real ranks in one process.
- The bounded top-k contract does not serve vLLM's `-1` whole-vocabulary prompt logprobs; that
  selection stays refused, now with the criterion documented above rather than "TP=4 has no
  all-position seam".
- Certified-head entries are refused rather than used with their upper bounds. A future certified
  merge would need to carry bounds through the log-sum-exp and prove the order against them; that is
  not this increment.

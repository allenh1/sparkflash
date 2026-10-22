# Serving surface, scheduler and deployment

Current-state handbook for `sparkflash serve`. Numbers are labelled **MEASURED** (recorded run),
**DESIGN** (declared contract/behaviour, not measured) or **ESTIMATE**. Operational process rules
live in `docs/RULES.md` and are not restated here.

## 1. Current state

- The default engine is the **host reference CPU engine**; `--engine device` exists but is
  window-gated. The TP=4 path is the four-rank host loop (see `four-rank.md`).
- Default concurrency is **one request at a time**; continuous batching is opt-in
  (`--continuous-batching`, `--max-batch N`). The driver queue is bounded at `JOB_QUEUE = 64`.
- The API surface is OpenAI-compatible with documented deviations; the Anthropic
  `/v1/messages` route is buffered-only.
- Paged KV is a host foundation, **not** the engine's KV backing yet.
- All serving evidence is host path / synthetic tiny plan (vocab 128) and has **no D41 bars**
  (the bar programme is parked), except the wire observations against the reference deployment,
  which are labelled MEASURED (wire).

## 2. Endpoint surface

Verdicts: IMPL = implemented and wire-tested; IMPL-b = implemented, covered below the wire;
PARTIAL = common case works, a documented vLLM behaviour differs; MISSING.

| endpoint | state | note |
| --- | --- | --- |
| `GET /health` | PARTIAL | 200 empty body; no engine-dead 503 signal |
| `GET /readyz` | IMPL | 200 while accepting, 503 from the drain signal |
| `GET /metrics` | PARTIAL | Prometheus text 0.0.4, project-prefixed families |
| `GET /version`, `GET /load` | MISSING | no instrumentator surface |
| `POST /v1/chat/completions` | IMPL-b | SSE, usage, logprobs, tools, stop, n>1 |
| `POST /v1/completions` | IMPL | token-id prompts on the wire; no nested int[][] |
| `POST /v1/responses` | PARTIAL | buffered only; no retrieve/cancel |
| `POST /v1/messages` | PARTIAL | buffered only; no `count_tokens` |
| `GET /v1/models`, `GET /v1/models/{id}` | IMPL | `owned_by = sparkflash` by design; retrieve is a documented superset |
| `POST /tokenize`, `POST /detokenize` | IMPL | string or messages form; OOV id -> `{"prompt":""}` |
| `GET /tokenizer_info` | MISSING | reference image 404s it, so no shape to match |
| embeddings / pooling / rerank / audio / LoRA | OUT-OF-SCOPE | generation-only server |

## 3. Request fields and behaviours

| area | state |
| --- | --- |
| sampling | temperature, seed, top_p, top_k, repetition/presence/frequency penalty implemented; penalties apply to raw logits before temperature. `top_p = 0.0` is a documented extension |
| not in schema | `ignore_eos`, `min_tokens`, `stop_token_ids`, `bad_words`, `allowed_token_ids`; unknown JSON is silently ignored |
| `min_p`, `logit_bias` | parsed and validated (integer keys, `-100..=100`, `0..=1`), then **refused typed** when they would change generation, matching the reference's 400 |
| `n > 1` | ordered choices, per-choice id `{id}-{choice}`, vLLM's seed+i, prompt counted once; TP=4 refuses typed (one shared sampling window) |
| `best_of` | accepted as an explicit no-op; `n` governs |
| `logprobs` / `top_logprobs` | IMPL-b; whole vocabulary via `-1` under `--max-logprobs`; cap text matches vLLM |
| `prompt_logprobs` | one pass over the prompt in bounded spans; position 0 null, echo implied; refused with `stream=true` |
| `logprob_token_ids` | supported with `MAX_LOGPROB_TOKEN_IDS` cap; sampled token reported first |
| stop | holdback across token boundaries, earliest match, tail flush, completing token counted; empty string -> 400 with vLLM's text |
| tools | DSML block lifted to `tool_calls`, buffered and streamed; `tool_choice` auto/none honoured, required/named refused typed; `parallel_tool_calls: false` reports only call 0 |
| streaming + tools + logprobs | one `StreamAssembler` drains parser deltas and queued logprobs; every generated token scored once |
| `stream_options` | `include_usage` final `choices: []` chunk; `continuous_usage_stats` puts cumulative usage on every chunk (needs include_usage) |
| `response_format` | `text` no-op; `json_object`/`json_schema` refused typed on chat and completions; constrained decoder exists but is not wired |
| reasoning | template side implemented (`enable_thinking`, `reasoning_effort`); reference parser behaviour not replicated |
| auth | static bearer key via `--api-key`/`SPARKFLASH_SERVE_API_KEY`; every route except `/health` and `/metrics`; constant-time compare, 401 envelope |
| CORS | on by default (`*`), `--cors-allow-origin` allow-list, `--no-cors`; applied outside bearer auth so preflight passes |
| errors | OpenAI envelope; `code` is the integer HTTP status, internal name kept as `inner_code`; 404 `model_not_found`; queue-full is a typed stream refusal (not HTTP 429) |

Deviations worth knowing: `logprobs = -1` on a non-streaming completion reports the whole
vocabulary (the reference dev build returns an empty dict); completion `top_logprobs` keys are
sorted by token id; chat `echo` is parsed but its prompt-prepend implication is not implemented;
the TP=4 path ignores stop strings.

## 4. Anthropic route

`POST /v1/messages` returns a buffered message only; `stream=true` is refused. There is no
`/v1/messages/count_tokens` and no model retrieve/list beyond the shared `/v1/models`. Treat it as
an adapter over the same router, not a second serving stack.

## 5. Scheduler and continuous batching

**Runtime default is the serial loop.** The scheduler crate's default policy is batched, but the
shipped CLI constant is the pre-batch-17 token loop because the endpoint measured it faster.

| knob | default | meaning |
| --- | ---: | --- |
| `JOB_QUEUE` | 64 | bounded driver queue; overflow is a typed 503 `refused` |
| `max_num_batched_tokens` | 8,192 | per-step token budget |
| `prefill_chunk_tokens` | 4,096 | per-sequence prefill chunk cap |
| KV page | 64 tokens | paged cache page unit |
| `--max-batch` | — | admission cap inside the scheduler; cannot exceed the queue |

Contract: plan then accept (`plan_step` -> `BatchStep` -> `accept_step` one token per producing op,
in op order). A step never exceeds the token budget; every chunk is contiguous. Decodes consume one
token each; a prefilling sequence contributes at most one chunk per step. Decode-first is the
default ordering. Both phases rotate a per-phase cursor, so a tight budget cannot starve a
prefilling request. Preemption picks the last-admitted running request with no op in this step,
returns its pages and queues it as swapped; policies are **recompute** (default, retain prefix and
re-forward the rest) and **swap** (reattach history). Swap bytes are accounted, not moved. Prefix
reuse shares complete 64-token blocks by `radix::block_hash`; the prefill plan starts at the end of
the reusable prefix, capped one token short of the prompt. The batcher owns prompt and generated
tokens (preemption must replay real history), not just counts.

Measured host A/B (same binary, same workload, synthetic plan; streaming ladder 1/8/32 clients x 2
requests x 64 tokens):

| run | concurrency | serial TTFT p50 | batched TTFT p50 | serial tok/s | batched tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20261021T005030Z` (530d776) | 32 | 27.295 s | 1.369 s | 68.32 | 62.94 |
| `20261021T011248Z` (dcf8d25) | 32 | 31.18 s | 1.44 s | 63.53 | 62.42 |
| `20261021T021856Z` (fc624a0) | 32 | 27.42 s | 1.58 s | 71.85 | 69.32 |

All MEASURED-host and unbarred (D41 parked); different host loads make these a shape check, not a speed comparison.
TTFT collapses 17-20x because queue wait disappears. Aggregate tokens/s does **not** improve
(-8% at 8/32 clients, -19% at concurrency 1: 70.19 -> 56.49 tok/s, 34.7 -> 42.5 ms TTFT). Batching
here is a latency/fairness change; the throughput upside needs device batch attention. Outputs are
identical across modes (3 fixed probes + 7 wire probes).

Chunked prefill: one `run_hybrid_sequence_chunk` per prompt chunk, layer-major, one attention pass
and one collective per chunk. Bit-identical to the token loop at 2/4/8/16/37 prompt tokens (0
differing logit bits); bench 24.6 s vs 33.1 s at 37 tokens (1.34x). In serve it is **slower**:
TTFT 43.62 s vs 38.96 s, so it stays opt-in and the default was flipped back to the token loop.

Not wired: `ContinuousBatcher` construction in the serve loop, prefix-cache awareness in the batch
loop, cross-request prefill-chunk interleaving, batch depth above 64, and device batching
(`--continuous-batching` is refused with `--engine device`).

## 6. Paged KV

Host modules only: `sparkflash-kv/src/paged.rs` (allocator) and
`sparkflash-device/src/paged_kv.rs` (device-attention operand seam), plus 16 paging/eviction tests
and 11 addressing/twin tests. Policies: FIFO free queue; block keys are the host radix tree's
FNV-1a `block_hash`; prefix sharing per complete 64-token block; a last-reference page keeps its
hash until reused; one 128-slot SWA tail per running sequence; preemption returns pages and tail;
truncation must end on a block boundary; fork shares the donor's complete blocks; refusals are
named (`Capacity`, `TailCapacity`, `PagingError::Invariant`, ...). `check_invariants` covers free
queue uniqueness, block-table length/duplication, reference counts, page state exclusivity and tail
disjointness, and is tested with deliberately corrupted state.

**Not serving backing.** The engine still keeps per-sequence `SequenceState` attention/KV;
`paged_serve`/`PagedSequenceState` is a test seam. The S3 gate ("format-2 paged KV in serve") is
**not met**. The device seam exposes `source_entry(sequence, layer, logical)` ->
`(page, entry_in_page, pool_index, byte_offset)` and absolute-position staging for the
`flash_mla_csa2_sm121` operands; device numerics are unverified. TP=4 has its own state, no paged
KV.

## 7. Deployment

- **Image**: two-stage `Dockerfile`; PID 1 is the server via `exec`; UID/GID 10001; vendored
  tokenizer at `/opt/sparkflash/reference/tokenizer.json`; `HEALTHCHECK` on `/health`, readiness
  is `/readyz` for the orchestrator (never point the container healthcheck at it); API key is an
  env var, never a build arg. Image not built in-tree — the Dockerfile is a definition.
- **Four-node launch**: `scripts/serve-four-node.sh print|launch|status|stop`, one process per
  rank over ssh. Rank command: `serve --checkpoint <ckpt> --tokenizer <tok> --tp4-rank R
  --tp4-world-size 4 --tp4-peers <ip0..ip3> --tp4-base-port 29601 --tp4-control-port 29701`;
  rank 0 binds `0.0.0.0:8000`, ranks 1-3 bind `127.0.0.1:0` (no HTTP). `print` is offline and
  contract-tested. The launch itself is only dry-run to date.
- **Device token windows**: `--device-token-dir` installs a claimed four-rank window;
  `--device-token-live` follows a parent directory and re-reads on run-id/mtime change;
  `--device-token-derived-claim` selects the derived (rehearsal) claim form. Responses carry
  `x-sparkflash-token-source` and `x-sparkflash-token-claim`; every other request falls back to
  the host twin. Served device bodies are UNMEASURED.
- **Drain**: SIGTERM/SIGINT -> `/readyz` 503 and `sparkflash_serving = 0` immediately, then after
  `SPARKFLASH_SERVE_DRAIN_DELAY_MS` (default 0) the listener closes; accepted connections finish
  within `SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS` (default 30000). Exit 0 on clean drain, 1 on
  `drain_timeout`. Orchestrator grace must exceed delay + timeout.
- **Cancellation**: client disconnect closes the reply channel; the driver treats the failed send
  as terminal and releases the slot. No separate cancel endpoint.
- **Backpressure**: full queue -> HTTP 503 `service_unavailable`, internal code `refused`; waiting
  gauge rolled back and `sparkflash_requests_rejected_total` incremented. No Retry-After.
- **Engram**: `--engram-dir` reads the packed NVMe store (264 B rows: 256 weight + 8 scale) via
  `O_DIRECT` windowed pread; two range requests per Engram layer per chunk, one batched row read
  per owning rank file, no cross-request cache. Read-overhead measurement is window work.

## 8. Metrics

Exported from the rank-0 HTTP port, unlabelled except where noted:

| family | type | note |
| --- | --- | --- |
| `sparkflash_requests_total` | counter | |
| `sparkflash_prompt_tokens_total` | counter | |
| `sparkflash_generation_tokens_total` | counter | engine work, not wire usage (a cancelled request can generate unreported tokens) |
| `sparkflash_global_kv_hit_tokens_total` | counter | |
| `sparkflash_swa_tail_cache_hits_total` | counter | |
| `sparkflash_requests_rejected_total` | counter | queue-full refusals |
| `sparkflash_num_requests_running` | gauge | admitted into execution; idle is exactly 0 |
| `sparkflash_num_requests_waiting` | gauge | accepted, not yet admitted |
| `sparkflash_serving` | gauge | 1 accepting, 0 draining |
| `sparkflash_time_to_first_token_seconds` | histogram | 22 vLLM buckets + `+Inf`; from `Job.enqueued_at` |
| `sparkflash_engram_lookups_total`, `sparkflash_engram_hit_rate` | gauge | label `layer` |
| `sparkflash_uma_free_gib` | gauge | label `rank` |
| `sparkflash_trace_active_scope_mask` | gauge | |

Accounting is enforced by placement: every admitted request passes exactly one terminal branch
(including panic), and decrements saturate. Census: 5,286/5,286 expected generation tokens and
91/91 TTFT samples in both modes (run `20261021T011248Z`), repeated as 5,298/5,298 and 103/103
(`20261021T021856Z`), idle gauges 0/0. TTFT `_sum` was 1590.74 s serial vs 96.51 s batched over
the same 91 samples. Deliberately absent: `vllm:` prefix-cache/GPU-cache/queue-time families and
`http_*` instrumentator metrics — omitted rather than exported as fabricated zeroes. `GET /load`
is absent.

## 9. Remaining gaps

| id | gap | owner / dependency |
| --- | --- | --- |
| S1 | constrained decoding not applied (decoder exists, no engine logits-processor seam) | serving; device claim needs a device processor |
| S2 | `response_format` on completions now refused like chat | closed host-side |
| S3 | TP=4 `n > 1` refused (one shared window) | device |
| S4 | device engine `n > 1` unproven; non-greedy 503s | device |
| S5 | `best_of` accepted as documented no-op | closed |
| S6 | streaming cache hit/miss tokens now derived | closed |
| S7 | static bearer key only (no OIDC/multi-key/rotation) | serving |
| S8 | no `http_*` router metrics, no `GET /load` | serving |
| S9 | prefix-cache, GPU KV, queue-time, swapped gauges absent | device accounting |
| S10 | unseeded `n == 1` still deterministic (seed None -> 0) | serving decision |
| D1 | paged KV + prefix cache not wired through serving | device |
| D2 | device continuous batching/chunked prefill absent | device |
| D3 | S7 needs 9/9 reference matches and `token_claimed` asserted (m1y14 and m1y34 both 2/9) | device |
| D4 | final gate: D38 vLLM A/B at concurrency 1/8/32 | QA + device |
| O1-O4 | 72 h soak not run; dashboards not checked in; image not built; no D41 bars | QA |

S1's remaining work is precise: add `allowed: Option<&TokenMask>` to the engine sampler options
(applied after penalties, before sampling), expose a token->bytes table, carry an opaque constraint
on `EngineRequest`, and replace the router's typed refusal with the compiled constraint. The
automaton refuses `enum`, `const`, unions, `anyOf`/`oneOf`, patterns and numeric bounds rather
than silently dropping them.

## 10. Parity plan

Parity is three gates, not token equality: **A** native arithmetic conformance against the vLLM
oracle with pre-declared distribution bars (top-1, top-k overlap, logprob deltas, KL; token equality
retired by D39); **B** serving behaviour (logprobs, format-2 paged KV, continuous batching +
chunked prefill, TP=4 like-for-like); **C** measured performance against the recorded vLLM TP=4
baseline on the same hardware, prompts and batch sizes.

Measurement protocol (fixed before bars): oracle = vLLM serving the released snapshot, TP=4,
`--max-logprobs 1000`, temperature 0, `prompt_logprobs` over pinned prompts; device side =
`--dump-logits` per-position JSONL + full-vocab f32 sidecar; comparison per position and
aggregate. The oracle must be verified idle before and after capture; captures at a forced +1-token
prompt length are the gate shape (the 1-token shape envelope is KL 0.025-0.235, p90 0.34-1.89 nats,
max 0.81-3.74 nats, top-100 overlap 0.68-0.83), and same-shape repeats are bit-identical.

WS-G stages: (1) quiet deployment baseline — MEASURED but on the **heretic** root
`12d58ad0...`; (2) distribution parity; (3) token accuracy/greedy agreement; (4) evaluation
parity (+-1 point). Since the served root is not the released `1bae0e10...`, the baseline is
`gate_eligible = false`; stage 2+ need either the released snapshot served (a GPU window) or an
equivalent released-root oracle capture. CI has a perf-regression gate that honestly SKIPs without a
valid baseline.

## Archived sources

serving-paged-kv.md, serving-paged-serve-wiring.md, serve-deployment.md, serve-ops-metrics.md,
serving-openai-parity-audit.md, serving-parity-plan.md, serving-logprobs-parity.md,
serving-s7-s8-plan.md, serving-s7-record-draft.md, openai-surface-gaps.md,
openai-surface-remaining-gaps.md, ws-f-api-20261021.md, ws-f-scheduler-20261021.md,
s3-s4-status-audit.md, s4-continuous-batching.md, s4-batch-chunked-prefill.md,
s4-host-continuous-batching.md, s6-engram-nvme-serving.md, prompt-logprobs-one-prefill.md,
streaming-tool-calls.md, task30-tokenizer-seam.md, ws-g-vllm-parity-plan-20261021.md

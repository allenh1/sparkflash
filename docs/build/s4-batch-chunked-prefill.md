# S4 host-side: completion batch semantics and chunked prefill (window-free)

**Status: task-40, host-side only, 2026-10-19.** Task-22 (*S4 continuous batching + chunked
prefill*, [serving-parity-plan.md](serving-parity-plan.md) section 2) is blocked on task-21 (S3,
paged format-2 device KV) and stays untouched. This page records the part of S4 that needs no
window and no device: the HTTP batch contract of `/v1/completions`, the scheduler's chunked
prefill policy with its host admission step, and the host invariant that chunking a prompt does
not change the model's output.

Nothing on this page is a performance claim. Section 5 names every measurement a window must
supply, and claims none of them.

## 1. What S4 has to mean, and what this slice can mean host-side

`serving-parity-plan.md` section 0 gate B lists "paging, continuous batching and chunked prefill,
and the same tensor parallelism" as serving behaviour. Three different things share the word
batching:

1. **HTTP batch semantics** - one request carries N prompts and answers N choices. This is a
   protocol contract, it is fully host-testable, and it is section 2.
2. **Chunked prefill** - a long prompt is prefilled in chunks, with the scheduler deciding which
   step runs which chunk and which decodes interleave. The *plan* and the claim that chunking
   preserves the result are host-testable (sections 3 and 4); the device launch sequence and its
   numerics are not.
3. **Continuous batching** - several sequences share device steps inside one engine launch. That is
   task-22, blocked on task-21's paged KV: it is device work and this slice does not simulate it.

The host path is the engine's own CPU `f32` reference (`sparkflash_engine`), driven by
`sparkflash-cli`'s serve. It runs one sequence at a time; "batch" below never means that several
sequences run concurrently on the reference path. It means the HTTP surface and the per-request
prefill policy.

## 2. `/v1/completions` batch semantics

### 2.1 vLLM's contract (evidence from the inspected vLLM image source)

The source inspected for this contract is the vLLM image's
`vllm/entrypoints/openai/completion/serving.py` (local extract under `/tmp/vllm/src/`), plus
the shared response builders it calls. The facts used here:

- **Non-streaming, N prompts.** One `CompletionResponse` is returned for the whole HTTP request.
  `choices` carries one `CompletionResponseChoice` per prompt; the choice `index` is the
  sequential choice position (with the unsupported expansion `n > 1`, the prompt index times
  `n`). Each engine request is issued with the id `{request_id}-{i}` and keeps its own
  `finish_reason` and `stop_reason`. `usage` is one aggregate block: prompt, completion and
  total tokens summed over the prompts. Per-request metrics are suppressed when the HTTP request
  has more than one prompt.
- **Streaming, N prompts.** One result generator runs per prompt and
  `merge_async_iterators(*generators)` interleaves them. Each streamed chunk carries a choice
  with `index = output.index + prompt_idx * num_choices` (so `index == prompt_idx` at
  `n == 1`), one final finish chunk per choice, and, when
  `stream_options.include_usage` is set, one final usage chunk (empty `choices`) whose usage is
  summed over the batch. Per-request metrics are only emitted for a single-prompt request.
- **Echo and logprobs are per prompt.** With `echo`, each choice's `text` is that prompt's text
  plus that prompt's generated text, and the per-choice prompt-logprob array describes that
  prompt.

`n` is not part of this server's schema (`CompletionRequest` has no such field), so the
choice-index arithmetic reduces to the prompt index. A nested prompt list (`int[][]`, a batch of
token-id prompts) is also not in the schema; a token-id batch is expressed as one `int[]` prompt.
Both are recorded as deviations, not as parity.

### 2.2 What the server now does

`crates/sparkflash-api/src/server.rs`:

- `CompletionPrompt { text, tokens }` normalizes the three accepted prompt forms. `Many(vec![])`
  is a 400 with code `invalid_prompt` and param `prompt` (vLLM refuses an empty list rather than
  answering a zero-choice body). The token-id form decodes through the tokenizer seam for `echo`
  and the summary, but the engine gets the caller's ids unchanged.
- One `EngineRequest` per prompt. The response id stays the HTTP id for a single prompt, and
  becomes `{id}-{i}` for a batch, matching vLLM's per-engine request ids. `stop`, `logprobs`,
  the sampling spec and the trace directive are shared by every prompt of the batch;
  `max_tokens` is each prompt's own completion cap.
- Every request is validated before the first one is driven, so a batch that the engine cannot
  serve fails as a 400/typed error before any work.
- **Non-streaming:** the prompts are driven in order, into `Vec<EngineOutput>`; the response body
  has one choice per output (index = prompt position, per-choice `finish_reason`, echo text and
  `prompt_logprobs`), one aggregate `usage`, and the aggregate
  `x_deepseek_spark_stats` extension. `Metrics::record_request` is called once for the HTTP
  request with the aggregate usage and plan, which is vLLM's "one request" accounting.
- **Streaming:** one `EngineEvent` stream per prompt, tagged with the prompt index, merged with
  `futures::stream::select_all` (`merge_async_iterators`' job). All per-prompt stream state
  (logprob buffering, echo, `text_offset` bookkeeping) is indexed by the prompt's choice index.
  Each choice gets its own finish chunk; the usage chunk is emitted once at the end with the
  summed usage and, for a batch, the aggregate stats. A single-prompt stream keeps the
  pre-task-40 usage-only chunk unchanged, so the batch change does not widen that wire shape.
- `stop_reason` is emitted as `null` on every non-streaming choice (vLLM's field). The reference
  engine has no stop-sequence matching, so a non-null value would be a fabrication. A streamed
  choice has no `stop_reason` in vLLM's chunk either, so the streamed shape does not grow one.

### 2.3 The `x_deepseek_spark_stats` extension in a batch

The extension is this project's, not vLLM's, so its batch rule is ours to state: for a
single-prompt request the engine's own stats value is returned **unchanged** (the
`x_deepseek_spark_stats` object is byte-identical to before this task; the body around it gains
vLLM's `stop_reason` field as noted in 2.2), and for several prompts the counters are summed, the
rates are recomputed from the
sums (never averaged), and the wall-clock fields take the batch's worst case:

- `kv_cache.*` counters sum; `swa_tail_hit` is any prompt's hit;
  `kv_cache_hit_rate = global_kv_cached_tokens / total_prompt_tokens`.
- Engram: per-layer counters sum, `hit_rate` is recomputed from the summed hits and lookups, and
  the two means are recomputed as lookup-weighted means.
- DSpark: `DSparkStepStats::from_counts` over the summed stage counts, so the acceptance ratio is
  derived from the batch's counts.
- Transport: counters and MB sums.
- Timing: `ttft_ms` is the slowest prompt's (a batch's TTFT is its last first token), throughput is
  total tokens over the summed per-prompt seconds, and `min_rank_uma_free_gib` is the minimum.

### 2.4 Deviations this slice records

| deviation | why | status |
| :--- | :--- | :--- |
| `n` / `best_of` / `suffix` / `user` absent from the schema | never implemented; task-20-scope freeze | unchanged, documented |
| nested `int[][]` prompt list refused | schema has one token-id form | unchanged, documented |
| `stop_reason` always `null` | reference engine has no stop-sequence matching | documented, vLLM field present |
| stop sequences are accepted but not matched by the reference engine | unchanged from task-20 | unchanged |
| per-request metrics not suppressed for a batch | this server's metrics are request-counted once with aggregate values; vLLM skips per-engine-request metrics | documented extension behaviour |
| batch prompts are driven sequentially on the host | reference engine is one sequence at a time | host shape, not a device claim |

## 3. Chunked prefill as the scheduler's prefill policy

`sparkflash_scheduler::prefill::ChunkedPrefillPlan` (new) is the policy object:

- `plan(prompt_tokens, admitted_step, config)` derives the chunk size and the inter-chunk decode
  interval from `SchedulerConfig` (`chunk_tokens` from `PREFILL_CHUNK_CHOICES`,
  `prefill_decode_interval`) by building the existing `bypass::PrefillTimeline`, so the chunk
  schedule keeps one source of truth. `chunks()`, `ranges()` (contiguous `Range<usize>` token
  windows, only the last short), `last_chunk_tokens()`, `chunk_step(i) = admitted_step +
  i * (decode_interval + 1)`, `last_chunk_step()`, `first_token_step()`.
- `with_chunk_tokens(...)` is the explicit-chunk host form: it lets the host invariant tests use
  prompts shorter than the smallest configured chunk (4096) without pretending the device policy
  allows that size. `plan` remains the policy entry point.
- `admitted_step` is the host-testable admission step: for a request admitted against a running
  background prefill, `bypass::admit` supplies it and the request's chunk steps then interleave
  with the background prefill's own stride. With `decode_interval == 0` (`with_chunk_tokens`)
  the chunks run back to back on consecutive steps, which is exactly what the host executor does.

The host executor is `sparkflash-cli`'s `prefill_chunked(engine, state, prompt, plan)`
(`crates/sparkflash-cli/src/serve.rs`): it walks `plan.ranges()` and calls
`InferenceEngine::prefill` on the same `SequenceState` for each chunk, returning the last chunk's
`StepOutput`. `run_request` uses `ChunkedPrefillPlan::plan(len, 0, &SchedulerConfig::default())`;
for any prompt at or under 4096 tokens (every current test and every current operator prompt) that
is one chunk and the call is exactly the old whole-prompt prefill. The prompt-logprobs pass is
unchanged: it already steps the prompt one token at a time, which is the degenerate chunk size 1,
so it is covered by the same invariant.

The device execution of this policy - which launch runs which chunk, how the inter-chunk decode
window is realized on device, how the paged KV sees chunk boundaries - is task-22/task-21 work and
is not claimed here. `docs/build/phase*-` and `window-runbook.md` own the window side.

## 4. The host invariant: chunking does not change the output

The claim proven on the host f32 path:

> For one prompt id sequence and one `ModelConfig`, prefilling the prompt in any contiguous
> chunking on one `SequenceState` produces the same logits, `head_input` and `next_token` as one
> whole-prompt prefill - bit for bit, not within a tolerance.

Evidence (all runnable without a GPU, vLLM or a window):

- `crates/sparkflash-engine/tests/chunked_prefill.rs`. `ModelConfig::tiny()`, the finite
  `common::FiniteSource` weights (no NaN/inf, so the bitwise comparison is not vacuous). Prompt
  lengths `[1, 2, 3, 5, window, window+1, window+3, 2*window+1]` - the tiny window is 8, so the
  ladder includes the ring wrap inside a chunk and across a chunk boundary - and every chunk size
  `1..=len`. It compares `logits` and `head_input` as `f32::to_bits()` vectors and
  `next_token` exactly, and also asserts the state holds the whole prompt.
- `crates/sparkflash-cli/src/serve.rs`, `chunked_prefill_matches_unchunked_on_the_driver_engine`:
  the same invariant on the engine the serve path actually constructs
  (`SyntheticWeights`), chunk sizes `[1, 2, 3, 4, 5, 8, 11]` for an 11-token prompt.
- `crates/sparkflash-scheduler/src/prefill.rs` unit tests: chunk coverage/contiguity, the explicit
  chunk form, the admitted-step shift, back-to-back chunks, and the refusal of an empty prompt or a
  zero chunk size.

Why this invariant and not a parity number: it makes a device chunked-prefill launch testable
against a whole-prompt host reference (compare the device chunked chain to the host
whole-prompt logits with the D40 numeric bars), instead of against a second chunked run whose
divergence would be indistinguishable from a chunking bug.

It also fixes what a divergence in chunked mode is allowed to mean. Because the host split is
arithmetically identical, **a window divergence between a device chunked chain and the
whole-prompt host reference at or across a chunk boundary is never an arithmetic effect of
chunking**: it is admissibility or launch order - which chunk ran at which step, what the
inter-chunk decode window actually executed, and how the paged KV was scheduled around the
boundary. That is the conclusion to start from, not "chunking changes the math".

## 5. Window measurements this slice does not make

Each item is a measurement the window must supply; none is claimed, estimated or bounded here.

1. **TTFT and throughput parity against the recorded vLLM TP=4 baseline** at the pinned prompt
   suite and batch sizes (1, 4, 16 at minimum), after D40's perf bars are declared. The host path
   measures only one CPU request at a time and reports no figure.
2. **Device chunked-prefill launch sequence and numerics**: which kernel launch runs chunk `i`,
   how the inter-chunk decode step is realized, and the device logits at chunk boundaries versus
   the whole-prompt host reference under the D40 bars - including chunk boundaries that cross the
   attention window wrap.
3. **Continuous batching (task-22)**: concurrent sequences sharing a device step, the scheduler's
   admission decisions on device, and numerics across batch sizes. Blocked by task-21 (paged
   format-2 device KV); not simulated here.
4. **Paged-KV behaviour under chunked prefill and batch**: page allocation/eviction for a batched
   sequence set, and whether the KV hit-rate plan (`PrefixMatchPlan`) survives chunking. Task-21.
5. **Per-choice stop-sequence finish reasons on device**: the reference engine has none, so
   `finish_reason`/`stop_reason` parity for stop-string matches remains unmeasured.
6. **Aggregate extension-stats accuracy on device**: the recomputed Engram means, TTFT-max and
   throughput-over-summed-seconds rules have only been exercised with host counters.
7. **Streamed chunk granularity and `text_offset` under a real batched scheduler**: the merged
   stream shape is tested against the mock/reference engine; the device event cadence is not.
8. **Metrics-endpoint parity**: that one HTTP request with N prompts increments
   `requests_total` once and adds the aggregate prompt/KV counters is host-tested; the same
   values under device transport/Engram counters are not.

## 6. Out of scope / non-goals

- task-21 (S3, paged format-2 device KV) and task-22 (S4 continuous batching + chunked prefill on
  device) are untouched and remain blocked as recorded on the board.
- No frozen protocol document from task-20/task-30 is edited; this page and code comments are new.
- No MR in this slice: the branch is handed to the Lead for the next batch fold.

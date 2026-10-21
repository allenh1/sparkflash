# Serving ops metrics: vLLM-shaped request-lifecycle families

Status: **IMPL host** (task-14, commit `dcf8d25`; re-measured with the schema-/3 census in
task-15, commit `fc624a0`), with measured accounting censuses under
`/home/user/ev-session-20261020/serve/`:

- task-14: `serve-s4-batching-census-20261021T011248Z` (schema
  `sparkflash-serve-s4-batching-census/2`, binary sha256 `036e7b05...`) - the four families and
  the accounting proof.
- task-15: `serve-s4-batching-census-20261021T021856Z` (schema
  `sparkflash-serve-s4-batching-census/3`, binary sha256 `4f53a655...`) - same accounting plus a
  prompt-logprobs one-pass probe; see section 5a and
  [prompt-logprobs-one-prefill.md](prompt-logprobs-one-prefill.md).

Everything in this document is measured on the **host reference CPU engine** with the synthetic
tiny plan (128-row vocabulary, non-finite logits that pin token id 0), labelled
**"host path, synthetic tiny plan (vocab 128), UNBARED / diagnostic"**. It is API-shape,
wiring and accounting evidence. It is **not** a device result and **not** a vLLM comparison.

S7 ops-skeleton additions (branch `task/serve-n1-ops`) are documented here but not part of the
task-14/15 census claim: two unlabelled families, `sparkflash_serving` (drain state gauge) and
`sparkflash_requests_rejected_total` (bounded-queue refusals), plus the drain/readyz wiring
described in [serve-deployment.md](serve-deployment.md). The census paragraphs below keep their
original "four families" wording because that is what those runs measured.

## 1. The exported request-lifecycle families

vLLM exposes request-lifecycle metrics from its engine stats. SparkFlash keeps the project's
`sparkflash_` prefix (no duplicate `vllm:` names) and documents the exact mapping. The first
four rows are the S4 parity families; the last two are the S7 ops-skeleton additions (drain state
and backpressure) and have no vLLM equivalent, which is stated in their help strings:

| family | type | vLLM equivalent | definition |
| :--- | :--- | :--- | :--- |
| `sparkflash_num_requests_running` | gauge | `vllm:num_requests_running` ("Number of requests in model execution batches.") | requests admitted into execution: the serial engine loop's current request, or a `BatchedRequest` holding an active batch slot |
| `sparkflash_num_requests_waiting` | gauge | `vllm:num_requests_waiting` ("Number of requests waiting to be processed.") | requests accepted into the driver's bounded job queue but not yet admitted, including jobs already read into the batch scheduler's pending deque |
| `sparkflash_generation_tokens_total` | counter | `vllm:generation_tokens_total` | every generated token id, incremented at the same point the response's `completion_tokens` is bumped |
| `sparkflash_time_to_first_token_seconds` | histogram | `vllm:time_to_first_token_seconds` ("Histogram of time to first token in seconds.") | seconds from driver queue entry to the first generated token; one sample per request that generated at least one token |
| `sparkflash_serving` | gauge | none (drain-aware addition; vLLM's `ServerStatus` is not a metric) | 1 while accepting requests, 0 from the moment the drain signal is received; the same atomic flag that turns `/readyz` 503 |
| `sparkflash_requests_rejected_total` | counter | none (backpressure addition) | requests refused before engine admission because the bounded job queue (64) was full; increments exactly where `request_enqueue_refused` rolls the waiting gauge back |

**Labels.** None of the six families above carries a label except where noted: the four S4 rows
are unlabelled scalars or a histogram, `sparkflash_serving` and
`sparkflash_requests_rejected_total` are unlabelled, and the older process families keep their
documented labels (`sparkflash_engram_lookups_total` and `sparkflash_engram_hit_rate` by
`layer`, `sparkflash_uma_free_gib` by `rank`). Nothing invents a per-model or per-rank label on
the request-lifecycle families, so a dashboard can sum them without a grouping decision.

Definition details that the code and the census pin down:

- **Running** is admission, not HTTP acceptance: a request waiting in the channel is *waiting*,
  and it becomes *running* at the instant the engine thread admits it (serial dequeue, or a batch
  slot opening). Idle is exactly 0/0.
- **Waiting** includes requests the batch scheduler has dequeued into its pending list but not yet
  admitted, so it answers "accepted but not executing" for both modes. A **refused** enqueue
  (queue full, typed 503 path) is rolled back immediately: a request that never entered the queue
  is never exported as waiting, and the refusal increments `sparkflash_requests_rejected_total`.
- **Serving** turns 0 on the drain signal, before the listener stops accepting (the pre-stop delay
  `SPARKFLASH_SERVE_DRAIN_DELAY_MS` is exactly that window). It is a gauge rather than a counter
  so a scrape always answers the readiness question, and it is fed by `bind_and_serve` in
  [serve.rs](../../crates/sparkflash-cli/src/serve.rs), not by the engine thread.
- **Generation tokens** counts engine work, not wire usage: a request that fails after generating
  some tokens contributes those tokens even though no usage is reported to its client. Prompt
  tokens are not counted.
- **TTFT** is measured from `Job.enqueued_at`, captured immediately before the channel send.
  It therefore includes the scheduler queue wait, the prefill and the first-token decode, and
  excludes HTTP parsing and tokenization. The existing wire field
  `x_deepseek_spark_stats.timing.ttft_ms` keeps its old definition (measured from the start of
  `run_request`, i.e. after dequeue), so no wire field changes meaning; the histogram is the
  larger, client-visible number.
- **Buckets** are vLLM's `TIME_TO_FIRST_TOKEN_BUCKETS` exactly (22 explicit bounds from 1 ms to
  2560 s; the encoded histogram adds `+Inf`). The help text names the vLLM semantics so the
  mapping is discoverable from a scrape.

## 2. How the driver feeds them

The exporter (`Metrics` in [metrics.rs](../../crates/sparkflash-api/src/metrics.rs)) owns a
private Prometheus registry. The router already exported request/prompt-cache families, but the
request lifecycle belongs to the **engine driver**, so `sparkflash serve` now builds the exporter
before the engine thread, passes it into the thread, and hands the same registry to the router:

- `AppState::new(engine)` still creates a fresh registry, so router-only tests keep their
  isolation; `AppState::with_metrics` ([state.rs](../../crates/sparkflash-api/src/state.rs#L52))
  replaces it.
- `ReferenceDriver::metrics()` ([serve.rs](../../crates/sparkflash-cli/src/serve.rs#L1341)) is
  called before the driver moves into the engine; the state then serves that registry
  (serve.rs#L4273 for the reference path, serve.rs#L4380 for the TP=4 path).

| lifecycle event | hook | site |
| :--- | :--- | :--- |
| accepted into the driver queue | `request_enqueued` | serve.rs#L1389 (reference), #L3794 (TP=4) |
| enqueue refused (queue full) | `request_enqueue_refused` | serve.rs#L1391, #L3796 |
| serial admission | `request_admitted` | serve.rs#L2318 |
| serial terminal (Ok/Err/panic) | `request_finished` | serve.rs#L2355 |
| batch slot admission | `request_admitted` | serve.rs#L2735 |
| batch terminal (Done/Err/panic) | `request_finished` | serve.rs#L2750, #L2755, #L2763 |
| generated token id | `record_generation_tokens(1)` | serve.rs#L3404 (serial), #L2663 (batch), #L4051 (TP=4) |
| first generated token | `observe_time_to_first_token` | serve.rs#L3366, #L2605, #L4018 |

The balance invariant is enforced by placement, not by sampling: every admitted request passes
through exactly one terminal branch in the same drop-and-forward flow, including the panic path of
the engine call, so `running` returns to zero. All decrements are saturating
(`decrement_gauge`), so a future unbalanced caller cannot export a negative or phantom request.
The TP=4 feature-gated rank-0 loop feeds the same hooks; it is instrumented, but this host census
does not exercise it (see section 7).

## 3. What is deliberately not exported

Honest gaps, recorded so their absence is not mistaken for parity:

- `vllm:` **prefix-cache hit/query counters**: the host reference path does not maintain the
  device prefix-cache accounting those families describe, and the existing project families
  (`sparkflash_global_kv_hit_tokens_total`, `sparkflash_swa_tail_cache_hits_total`,
  `sparkflash_prompt_tokens_total`) already cover the cache accounting this path has. No
  `vllm:num_prefix_cache_*` names are invented.
- **GPU KV-cache usage and queue-time histograms**: there is no device KV pool on the host
  reference engine, so any value would be fabricated.
- **Swapped/preempted request gauges**: the host scheduler has no preemption; omitted rather than
  exported as a misleading zero.
- **`http_*` instrumentator-style router metrics and GET /load**: vLLM serves these from its
  OpenAI-facing middleware; our /metrics is the API process's registry. That is a host-fixable
  follow-up, out of this increment.
- No `vllm:` prefix duplication: the mapping table above is the compatibility statement.

## 4. Batched wire probes

Batching may interleave engine work, but it must not change the API payload. The census runs the
same seven client requests against the serial and the `--continuous-batching` servers and records
status, content type, payload shape (keys, types and array lengths) and stable values (everything
except per-process `id`, wall-clock `created` and the `timing` block):

| probe | expected | result (both modes identical) |
| :--- | :--- | :--- |
| buffered `prompt_logprobs` 0 and 1 | typed per-position payload, 3 completion tokens | 200, shape and values identical |
| buffered stop string | stop cuts the text, `finish_reason` stop, 1 token | 200, identical |
| SSE stop string | same stop behaviour on the stream, `[DONE]` | 200, frames identical |
| SSE sample `logprobs` 2 | logprob objects per token frame | 200, frames identical |
| SSE explicit `prompt_logprobs` 1 | vLLM's refusal: typed 400, no stream | 400, error payload identical |
| SSE `prompt_logprobs` 0 | accepted and dropped (vLLM's rule; chunk has no field) | 200, frames identical and no `prompt_logprobs` field |

The wire-level e2e tests in
[serve_http.rs](../../crates/sparkflash-cli/tests/serve_http.rs) pin the same behaviour:
`metrics_report_the_ops_families_the_driver_feeds` (#L733), the batched gauge test
`batched_metrics_report_two_running_requests_and_one_waiting_slot` (#L873) and
`batched_wire_probes_match_the_serial_payload_for_logprobs_and_stop` (#L940). Task-15 keeps the
same seven probes unchanged and adds, beside them, the one-pass prompt-logprobs probe described in
section 5a (it checks every prompt position and records diagnostic timing; the seven wire probes stay
the API-shape contract).

## 5. Measured census

Command (defaults: streaming ladder 1/8/32 clients x 2 requests, 64 max tokens; three fixed
buffered probes with 8 max tokens), run on a clean tree:

    cd /home/user/wt/serve-parity
    python3 scripts/serve-s4-batching-census.py

Run identity and manifests:

- commit `dcf8d25` (branch `task/serve-openai-parity`, `git status --porcelain` empty in the
  artefact), binary `target/debug/sparkflash` sha256
  `036e7b05086d4ea6bab125e97ef81dab3af4b871df89f6af0c00a44cbf9d5843`
- artefact set `serve-s4-batching-census-20261021T011248Z` under
  `/home/user/ev-session-20261020/serve/`; its `.sha256` file pins the binary, JSON,
  Markdown and both server logs:
  json `401acf790e57a2b08e1118f5142fb700f28f204b7b7b27a7f9778c9e8eb4a56e`,
  markdown `747ae150b7ee9ba91f9bfa9f10b893d59a8470a90afdfe74507e672496025814`,
  serial log `6f926770d11054386eabd2f91defe5f3cba26da79bf5c60d24709aac82ac79ae`,
  batched log `95311805359f7adaac71bbb33338d7d5ded2fe19b1bc21c4f90e738db187b904`
- host loadavg over the run 8.80/9.34/8.82 on 88 CPUs (loopback host census; no benchmark
  controls, no frozen floor).

Ops-metrics accounting proof (identical in both modes because the workload is identical):

| mode | expected generation tokens | observed counter delta | expected TTFT samples | observed count delta | idle running | idle waiting | ok |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| serial | 5286 | 5286 | 91 | 91 | 0 | 0 | true |
| batched | 5286 | 5286 | 91 | 91 | 0 | 0 | true |

- All four families are present in both modes, help text identical, and the TTFT bucket bounds are
  identical (the 22 vLLM bounds plus `+Inf`).
- The expected values are computed from the responses themselves: fixed probes 24 tokens + wire
  probes 14 tokens + ladder 5248 tokens = 5286; 3 fixed + 6 token-producing wire probes + 82
  accepted ladder streams = 91 samples.
- The TTFT `_sum` delta is itself informative: 1590.74 s serial versus 96.51 s batched over the
  same 91 samples, i.e. the histogram records the queue wait the ladder measures (32-client TTFT
  p50 31.18 s serial -> 1.44 s batched, aggregate client tokens/s 63.53 -> 62.42). Those ladder
  numbers are the host census's existing S4 measurement, reproduced on this commit; they are not
  a throughput claim.
- Fixed buffered probes remain text-identical across modes, and all seven wire probes are
  identical in shape and stable values.

### 5a. Task-15 re-run with the prompt-logprobs one-pass probe

The schema-/3 census re-ran the same host A/B workload with a prompt-logprobs probe added: a
buffered `prompt_logprobs: 1` request against the same request without prompt logprobs, six
repetitions each, on an 8-token prompt (`max_tokens: 1`). Every position is checked (one entry per
prompt token, position 0 null, each later entry carries the prompt token itself with a numeric
`logprob`, `rank >= 1` and a decoded string, and the payload is stable across repetitions).

- commit `fc624a0` (branch `task/serve-openai-parity`, clean tree in the artefact), binary
  `target/debug/sparkflash` sha256
  `4f53a6551311d6c84ff29a2e4997d65f373278c912c066d63ae5eed854f18e45`; artefact set
  `serve-s4-batching-census-20261021T021856Z` (`.sha256` pins the binary, JSON, Markdown and both
  server logs; json `36138965fb783b14f0f7d5991c79e2a5db66b2650f414137b300af9ec8f8a8a4`,
  markdown `e54ef8fc32efde459ffeea607ad3f36352c439a93280e37a3fb5a1faaaf59e34`); host loadavg
  5.19/6.80/6.99 on 88 CPUs (UNBARED / diagnostic).
- Both modes: 8 positions, 7 scored, position 0 null, every position correct, stable across runs,
  payload identical across modes. Diagnostic p50 client elapsed (loopback included, tiny plan):
  serial 49.29 ms -> 52.23 ms with prompt logprobs; batched 33.02 ms -> 35.87 ms. This is a
  repeatability record, not a speedup or latency claim: the modes ran at different times and their
  baselines differ, so they are not compared with each other.
- Accounting now includes the probe's 12 token-producing requests: 5298 expected = 5298 observed
  generation tokens and 103 expected = 103 observed TTFT samples in both modes, idle gauges 0/0;
  all four families present with identical help text and bucket bounds.
- The ladder reproduces the S4 shape (32-client TTFT p50 27.42 s serial -> 1.58 s batched, client
  tokens/s 71.85 -> 69.32); the fixed probe texts and all seven pre-existing wire probes remain
  identical across modes. Host load differs from the task-14 run, so the two censuses are a shape
  check, not a comparison.

## 6. Tests and gates

- Unit: `ops_families_export_vllm_shaped_state` in
  [metrics.rs](../../crates/sparkflash-api/src/metrics.rs#L345) pins the exported names, help
  text, gauge transitions, refusal rollback, the backpressure counter, the serving gauge, histogram
  count/sum/buckets, dropped non-finite samples, and that an unbalanced decrement cannot go below
  zero. The drain behavior itself is covered end to end by
  `sigterm_drains_in_flight_streams_and_turns_readiness_off` in
  [serve_http.rs](../../crates/sparkflash-cli/tests/serve_http.rs) and by the router
  `readyz` test in [router.rs](../../crates/sparkflash-api/tests/router.rs).
- Wire e2e: the three tests listed in section 4 spawn the real binary on loopback and assert
  counter/usage agreement, live running/waiting gauges (two running + one waiting under
  `--max-batch 2`), TTFT count, histogram bucket export, and serial-vs-batched payload identity.
- Census unit tests: `scripts/tests/test_serve_s4_batching_census.py` covers the ops-metrics
  parser, bucket ordering, volatile stripping, shape comparison, wire-probe comparison, the
  prompt-logprobs payload checks and comparison, and the markdown sections.
- Prompt-logprobs one-pass tests: the engine row-equality/bounds tests, the driver old-vs-new
  equality test and the new buffered e2e probe are listed in
  [prompt-logprobs-one-prefill.md](prompt-logprobs-one-prefill.md) section 4.
- Gates run on the task-14 commit: `cargo fmt --all -- --check`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo clippy -p sparkflash-cli --all-targets --features
  tp4-serve -- -D warnings`, `cargo test -p sparkflash-api`, `cargo test -p sparkflash-cli`
  and `python3 -m unittest discover -s scripts/tests` all green.
- The same gates were re-run green on the task-15 commit `fc624a0` (250 python tests); no node,
  GPU or device action was taken in either task.

## 7. Non-claims and what remains

- These metrics prove wiring and accounting on the host reference engine. They do **not** say
  anything about device batch attention, paged device KV, device TTFT/throughput, or vLLM's
  scheduler behaviour; the census compares our serial mode with our batched mode on one host.
- The TP=4 feature-gated path feeds the same hooks (rank-0 queue and generation), but no TP=4
  run was performed here; its families are not part of the measured claim.
- Host-fixable follow-ups: `http_*` instrumentator-style request counters/latencies and GET
  /load; prefix-cache hit/query counters when the host path has an accounting source for them;
  scheduler policy work. Prompt-logprobs one-prefill landed in task-15
  ([prompt-logprobs-one-prefill.md](prompt-logprobs-one-prefill.md)); the remaining prompt-logprobs
  item is the TP=4 all-position rank contract, which is documented as a refusal rather than
  approximated.
- Device-M1 window: every metric that describes device batch execution, and the S4 gate itself.

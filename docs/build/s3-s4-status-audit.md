# S3/S4 serving-parity status audit (task-15, host-side)

**Status: 2026-10-21, task-15 with the task-14 increment, branch task/serve-openai-parity, based on
c3b8ae9.** One-page audit of the S3/S4 checklist in
[serving-parity-plan.md](serving-parity-plan.md) against the current code, plus what these tasks
closed and what remains. Evidence lines are current at commit fc624a0 (task-15) unless a row names
an earlier task commit (task-14 dcf8d25, task-13 530d776). No device number is claimed; items that
need the device M1 token or a device window are marked **window**.

## 0. What this task did

Made the S4 host continuous-batching scheduler reachable from the reference HTTP path behind
--continuous-batching / --max-batch N (off by default), and measured it against the serial path in a
single A/B census. The full record, numbers, limitations and reproduction are in
[s4-host-continuous-batching.md](s4-host-continuous-batching.md). Tests and the census were green
at 530d776; the default serial path is byte-identical to before this task.

Task-14 then added the vLLM-shaped request-lifecycle ops families (running/waiting gauges,
generation-token counter, TTFT histogram), fed by the engine driver on both the serial and batching
paths, and extended the S4 census to schema /2 with a /metrics accounting proof and batched wire
probes for prompt_logprobs and stop. The design record and the measured evidence are in
[serve-ops-metrics.md](serve-ops-metrics.md); the census that backs it is
serve-s4-batching-census-20261021T011248Z on commit dcf8d25: 5286 expected = 5286 observed
generation tokens and 91 expected = 91 observed TTFT samples in both modes, idle gauges 0/0, all
seven wire probes identical across modes.

Task-15 then replaced the O(n) token-at-a-time prompt-logprobs pass with one pass over the prompt:
the engine returns the LM-head rows of selected positions from the same forward that consumes the
chunk (`PositionLogits` / `prefill_with_position_logits`), and the driver asks only for the rows
that score the prompt tokens, in bounded contiguous spans. The payload shape, decoded strings and
final logits/head_input bits are compared with the retained token-at-a-time reference, and the
engine refuses a position it cannot serve from the pass rather than answering with a different
number. Design, equality evidence and non-claims are in
[prompt-logprobs-one-prefill.md](prompt-logprobs-one-prefill.md); the measured evidence is
serve-s4-batching-census-20261021T021856Z on commit fc624a0: schema /3, both modes' prompt-logprobs
payloads correct and identical, 5298/5298 generation tokens, 103/103 TTFT samples and idle gauges
0/0 in both modes.

## 1. S3 - paged format-2 device KV in the serving path (task-21, task-44)

| checklist item | state | evidence | remaining |
| :--- | :--- | :--- | :--- |
| paged block/table storage and invariants | **IMPL host**, unit-tested | sparkflash-kv/src/paged.rs: PagedGeometry:53, BlockTable:100, PagedKvCache:239, check_invariants:604 | device-resident pool and device paging invariants under real traffic are **window** |
| serving-side sequence wiring seam | **IMPL host**, crate-internal | serve::paged_serve serve.rs:4396; PagedSequenceState:4512; DenseSequenceState:4708 | not used by the HTTP engine/KV path, which still keeps per-sequence host/device attention state; wiring the engine's KV to the paged pool is S3 **window** work |
| format-2 CSA2 paged read attention | kernel planned/documented, device numerics unverified | design in serving-paged-kv.md; CSA2 entry is in the kernel probe set (syntax/static-assert only) | device numerics vs the paged mirror and vLLM-style paging invariants: **window** |
| prompt logprobs at prefill | **IMPL host (task-15 serial, task-18 TP=4)**: one pass over the prompt in bounded spans, every position's distribution computed once; on the Phase-1 TP=4 path each rank screens its local head shard and the bounded host merge reports a position only when its f32 order is provable | serial: serve.rs prompt_logprobs_pass/spans/row budget, engine forward/mod.rs prefill_with_position_logits; TP=4: engine prompt_ranks.rs, engine head_rank_logits and chunk position_head_inputs, tp4_host.rs prefill_prompt_logprobs, serve.rs control frame + payload builder; payload and bit equality against the serial host path; serial census 20261021T021856Z (schema /3), TP=4 census tp4-prompt-logprobs-census-* and [prompt-logprobs-one-prefill.md](prompt-logprobs-one-prefill.md), [tp4-prompt-logprobs-rank-contract.md](tp4-prompt-logprobs-rank-contract.md) | device serving-path measurement: **window**; the TP=4 path refuses (never approximates) whole-vocabulary -1, top counts above the bounded budget, certified upper-bound entries, a position whose f32 order cannot be proven, and broken shard layouts (criterion documented in the contract) |
| paged KV in TP=4 | **MISSING** | TP=4 path has its own state (serve.rs:3627, 3815, 3876) | S3/S5 **window** work |

Verdict: S3 is a host foundation plus test seam, not a serving-path paged KV. The engine still owns
SequenceState attention/KV per sequence, so the S3 gate item "format-2 paged KV in serve" is **not
met**; it is blocked on the device KV backing (task-21, device M1 token).

## 2. S4 - continuous batching + chunked prefill (task-22)

| checklist item | state | evidence | remaining |
| :--- | :--- | :--- | :--- |
| continuous batching reachable from serve | **IMPL host (this task)**, opt-in | serve.rs:2252 engine_thread, 2275 selects batch loop, 2647 run_continuous_engine_loop, 2371 BatchedRequest, 176/169 flags | always-on device batching, batch attention and batch-depth economy: **window** |
| batching does not change output | **IMPL host**, tested | unit tests serve.rs:6001, 6032; wire e2e serve_http.rs:555; census fixed probes identical across modes | device numerics across batch sizes (host identity is proven): **window** |
| chunked prefill policy | **IMPL host**, per request, both paths | prefill_chunked serve.rs:3101; policy sparkflash-scheduler/src/prefill.rs (ChunkedPrefillPlan:300) | cross-request prefill-chunk interleaving and per-chunk batching: host-fixable follow-up; device chunked numerics: **window** |
| TTFT / throughput behaviour | **measured host proxy** only | s4-host-continuous-batching.md and the 20261021T005030Z census: 32-client TTFT p50 27.30 s serial -> 1.37 s batched, tokens/s 68.32 -> 62.94; the task-14 census 20261021T011248Z reproduces the shape: 31.18 s -> 1.44 s, tokens/s 63.53 -> 62.42 | device/vLLM TP=4 bars (task-25): **window** |
| queue/backpressure | bounded queue and typed refusals | JOB_QUEUE=64 serve.rs:169; REFUSED path + batch-aware message; batch admission <= max_batch; the accepted-vs-admitted distinction is exported as sparkflash_num_requests_waiting/running (task-14, section 3) | none for the host surface; device queueing behaviour: **window** |

Verdict: S4 host behaviour is now reachable and measured; the S4 gate itself ("numerics across
batch sizes + perf") is **not met**, because both halves of it are device measurements. The host
slice removes the "not reachable" objection and gives the window a measured baseline to beat.

## 3. Ops surface (serving parity, not S3/S4-gated but in the same checklist)

| item | state | evidence | remaining |
| :--- | :--- | :--- | :--- |
| /metrics families | **IMPL host (task-14)**: existing cluster/cache families plus the request-lifecycle ops families | metrics.rs: TIME_TO_FIRST_TOKEN_BUCKETS:35, collectors new():120-145, hooks request_enqueued:196/request_admitted:212/request_finished:221/record_generation_tokens:231/observe_time_to_first_token:241; AppState::with_metrics state.rs:52; driver hook sites serve.rs:1389,1391,2318,2355,2605,2663,2735,3366,3404; measured in [serve-ops-metrics.md](serve-ops-metrics.md), census 20261021T011248Z (5286/5286 generation tokens, 91/91 TTFT samples, idle gauges 0/0 in both modes) and re-measured in task-15's census 20261021T021856Z (5298/5298, 103/103, idle 0/0) | vllm: prefix-cache hit/query counters, GPU cache usage and queue-time histograms need an accounting source and are deliberately not fabricated; http_* instrumentator metrics and GET /load are still absent; device scheduler metrics: **window** |
| health / models card / SSE framing | IMPL | S2 audit [serving-openai-parity-audit.md](serving-openai-parity-audit.md) sections 1, 4, 8, 12 and its e2e/router tests | none for the host surface |
| logprobs / streaming surface | IMPL-b | same audit sections 5 and 12; batched wire probes for prompt_logprobs and stop are measured identical across modes (census schema /3, task-14 then task-15); prompt_logprobs now runs as one pass over the prompt with the payload and bits matched to the retained token-at-a-time reference (task-15, [prompt-logprobs-one-prefill.md](prompt-logprobs-one-prefill.md)) | TP=4 prompt logprobs (needs the all-position rank contract, refused today) and device serving-path measurement: **window** |
| 1M context (model context window) | IMPL admission; unmeasured at scale | engine.rs:67 max_seq_len (SPARK_MAX_SEQ_LEN bounded by MODEL_CONTEXT_TOKENS); server.rs:367 resolve_max_tokens_impl with the OpenAI context-length error | a real long-prefill/paged-KV run at scale: **window** |
| Engram NVMe in the serving path | IMPL host packed-store read path | serve.rs:957-1070 (packed Engram tables served through sparkflash_engram::store::DirectTable rows); design in [s6-engram-nvme-serving.md](s6-engram-nvme-serving.md) | measured read overhead/overlap in the serving path: **window** (task-24) |

## 4. What remains, split by dependency

Host-fixable, no device window (ordered by value):

1. **Scheduler policy**: prefill-chunk interleaving, decode-first/admission policy, prefix-cache
   awareness in the batch loop, batch depth above JOB_QUEUE.
2. **Router-level vLLM instrumentator surface**: http_* request counters/latencies and GET /load;
   vLLM serves these in front of the engine, our /metrics is the API registry.
3. **Remaining ops families** (section 3): vllm: prefix-cache hit/query counters, GPU KV-cache
   usage and queue-time histograms. They need an accounting source first; until then their absence
   is documented rather than exported as fabricated zeroes.
4. **TP=4 prompt logprobs all-position rank contract**: **landed host-side in task-18**
   ([tp4-prompt-logprobs-rank-contract.md](tp4-prompt-logprobs-rank-contract.md)). The Phase-1
   TP=4 path now scores every prompt position through rank-local head-shard screening, three
   bounded candidate/count/partial-sum exchanges over the forward's existing all-gather, and a
   deterministic host merge that reports the serial order only when its f32 boundary proves it and
   refuses with a typed criterion otherwise. Remaining: measurement on a real four-process serve
   deployment and the device serving path: **window**.

Done in task-14: the running/waiting gauges, generation-token counter and TTFT histogram (section
3), and the batched-path wire probes for prompt_logprobs and stop (census schema /2).

Done in task-15: prompt_logprobs one-prefill (section 1 row and section 3 row), with the retained
token-at-a-time reference as the equality contract, and the schema-/3 census probe that checks every
prompt position and records diagnostic timing.

Window / device-M1 dependent (must not be claimed from host evidence):

1. S3: device paged format-2 KV, engine KV/attention backed by the paged pool, paging invariants
   under real traffic.
2. S4: device batch attention, chunked-prefill numerics across batch sizes, device TTFT/throughput,
   vLLM TP=4 performance bars (task-25).
3. S5: TP=4 numerics and transport (task-23).
4. S6: Engram NVMe read-overhead measurement in the serving path (task-24).
5. Anything that would call the current host numbers a parity or a floor: they are UNBARED /
   diagnostic (D41).

## 5. One-line verdict

S2 accepted on the host path and the request-lifecycle ops surface is implemented and accounted on
both the serial and batching paths (task-14); prompt_logprobs now runs in one pass with the payload
and bits matched to the retained token-at-a-time reference (task-15); S3 is host foundation only
(device paged KV outstanding); S4 host continuous batching is reachable, tested and measured
(task-13) but its gate needs device numerics and device performance; S5/S6 and all device claims
remain on the device M1 token/window.

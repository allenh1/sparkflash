# S4 host continuous batching: reachable and measured (task-13)

**Status: task-13, host-side only, 2026-10-21.** Branch task/serve-openai-parity, based on
c5be114. This page records the one S4 gap this task closed: the host continuous-batching scheduler
that existed as a crate-internal seam ([s4-continuous-batching.md](s4-continuous-batching.md)) is now
reachable from the reference HTTP serve path behind an explicit flag, and its A/B behaviour is
recorded in one measured census artefact. Nothing here is a device result, a vLLM comparison, a
parity claim or a frozen floor: the numbers are host path, synthetic tiny plan (vocab 128), UNBARED /
diagnostic (D41), and the device items stay with the window/device M1 token.

A later task-14 increment (commit dcf8d25) added the vLLM-shaped request-lifecycle ops families to
both engine paths and extended this census to schema /2 with a /metrics accounting proof and batched
wire probes; that record and its measured run are in
[serve-ops-metrics.md](serve-ops-metrics.md). The 005030Z run below remains the task-13 S4
measurement; the schema-/2 run re-measures the same workload and adds the ops evidence.

## 1. The gap

The reference engine thread has always run one Job at a time
([serve.rs:2252](../../../crates/sparkflash-cli/src/serve.rs#L2252), serial loop at 2278). Concurrent HTTP
requests therefore queue behind the running request; the accepted S2 census measured 32-client
streaming TTFT p50 of 32.11 s (and this run measures 27.3 s on a busier host). S4 asks for continuous
batching, and two host seams existed from task-22/task-44 - serve::continuous
([serve.rs:5081](../../../crates/sparkflash-cli/src/serve.rs#L5081)) and serve::paged_serve
([serve.rs:4396](../../../crates/sparkflash-cli/src/serve.rs#L4396)) - but the HTTP path never used them.
This task makes the scheduler reachable and measures what changes.

## 2. What landed

- --continuous-batching on the serve subcommand: opt-in, default **off**. The default serial path,
  its outputs, the S2 census and every earlier receipt are unchanged.
- --max-batch N: default 32 ([serve.rs:176](../../../crates/sparkflash-cli/src/serve.rs#L176)), valid range
  1..=JOB_QUEUE (64, [serve.rs:169](../../../crates/sparkflash-cli/src/serve.rs#L169)). The flag is refused
  with the device engine, with the TP=4 group, with --max-batch 0, with --max-batch greater than the
  queue, and with --max-batch when --continuous-batching is absent
  ([serve.rs:601](../../../crates/sparkflash-cli/src/serve.rs#L601)).
- Scheduler: run_continuous_engine_loop
  ([serve.rs:2647](../../../crates/sparkflash-cli/src/serve.rs#L2647)) and BatchedRequest
  ([serve.rs:2371](../../../crates/sparkflash-cli/src/serve.rs#L2371)).

## 3. How the opt-in scheduler runs

The engine thread selects the batch loop after the ready handshake
([serve.rs:2275](../../../crates/sparkflash-cli/src/serve.rs#L2275)); the serial loop below it is
untouched. The batch loop:

- keeps a FIFO VecDeque of pending jobs and a Vec of active slots, admits up to max_batch pending
  jobs every round, and never blocks on a new job while requests are active - it drains try_recv and
  runs the round;
- gives every active request one scheduler step per round: a request in Prefill runs prefill_chunked
  ([serve.rs:3101](../../../crates/sparkflash-cli/src/serve.rs#L3101)) for its whole prompt and moves to
  Decode; a request in Decode emits one token and runs one engine.decode on its own SequenceState.
  Round-robin across slots is the fair service order;
- reconstructs all per-request state the serial run_request
  ([serve.rs:3217](../../../crates/sparkflash-cli/src/serve.rs#L3217)) kept: SequenceState, streaming
  detokeniser state (tokenizers::step_decode_stream, mirroring DecodeStream exactly), stop buffer,
  DSpark accounting, sample/prompt logprobs, first-token timing;
- applies each request's sampler with the non-consuming set_sampling/set_sampler_options immediately
  before the engine call, because the engine is owned by this single thread and a different slot's
  sampler must not leak into this call;
- keeps the same Reply channel contract (started/delta/done/failed) and per-request catch_unwind, so
  the HTTP/SSE surface is unchanged and one panicking request does not take its neighbours down;
- differs from the serial loop in one measured way: after a request's token budget is spent the
  serial loop runs one final decode whose output is never used; the batch loop emits the last token
  and finishes, so it does one less decode per request. This is documented in the code and listed as
  a limitation, not hidden.

Chunked prefill remains the same per-request policy call as the serial path; cross-request
prefill-chunk interleaving is not implemented (a prefill is one scheduler step, and with the default
4096-token chunk a short host prompt is one chunk). Prefill of one request therefore delays one round
of the other requests' decode by one engine prefill call - the device scheduler's chunked
interleaving is still window work.

## 4. Tests and gates (all host, all green at 530d776)

- Scheduler output identity: continuous_batching_admits_both_requests_and_keeps_their_output
  ([serve.rs:6001](../../../crates/sparkflash-cli/src/serve.rs#L6001)) - two concurrent requests finish in
  short-first order and each stream equals the serial driver's output.
- Logprobs and stop strings across batching: continuous_batching_carries_prompt_logprobs_and_stop_strings
  ([serve.rs:6032](../../../crates/sparkflash-cli/src/serve.rs#L6032)) with a concurrent filler request.
- Flag parse and refusal: continuous_batching_flags_parse_and_their_misuses_are_refused
  ([serve.rs:6959](../../../crates/sparkflash-cli/src/serve.rs#L6959)).
- Wire e2e: continuous_batching_answers_concurrent_requests_with_the_serial_paths_output
  ([serve_http.rs:555](../../../crates/sparkflash-cli/tests/serve_http.rs#L555)) - four concurrent
  completions against --continuous-batching --max-batch 4 produce exactly the serial server's text,
  completion count and "length" finish reason.
- Ops metrics under load (task-14): metrics_report_the_ops_families_the_driver_feeds
  ([serve_http.rs:733](../../../crates/sparkflash-cli/tests/serve_http.rs#L733)),
  batched_metrics_report_two_running_requests_and_one_waiting_slot
  ([serve_http.rs:873](../../../crates/sparkflash-cli/tests/serve_http.rs#L873)) and the batched wire
  probes ([serve_http.rs:940](../../../crates/sparkflash-cli/tests/serve_http.rs#L940)): 12 serve_http
  tests at commit dcf8d25.
- Gates run at 530d776: cargo fmt --all -- --check; cargo clippy --workspace --all-targets -- -D
  warnings; cargo test -p sparkflash-cli (230 lib tests plus 24 integration tests, including the 9
  serve_http tests); python3 -m unittest discover -s scripts/tests (9 tests in
  [test_serve_s4_batching_census.py](../../../scripts/tests/test_serve_s4_batching_census.py) plus the
  pre-existing script tests, exit 0). At dcf8d25 the same gates are green with 12 serve_http tests
  and 17 census-script tests.
- Census script: [serve-s4-batching-census.py](../../../scripts/serve-s4-batching-census.py).

## 5. Measured A/B census

One run per mode, same binary, same host, serial first and batched second. Streaming ladder 1/8/32
concurrent clients, two streaming completions per client, 64 max tokens, token-id prompt [1..8],
skip_special_tokens: false so the synthetic plan's only generated token (BOS) is visible and TTFT has
a text delta to measure. TTFT is the client's first non-empty text delta; end-to-end is the request
close.

- label: **host path, synthetic tiny plan (vocab 128), UNBARED / diagnostic**
- run: serve-s4-batching-census-20261021T005030Z
- commit 530d7766ca2f865fe0d7edf99c7a880060ba74c3 (branch task/serve-openai-parity); the JSON
  records two dirty entries, both documentation edits made after the commit, and pins the measured
  build by sha256, so the binary is unambiguous
- binary sha256 a65cbe86db2c40912e4fa2b4bd6a0831dbb0f6e583bc2d8ebf53844e85b65c0c
- host: 88 CPUs, loadavg start 13.54/8.92/6.93 (serial), 11.36/9.43/7.28 (batched); other work was on
  the host during the run, so the numbers are regression-tracking evidence, not a benchmark
- artefacts (sha256 pins):
  - JSON /home/user/ev-session-20261020/serve/serve-s4-batching-census-20261021T005030Z.json
    (sha256 301fe762efc491329e80bd8e5b33116f416ca39f6b0e2dfaaccea8c96f19598c)
  - Markdown /home/user/ev-session-20261020/serve/serve-s4-batching-census-20261021T005030Z.md
    (sha256 e5736afff0b48b9825a61cb6624db0189962d225803f84b82b6fc5f7d836bdf3)
  - server logs ...-server-serial.log (sha256
    ffc96125f624d4f1bc4fd4c2522086dbfe6a4f0780d2fac90e4522fe43d4bea1) and ...-server-batched.log
    (sha256 372539f7a1e949d9113f2e0de64e1f802bfedf9371cfa8d4092dc1828ed61bf8)
  - the run also writes a .sha256 file pinning the binary and all five artefacts

| concurrency | requests | tokens | serial TTFT p50 | batched TTFT p50 | TTFT p50 ratio | serial tokens/s | batched tokens/s | tokens/s ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 128 | 34.7 ms | 42.5 ms | 1.22 | 70.19 | 56.49 | 0.805 |
| 8 | 16 | 1024 | 6.281 s | 0.356 s | 0.057 | 70.74 | 65.03 | 0.919 |
| 32 | 64 | 4096 | 27.295 s | 1.369 s | 0.050 | 68.32 | 62.94 | 0.921 |

All 82 ladder requests per mode were accepted (0 refused, 0 failed, 0 panics; every SSE stream
terminated; one distinct text per rung because the plan emits one token id). Server counters moved by
sparkflash_requests_total +85 per mode - the 82 ladder requests plus the 3 fixed probes - and each
mode's log has 85 DSpark evidence lines, i.e. the same work was done in both modes.

Output identity: three fixed buffered completions (token-id prompts, temperature 0, seed 0, explicit
skip_special_tokens: false, 8 max tokens) were sent through each mode; text, finish reason ("length")
and completion count (8) are identical across modes. Batching reorders work, it does not change it on
this path.

Task-14 re-ran the same workload with the schema-/2 census on a clean tree: run
serve-s4-batching-census-20261021T011248Z, commit dcf8d25, binary sha256 036e7b05..., host loadavg
8.80/9.34/8.82 on 88 CPUs. It reproduces the shape (32-client TTFT p50 31.18 s serial -> 1.44 s
batched, aggregate client tokens/s 63.53 -> 62.42) and adds the ops-metrics accounting: 5286
expected = 5286 observed generation tokens and 91 expected = 91 observed TTFT samples in both modes,
idle running/waiting gauges 0/0, all seven buffered/SSE prompt_logprobs and stop wire probes
identical in shape and stable values across modes. The two runs are not a speed comparison (host
load differs); both are regression-tracking evidence. The measured ops evidence is documented in
[serve-ops-metrics.md](serve-ops-metrics.md).

Task-15 re-ran the same workload again with the schema-/3 census on a clean tree: run
serve-s4-batching-census-20261021T021856Z, commit fc624a0, binary sha256 4f53a655..., host loadavg
5.19/6.80/6.99 on 88 CPUs. The ladder reproduces the shape (32-client TTFT p50 27.42 s serial ->
1.58 s batched, aggregate client tokens/s 71.85 -> 69.32), all seven pre-existing wire probes and
the fixed probe texts remain identical across modes, and the new prompt-logprobs probe reports the
same correct, payload-identical one-pass result in both modes (8 positions, 7 scored, position 0
null, stable across repetitions). Its 12 requests are included in the accounting: 5298/5298
generation tokens and 103/103 TTFT samples in both modes, idle gauges 0/0. The one-pass design and
that probe are documented in [prompt-logprobs-one-prefill.md](prompt-logprobs-one-prefill.md). As
with the earlier runs, different host loads make the three censuses a shape check, not a
comparison.

Reading the numbers honestly:

- The TTFT collapse (17.6x at 8 clients, 19.9x at 32) is the queue wait disappearing: every admitted
  request starts its prefill as soon as the batch admits it instead of waiting for all earlier
  requests to finish 64 tokens. This is the S4 serving-behaviour effect the flag is for.
- Aggregate tokens/s does **not** improve: it falls ~8% at 8/32 clients and ~19% at concurrency 1.
  One CPU engine still does the same total work, and the scheduler adds round-robin switching, a
  per-op sampler switch and per-step bookkeeping. Batching is a latency/fairness change here, not a
  throughput change - the throughput upside belongs to device batch attention, which this path does
  not model.
- At concurrency 1 the batch-of-one path is slower than serial (TTFT 34.7 -> 42.5 ms, tokens/s 70.19
  -> 56.49): with a single request there is no queue to remove, only the per-op switch and the
  missing final decode. This is why the flag is opt-in and not the default.
- End-to-end p50 at 32 clients is slightly worse in batched mode (28.20 s -> 32.37 s): every request
  now gets a fair 1/32 share and starts immediately, so the last token arrives about when it would
  under serial admission, while TTFT is 20x lower. Both numbers are in the JSON; neither is
  cherry-picked.
- Host loadavg differed between the two halves (13.5 vs 11.4 at the switch, both above the 8-9
  baseline), so the throughput ratio is a same-day diagnostic, not a controlled benchmark.

What this census cannot say: anything about device batch attention, paged device KV, device TTFT or
device throughput; anything about vLLM; anything about real model quality (the synthetic plan's
logits are non-finite and pin token id 0, so the fixed probes prove wiring/identity, not numerics);
and anything about a frozen floor - it is UNBARED / diagnostic.

## 6. Remaining S4 work, and what needs device M1

Host-fixable follow-ups (no device needed):

1. Scheduler/queue metrics: **done in task-14** - the running/waiting gauges, generation-token
   counter and TTFT histogram are fed by both engine paths and accounted against responses in the
   schema-/2 (and later /3) census ([serve-ops-metrics.md](serve-ops-metrics.md)). Remaining ops families that need
   an accounting source (prefix-cache counters, GPU cache/queue-time histograms) stay documented as
   omitted, not fabricated.
2. Prefill-chunk interleaving and a decode-first/admission policy; today a prefill is one whole round
   for the other slots, and admission is FIFO with no output-length awareness.
3. Batch depth above the 64-entry job queue, and prefix-cache interaction with batching (the serial
   path's prefix match is not consulted by the batch loop).
4. A batched-path counterpart to the 20-probe capability census: **partly done in task-14** - buffered
   and SSE prompt_logprobs/stop/logprobs probes are measured identical across modes; **task-15**
   then moved prompt_logprobs to one pass over the prompt (schema-/3 one-pass probe, both modes
   correct and payload-identical; [prompt-logprobs-one-prefill.md](prompt-logprobs-one-prefill.md)).
   The remaining broad 20-probe adaptation (chat/tools probes are host-limited on the synthetic
   plan) is still open.

Window / device-M1 dependent (not claimed, not started here):

1. Device batch attention and real paged format-2 KV backing - the engine's KV/attention state is
   still per-sequence host/device state, not the paged pool; S3 task-21.
2. Chunked-prefill numerics across batch sizes on device (this page only proves host output
   identity), and cross-request chunk interleaving at device speed.
3. Device/vLLM TP=4 performance parity (task-25): real TTFT, throughput, batching efficiency.
4. Engram NVMe read-overhead measurement in the serving path (task-24) and the S5 TP=4 numerics.

## 7. Reproduction

    cargo build -p sparkflash-cli
    python3 scripts/serve-s4-batching-census.py \
        --evidence-dir /home/user/ev-session-20261020/serve
    cargo test -p sparkflash-cli --lib continuous_batching
    cargo test -p sparkflash-cli --test serve_http continuous_batching
    python3 -m unittest discover -s scripts/tests

The census starts both servers on 127.0.0.1:0, never contacts the live vLLM instance and never leaves
loopback.

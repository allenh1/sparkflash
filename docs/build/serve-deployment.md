# Serving deployment skeleton: container, four-node launch, drain and backpressure

Status: **ops skeleton, host-only**. This block contains source and scripts that are tested on
this host, plus one definition (the container image) that is not built here, and documentation.
None of it was exercised on a cluster, in a container runtime, or on a device. That distinction is
deliberate: the four-node launch was only dry-run (the commands it prints are the contract), and
no metric value here is a serving measurement. Labels for anything that would be measured on a
device stay UNMEASURED until a device window actually records them.

Related documents: [serve-ops-metrics.md](serve-ops-metrics.md) (the Prometheus families,
definitions and buckets), [openai-surface-remaining-gaps.md](openai-surface-remaining-gaps.md)
(the API surface still missing or unproven), and [serving-s7-s8-plan.md](serving-s7-s8-plan.md)
(where the D38 vLLM A/B sits in the S7/S8 gate).

## 1. Container image

Files: [Dockerfile](../../Dockerfile), [deploy/container-entrypoint.sh](../../deploy/container-entrypoint.sh),
[.dockerignore](../../.dockerignore).

The image is a two-stage build of the checked-in workspace. The default image serves the host
reference path (no CUDA, no GPU, no checkpoint baked in); a TP=4 or device-capable binary is the
same Dockerfile with build args on a suitable build host:

    docker build -t sparkflash-serve .
    docker build --build-arg SPARKFLASH_FEATURES="tp4-serve cuda" -t sparkflash-serve:tp4 .

The default command is the synthetic tiny plan, so a bare image answers `/health` and serves the
OpenAI surface without a checkpoint:

    docker run --rm -p 8000:8000 sparkflash-serve

A real deployment mounts the checkpoint and its tokenizer and overrides the command:

    docker run --rm -p 8000:8000 \
      -v /models/deepseek-v4.1-flash:/models/ckpt:ro \
      -e SPARKFLASH_SERVE_API_KEY=... \
      sparkflash-serve serve --checkpoint /models/ckpt \
        --tokenizer /models/ckpt/tokenizer.json --bind 0.0.0.0:8000

Image properties the deployment depends on:

- **PID 1 is the server.** The entrypoint ends in `exec /usr/local/bin/sparkflash ...`, so
  SIGTERM/SIGINT reach the serve process and the drain described in section 3 runs. A shell
  wrapper without `exec` would eat the signal and the orchestrator would eventually SIGKILL.
- **Non-root.** UID/GID 10001, workdir `/opt/sparkflash`; `/models` is created writable only so a
  mounted model directory can be opened read-only by the operator's runtime.
- **Vendored tokenizer.** The reference tokenizer ships at
  `/opt/sparkflash/reference/tokenizer.json`, which is the fallback the binary resolves from its
  working directory; a real model still passes `--tokenizer`.
- **Liveness is `curl -fsS http://127.0.0.1:8000/health`** (`HEALTHCHECK`). Readiness is
  `/readyz` and belongs to the orchestrator (section 3.1); a container healthcheck must not point
  at `/readyz`, or a drain would look like a crash.
- **The API key is an environment variable** (`SPARKFLASH_SERVE_API_KEY`), never a build arg or a
  layer. The key is a static local key only; rotation/multi-key is a remaining gap.

Untested here: no image build was run on this host (the workspace build needs the full toolchain
and the image is not the deliverable this block can validate). The Dockerfile is the definition,
not an evidence artefact.

## 2. Single-command four-node launch

File: [scripts/serve-four-node.sh](../../scripts/serve-four-node.sh). It builds the same one
command per rank and runs it over ssh; it is the host TP=4 path, one process per rank:

    cargo build --release --features tp4-serve -p sparkflash-cli

    export SPARK_NODES="192.0.2.80 192.0.2.88 192.0.2.59 192.0.2.68"
    export SPARKFLASH_CHECKPOINT=/models/deepseek-v4.1-flash
    export SPARKFLASH_TOKENIZER=/models/deepseek-v4.1-flash/tokenizer.json
    scripts/serve-four-node.sh print     # offline: print the four commands
    scripts/serve-four-node.sh launch    # start four ranks
    scripts/serve-four-node.sh status    # pgrep on each node
    scripts/serve-four-node.sh stop      # SIGTERM to each rank (the drain signal)

The command it prints for rank R is exactly:

    <bin> serve --checkpoint <ckpt> --tokenizer <tokenizer> \
      --tp4-rank R --tp4-world-size 4 --tp4-peers <ip0,ip1,ip2,ip3> \
      --tp4-base-port 29601 --tp4-control-port 29701 \
      --bind 0.0.0.0:8000      # rank 0 only; ranks 1-3 bind 127.0.0.1:0 (no HTTP)

Environment knobs (all also documented in the script header): `SPARKFLASH_BIN`,
`SPARKFLASH_HTTP_BIND`, `SPARKFLASH_TP4_BASE_PORT`, `SPARKFLASH_TP4_CONTROL_PORT`,
`SPARKFLASH_LOG_ROOT`, `SPARKFLASH_RUN_ID`, `SPARKFLASH_DRAIN_DELAY_MS`,
`SPARKFLASH_DRAIN_TIMEOUT_MS`, `SPARKFLASH_DEVICE_TOKEN_DIR`, `SPARKFLASH_DEVICE_TOKEN_LIVE`,
`SPARKFLASH_DEVICE_TOKEN_DERIVED_CLAIM`, `SSH_USER`, `SSH_OPTS`. Rank 0 exports the drain knobs to
its process; every rank gets the same environment so the command builder has one path. The device
token window flags are rank-0 only (workers never detokenize).

### 2.1 Serving a device token window

A claimed four-rank window can be served instead of the host twin. The window must record the exact
prompt it answered, and since W4 chat carries the rendered prompt's ids, so a chat request can be
served from it too:

    export SPARKFLASH_DEVICE_TOKEN_DIR=/var/lib/spark/windows     # one window, or a parent
    export SPARKFLASH_DEVICE_TOKEN_LIVE=1                         # follow it as a live directory
    scripts/serve-four-node.sh launch

- Without `--device-token-live` the directory must already hold one complete claimed window; it is
  loaded once at startup and the claim state is printed.
- With it, the directory is a parent: the newest subdirectory holding a complete four-rank token
  pair is served, re-read when the rank-0 token file or `run.meta` run id changes, and every other
  request (different prompt, no window yet, refused claim) falls back to the host twin.
- `SPARKFLASH_DEVICE_TOKEN_DERIVED_CLAIM=1` selects the derived claim form (S7 rehearsal) for a
  window whose writer did not assert `token_claimed`.

A claimed response carries `x-sparkflash-token-source: <source> (UNMEASURED)` and
`x-sparkflash-token-claim: explicit|derived`. The source name distinguishes the one-shot envelope
(`device-chain-envelope`) from the live follower (`device-chain-live-window`).

#### Producing the window

The window is written by the device arm, not by `serve`. One `tp4-smoke` run per rank, all four
into the same evidence directory:

    for R in 0 1 2 3; do
      sparkflash device-chain --run-stage tp4-smoke \
        --checkpoint "$SPARKFLASH_CHECKPOINT" \
        --tp-rank "$R" --tp-world-size 4 \
        --comm-backend tcp --comm-peers "$PEERS" --comm-port 29601 \
        --evidence-dir "$WINDOW/rank$R"
    done

That writes `tp4-forward-embed-r{R}.json` and `tp4-token-r{R}.json` per rank, which is exactly the
pair `DeviceChainEnvelope::load` reads (flat, or under `rank{R}/`, both accepted). Then:

    export SPARKFLASH_DEVICE_TOKEN_DIR="$WINDOW"   # or its parent, with _LIVE=1
    scripts/serve-four-node.sh launch

**The exact-prompt rule is the whole contract.** A window answers one prompt. For `/v1/completions`
that prompt is the `int[]` you send; for `/v1/chat/completions` it is the *rendered* chat prompt
(template + tokenizer), so the window must have been recorded for the same message list and the same
template settings. Anything else falls back to the host twin, which is correct behaviour and not a
failure. A window whose claim is refused (no `token_claimed`, failed reference matches) serves
nothing until the device arm writes one that claims.

### 2.2 Boot output

Startup prints the reference server's resolved-decision lines plus a shard-loading progress bar:

    (APIServer pid=273) INFO 10-22 14:53:36 [serve.rs:5021] Resolved architecture: DeepseekV41ForCausalLM
    (APIServer pid=273) INFO 10-22 14:53:36 [serve.rs:5024] Using max model len 1048576
    (APIServer pid=273) INFO 10-22 14:53:36 [serve.rs:5028] Prefill mode: token-loop (...)
    (APIServer pid=273) INFO 10-22 14:53:52 [tp4_host.rs:970] Resolved architecture: DeepseekV41ForCausalLM
    Loading checkpoint tensors:  62% Completed | 496/800 [00:22<00:14,  1.24it/s]
    (APIServer pid=273) INFO 10-22 14:56:10 [serve.rs:5250] Starting server on http://0.0.0.0:8000
    (APIServer pid=273) INFO 10-22 14:56:10 [serve.rs:5254] Application startup complete.

`[file:line]` names the branch that decided the value, so a reader can jump to it. Nothing in these
lines is a measurement. `SPARKFLASH_BOOT_LOG=0` (or `off`/`quiet`) silences them; `progress`
keeps only the bar. The bar is rewritten in place on a terminal and printed one line per 10% into a
redirected log.


Contracts this skeleton fixes without running it:

- The rank command line is the one `Tp4ServeOptions` accepts (checkpoint required, peers in rank
  order, base port + rank for the engine collective, control port on rank 0). A rank command that
  is wrong at parse time fails before it opens a socket; the print mode lets a reviewer see it
  offline.
- `stop` sends SIGTERM, not SIGKILL, to the exact `--tp4-rank R` process; rank 0 turns
  `/readyz` 503 and finishes accepted requests before exiting (section 3).
- `print` is offline: [scripts/tests/test_serve_four_node.py](../../scripts/tests/test_serve_four_node.py)
  pins the four commands, the rank-0 bind, peer ordering, port derivation and the refusals for a
  topology that is not four nodes.

Not tested here: no `launch`/status/stop was run against a cluster, and no four-node serving
was performed in this block. A device window (and the D38 vLLM A/B at concurrency 1/8/32) is the
gate that measures this topology; until then it is a launch harness, not a result.

## 3. Graceful drain, cancellation and backpressure

### 3.1 Drain (SIGTERM / SIGINT)

Implementation: `bind_and_serve` in [serve.rs](../../crates/sparkflash-cli/src/serve.rs) installs
`tokio::signal::unix` handlers for SIGTERM and SIGINT before the server runs, and
`AppState::begin_drain` flips the readiness flag. The sequence is:

1. Signal arrives. The process prints
   `serve_timeline phase=drain_start epoch_ms=... timeout_ms=...`.
2. `/readyz` returns **503** immediately (the same atomic flag; no engine work), and
   `sparkflash_serving` turns **0**. `/health` stays 200 and `/metrics` keeps working, because the
   process is alive and finishing work.
3. After `SPARKFLASH_SERVE_DRAIN_DELAY_MS` (default **0**; set a few seconds when a load balancer
   or endpoint controller needs the readiness change to propagate), the listener stops accepting.
   The process prints `phase=drain_accepting_stopped`.
4. Connections already accepted finish normally: an SSE stream completes its frames and
   `[DONE]`, a buffered response completes. The drain deadline is
   `SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS` (default **30000**).
5. When the last connection closes, `phase=drain_complete` is printed and the process exits **0**.
   If the deadline expires first, `phase=drain_timeout forced_exit=1` is printed and it exits
   **1**: a forced drain is not a clean drain and must be visible.

Environment knobs (positive milliseconds; unset or invalid keeps the default):

| variable | default | meaning |
| :--- | ---: | :--- |
| `SPARKFLASH_SERVE_DRAIN_DELAY_MS` | 0 | grace between readiness turning 503 and the listener closing |
| `SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS` | 30000 | budget for accepted connections to finish after the listener stops |

The orchestrator's termination grace period must exceed `DRAIN_DELAY_MS + DRAIN_TIMEOUT_MS` (or
the runtime will SIGKILL mid-drain). Kubernetes example: `terminationGracePeriodSeconds: 45`,
`SPARKFLASH_SERVE_DRAIN_DELAY_MS=5000`, `SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS=30000`. A
`preStop` hook is not needed; the delay knob is the same mechanism in-process.

Probes: `GET /health` is liveness (always 200 while the process runs); `GET /readyz` is
readiness (200 while accepting, 503 from the drain signal); both are public like `/metrics`.
Evidence on this host: `sigterm_drains_in_flight_streams_and_turns_readiness_off`
in [serve_http.rs](../../crates/sparkflash-cli/tests/serve_http.rs) spawns the real binary,
starts a delayed SSE stream, sends SIGTERM, observes `/readyz` 503 and `/health` 200 while the
stream finishes `[DONE]`, then asserts exit code 0 and the three timeline lines.

### 3.2 Cancellation (client disconnect)

Already wired in the engine driver, not a new mechanism: the request's reply channel is closed
when the HTTP body is dropped, and the engine loop treats a failed send as this request's terminal
error (`"the client stopped reading this response"`) in both the serial loop and the
continuous-batching loop. The terminal branch calls `request_finished`, so
`sparkflash_num_requests_running` returns to 0 and the queue slot is released; the next request
is admitted normally. A cancelled request may have generated tokens that no usage block reports,
which is why `sparkflash_generation_tokens_total` can exceed the sum of wire usages (documented
in serve-ops-metrics.md section 1). No separate cancel endpoint exists; the client disconnect is
the cancel signal.

### 3.3 Backpressure (bounded queue)

The driver queue is bounded (`JOB_QUEUE = 64`). When it is full, `try_send` fails, the
waiting gauge is rolled back, `sparkflash_requests_rejected_total` increments, and the request
is answered with the typed engine refusal: HTTP **503 service_unavailable**, internal engine code
`refused`, with a message naming the bounded queue and saying to retry when in-flight requests
finish. This is refusal rather than unbounded buffering or an untyped timeout, and it is visible
in metrics. No client-side retry/backoff header is added yet (Retry-After is a possible
follow-up); a caller should retry with its own budget. `--max-batch` is a separate, smaller
admission cap inside the scheduler and cannot exceed the queue.

### 3.4 Prometheus names, labels and scrape

The families and their definitions live in [serve-ops-metrics.md](serve-ops-metrics.md); the
complete set exported by `sparkflash serve` is:

| metric | type | labels |
| :--- | :--- | :--- |
| `sparkflash_requests_total` | counter | none |
| `sparkflash_prompt_tokens_total` | counter | none |
| `sparkflash_global_kv_hit_tokens_total` | counter | none |
| `sparkflash_swa_tail_cache_hits_total` | counter | none |
| `sparkflash_generation_tokens_total` | counter | none |
| `sparkflash_requests_rejected_total` | counter | none |
| `sparkflash_num_requests_running` | gauge | none |
| `sparkflash_num_requests_waiting` | gauge | none |
| `sparkflash_serving` | gauge | none |
| `sparkflash_time_to_first_token_seconds` | histogram | none (22 vLLM buckets + `+Inf`) |
| `sparkflash_engram_lookups_total` | gauge (per layer) | `layer` |
| `sparkflash_engram_hit_rate` | gauge (per layer) | `layer` |
| `sparkflash_uma_free_gib` | gauge (per rank) | `rank` |
| `sparkflash_trace_active_scope_mask` | gauge | none |

Scrape `/metrics` on the rank-0 HTTP port; it is public like `/health` (a scraper has no
bearer key). Suggested alerting shapes: `sparkflash_serving == 0` (draining or stuck), a
rising `rate(sparkflash_requests_rejected_total[5m])` (backpressure),
`sparkflash_num_requests_waiting` near 64 (queue pressure), and
`histogram_quantile(0.99, ...time_to_first_token_seconds_bucket)` against an SLO the operator
measured — not a value from this repository's host census.

## 4. Non-claims

- No container was built and no four-node launch was executed in this block; both are definitions
  plus offline contract tests.
- The host drain test measures behavior on the reference engine with the scripted stub; it does
  not measure device drain latency, SSE drain semantics under TP=4, or behavior under a load
  balancer.
- No throughput, TTFT, token-acceptance, GPU, device or vLLM comparison claim is made here. The
  D38 vLLM A/B at concurrency 1/8/32 remains the final S7/S8 gate and is owned by the device
  window.

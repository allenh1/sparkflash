# n > 1 choices, response_format and bearer auth (task-20)

Status: landed on branch task/serve-openai-parity. Label for all measured evidence:
host path, synthetic tiny plan and debug-build scripted stub, UNBARED / diagnostic.

This note records the three OpenAI surface gaps task-20 closed, the wire contract each one now
serves, the tests that pin it, and what is deliberately still missing. The actionable residual
tracker (owner and device dependency per item) is
[openai-surface-remaining-gaps.md](openai-surface-remaining-gaps.md).

## 1. n > 1 choices

### Wire contract

The reference engine serves one sequence per generate() call, so n choices are n sequential
generate() calls. chat/completions and completions both accept n, validate it first, and return
choices ordered by index 0..n-1.

Buffered chat:

- choices[i].index == i, one choice object per generate() output.
- The prompt is counted once in usage.prompt_tokens and completion tokens are summed over the
  choices; total_tokens stays prompt + completion.
- Per-choice request id "chatcmpl-...-{i}" ("{id}" only when n == 1) and per-choice seed
  seed + i, matching vLLM's per-choice seed derivation.
- Sampling options, stop sequences, tool parsing and logprobs validate per choice; any refused
  choice refuses the whole request before generation.

Buffered completions: the choice index is prompt-major, index = prompt_index * n + choice_index,
so multi-prompt and multi-choice requests stay ordered the way vLLM orders them. Request ids are
"{id}" for one prompt and n == 1, "{id}-{prompt}" for multiple prompts and n == 1, and
"{id}-{prompt}-{choice}" otherwise; the seed derivation is the same seed + i.

Streaming:

- chat: one assistant role chunk at index 0, then each choice's content and tool-call deltas at
  its own index, then its finish chunk. The single usage chunk (choices: []) is emitted after all
  choices when stream_options.include_usage is set; the aggregated stats ride on the last finish
  chunk. A terminal parser refusal ends the stream with the error frame and [DONE] without a
  finish or usage chunk, exactly like the single-choice path.
- completions: one stream tagged per request index; the chunk index is
  prompt_index * n + choice_index and the aggregated usage chunk follows the same
  prompt-once/summed-completion rule.
- n == 1 keeps the pre-task-20 bytes: request id, seed, engine usage/stats placement and the
  single role chunk are unchanged (the byte-exact single-choice router test and prior wire
  assertions still pass untouched).

### Engine seam

The API trait gained independent_choices(&self) -> bool, default true, documented on the trait
beside vocab_size. The router refuses n > 1 typed when the engine reports false. The TP=4 driver
overrides it to false because its tensor-parallel group samples one shared window: independent
choices would silently correlate. The refusal is invalid_request_error with param n and the
internal subcode unsupported_parameter; its message names the shared sampling state across ranks
and the temperature-plus-seed TP=4 sampler, and says the request is refused rather than returning
choices that were not independently sampled. (The wire envelope's code stays the integer 400 --
typed subcodes are internal.) The device engine path keeps the default true; no per-request
independent sampling was added there, so n > 1 on that path is a residual exposure noted in
section 6.

### Tests

- crates/sparkflash-api/tests/router.rs: chat_n_two_choices_buffered_counts_the_prompt_once,
  chat_n_choices_carry_per_choice_ids_and_seeds, chat_n_zero_is_refused_before_generation,
  chat_n_above_one_is_refused_on_an_engine_with_shared_sampling,
  chat_n_two_streams_each_choice_and_aggregates_usage,
  completions_n_two_choices_per_prompt_keep_prompt_major_indexes,
  completions_n_two_streams_choices_and_one_prompt_usage,
  completions_n_zero_is_refused_before_generation.
- crates/sparkflash-cli/tests/serve_http.rs over the real socket:
  wire_chat_n_multiple_choices_over_the_scripted_wire, wire_chat_n_stream_over_the_scripted_wire,
  wire_completions_n_choices_serial_and_batched, wire_completions_n_stream_serial_and_batched,
  wire_response_format_json_and_n_zero_are_typed_400s.
- scripts/openai-surface-census.py: n > 1 completion and chat probes over HTTP, serial and
  --continuous-batching --max-batch 2, buffered and SSE, with the usage relations and ordering
  asserted per run.

## 2. response_format

The schema now parses {"type":"text"}, {"type":"json_object"} and
{"type":"json_schema","json_schema":{"name":...,"schema":...}}. Nothing in the decode path
constrains tokens, so the server cannot honour the JSON modes: it refuses them typed rather than
pretending.

- {"type":"text"} is accepted as the no-op it means.
- json_object and json_schema -> 400 invalid_request_error, param response_format, internal
  subcode unsupported_parameter, message naming the absent constrained decoding (vLLM's xgrammar
  path / no decoder logits processor) and explaining that rendering the schema into the prompt is
  only an instruction, so the request is refused instead of returning unconstrained JSON that may
  violate the caller's guarantee.
- A malformed json_schema wrapper (for example missing name) is the generic invalid-body 400.
- The completions request has no response_format field: an unknown JSON key is still silently
  ignored there, recorded as a residual gap.

Constrained decoding (xgrammar/guided decoding) remains a named gap in the audit, section 7. This
is deliberate honest-refusal scope, not an implementation claim.

Tests: response_format_text_is_a_no_op_and_json_modes_are_refused_typed,
response_format_malformed_json_schema_is_an_invalid_body in router.rs;
wire_response_format_json_and_n_zero_are_typed_400s in serve_http.rs, plus the census refusal
probes.

## 3. Bearer auth

serve gained --api-key KEY, with the environment fallback SPARKFLASH_SERVE_API_KEY, validated at
parse time (an empty or whitespace key is refused: an empty bearer token would be both
meaningless and the only accepted one). AppState carries the key as Option<Arc<str>>; the default
is None and every pre-task-20 deployment therefore keeps the open server unchanged.

When a key is configured, one axum middleware requires "Authorization: Bearer <key>" on every
route except /health, /readyz and /metrics (open for liveness, readiness probes and scrapers). The scheme compare is
case-insensitive and the token compare is constant-time over equal-length byte slices. A missing,
malformed or wrong credential yields the documented envelope:
{"error":{"message":"Unauthorized","type":"invalid_request_error","param":null,"code":401}}.
The protected set includes /v1/models and the admin/trace routes; no route is silently exempted
beyond health, readiness and metrics.

Residual: this is a local static bearer key, not OIDC, multi-key or scoped authorization. It is
single-process configuration; no rotation or per-key accounting exists.

Tests: bearer_key_protects_every_route_except_health_and_metrics and
no_api_key_keeps_the_server_open in router.rs;
wire_bearer_key_protects_routes_and_keeps_health_open in serve_http.rs, and the census keyed-mode
probes (health/metrics 200, models 401 with the exact envelope, wrong scheme/wrong key 401,
correct key 200, generation 401/401/200).

## 4. Evidence

Measured census: scripts/openai-surface-census.py writes JSON + markdown + sha256 under
/home/user/ev-session-20261020/serve/. Current run:
openai-surface-census-20261021T040922Z (JSON sha256 c0b1d8793a92523e6fe66d475da1ca03c00c0773c3f07389b765933e34fcb788,
markdown sha256 96812ceb0e57310057749497e2212d42c809098fd391bc93a23220ddd18f3167), 3 runs x 74
named checks all passing, serial and batched payloads equal by canonical hash, serve binary sha256
f1dfb7b10d07f476658cffff75d606460f59f0700bec43db27d43e0175efde02, captured at git e909f310 with the
slice uncommitted (the census pins its source manifests to the measured working tree; the task-20
commit follows the probe, so those manifest hashes describe the pre-commit docs state).

Gates run on the same slice: cargo fmt --all -- --check; cargo clippy --workspace --all-targets --
-D warnings; cargo clippy -p sparkflash-cli --all-targets --features tp4-serve -- -D warnings;
cargo test -p sparkflash-engine (198), -p sparkflash-api (router 83 among the suite), -p
sparkflash-cli (lib 235, serve_http 21), -p sparkflash-cli --features tp4-serve (lib 247,
serve_http 21); python3 -m unittest discover -s scripts/tests (289 tests, including 17 for this
census script).

## 5. Residual gaps and non-claims

- **Follow-up (host, branch `task/serve-n1-ops`): unseeded `n > 1` now draws a fresh seed per
  choice.** Before that change every unseeded choice reached the driver with seed `None`, which
  the host driver maps to 0, so the choices were copies of one sample. `AppState::choice_seed`
  draws one seed per choice only when the caller named no seed and asked for `n > 1`; a seeded
  request keeps `seed + i` and `n == 1` keeps the byte-compatible `None` path. This closes the
  **host reference path** only: the device and TP=4 rows below remain open.
- Constrained decoding is absent: json_object/json_schema are refused, never emulated.
- response_format on the completions request is still absent from the schema; unknown JSON is
  silently ignored there.
- n > 1 on the TP=4 driver is refused at the router (independent_choices() = false); the device
  engine path keeps the trait default true but refuses non-greedy sampling, so n > 1 with
  temperature > 0 ends in a 503 there and greedy choices are deterministic copies. Both rows are
  unproven for independent sampling; see [openai-surface-remaining-gaps.md](openai-surface-remaining-gaps.md)
  rows S3/S4.
- No best_of; no n > 1 with prompt_logprobs beyond what the per-choice code paths already expose.
- Streaming completions deliberately keep prompt_cache_hit/miss at 0 (pre-existing streaming
  shape); buffered responses derive them.
- Bearer auth is a static local key only (no OIDC, multi-key, rotation, scopes).
- All evidence is host path with the synthetic tiny plan (vocab 128) or the debug-build scripted
  stub, UNBARED. No CUDA, device, GPU, cluster or vLLM run; no throughput, TTFT, model-quality or
  vLLM-comparison claim. The TP=4 refusal is router-unit coverage, not a TP=4 wire probe.

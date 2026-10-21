# OpenAI serving parity audit (task-3 / WS-E), 2026-10-20

**Scope.** Row-level comparison of SparkFlash's serve HTTP surface with the OpenAI-compatible
surface of the vLLM the cluster actually runs as the oracle. Reference source is the oracle
image's own vLLM extraction at /tmp/vllm/src (read-only, recent main dev version; the extraction
covers vllm/entrypoints, vllm/sampling_params.py, vllm/logprobs.py and vllm/exceptions.py, so
parser internals that were not extracted are cited at their launch flags instead of their classes).
Our side is the reference CPU serve path at commit 3f4cb3a (this task's stop and health work is
in b957aa3 and d728b9a), with the file:line locations in the tables.

**Oracle configuration (the thing being matched).** vLLM TP=4 DSpark serving the heretic
snapshot with --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser
deepseek_v41, speculative-config dspark k=5. Its /metrics shows SpecDecoding mean acceptance
length 3.2-5.8; that is calibration for the oracle, not a SparkFlash result.

**Boundary.** Every SparkFlash number in this audit's census companion is a HOST PATH number from
the synthetic tiny plan (vocab 128, no checkpoint, no device, no CUDA) and is labelled
"host path, synthetic tiny plan (vocab 128), UNBARED / diagnostic". D41 has no frozen floor, so
nothing here is a gate input, a device claim or a qualification result. No vLLM number is
reproduced and the live server on 192.0.2.80:8000 is never contacted. Host-path e2e generation
tests use token-id prompts because the synthetic vocabulary cannot embed the released tokenizer's
real ids; chat/text-prompt rows below are therefore router-level (MockEngine) or marked host-limited.

**Verdict legend.** IMPL = implemented and tested at the wire level in this checkout;
IMPL-b = implemented and covered by router/unit tests below the wire (the synthetic host cannot
run the released tokenizer path end to end); PARTIAL = the common case works but a documented
vLLM behaviour is absent or different; MISSING = not implemented; HOST-LIMIT = cannot be
exercised on this host (not a parity result); OUT-OF-SCOPE = a vLLM endpoint family this
generation server does not target.

## 1. Endpoint-level summary

| vLLM endpoint | vLLM reference | SparkFlash | Verdict |
| :--- | :--- | :--- | :--- |
| GET /health | instrumentator/health.py:22-38 | server.rs:225, 2450 | PARTIAL: 200 with empty body matches; the 503 EngineDeadError branch is documented but no engine-dead signal exists on this path |
| GET /metrics | instrumentator/metrics.py:67-81 | server.rs:2612 | PARTIAL: Prometheus text 0.0.4, correct content type; the request-lifecycle families (running/waiting gauges, generation-token counter, TTFT histogram) are **IMPL host** in task-14 with measured accounting ([serve-ops-metrics.md](serve-ops-metrics.md)); still no vllm:* names and no http_* instrumentator metrics, and GET /load is absent |
| GET /version | instrumentator/basic.py:53 | absent | MISSING |
| GET /load | instrumentator/basic.py:30 | absent (server_load_metrics unavailable) | MISSING |
| POST /v1/chat/completions | openai/chat_completion/api_router.py:42 | server.rs:220, 1055 | IMPL-b (see field matrix) |
| POST /v1/chat/completions/batch | openai/chat_completion/batch_serving.py | absent | OUT-OF-SCOPE (dev batch route) |
| POST /v1/completions | openai/completion/api_router.py:36 | server.rs:221, 1517 | IMPL (token-id prompts on the wire; see matrix) |
| POST /v1/responses | openai/responses/api_router.py:49 | server.rs:222, 2228 | PARTIAL: buffered response only (stream=true refused); no GET /v1/responses/{id} and no /cancel |
| POST /v1/messages | entrypoints/anthropic/api_router.py:51 | server.rs:223, 2335 | PARTIAL: buffered message only (stream=true refused); /v1/messages/count_tokens (:89) missing |
| GET /v1/models | openai/models/api_router.py:20 | server.rs:224, 2454 | IMPL (ModelCard members root/parent/max_model_len/permission added; owned_by remains sparkflash, an intentional divergence from vllm) |
| POST /tokenize, /detokenize, GET /tokenizer_info | serve/tokenize/api_router.py:35,61,90 | absent | MISSING (tokenizer seam exists internally; no HTTP surface) |
| /v1/embeddings, /pooling/*, /classify, /score, /v1/score, /rerank, /v1/rerank, /v2/rerank | embedding/pooling routers | absent | OUT-OF-SCOPE: the released V4.1 serving target is generation; this server serves no embedding head |
| /v1/audio/* | audio routers | absent | OUT-OF-SCOPE |
| LoRA / RLHF / sleep / wake / dev routes | serve/dev/* | absent | OUT-OF-SCOPE for this batch |

## 2. /v1/chat/completions field matrix

| field / behaviour | vLLM reference | SparkFlash | Verdict |
| :--- | :--- | :--- | :--- |
| model, messages (string content) | chat/protocol.py: model, messages | schema.rs:12 ChatCompletionRequest; server.rs:1055 | IMPL-b |
| role system/user/assistant/tool, tool_calls echo | chat/protocol.py message union | schema.rs ChatMessage | IMPL-b |
| content parts: text / image_url | chat/protocol.py content union; multimodal backend | schema.rs ContentPart; image refused with typed missing capability (server.rs find_image_carrier) | PARTIAL: text parts served in template; images refused by design on the reference engine |
| stream, stream_options.include_usage | api_utils.py:289-303; chat/serving.py:510-511,787-816 | server.rs:1207-1400 | IMPL-b: SSE data frames, include_usage final chunk with choices=[], data: [DONE] |
| stream_options.continuous_usage_stats | api_utils.py:289-303 | absent | MISSING |
| max_tokens / max_completion_tokens precedence | api_utils.py:169 get_max_tokens | server.rs:321 resolve_max_tokens | IMPL-b |
| temperature | sampling_params.py | schema.rs; EngineRequest.temperature | IMPL |
| top_p | sampling_params.py:256-262 nucleus filter; range :583-585 | schema.rs; server.rs validate_top_p; engine forward/mod.rs SamplerOptions::top_p + candidate_filter | IMPL: the reference sampler applies the nucleus over the temperature-scaled probabilities; top_p=0.0 is our documented extension (vLLM refuses it) and keeps the single most likely token |
| top_k | sampling_params.py:259-261 load; :583-599 range | schema.rs; server.rs validate_top_k; forward/mod.rs SamplerOptions::top_k | IMPL: 0/-1 disable, >=1 keep that many most likely; values below -1 refused with vLLM's message |
| min_p | sampling_params.py:262, :599 | absent from schema (unknown JSON silently ignored) | MISSING |
| repetition_penalty, presence_penalty, frequency_penalty | sampling_params.py:240-253, ranges :547-563 | schema.rs; server.rs validate_repetition_penalty/validate_penalty; forward/mod.rs SamplerOptions + apply_penalties | IMPL: penalties applied to the raw logits before temperature; repetition over prompt+generated, presence/frequency over generated only |
| logit_bias | sampling_params.py logit_bias | absent from schema | MISSING |
| bad_words, allowed_token_ids, ignore_eos, min_tokens, stop_token_ids | sampling_params.py | absent from schema | MISSING |
| seed | sampling_params.py seed | schema.rs; EngineRequest.seed; deterministic at temperature 0 and 1 per seed (census probe) | IMPL: seed is honoured by the reference sampler; unlike vLLM it is engine-global, not a per-request RNG handle |
| n (n>1 / best_of) | sampling_params.py:229 | schema.rs n; server.rs resolve_choice_count + per-choice request plan; chat/completions buffered and streaming assembly; TP=4 driver independent_choices() = false | IMPL-b for the reference path (task-20): ordered choices, per-choice request id "{id}-{choice}" (completions "{id}-{prompt}-{choice}"), vLLM seed+i, prompt counted once + completion summed, per-choice finish chunks in streaming; TP=4 refuses n>1 typed (param n) because its group samples one shared window |
| stop (string or list), empty-string validation | sampling_params.py:525 holdback max(len)-1, :645 empty-stop error | serve.rs:2629 StopBuffer, :1193/:1633 normalize_stop -> 400 "stop cannot contain an empty string." | IMPL: holdback across token boundaries, earliest match, tail flush, finish_reason stop, completing token counted; streaming and logprobs covered |
| logprobs, top_logprobs (chat shape: content entries with bytes/top_logprobs) | logprobs.py; chat protocol logprobs models; sampling_params.py:810-824 cap | server.rs:661-960, 1140-1200 | IMPL-b: cap refusals with vLLM text, prompt_logprobs and logprob_token_ids; prompt_logprobs refused when stream=true (server.rs:803-817) |
| logprob_token_ids cap MAX_LOGPROB_TOKEN_IDS | sampling_params.py | server.rs:72, 772 | IMPL-b |
| echo (chat) | chat/protocol.py | schema.rs echo doc: parsed, used only for vLLM's prompt_logprobs side effect | PARTIAL (deliberate: no prompt prepend) |
| response_format / structured outputs / guided decoding | vLLM structured-output backend (xgrammar) | schema.rs ResponseFormat (text/json_object/json_schema); server.rs refuse_unenforceable_response_format; still not read by encoding/seam.rs encode_chat | PARTIAL (task-20): {"type":"text"} accepted as a no-op, json_object/json_schema refused typed 400 param response_format naming absent constrained decoding (no silent best effort that could return invalid JSON); constrained decoding itself is a named gap |
| tools, tool_choice, parallel_tool_calls | chat/protocol.py; launchers/cli_args.py:106-119,432 (--enable-auto-tool-choice requires --tool-call-parser); chat_utils.py renders tools | schema.rs tools Option<Value>; encoding/seam.rs:406 attach_tools renders them; server.rs apply_deepseek_v41_tool_parser and chat_stream_response's incremental ToolCallStreamParser; tool_parser.rs ports the reference decoder's DSML parse and the streaming delta parser (task-19, [streaming-tool-calls.md](streaming-tool-calls.md)) | IMPL-b: tools reach the prompt, a buffered answer's DSML block is lifted, and the same block is streamed as tool_calls deltas (id+name start, argument fragments, finish_reason tool_calls, typed tool_call_parse_error for malformed/unterminated blocks); `tool_choice`/`parallel_tool_calls` are absent, the reference engine emits no tool-call channel of its own, and streaming sample logprobs combined with offered tools keeps the parser disabled (documented residual gap) |
| reasoning_effort / enable_thinking / chat_template_kwargs | vLLM reasoning parsers; launcher --reasoning-parser | schema.rs; encoding/seam.rs:449 apply_chat_template_kwargs; reasoning validation crate | IMPL-b (template side); reasoning channel exists but the released oracle's deepseek_v41 parser behaviour is not replicated |
| skip_special_tokens | chat/protocol.py:274 default true | both request schemas; EngineRequest.skip_special_tokens; serve.rs run_request decode_stream(job.skip_special_tokens) | IMPL: absent means true, so a generated special token is hidden; an explicit false passes the raw decoded text through (serve_http cover both directions) |
| user, metadata | chat/protocol.py | user parsed; metadata absent | PARTIAL |
| response fields: object=chat.completion, choices[].message, finish_reason, usage | chat/protocol.py | server.rs chat response assembly | IMPL-b; finish_reason mapping differs for loop-abort (our extra LoopAbort maps to stop) |
| chunk shape: every chunk carries all response keys with nulls (usage null except final) | chat/protocol.py chunk model | our chunks omit keys that are null | PARTIAL (shape difference only) |

## 3. /v1/completions field matrix

| field / behaviour | vLLM reference | SparkFlash | Verdict |
| :--- | :--- | :--- | :--- |
| prompt: string, string list, int[] | completion/protocol.py prompt union | schema.rs PromptSpec (Text, Many, Tokens) | IMPL: nested int[][] token batches are not accepted (documented) |
| echo, echo + prompt_logprobs implication, echo + max_tokens=0 | completion/protocol.py echo block | server.rs:1562-1650 | IMPL-b |
| max_tokens, logprobs, prompt_logprobs, logprob_token_ids | sampling_params.py; completion/protocol.py | server.rs:1517-1720 | IMPL-b (see section 5) |
| stop | sampling_params.py:525 | serve.rs StopBuffer | IMPL (new: was accepted and ignored before this task) |
| suffix | completion/protocol.py | absent | MISSING |
| best_of / n | sampling_params.py:229 | absent | MISSING |
| seed, temperature | sampling_params.py | EngineRequest | IMPL / temperature only as section 2 |
| skip_special_tokens | completion/protocol.py:82 default true | schema.rs; EngineRequest; serve.rs decode_stream | IMPL: default true hides special tokens, explicit false passes them through |
| streaming + usage | api_utils.py:289 | server.rs:1693-1720; e2e test | IMPL |

## 4. Streaming (SSE) and usage

| behaviour | vLLM reference | SparkFlash | Verdict |
| :--- | :--- | :--- | :--- |
| framing: data: <json>\n\n, terminator data: [DONE]\n\n | chat/serving.py, sse_keep_alive.py | server.rs sse_frame/sse_done, byte-exact router test | IMPL |
| content type text/event-stream, cache-control | instrumentator/middleware | server.rs sse_response | IMPL |
| include_usage: final chunk choices=[], usage object | api_utils.py:289-303 | server.rs:1393, 1497; e2e test checks it | IMPL |
| final content chunk with empty text and finish_reason | chat/serving.py:787-816 | completion stream emits it (observed) | IMPL |
| SSE keep-alive comment when idle | sse_keep_alive.py:21-55 | server.rs sse_with_keep_alive + SSE_KEEP_ALIVE_COMMENT, AppState::with_sse_keep_alive (default 15 s, opt-out); router tests an_idle_stream_is_kept_alive_with_comment_frames and the_keep_alive_frames_can_be_turned_off | IMPL |
| cancellation on client disconnect | with_cancellation / engine abort | serve.rs:2760-2884 "the client stopped reading this response"; engine thread stops | IMPL (host path; device abort not wired) |
| thread safety / one request at a time | continuous batching in vLLM | serve.rs engine thread; JOB_QUEUE=64; opt-in --continuous-batching scheduler (serve.rs:2252-2700) | PARTIAL: default remains one-at-a-time; opt-in host batching (task-13, [s4-host-continuous-batching.md](s4-host-continuous-batching.md)) is measured; device batching is window work |

## 5. Logprobs family

| behaviour | vLLM reference | SparkFlash | Verdict |
| :--- | :--- | :--- | :--- |
| sample logprobs, top-k alternatives, top_logprobs | logprobs.py; protocol logprob models | server.rs:661-960, 931 | IMPL-b; shape matches vLLM (tokens/token_logprobs/top_logprobs/text_offset/bytes) |
| whole vocabulary via -1 bounded by max_logprobs | sampling_params.py:813-824 | server.rs:724-736 refusal text; census probe asks -1 under --max-logprobs 1000 | IMPL |
| over-cap refusal text | sampling_params.py:821-824 | same message shape | IMPL |
| prompt_logprobs (chat and completions), position 0 null, echo implied | sampling_params.py:860-869; completion/protocol.py | server.rs:803-817, 1140-1200; one-pass scoring in serve.rs with engine rows in forward/mod.rs (task-15, [prompt-logprobs-one-prefill.md](prompt-logprobs-one-prefill.md)); TP=4 rank contract in engine prompt_ranks.rs + tp4_host.rs + serve.rs control frame (task-18, [tp4-prompt-logprobs-rank-contract.md](tp4-prompt-logprobs-rank-contract.md)) | IMPL-b: payload unchanged; every position's distribution computed once per prompt instead of one re-forward per position, equality pinned against the retained token-at-a-time reference; the Phase-1 TP=4 path serves a bounded top-k through its all-position rank contract and refuses -1, over-wide counts, certified-bound entries and unprovable orders with the documented criterion |
| prompt_logprobs with stream=true refused | vLLM logs/refuses this combination | server.rs:803-817 | IMPL |
| logprob_token_ids (vLLM extension), MAX_LOGPROB_TOKEN_IDS cap | sampling_params.py | server.rs:72, 745-800 | IMPL-b |
| streaming logprobs per token, after that token's delta | chat/serving.py logprobs streaming | serve.rs run_request ordering; router tests | IMPL-b; with offered tools the parser is disabled so the raw DSML text stays content in that combination (documented residual gap, [streaming-tool-calls.md](streaming-tool-calls.md)) |

## 6. Sampling and decoding parameters

Summary: temperature, seed, top_p, top_k, repetition_penalty, presence_penalty and
frequency_penalty now change generation on the reference path. The sampler applies the vLLM
penalties to the raw logits before temperature, then the candidate filters (forward/mod.rs
`SamplerOptions` + `sample`), and a request that sets none of them goes through the untouched
`sample_gumbel` fast path, so the default output is byte-identical to the pre-extension driver
(proven by the unit equivalence test and the serve_http default-identity case). Still absent from
the schema: min_p, logit_bias, bad_words, allowed_token_ids, ignore_eos, min_tokens and
stop_token_ids; serde silently ignores those unknown JSON keys. The TP=4 host path keeps its
temperature/seed-only sampler, so this row describes the reference path only. The
20261020T222156Z census on the d7c9622 binary records the reference path over HTTP: sampler
options are accepted and out-of-range values are refused with vLLM's text, while the top_p
effect stays indistinguishable because the synthetic plan's logits are non-finite (unit tests,
not the wire probe, prove the filter).

## 7. Tools, structured outputs, reasoning

| behaviour | vLLM reference | SparkFlash | Verdict |
| :--- | :--- | :--- | :--- |
| tool definitions rendered into the prompt | chat_utils.py tool rendering | encoding/seam.rs:406 attach_tools | IMPL-b |
| auto tool choice + --tool-call-parser deepseek_v41 | launchers/cli_args.py:106-119,432; tool parsers package | server.rs apply_deepseek_v41_tool_parser and chat_stream_response; tool_parser.rs ports reference/encoding/encoding.py's generation-side DSML parse and carries the incremental ToolCallStreamParser | IMPL-b for buffered and streaming chat (task-19): content before the block stays content, calls become tool_calls, finish_reason tool_calls; streaming emits call-start and argument-fragment deltas and a typed tool_call_parse_error for a malformed/unterminated block; streaming logprobs keeps the parser off (residual gap) |
| tool_calls in response, finish_reason tool_calls, streaming tool-call deltas | chat protocol | server.rs:570-637 assembles engine events and chat_stream_response maps StreamToolEvent onto chunks; apply_deepseek_v41_tool_parser lifts DSML text on buffered chat when tools were offered; parser unit, router and scripted serve_http wire tests | IMPL-b buffered and streaming (task-19, [streaming-tool-calls.md](streaming-tool-calls.md)) |
| parallel_tool_calls | chat/protocol.py | absent | MISSING |
| reasoning parser / reasoning_content channel | launcher --reasoning-parser deepseek_v41 | reasoning channel exists in our protocol; parser replication absent | PARTIAL |
| response_format / grammar-constrained decoding | vLLM structured output | chat only: schema.rs ResponseFormat + typed refusal; the completions request has no field (unknown JSON silently ignored) | PARTIAL (task-20): chat refuses json_object/json_schema typed; completions still accepts-and-ignores; constrained decoding absent |

## 8. Models card and metrics

| behaviour | vLLM reference | SparkFlash | Verdict |
| :--- | :--- | :--- | :--- |
| GET /v1/models object list, ModelCard members | engine/protocol.py:80-88, models/api_router.py:20 | server.rs:2454-2530 | IMPL: id/object/created/owned_by/root/parent/max_model_len/permission; owned_by=sparkflash differs from vLLM's vllm by design |
| /metrics Prometheus content type and family naming | instrumentator/metrics.py:67-81 | server.rs:2612; task-14 families in metrics.rs; measured in [serve-ops-metrics.md](serve-ops-metrics.md) | PARTIAL: the vLLM-shaped request-lifecycle metrics now exist under the project prefix (sparkflash_num_requests_running/waiting, sparkflash_generation_tokens_total, sparkflash_time_to_first_token_seconds with vLLM's buckets), fed by the engine driver and accounted in the schema-/2 (and task-15 schema-/3) census; still no vllm:* naming, no http_* histograms, no GET /load, and no prefix-cache/GPU-cache ops families (deliberately omitted rather than fabricated) |

## 9. Errors and status semantics

| behaviour | vLLM reference | SparkFlash | Verdict |
| :--- | :--- | :--- | :--- |
| envelope {"error":{message,type,param,code}} | engine/protocol.py:54-62 | sparkflash-api/src/error.rs | IMPL shape |
| error.code | ErrorInfo.code is int (the HTTP status), engine/protocol.py:54-62 | error.rs serializes code as status.as_u16(); the internal code name is kept for diagnostics only (inner_code) | IMPL: same wire type; tests assert 400/404 integers |
| 404 unknown model, code model_not_found | vLLM model router | server.rs:301; census probe | IMPL |
| 400 validation messages | sampling_params.py etc. | server.rs:321-430, 724-800, 803-817 | IMPL for covered fields (vLLM message text copied for logprobs and stop) |
| 503 engine-dead vs 503 synthetic embed failure | managed by engine client | server.rs engine_error path; synthetic host raises 503 engine_error for real-tokenizer ids | PARTIAL: same status, different cause; /health does not report it |
| 429/queue-full semantics | vLLM engine scheduler | typed refusal code "refused" on the stream (serve.rs:266) | PARTIAL: not an HTTP 429 |

## 10. Serving behaviour

| behaviour | vLLM reference | SparkFlash | Verdict |
| :--- | :--- | :--- | :--- |
| per-request timeout | no vLLM request timeout; SSE keep-alive instead | none | n/a |
| queue/backpressure | engine queue + scheduler | bounded JOB_QUEUE=64 and REPLY_BUFFER=32, typed refusal | PARTIAL |
| abort propagation | engine abort on disconnect | driver stops when the reply receiver drops | IMPL (host) |
| auth (--api-key) | launcher flag | serve.rs ServeArgs --api-key / SPARKFLASH_SERVE_API_KEY; AppState.with_api_key; server.rs require_bearer_auth middleware | IMPL-b (task-20): no key keeps the server open exactly as before; with a key every route except /health and /metrics requires "Bearer <key>" (scheme case-insensitive, constant-time compare); 401 envelope {"message":"Unauthorized","type":"invalid_request_error","param":null,"code":401}; static key only, no OIDC/multi-key/scopes |
| CORS / middleware | instrumentator middleware | absent | MISSING |
| request id shape | chatcmpl-<uuid>/cmpl-<uuid> | cmpl-dsv41-NNNNNNNN / chatcmpl-dsv41-NNNNNNNN | PARTIAL (stable counter, not uuid) |
| stop_strings on the TP=4 host path | same vLLM rule | serve.rs Job carries stop, but run_tp4_request (:3359) does not apply it | HOST-LIMIT / documented gap: the TP=4 feature path is Phase-1 host scaffolding; default reference path is wired |

## 11. Highest-value next items

1. **Streaming tool-call parsing - landed in task-19.** The V4.1 DSML block is now parsed
   incrementally on the SSE path: `ToolCallStreamParser` in tool_parser.rs consumes content
   deltas, holds back only marker-prefix bytes, and emits tool-call start/argument fragments;
   `chat_stream_response` maps them to `delta.tool_calls` chunks, overrides the final choice to
   `finish_reason: tool_calls`, and ends a malformed or unterminated block with one typed
   `tool_call_parse_error` SSE error frame followed by `[DONE]`. Split-at-every-boundary property
   tests, router streamed-vs-buffered equality and a debug-build scripted wire seed pin it;
   design note [streaming-tool-calls.md](streaming-tool-calls.md). Residual: streaming sample
   logprobs combined with offered tools keeps the parser disabled (raw DSML content stays
   content in that combination).
2. **Sampler on the TP=4 host path.** The reference path honours top_p/top_k/penalties; the TP=4
   feature path still uses temperature+seed only and documents that gap beside the stop-string
   gap.
3. **Remaining OpenAI surface:** n>1, response_format/guided decoding and --api-key auth landed
   in task-20 (design note [openai-surface-gaps.md](openai-surface-gaps.md)): n>1 choices are
   served ordered with vLLM's seed+i and prompt-once/summed usage on both buffered and streaming
   chat/completions (TP=4 refuses n>1 typed), response_format json modes are refused typed rather
   than silently ignored, and a configured bearer key protects every route except /health and
   /metrics. Still open: constrained decoding itself (xgrammar/guided decoding), response_format
   on the completions request, and the sampler fields
   min_p/logit_bias/bad_words/allowed_token_ids/ignore_eos/min_tokens/stop_token_ids.
   (Landed in task-18: the idle SSE keep-alive comment frame.)

Landed in this unit: the sampler extension, skip_special_tokens (default true, explicit false
passthrough), the integer error code, and the buffered deepseek_v41 tool-call parser - each with
unit/router/e2e coverage as noted in section 12.

Landed in task-18: the TP=4 all-position prompt-logprobs rank contract
([tp4-prompt-logprobs-rank-contract.md](tp4-prompt-logprobs-rank-contract.md)) and the idle SSE
keep-alive comment frame (`sse_keep_alive`). The sampler on the TP=4 host path is explicitly not
attempted here: its sampling/penalty code path is the device-window seam, so it stays window work
rather than being approximated on the host.

Landed in task-20: the OpenAI surface gaps above
([openai-surface-gaps.md](openai-surface-gaps.md)). n>1 now returns ordered choices with per-choice
request ids and vLLM's seed+i derivation, counts the prompt once and sums completion tokens, and
streams per-choice finish chunks with one aggregate usage chunk; the TP=4 driver reports
independent_choices() = false so the router refuses n>1 typed (its group samples one shared
window). response_format parses text/json_object/json_schema: text is a no-op, json modes are a
typed 400 on param response_format instead of being silently ignored, and a malformed json_schema
wrapper is a body error. A bearer key via --api-key or SPARKFLASH_SERVE_API_KEY puts every route
except /health and /metrics behind a constant-time compared "Bearer" header with the documented
401 envelope; no key keeps the server open exactly as before. Evidence: router and debug-build
scripted serve_http wire tests plus the measured openai-surface-census artefact (section 12); n==1
responses and streams stay byte-identical (single-choice tests unchanged).

Landed in task-19: incremental streaming of V4.1 DSML tool-call deltas on the chat SSE path
([streaming-tool-calls.md](streaming-tool-calls.md)): call-start and argument-fragment deltas,
`finish_reason: tool_calls` on the final choice chunk, and a typed `tool_call_parse_error` for a
malformed or unterminated block instead of a silent drop. Evidence: split-at-every-boundary parser
property tests, router-level streamed-vs-buffered equality and keep-alive interleave tests, a
debug-build scripted-engine wire seed in serve_http.rs, and the measured
`tool-call-streaming-census-*` artefact. The streaming-logprob-plus-tools combination stays a
documented residual gap.

## 12. Evidence and reproduction

- Wire-level e2e: crates/sparkflash-cli/tests/serve_http.rs spawns the real binary on
  127.0.0.1:0 and checks /health, token-id completion, stop truncation, empty-stop 400, SSE
  usage/[DONE], the models card, the sampler acceptance/identity case, and skip_special_tokens
  in both directions (default hides the generated BOS token, explicit false shows it). Task-14 adds
  the ops-metrics e2e (counters matching usage, live running/waiting gauges, TTFT histogram) and the
  serial-vs-batched wire probes for prompt_logprobs and stop (serve_http.rs tests at lines 733, 873
  and 940). Task-15 adds the one-pass prompt-logprobs coverage: engine row-equality/bounds tests in
  `crates/sparkflash-engine/tests/prompt_logprobs_prefill.rs`, the old-vs-new driver equality test
  and span test in serve.rs (`:7809`, `:7892`) and the buffered e2e
  `buffered_prompt_logprobs_score_every_prompt_position_once` (serve_http.rs:1100). Task-19 adds
  the debug-build scripted-engine seed (SPARKFLASH_SERVE_SCRIPT; serve.rs, debug builds only) and
  two wire tests: the streamed DSML tool_calls array equals the buffered array byte for byte with
  keep-alive comments interleaved on the real socket, and malformed/unterminated blocks end in the
  typed tool_call_parse_error SSE frame. Because the seed replaces only the engine (tokenizer seam,
  router, metrics and bind stay production), the wire tests still exercise the shipped server.
- Router-level: crates/sparkflash-api/tests/router.rs (SSE byte-exact framing, logprobs shapes and
  refusals, chat/completions validation, health, empty stop, integer error codes, sampler
  pass-through/validation, deepseek_v41 DSML parsing with and without tools, and streaming DSML
  deltas: streamed-vs-buffered tool_calls equality, multi-call index/order, malformed and
  unterminated typed refusals, keep-alive/delta interleave).
- Unit-level: crates/sparkflash-api/src/tool_parser.rs tests both DSML shapes: the buffered parse
  (order, string vs JSON parameters, malformed block left as content) and the incremental streamed
  parse (one-shot buffered equality, split at every UTF-8 boundary, character-at-a-time feed,
  malformed/unterminated/trailing-text refusals, the 4 MiB bound, sticky refusal).
- Driver-level: crates/sparkflash-cli/src/serve.rs tests (StopBuffer unit tests and driver stop
  tests, logprob ordering, usage counts).
- Measured host census: scripts/serve-parity-census.py writes JSON+MD+server log under
  /home/user/ev-session-20261020/serve/. The current run is
  serve-parity-census-20261020T222156Z (commit d7c9622, branch task/serve-openai-parity,
  binary sha256 411dee72...): 20 capability probes, one single streaming request and a
  1/8/32-client streaming ladder with client-measured TTFT and tokens/s, all labelled
  "host path, synthetic tiny plan (vocab 128), UNBARED / diagnostic". The JSON records one
  dirty entry: the census-script adaptation used to probe the new OpenAI default (that edit is
  in this commit); the measured binary is exactly the d7c9622 build. Host loadavg during the
  capture was 13.3/12.7/11.2 on 88 CPUs. Headline numbers: single stream 64 tokens TTFT
  33.7 ms / 0.966 s end to end / 66.2 client tokens/s; ladder aggregate 68.8 / 72.3 / 63.1
  client tokens/s at 1/8/32 clients with TTFT p50 35.6 ms / 6.12 s / 32.11 s (the growth is
  queue wait behind one engine thread, which is the serving behaviour being recorded, not a
  throughput collapse). Every ladder rung completed: 164/164 accepted, 0 refused, 0 failed;
  the server counters moved by sparkflash_requests_total +178 and
  sparkflash_prompt_tokens_total +1424. A pre-extension run 20261020T203700Z (commit 3f4cb3a,
  binary sha256 500257be..., loadavg 20.2/17.3/14.9) measured the same workload at 52.2
  single / 57.8 / 61.3 / 59.9 ladder client tokens/s. The two runs are not a speed
  comparison: host load differed and the loopback census has no benchmark controls; both are
  recorded for regression tracking and parity triage only.
- Ops metrics and batched wire probes (task-14): [serve-ops-metrics.md](serve-ops-metrics.md) records
  the four request-lifecycle families and the schema-/2 S4 census
  serve-s4-batching-census-20261021T011248Z (commit dcf8d25, binary sha256 036e7b05..., loadavg
  8.80/9.34/8.82 on 88 CPUs). Both modes exported all four families; the generation-token counter
  delta was 5286 = the summed response completion tokens, the TTFT histogram count delta 91 = the
  number of token-producing requests, and the idle running/waiting gauges were 0/0. The ladder
  reproduced the S4 shape (32-client TTFT p50 31.18 s serial -> 1.44 s batched) and all seven
  buffered/SSE prompt_logprobs and stop wire probes were identical in payload shape and stable
  values across modes. This is host-path API-shape/accounting evidence, not a vLLM or device
  measurement.
- Prompt logprobs in one pass (task-15): [prompt-logprobs-one-prefill.md](prompt-logprobs-one-prefill.md)
  records the engine row seam (`prefill_with_position_logits`), the driver span path, the equality
  contract against the retained token-at-a-time reference (payload equality and f32-bit equality of
  the final logits/head_input) and the schema-/3 census
  serve-s4-batching-census-20261021T021856Z (commit fc624a0, binary sha256 4f53a655..., loadavg
  5.19/6.80/6.99 on 88 CPUs). Both modes scored every prompt position correctly, with identical
  payloads (8 positions, 7 scored, position 0 null, stable across repetitions); prompt logprobs are
  now measured as one pass over the prompt rather than one re-forward per position. The probe's
  requests are included in that census's accounting: 5298/5298 generation tokens and 103/103 TTFT
  samples in both modes, idle gauges 0/0.

  Task-18 (TP=4 all-position rank contract): the Phase-1 TP=4 host path no longer refuses prompt
  logprobs wholesale. Each rank screens its local LM-head shard, the ranks exchange bounded candidate
  blocks and integer greater-than-own counts over the forward's existing all-gathers, and a
  deterministic host merge reports the serial top order only when its f32 boundary proves it. The
  criterion and every refusal case are in
  [tp4-prompt-logprobs-rank-contract.md](tp4-prompt-logprobs-rank-contract.md); the measured
  rank-path evidence is `tp4-prompt-logprobs-census-*` under `/home/user/ev-session-20261020/serve/`
  (host path, synthetic 40-layer tiny twin fixture, UNBARED): four loopback ranks, every rank's merged
  payload bit-identical, top order exactly the serial host order, logprob delta within the pinned
  1 ULP (measured 0 on this plan), the bounded-budget refusal identical on every rank with the group
  recovering for the next request, plus a control-channel e2e pinning the `Request.prompt_logprobs`
  frame and the leading payload event. This remains host-path API/equality evidence, not a device or
  vLLM claim; a real four-process serve deployment on the released checkpoint is still window work.
- Streaming tool-call deltas (task-19): [streaming-tool-calls.md](streaming-tool-calls.md) records
  the incremental parser, the SSE mapping and the measured run
  `tool-call-streaming-census-20261021T034618Z` under `/home/user/ev-session-20261020/serve/`
  (json sha256 59ad82ec..., md sha256 e8f1c5fb..., test binary 06ec9667..., serve binary
  7ee2ea50...; an earlier pre-final capture 20261021T033527Z is retained beside it). Three runs on
  the loopback socket with the real serve binary and router and a debug-build scripted stub engine
  (UNBARED): in every run the streamed tool_calls array equals the buffered array byte for byte
  (content equal, final finish_reason tool_calls, arguments {"city": "Beijing", "days": 3}); the
  stream carries 5 content deltas, 1 call-start delta and 3 argument-fragment deltas in 11 SSE data
  frames, and 169/170/169 keep-alive comment frames interleave at monotonic whole-frame positions
  with none before the first data frame and none after the terminator; streamed request wall was
  1092/1097/1090 ms. The census JSON records git commit 86f1249 because the slice was uncommitted at
  capture; its source manifests pin the measured working-tree bytes and the task-19 commit adds docs
  after the probe. Host path, scripted stub, synthetic DSML fixture: parser/wire evidence, not a
  model, device or vLLM claim.
- What the current census shows matching vLLM on the reference path: health empty body,
  models card, stop truncation with finish_reason stop and the completing token counted,
  empty-stop 400, whole-vocabulary logprobs under the 1000 cap and over-cap refusal,
  determinism, seed reproducibility, skip_special_tokens (the default hides the synthetic
  plan's only generated special token, so text is empty; an explicit false returns the BOS
  text), sampler options accepted (top_k and the three penalties return 200), out-of-range
  sampler input refused with vLLM's range text and error.code == 400 (integer), and unknown
  model 404 with error.code == 404. top_p still cannot be distinguished on this wire: the
  synthetic plan's logits are non-finite, so every step pins token id 0 and the two top_p
  probes emit the same text; the census records "indistinguishable on non-finite synthetic
  logits", not evidence that top_p is dropped - the candidate filter is proven by the
  finite-logits unit tests. At the time of that census the remaining recorded gaps were
  response_format/min_p accepted and ignored (no schema fields) and n=2 refused; task-20 has
  since made n>1 served and response_format a typed refusal (audit sections 7 and 11 and the
  task-20 census below), while min_p is still absent. The earlier chat/tools probes were
  host-limited 503s because the synthetic vocabulary cannot embed released tokenizer ids;
  task-19's debug-build scripted wire seed now exercises the buffered and streaming
  deepseek_v41 parser over the real socket without a checkpoint (debug-only seam, documented
  in [streaming-tool-calls.md](streaming-tool-calls.md)).
- Sampler scope: on the default reference path top_p, top_k, repetition_penalty,
  presence_penalty and frequency_penalty are honoured (candidate filters around sample_gumbel;
  the default fast path is byte-identical to the pre-extension driver). The TP=4 feature path
  still samples with temperature and seed only; the buffered/streaming DSML parser sits above
  the engine seam and applies to both paths, but the TP=4 path itself emits no tool-call
  channel and is not measured here. The census measures the default reference path only, so
  it supports no claim about the TP=4 path.
- Sampler default-path wire capture (separate from the census): 8 replay requests recorded
  before and after the sampler commit,
  /home/user/ev-session-20261020/serve/sampler-extension-20261020T214016Z/baseline.json
  (commit 304ef5c) and .../sampler-extension-after-20261020T220543Z/after.json (commit 1f39eaa,
  binary sha256 944a128c...). All eight cases are byte-identical pre/post (status 200, text
  sha256 a800530c..., same usage): with non-finite synthetic logits every case pins token id 0,
  so the capture shows wire acceptance and no default-path regression, NOT knob effects; both
  files carry the label "host path, synthetic tiny plan (vocab 128), UNBARED / diagnostic".
- Measured host batching census (task-13, S4 slice): scripts/serve-s4-batching-census.py writes
  JSON+MD+sha256+server logs under /home/user/ev-session-20261020/serve/. The current run is
  serve-s4-batching-census-20261021T005030Z (commit 530d776, branch task/serve-openai-parity, binary
  sha256 a65cbe86db2c40912e4fa2b4bd6a0831dbb0f6e583bc2d8ebf53844e85b65c0c; two dirty entries, both
  documentation edits made after the commit). It measures the same streaming 1/8/32-client ladder in
  serial and opt-in batched mode: 32-client TTFT p50 27.295 s serial -> 1.369 s batched (19.9x lower)
  while aggregate client tokens/s 68.32 -> 62.94 (0.921x); three fixed buffered completions are
  byte-identical across modes; 85 requests per mode, 0 refused/failed/panics. Full record:
  docs/build/s4-host-continuous-batching.md. Same label and same non-claims: host path, synthetic
  tiny plan (vocab 128), UNBARED / diagnostic; no device TTFT/throughput claim and no vLLM
  comparison.
- OpenAI surface-gap wire census (task-20): scripts/openai-surface-census.py starts the real serve
  binary on loopback -- the synthetic tiny plan for token-id completion probes, the debug-build
  scripted stub for chat -- and speaks OpenAI HTTP to it. It probes n>1 buffered and streamed in
  serial and in --continuous-batching --max-batch 2 mode, the response_format and n=0 typed 400s,
  and bearer-key mode. Current run: openai-surface-census-20261021T040922Z under
  /home/user/ev-session-20261020/serve/ (JSON sha256 c0b1d879..., markdown sha256 96812ceb...);
  3 runs x 74 named checks all passing, with the serial and batched completion payloads equal by
  canonical hash. Serve binary sha256 f1dfb7b10d07f476658cffff75d606460f59f0700bec43db27d43e0175efde02,
  git e909f310 (the slice was uncommitted at capture, so the census pins its source manifests to
  the working tree and the task-20 commit adds this audit line after the probe -- the task-19
  pattern). The probe asserts ordered choice indexes [0, 1], prompt-counted-once / summed
  completion usage for n=2 chat and completions, per-choice streaming finish chunks with a single
  chat role chunk and one aggregate usage chunk, json_object/json_schema 400 param
  response_format, n=0 400 param n, the exact 401 envelope with /health and /metrics open, and the
  no-key open mode returning 200. Label: host path, synthetic tiny plan (vocab 128) and scripted
  stub, UNBARED; no device, GPU, vLLM or throughput claim.
- Reproduce: cargo test -p sparkflash-cli --test serve_http ; cargo test -p sparkflash-api --test
  router ; python3 scripts/serve-parity-census.py ; python3 scripts/serve-s4-batching-census.py ;
  python3 scripts/tool-call-streaming-census.py --runs 3 ; python3 scripts/openai-surface-census.py
  --runs 3.


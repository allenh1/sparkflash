# Serving logprobs (vLLM parity)

Branch `task/serve-logprobs`, task-20 (S2). Scope: the OpenAI-protocol logprobs surface of the
SparkFlash serving path, measured against the vLLM image the cluster runs as the behavioural
oracle. No device code, no engine numerics, no tokenizer work: the HTTP protocol layer and the CPU
reference driver are the whole change.

## What existed before

`crates/sparkflash-api` served the OpenAI request/response schema, the router and the engine seam
(`InferenceEngine`, `EngineEvent`) against `MockEngine` and the device engine. `EngineRequest`
carried no logprob fields, `StepOutput` carried raw logits for one position only, and neither
`logprobs` nor `prompt_logprobs` was parsed. `crates/sparkflash-cli/src/serve.rs` (the reference
driver) prefill+decoded through `sparkflash-engine` and streamed text deltas with per-request
timing stats. Both completion and chat responses were shaped, validated and streamed already; the
logprob work slots into those paths.

## Oracle

vLLM `0.1.dev20904+g179dd0fa9` (v1 engine) from
`vllm/vllm-openai:deepseekv41-flash-0909`, launched by the cluster recipe with `--max-logprobs 1000`.
The semantics below were read out of the image's source (extracted to `/tmp/vllm/src`) rather than
inferred: `vllm/sampling_params.py`, `vllm/entrypoints/openai/completion/{protocol,serving}.py`,
`vllm/entrypoints/openai/chat_completion/{protocol,serving}.py`, `vllm/logprobs.py`,
`vllm/v1/engine/logprobs.py`, `vllm/v1/sample/sampler.py`.

Key oracle facts the implementation mirrors:

* default `logprobs_mode="raw_logprobs"`: a token's logprob is `log_softmax(raw logits, float32)`
  *before* temperature, top-p or any penalty, so the same token scores the same at any temperature;
* `rank` is a competition rank: `1 + count(logit > this logit)` (ties share a rank);
* completion `top_logprobs` is a dict of `logprobs + 1` entries with the sampled token first, then
  the top-k alternatives, duplicates collapsed; it is *not* a "top-k excluding the sample" list;
* chat `top_logprobs` is a list of the first `top_logprobs` entries of that dict, sampled token
  first, and empty when `top_logprobs` is 0 or absent;
* `logprobs = -1` means the whole vocabulary, "unsorted and unranked" (rank omitted);
* a logprob below the floor is clamped to `-9999.0` ("`max(lp, -9999.0)`");
* `prompt_logprobs` is one list per prompt token, position 0 is `null`, and each position is an
  id-keyed dict `{id: {logprob, rank, decoded_token}}`;
* `text_offset` is the *character* offset of the token in the choice text (not bytes, not tokens),
  and it keeps advancing across streamed chunks;
* `echo` prepends the prompt text and its logprobs; with `max_tokens: 0` it is the
  echo-without-generation request, and the finish reason is `length`;
* completion `echo` without an explicit `prompt_logprobs` uses `logprobs` as the prompt count;
* an explicit `logprob_token_ids` list *replaces* the natural top-k selection, and both protocol
  layers drop `logprobs`/`top_logprobs` from the sampling parameters while the list is present, so
  the cap and the `logprobs == len(ids)` sampling rule are unreachable over HTTP;
* `prompt_logprobs` is refused with `stream: true` unless the value is 0 (vLLM accepts 0 and
  silently computes-then-drops it: a completion stream chunk has no `prompt_logprobs` field);
* `logprobs`, `prompt_logprobs` and `top_logprobs` are quoted in the error text and the cap error names
  `sample` or `prompt`: `"Requested sample logprobs of N, which is greater than max allowed: M"`.

## Request surface

`CompletionRequest` gained `logprobs: Option<i32>`, `prompt_logprobs: Option<i32>`,
`logprob_token_ids: Option<Vec<u32>>` and honours `echo`; `ChatCompletionRequest` gained
`logprobs: Option<bool>`, `top_logprobs: Option<i32>`, `prompt_logprobs: Option<i32>`,
`logprob_token_ids: Option<Vec<u32>>`, `echo: Option<bool>` (parsed for protocol completeness;
its implication is documented as not implemented, see deviations).

Validation runs *before* generation, so an impossible request is a 400 and never a truncated answer:

| case | error code | message (vLLM spelling) |
| --- | --- | --- |
| `logprobs < -1` | `invalid_logprobs` | logprobs must be a positive value or -1. |
| `prompt_logprobs < -1` | `invalid_logprobs` | prompt_logprobs must be a positive value or -1. |
| `top_logprobs < -1` | `invalid_logprobs` | top_logprobs must be a positive value or -1. |
| count or `-1` over `--max-logprobs` | `invalid_logprobs` | Requested sample/prompt logprobs of N, which is greater than max allowed: M |
| `-1` but the engine reports no vocabulary size | `invalid_logprobs` | logprobs = -1 requests the whole vocabulary, but this engine does not report its vocabulary size, so the request cannot be bounded |
| `logprob_token_ids` without `logprobs` | `invalid_logprobs` | completion: when using logprob_token_ids, logprobs must be set. chat: ... must be set to true. |
| more than 128 ids | `invalid_logprobs` | Requested logprob_token_ids of length N, which is greater than max allowed: 128 |
| an id outside the vocabulary | `invalid_logprobs` | token_id(s) [1, 2] in logprob_token_ids contain out-of-vocab token ids. Vocabulary size: V |
| chat `top_logprobs != 0` without `logprobs: true` | `invalid_logprobs` | when using top_logprobs, logprobs must be set to true. |
| `prompt_logprobs > 0` or `-1` with `stream: true` | `unsupported_parameter` | prompt_logprobs are not available when stream=True. |

The server cap is `AppState::max_logprobs`, default `DEFAULT_MAX_LOGPROBS = 20` (vLLM's
`--max-logprobs` default), settable through `AppState::with_max_logprobs` and the CLI's
`--max-logprobs` flag (accepted values: a positive number or `-1`, where `-1` becomes
`usize::MAX` and means "no cap": `--max-logprobs must be a positive value or -1 (-1 means no
cap)`).

## Response surface

Completion choice: `logprobs` is always present (`null` when not requested) and is
`{tokens, token_logprobs, top_logprobs, text_offset}`; `prompt_logprobs` is always present (`null`
when not computed) so a client can key on the field. Chat: `choices[0].logprobs` is always present
(`null` when not requested) as `{content: [{token, logprob, bytes, top_logprobs}]}`, and the
top-level `prompt_logprobs` (list of per-prompt-token dicts) is present only when computed.
Fields vLLM's dev build adds as `null` (`stop_reason`, `token_ids`, `prompt_token_ids`,
`routed_experts`, `prompt_text`) are omitted; clients treat absent and `null` alike and the
existing byte-exact SSE test depends on not emitting them.

`prompt_logprobs` example (position 0 is always `null`):

```json
[
  null,
  {"22": {"logprob": -0.125, "rank": 1, "decoded_token": "b"}},
  {"23": {"logprob": -0.25,  "rank": 2, "decoded_token": "c"},
   "24": {"logprob": -0.0625, "rank": 1, "decoded_token": "d"}}
]
```

`echo` text is the prompt text followed by the generated text, its `tokens` list is the prompt
tokens followed by the generated ones, its `token_logprobs[0]` is `null` (position 0 has no
conditioning token), and the prompt positions appear in `prompt_logprobs` too.
`max_tokens: 0` + `echo` returns the prompt text, no generated tokens, `finish_reason:
"length"`. In a completion stream the echo text and all prompt positions ride in the first chunk
and later chunks continue from the characters already sent. Temperature 0 is reproducible: the
test suite runs the same request twice and compares every logprob entry.

## Where it lives

* `crates/sparkflash-api/src/engine.rs` — seam types: `LogprobSelection` (`Top(n)` / `All`),
  `LogprobsRequest` (`sample`, `prompt`, `token_ids`), `TokenLogprob` (sampled entry plus `top`
  alternatives), `LogprobAlternative`, `TokenString`, `PromptLogprobs`, the
  `EngineEvent::TokenLogprobs` / `EngineEvent::PromptLogprobs` events, the `InferenceEngine::vocab_size`
  method (default `None`, i.e. "cannot serve `-1`"), and `MockEngine` builders for the tests.
* `crates/sparkflash-api/src/server.rs` — protocol resolution/validation, chat and completion
  assembly (including echo, output clamps, `bytes`, char offsets, streaming), everything the wire
  needs.
* `crates/sparkflash-api/src/{schema,state,lib}.rs` — request fields, `max_logprobs` state, re-exports.
* `crates/sparkflash-cli/src/serve.rs` — the reference driver's logprob math (log-softmax with an
  f64 accumulator, competition ranks, deterministic top-k, collapse, byte-fallback decoding,
  prompt pass) and `--max-logprobs`; `vocab_size` comes from `recipe.config.vocab_size`.

## Cost

The math itself is cheap: one `log_softmax` over the vocabulary per generated token, and per token
position one log-softmax over the prompt row. `prompt_logprobs` is the expensive part *today*:
the reference engine reports logits only for the position it just fed, so scoring prompt position
`i` needs the forward pass after `tokens[..i]`, i.e. one forward pass per prompt token (O(n) instead
of the one batched prefill). It is gated: no prompt logprobs are requested by default, and a request
that asked for none produces no logprob events at all (pinned by a test).

Measured on the host CPU reference driver, debug build, synthetic tiny plan (vocab 128, 32-token
prompt, 32 generated tokens). These numbers show the *shape* of the cost, not production cost:

| request | ttft_ms | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: |
| none | 43.3 | 740 | 41 |
| prompt logprobs only | 802.0 | 40 | 31 |
| sample top-2 | 61.9 | 517 | 37 |
| sample all (vocab 128) | 53.6 | 597 | 40 |
| sample top-2 + prompt | 723.1 | 44 | 42 |

So on this plan prompt logprobs cost ~25 ms per prompt token against ~1.35 ms per prompt token for a
batched prefill (18.5x TTFT for 32 tokens), while sample logprobs cost ~1.7 ms per generated token
in the same debug build. The device chain keeps the KV cache, so once it reports every position's
logits from one prefill this becomes one pass; that is the S3 follow-up recorded below.

## Deviations from vLLM (and why)

1. **`logprobs = -1` on a non-streaming completion reports the whole vocabulary.** The image's v1
   build returns an *empty* `top_logprobs` dict for that request (a dev-build bug: the gathering
   path caps the dict at the sampled token). We implement the documented "unsorted and unranked"
   all-vocab answer — id-ordered, `rank` omitted, entries clamped — because an empty dict silently
   loses the answer the client asked for. Both behaviours are pinned by tests here; the branch
   document is the flag.
2. **Chat `echo`.** vLLM derives chat `prompt_logprobs` from `top_logprobs` and prepends the final
   message to the continuation. This router does not render a chat template into the prompt yet
   (that is the tokenizer seam, task-30), so chat `echo` is parsed but its implication is not
   implemented; explicit `prompt_logprobs` works in chat exactly as in completions.
3. **`prompt_logprobs` serialization carries only the OpenAI fields.** vLLM's dev build serializes a
   pile of `null` extension fields on every choice; omitting absent fields is equivalent for typed
   clients and is required by the existing byte-exact SSE test.
4. **Object key order in the completion `top_logprobs` dict is sorted by token id**, because
   `serde_json::Map` is a `BTreeMap`; vLLM emits insertion order (sampled token first). Parsed
   values are identical; a byte-for-byte comparator (task-14) must compare objects, not key order.
5. **Completion with `logprob_token_ids` always reports the sampled token first** (as the vLLM chat
   docs describe: "in addition to the sampled token"). The image's completion path instead indexes
   the id dict with the sampled token and would fail if that id is not in the request; being
   lenient here cannot lose information.
6. **`prompt_logprobs: 0` + `stream: true` is accepted and dropped**, matching vLLM; it costs the
   O(n) prompt pass even though no stream field can carry it. Worth knowing when reading traces.
7. **The error `code` is `invalid_logprobs`/`unsupported_parameter`.** vLLM's OpenAI layer
   reports its own `BadRequestError`-style code; the message text and the `param` field match vLLM,
   the machine-readable code is this server's typed error taxonomy.
8. **Device engine vocabulary size.** `DeviceEngine` does not forward the recipe's vocabulary size,
   so `--engine device` refuses `logprobs = -1` with the "no vocabulary size" error. Delegating
   `vocab_size` through the device wrapper is a one-line follow-up in a crate this branch does not
   touch (its code is owned by the device chain work).
9. **Chat `top_logprobs = -1`** is accepted and means the whole vocabulary (selection `All`) rather
   than being refused; vLLM accepts it too. The list form (first N entries of the dict) is used for
   every positive N.
10. **Reasoning/tool channels.** `MockEngine` scripts logprobs per content piece, and the reference
    driver only generates content tokens; there is no reasoning channel in the engine yet, so the
    "logprobs cover every generated token regardless of channel" property is structural, not yet
    exercised. When a reasoning channel lands, its tokens must carry logprobs too.

## Known limitations and follow-ups

* **Prompt tokens on the real HTTP path are placeholders today.** This router still summarizes the
  prompt to a placeholder token list (Milestone 1 behaviour); the logprob layer is honest about
  whatever id list it is given, so `logprobs` values, ranks and decoded strings for prompt
  positions are only meaningful once the tokenizer seam (task-30) supplies real ids. Everything
  here is tested at the seam with `MockEngine` and on the synthetic plan, where token strings are
  the tokenizer's real decodes for the ids they score.
* **One pass for prompt logprobs** when the device chain keeps KV and reports all-position logits
  (S3, task-21).
* **`--max-logprobs` is per-process**: vLLM's flag is also per-request-capable; no per-request
  override exists in the OpenAI protocol, so this is identical.
* The math is computed on `f32` logits with an `f64` accumulator and then rounded once to `f32`;
  vLLM computes `log_softmax` on float32 directly. Differences are at the float32 epsilon and are
  covered by the numeric acceptance bars, not token equality.

## What is and is not meaningful today

* **The HTTP path still has Milestone-1 placeholder prompt ids until task-30 (tokenizer seam).**
  Consequence, stated plainly: the *values* of `prompt_logprobs` positions returned over HTTP are
  today computed over the wrong conditioning contexts and mean nothing as model behaviour; only the
  protocol shape is meaningful. Generated-token logprobs are real log-softmax over real logits, but
  the prompt they condition on is the placeholder id list. Do not read prompt-position numbers from
  the HTTP server as evidence about the model.
* **The D40 device-vs-oracle acceptance path is unaffected.** It does not consume HTTP prompt
  tokens: it joins on device positions and compares device logits/logprobs directly, so nothing in
  the acceptance campaign depends on the placeholder prompt ids.
* **`prompt_logprobs` is gated and off by default.** No request gets it unless it asks, and a
  request that asks for no logprobs emits no logprob events at all. Measured cost when it is on: one
  forward pass per prompt token (18.5x TTFT for a 32-token prompt on the synthetic driver, see the
  cost table above). The one-prefill fix is task-21/S3, so the O(n) path must not be enabled by
  default anywhere in the serving stack.

## Tests and gates

Host only, no GPU, no checkpoint:

```
cargo fmt --check                                   # exit 0
cargo check --all-targets                           # exit 0
cargo clippy --all-targets -- -D warnings           # exit 0
cargo test --workspace                              # exit 0
```

New coverage:

* `crates/sparkflash-api/tests/router.rs` (13 cases): completion logprob shape (tokens,
  token_logprobs, top_logprobs, text_offset, sampled-first dict, `-inf` clamp), prompt logprobs
  id-keyed with `null` position 0, echo text/tokens/offsets, echo with `max_tokens: 0` (finish
  `length`, zero completion tokens), chat list shape with `bytes`, chat `logprobs: true` alone
  (empty alternatives), chat prompt logprobs, all validation 400s above, explicit-id precedence,
  the lifted cap, streaming chat logprobs, streaming completion echo offsets.
* `crates/sparkflash-cli/src/serve.rs` unit tests: log-softmax is a distribution over the raw
  logits, competition ranks, deterministic tie-break by token id, sampled-first + alternatives
  dict, explicit ids replace the natural top-k, whole-vocabulary id order and unranked entries,
  byte-fallback decoding, no events when no logprobs were requested, prompt logprobs delivered
  after `Started` with `null` at position 0, temperature-0 reproducibility, whole-vocabulary
  coverage of the configured vocabulary.

## Files changed

`crates/sparkflash-api/src/{engine,schema,server,state,lib}.rs`,
`crates/sparkflash-api/tests/router.rs`, `crates/sparkflash-cli/src/serve.rs`,
`crates/sparkflash-device/tests/engine_offline.rs` (one struct field), and this document.

# deepseek-recipe 0.2 — adoption and API-surface parity

Status: investigation, 2026-10-20. Author: adoption/API-surface slice.
Scope: inventory the new official workspace, decide what we can adopt, and map its
request/response/streaming/usage surface onto `sparkflash-api`'s OpenAI-compatible
router ([server.rs](../../../crates/sparkflash-api/src/server.rs)). Rendering parity has its own
slice; this report only goes into rendering as far as the adopt/port decision requires, and it
does so with measurements rather than reading.

No repository file outside this document was modified. The experiments ran in a scratch crate in
`/tmp/recipe-probe` against the read-only clone at `/home/user/deepseek-recipe`
and our own `reference/tokenizer.json`.

---

## 0. TL;DR

1. **0.2.0 is published and permissively licensed.** All five crates (`recipe`, `core`, `encoding`, `image`, `macros`)
   are on crates.io at 0.2.0 (recipe published 2026-10-19T16:39Z), MIT, and the whole protocol
   crate builds with our toolchain (rustc 1.98.1) in ~34 s cold. Adoption is a version bump plus
   API migration, not a vendoring project.
2. **Do not adopt their renderer.** Recipe 0.2's V4.1 prompt renderer is *not* the released
   checkpoint/vLLM encoder. On our five golden vectors it differs from
   `reference/encoding/tests/test_output_*.txt` in **5/5** cases, with token-level deltas
   of +2, +2, −203, +1, −4 ids. Our [render.rs](../../../crates/sparkflash-api/src/encoding/render.rs)
   matches all five byte-for-byte; swapping it would invalidate every measured prompt.
3. **Their bundled tokenizer is not the checkpoint's.** `static/tokenizers/v41/tokenizer.json`
   (sha256 81f64d12…) renames id 129264 from `<｜deepseek_image｜>` (checkpoint, special=false) to
   `<｜image｜>` (special=true). Our `reference/tokenizer.json` (c90dfa01…) is the
   served file. If we adopt their conversion we must override `SpecialToken::Image`
   (`ConversionOptions::with_special_tokens`) or image requests render a token the
   checkpoint tokenizer does not contain.
4. **Their tokenizer *integration* is equivalent for ordinary prompts.** `PromptTokenizer`
   (encode plain regions without the added vocabulary, encode each explicit special range by id)
   and our `HfTokenizerEncoder` (`Tokenizer::encode(text, false)`) produce identical ids on a
   normal two-turn chat (42 == 42, 12 special ranges). They differ only when message text spells a
   special token: 9 ids for us vs 12 for them (`<｜User｜>` becomes one id vs 3 plain ids).
   Our behaviour is what vLLM does; theirs is the semantic fix. Keep ours for parity.
5. **The highest-value adoptable layer is the protocol layer** — the three request converters,
   the three chunk generators, `StreamProcessor`, strict json_schema normalization, and
   `OutputConstraint` — because we hand-roll all of it and are missing Responses/Messages SSE,
   tools, response_format, prefix continuation, deferred tool loading and tool search. Recipe
   explicitly does **not** implement constrained decoding; the constraint is only derived and
   handed to the backend.
6. **Real parity bugs found on our side**, independent of adoption:
   `max_tokens` default 4096 vs their 8192/65536/131072 (and cap 65536 vs 393216);
   thinking-mode `temperature`/`top_p` not normalized (they force 1.0 and `max(top_p, 0.95)`);
   usage lacks `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens`;
   Anthropic `input_tokens` includes cached tokens where Anthropic/recipe exclude them;
   three different reasoning-effort scales exist inside our own tree (25/50/75/100 on the HTTP
   path, 50/75/75/100 on the Conversation path, and a 75 default in `reasoning.rs`).

---

## 1. Inventory

### 1.1 Workspace at a glance

| crate | v0.2.0 | publish | license | normative deps | notes |
| --- | --- | --- | --- | --- | --- |
| `deepseek-recipe` | yes | crates.io | MIT | async-stream, base64, data-url, core, macros, indexmap, itertools, jsonschema 0.58, rand 0.9, serde, serde_json(preserve_order), strum_macros, tokenizers 0.23 (http), tokio-stream, xxhash-rust | protocol conversion + output parsing; 11.4k lines |
| `deepseek-recipe-core` | yes | crates.io | MIT | serde, serde_json | Conversation/Prompt/InputMessage/SpecialTokens; 1.5k lines |
| `deepseek-recipe-encoding` | yes | crates.io | MIT | aho-corasick, core, serde, serde_json, thiserror, tokenizers 0.23 (http), tracing | V4/V4.1 renderers + PromptTokenizer; 1.6k lines |
| `deepseek-recipe-macros` | yes | crates.io | MIT | proc-macro2, quote, syn (full, extra-traits) | `prompt!` template macro, `UntaggedDeserialize` derive |
| `deepseek-recipe-image` | yes | crates.io | MIT | data-url, core, futures, thiserror, tokio, tokio-stream, tracing; optional image/infer/opencv/reqwest | default features `opencv-preprocess` + `reqwest-fetch`; OpenCV via C++ toolchain |
| `deepseek-recipe-python` | yes (wheel) | PyPI | MIT | pyo3 0.27 abi3-py310 | cdylib `_native`; not relevant to our Rust server |
| `server-rs` | unpublished | — | MIT | axum 0.8, recipe, core, encoding, image | 273-line reference integration; the best template we have |
| `encoding-decoding-demo` | unpublished | — | MIT | axum, recipe, core, encoding | local prompt inspector |

Workspace facts: `edition = "2024"`, `resolver = 3`, `rust-toolchain.toml` pins
**1.97.1**, rustfmt `style_edition = 2024`. Our workspace is edition 2024 on 1.98.1 with
`rust-version = "1.88"`. The crate code uses let-chains (`if … && let`), already
available on 1.88+, and built cleanly under 1.98.1 — no toolchain obstacle. No `no_std`, no
wasm story: everything is std + tokio + tokio-stream (`StreamProcessor::into_protocol_stream`
is an `async_stream`). That is fine for our native single binary, and it does mean the
protocol layer cannot be reused on a wasm edge, which we do not need.

### 1.2 Dependency weight

A cold build of `deepseek-recipe` 0.2.0 + `encoding` + `core` + `macros` in a scratch crate
resolved **223 packages**, of which **20 are not already in our workspace lock**:

```
allocator-api2, borrow-or-share, data-encoding, data-url, deepseek-recipe-macros,
email_address, fluent-uri, jsonschema-regex, jsonschema-value, micromap, outref,
ref-cast, ref-cast-impl, referencing, strum, unicode-general-category, uuid-simd,
vsimd, xxhash-rust
```

Almost all of that is `jsonschema 0.58`'s tree. `tokenizers 0.23` (and its
`hf-hub`/`ureq` chain) is already in our lock, so the marginal cost of the protocol crate
is roughly **+20 crates, ~1.5 s extra link time, no new native library**. By contrast
`deepseek-recipe-image` with default features drags in `opencv` (system C++/clang, a
multi-minute build, a shared-library dependency at run time) — a hard no for a single static
serve binary until vision exists at all.

### 1.3 Versions and publish state (verified)

* crates.io sparse index: `deepseek-recipe 0.2.0`, pubtime `2026-10-19T16:39:17Z`,
  cksum `7e9a3c71…`; `0.1.0` was published `2026-09-10T08:06:34Z`.
* `cargo search` reports 0.2.0 for recipe, core, encoding and image; macros is a
  proc-macro member of the same release. So **no git dependency or vendoring is needed**.
* Our workspace pins `deepseek-recipe = "0.1"`, i.e. `>=0.1.0, <0.2.0`: **the new
  release is invisible to us today**. Our lock resolves 0.1.0 for recipe, core and encoding
  (`Cargo.lock` lines 578–610).

### 1.4 What we already take from 0.1

`sparkflash-api` already depends on `deepseek-recipe`, `deepseek-recipe-core` and
`deepseek-recipe-encoding` (workspace manifest: "Official DeepSeek V4.1 Rust encoding &
protocol toolkit") but uses only:

* `v4::{BOS,SYSTEM,USER,ASSISTANT,LATEST_REMINDER,EOS,DSML,THINKING_START,THINKING_END}_TOKEN` constants,
* `Conversation`/`InputMessage`/`ToolCall`/`ToolDefinition`/`ToolChoice`/`ReasoningEffort`/`ResponseFormat`,
* `ImageSource`/`ImageDetail`/`is_http_url`,
* `TokenizerEncoder`/`RenderedPrompt`/`EncodingError`,
* `util::json_formatter::stringify_python_style`.

The protocol crate (`deepseek_recipe::…`) is **declared and never imported** — `grep deepseek_recipe`
in `crates/` finds 19 hits, all core/encoding. So we pay for the dependency in build time but
get none of its request/response/streaming value.

### 1.5 0.1 → 0.2 API breaks that hit us

From `CHANGELOG.md` (`Migrating from 0.1`) and verified against the source:

| 0.1 API we use | 0.2 replacement | our call sites |
| --- | --- | --- |
| `util::json_formatter::stringify_python_style` | `core::json::stringify_python_style` (module `util` gone) | render.rs:20 |
| `v4::*_SP_TOKEN` / `THINKING_*` string consts | `core::special_tokens::{SpecialToken, SpecialTokens, default_special_tokens}` + `.token(SpecialToken::X)` (or `SpecialTokensMap`) | render.rs, encoding/mod.rs, seam.rs |
| `TokenizerEncoder::encode_ids(&self, text)` | `encode_ids(&self, text, special_ranges: &[Range<usize>])` | HfTokenizerEncoder impl, SparkV41Encoding::encode |
| `RenderedPrompt { prompt, image_sources }` | adds `special_ranges` | encoding/mod.rs:474 |
| `InputMessage::System { content }` / `User { content }` / `Tool { content, tool_call_id }` | adds `image_sources`, `System.tools`, `Assistant.prefix` | encoding/mod.rs::input_message_to_encoding, tests |
| `ToolDefinition { name, description, parameters, strict }` | adds `defer_loading` | encoding/mod.rs, tests |
| `ImageSource::{DataUrl,Url,Bytes}{…}` | each variant adds `path: String` | case.rs, tests |
| `ResponseFormat` | adds `Regex(String)` | matches in encoding/mod.rs |
| `PromptEncoding` trait | unchanged name/semantics, but `encode` now needs ranges | tests |

Rough size: one focused day to compile, plus golden re-verification, across
`encoding/mod.rs`, `encoding/render.rs`, `encoding/seam.rs`, `encoding/case.rs` and
`tests/encoding_*.rs`. Nothing in the engine or CLI changes.

---

## 2. Rendering: recipe 0.2 does **not** match the official encoder

Method. I built `/tmp/recipe-probe` against the 0.2 path crates, converted our five
reference fixture inputs into 0.2 `Conversation` values (representable subset; the fixtures are
the same ones our golden tests use), rendered with `DeepseekV41Encoding::render_conversation`,
compared byte-for-byte with `reference/encoding/tests/test_output_*.txt`, then tokenized both
sides with `reference/tokenizer.json` (`Tokenizer::encode(text, false)`).

| vector | recipe 0.2 vs golden | ids recipe/golden | first id diff | cause |
| --- | --- | --- | --- | --- |
| 1 tool-call thinking | differs at byte 61; 2500 vs 2476 B | 550 / 548 | 8 | effort rendered **75** where the encoder renders **50** |
| 2 multi-turn chat | differs at byte 294; 319 vs 294 B | 42 / 40 | none (prefix) | trailing `<｜Assistant｜></think>` appended after a final assistant message |
| 3 mid-conversation system with tools | differs at byte 246; 1549 vs 2481 B | 376 / 579 | 58 | mid-conv tools become `### Updated Tool Schemas` instead of the full `## Tools` instructions |
| 4 quick-instruction task | differs at byte 527; 585 vs 574 B | 102 / 101 | 98 | no `<｜action｜>` task token support, plus the trailing assistant header |
| 5 vision | differs at char 187; 400 vs 408 B | 70 / 74 | 50 | multi-part user content is concatenated; the reference inserts `\n\n` between parts around image placeholders |

These are not cosmetic. Vector 3 is a **203-token** prompt difference; vector 1 changes an id at
position 8. Both are exactly the kind of thing our endpoint measurements would attribute to the
engine.

The concrete rendering differences, in the order they matter:

1. **Reasoning effort scale.** 0.2 renders `low=50, high/xhigh/none=75, max=100`
   (dsv41.rs:21-30). The checkpoint encoder renders `low=25, high=50, xhigh=75, max=100`
   with default `high` (`reference/encoding/encoding.py`, `REASONING_EFFORT_MAPPINGS`).
   Our HTTP path renders 25/50/75/100 ([seam.rs:100-114](../../../crates/sparkflash-api/src/encoding/seam.rs)),
   our `Conversation` path renders 50/75/75/100
   ([encoding/mod.rs:284-293](../../../crates/sparkflash-api/src/encoding/mod.rs)), and
   `reasoning.rs`/`resolve_reasoning_effort` exposes a *third* table (low=50, high=75,
   default 75). Three scales in one tree is a parity hazard regardless of recipe.
2. **Trailing assistant header.** 0.2 always ends with `<｜Assistant｜>` + marker unless the last
   message is a prefix continuation. The reference appends the header to the *preceding* message and
   leaves a final assistant message bare. Our render.rs follows the reference.
3. **Mid-conversation tool schemas.** 0.2 emits a short `### Updated Tool Schemas` block
   (its deferred-tool-loading feature). The reference emits the whole `## Tools` instruction
   block again. Whichever is right for the next API, it is not the prompt our endpoint served.
4. **No task tokens.** 0.2's `InputMessage` has no representation for `<｜action｜>`,
   `<｜query｜>`, …; the checkpoint encoder does, and our `EncodingCase` models them.
5. **Multi-part joining.** Reference `process_image_messages` joins user parts with `\n\n`;
   0.2 concatenates text and image placeholders directly.
6. **Image spelling.** 0.2 uses `<｜image｜>` (and ships a matching modified tokenizer).
   The checkpoint uses `<｜deepseek_image｜>`, and id 129264 is that spelling with
   `special: false`.

Verdict: **our renderer stays**. Recipe 0.2's renderer is a different normalization (most likely the
public API service's, not the released vLLM image's), and adopting it would silently rewrite every
prompt we use as evidence.

---

## 3. Request/response conversion: what recipe has vs what our router has

### 3.1 Chat Completions

Recipe (`protocol/openai/chat_completion/{schema,convert}.rs`, a 611-line converter) supports and
validates: typed messages (text, image_url/file, tool_calls, tool results), `thinking` ({type:
enabled/disabled, budget_tokens}) overriding `reasoning_effort`, `tools`/`tool_choice`
(none/auto/required/named), `response_format` text/json_object/json_schema/regex,
`logprobs`/`top_logprobs` ([0,20]), `max_tokens`, `seed`, `n` (only 1),
`stream`/`stream_options` (`include_usage` requires stream=true), `stop`
(string/array), `prefix: true` on a final assistant message, strict-tool normalization and the
strict/non-strict mixing rules, tool-name validation (<=128 bytes, uniqueness), reasoning-content
passback rules, and billing-header stripping from system content. Model names are resolved through
an explicit registry and an unknown model is a 400 that lists the accepted names.

Ours ([schema.rs](../../../crates/sparkflash-api/src/schema.rs),
[server.rs:948](../../../crates/sparkflash-api/src/server.rs)): typed `messages` with `role`/
`content`/`name`/`tool_calls`/`tool_call_id`; `stream`/`stream_options`; `max_tokens` +
`max_completion_tokens`; `temperature`, `top_p`; logprobs (a superset: `logprob_token_ids`,
`prompt_logprobs`, `echo`); `tools` accepted **as a raw `Value` and never validated**;
`stop`; `seed`; `reasoning_effort`/`enable_thinking`/`chat_template_kwargs` via
[reasoning.rs](../../../crates/sparkflash-api/src/reasoning.rs); `n` (1 only); `user`. Missing
entirely: `response_format`, `tool_choice`, `prefix`, `thinking.budget_tokens`,
per-message `reasoning_content` in the typed schema (the tokenizer seam reads it from the raw
body, so the HTTP path keeps it), `file` content blocks, the `developer` role, and any
validation of tool schemas or tool-call/result pairing.

### 3.2 Responses

Recipe: `ResponsesRequest` with input items (message content incl. `input_image`, `input_text`,
unencrypted `reasoning_text`, function-call/function-call-output, `tool_search_output`),
`instructions`, `text.format` json_schema, `reasoning.effort`, `tools` including
functions, **namespaces**, the `apply_patch` custom tool and client-executed `tool_search`,
plus full SSE (`response.created`, `response.output_item.added`,
`response.output_text.delta`, `response.function_call_arguments.delta`, `response.completed`, …)
and a complete-response accumulator.

Ours ([server.rs:2104](../../../crates/sparkflash-api/src/server.rs)): non-streaming only —
`stream: true` is a 400 `unsupported_parameter` — `input` summarized to text (the tokenizer seam
has a `summarize_responses_input`), one `message` output item plus a `reasoning` item
(`summary`, not `reasoning_text`), no function_call items, no tools, no `text.format`,
no namespaces, no tool search, no reasoning history, no `incomplete_details` (finish reason maps only
through `status`).

### 3.3 Messages (Anthropic)

Recipe: typed Messages request with system string/blocks, images, `tool_use`/`tool_result`, thinking
blocks, `output_config`, server-tool rejection policy, and full SSE (`message_start`,
`content_block_start`, `content_block_delta` with `thinking_delta`/`text_delta`/`input_json_delta`,
`message_delta`, `message_stop`). Usage is `input_tokens` (non-cached) +
`cache_read_input_tokens` + `cache_creation_input_tokens=0` + `output_tokens` + `service_tier`.

Ours ([server.rs:2206](../../../crates/sparkflash-api/src/server.rs)): non-streaming only (400 on
`stream`), no tools, no thinking blocks in, thinking out as
`{"type":"thinking","thinking":…,"signature":""}`, and **`input_tokens` = prompt_tokens including cached**
(see §5.3).

---

## 4. Streaming and usage accounting

Recipe's pipeline is `InferenceChunk` -> `StreamProcessor` (output parser) -> `ChunkGenerator`
(protocol chunks) with usage accumulated in the processor. Interesting properties, measured with a
one-chunk-per-delta harness (see §8):

* `InferenceChunk::{Ready, Text{token_count}, Tokens{token_ids, logprobs}, Finish}`;
  `Ready` carries `PromptUsage { prompt_tokens, prompt_cache_hit_tokens }`.
* `CompletionUsage { completion_tokens, reasoning_tokens: Option<usize> }`.
  `reasoning_tokens` is **only** produced in thinking mode and **only** when a tokenizer is attached:
  the processor re-encodes the emitted reasoning text to count it. Without a tokenizer the field
  is `None` and the Chat Completions usage omits `completion_tokens_details` entirely.
* `completion_tokens` counts every processed token including discarded markup and the whole
  chunk that contains a stop sequence; trailing undecoded ids are excluded.
* Chat usage wire shape (probe output):
  `{"prompt_tokens":12,"completion_tokens":7,"total_tokens":19,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":3},"prompt_cache_hit_tokens":4,"prompt_cache_miss_tokens":8}`.
* Streaming deltas: first chunk `delta{role:assistant, content:null, reasoning_content:""}`;
  reasoning chunks `{content:null, reasoning_content:"…"}`; content chunks
  `{content:"…", reasoning_content:null}`; final chunk `{content:"", reasoning_content:null, finish_reason}`
  plus `usage` when the request is non-streaming **or** `stream_options.include_usage` is set.
  Intermediate chunks serialize `"usage": null` (field present), and the response includes
  `system_fingerprint: null`.
* Stop-sequence semantics: matching happens in answer content and detected JSON output only;
  reasoning, tool markup and JSON-adjacent text are excluded; a partial match backtracks.

Ours ([chat_stream_response, server.rs:1268](../../../crates/sparkflash-api/src/server.rs)):

| aspect | recipe 0.2 | ours |
| --- | --- | --- |
| first delta | `role + content:null + reasoning_content:""` | `{"role":"assistant"}` |
| per-chunk `usage` field | present, `null` until final | absent |
| `prompt_cache_hit_tokens`/`miss` | yes | no |
| `completion_tokens_details` | omitted outside thinking / without tokenizer | always present (0) |
| `system_fingerprint` | yes (null) | absent |
| `[DONE]` sentinel | `sse_done_data()` (chat: `[DONE]`, responses: none) | `data: [DONE]` always |
| Responses / Messages SSE | full event sets | 400 today |
| tool-call ids | `call_{index:02}_{ET_}{20 alnum}{checksum:04}` | engine-supplied ids, no ET marker |
| stop-sequence backtracking across chunks | implemented in parser | handled by engine output slicing (not verified here) |

The `ET_` marker is worth spelling out: recipe encodes "thinking-mode turn that produced no
reasoning content" into the tool-call id so that a later request's conversion can distinguish it
from "client dropped the reasoning". We have neither the marker nor the validation, which is fine
for one-shot requests and wrong for tool loops that round-trip through us.

---

## 5. max_tokens, sampling defaults, and usage — the concrete contradictions

### 5.1 resolve_max_tokens

Recipe (`request/mod.rs:162-270`):

* defaults: **8192** outside thinking, **65536** in thinking, **131072** in thinking with
  `reasoning_effort: max`;
* an explicit `max_tokens` is kept, then lowered to `context_limit - prompt_tokens`, then
  raised to at least 1;
* `prompt_tokens + requested > context_limit` is a 400 with the vLLM-style text
  `This model's maximum context length is N tokens. However, you requested T tokens (P in the messages, M in the completion). Please reduce the length of the messages or completion.`;
* the `max_tokens` upper bound defaults to `393_216` in both modes
  (`DEFAULT_MAX_TOKENS_UPPER_BOUND`); Chat Completions/Responses reject above it, Messages lowers
  instead.

Measured with the probe:

```
default thinking=true effort=Some(High) temperature=Some(1.0) top_p=Some(1.0) max_tokens=None
thinking:        resolve_max_tokens(1234, 1048576) = 65536
chat:            resolve_max_tokens(1234, 1048576) = 8192
thinking+max:    resolve_max_tokens(1234, 1048576) = 131072
explicit 4096:   resolve_max_tokens(1000000, 1048576) = 4096
prompt at limit: resolve_max_tokens(1048576, 1048576) = Ok(1)   # no max_tokens, total == limit
over context:    Err("This model's maximum context length is 1000000 tokens. …")
```

Ours ([server.rs:314](../../../crates/sparkflash-api/src/server.rs),
`DEFAULT_MAX_TOKENS = 4096`, `MODEL_MAX_OUTPUT_TOKENS = 65_536`):
absent budget -> **4096 always** (thinking or not, any effort); `max_tokens: 0` or
`> 65536` is a 400; no context-length interaction at the router (the engine validates
sequence length separately, and `/v1/models` advertises `max_output_tokens: 65536`). A
thinking request with no `max_tokens` therefore gets **16x fewer tokens** from us than from a
recipe-based server, and the vLLM image's own default (fill the remaining context) is a *third*
rule. This is measurement-affecting: a truncated answer changes `finish_reason`,
`completion_tokens` and the visible text.

**Open question requiring a measurement, not a guess:** what does the reference vLLM image do when
`max_tokens` is absent? Test plan in §10.

### 5.2 Sampling parameters

Recipe normalizes: `temperature` = 1.0 in thinking mode (the request value is ignored), otherwise
the request value or 1.0; `top_p` = max(requested, 0.95) in V4.1 thinking mode, 1.0 outside
(`ModelVersion::resolve_top_p`).

Ours: `temperature.unwrap_or(1.0)`, `top_p.unwrap_or(1.0)` in every mode
([server.rs:377-378](../../../crates/sparkflash-api/src/server.rs)). A client that sends
`temperature: 0.6, top_p: 0.7` with thinking on gets near-greedy sampling from us and 1.0/0.95
from a recipe-based server. Whether the vLLM image matches recipe or us is unverified; either way
the two candidate oracles disagree, so this must be decided explicitly and recorded.

### 5.3 Usage fields

Ours ([telemetry_types.rs:41](../../../crates/sparkflash-api/src/telemetry_types.rs)) emits
`prompt_tokens`, `completion_tokens`, `total_tokens`,
`prompt_tokens_details{cached_tokens, uma_cached_tokens, nvme_cached_tokens, image_tokens}`,
`completion_tokens_details{reasoning_tokens, accepted_prediction_tokens, rejected_prediction_tokens}`.
Recipe emits `prompt_tokens_details.cached_tokens` and
`completion_tokens_details.reasoning_tokens` only, plus the DeepSeek top-level
`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`. The two top-level cache fields are the
ones the official API documents; clients that compute cache savings read them. Adding them is a
five-line change in `OpenAiUsageWithCache` (computed as `cached_tokens` and
`prompt_tokens - cached_tokens`), and it is low risk.

Anthropic: recipe's `MessagesUsage::from` sets
`input_tokens = prompt_tokens.saturating_sub(prompt_cache_hit_tokens)`, with
`cache_read_input_tokens = prompt_cache_hit_tokens`. Ours sets `input_tokens = prompt_tokens`
while also reporting `cache_read_input_tokens = cached_tokens`
([server.rs:2273-2278](../../../crates/sparkflash-api/src/server.rs)) — the same tokens are counted
twice, so any client's cost accounting for `/v1/messages` is wrong by the cache-hit amount.
This is a plain bug against Anthropic's semantics, not a taste question.

---

## 6. Structured output and constrained decoding

Recipe derives an `OutputConstraint` per request (`protocol/output_constraint.rs`):

* tools present and tool_choice != none -> `ToolCalls { tools, required }` (`kind()` =
  `tool_call`/`tool_call_required`), with strict-tool mixing rules (all strict or none; a
  strict tool forbids any `response_format`, including `text`);
* otherwise `response_format`: json_object -> `Json { schema: None }`; json_schema ->
  `Json { schema: Some(normalized) }`; regex -> `Regex(pattern)`; text -> none;
* `OutputConstraint::schema()` serializes a constraint string; `kind()` is
  `json`/`regex`/`tool_call`;
* **no inference-side enforcement.** The README says the caller decides whether its backend applies
  it by constrained decoding, and there is no FSM/mask/logit-bias code anywhere in the workspace.

Two things are still valuable to us:

1. **json_schema validation and strict normalization** (`protocol/json_schema.rs`, 366 lines):
   deserialize into a typed `JsonSchema`, then for strict schemas require every property in
   `required`, forbid `additionalProperties: true`, reject empty nested objects, reject
   non-object top levels, keep `$def`, validate syntax with
   `jsonschema::options().offline().build()`. The probe shows the effect: a schema
   `{"type":"object","properties":{"a":{"type":"string"}}}` becomes the constraint string
   `{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false,"required":["a"]}`
   while the *prompt* keeps the schema as written (Python-separator JSON inside
   `## Response Format:`). Note the empty-system prelude for json_object produces
   `<System>\n\n\n\n## Response Format:` — worth matching byte-for-byte if we ever render it.
2. **A clean place to attach a constrained decoder**: `InferenceOptions::output_constraint` is
   already the interface a backend is expected to consume, including the tool-call case. That is
   the shape our sampler-side plumbing should expose.

Plumbing effort into our sampler is the expensive part and recipe gives us nothing for it: a
tokenizer-driven trie/FSM over the vocabulary per constraint kind (JSON, JSON Schema, regex,
tool-call DSML with per-parameter `string="true|false"` typing), a logit mask applied inside the
sampling loop of the device chain, prefix reuse across decode steps, thinking-mode handling (the
constraint starts after `</think>`), and stop/EOS legality. That is a multi-week feature
(3–6 weeks for a careful implementation with tests), not an adoption.

---

## 7. Tokenizer integration

`PromptTokenizer::new(&Tokenizer)` clones the tokenizer once, removes its added vocabulary
for plain text, and keeps the added vocabulary for explicit special ranges; `encode_ids` batches
the plain regions and splices in one id per range. Our `HfTokenizerEncoder` calls
`tokenizer.encode(text, false)` on the whole prompt.

Measured on the recipe-0.2-rendered vector 2 prompt (12 special ranges):
`ids_with_ranges == ids_hf_encode` (42 == 42). On a user message that literally contains
`<｜User｜>`: 12 ids with ranges vs 9 with our path — the special spelling becomes 1 id for us
(and for vLLM, which tokenizes the rendered string) and 3 plain ids for recipe. So the integration
is equivalent for normal traffic and deliberately different for a control-token-injection case;
ours is the more vLLM-faithful of the two.

The concrete gotcha is the **bundled tokenizer**: `static/tokenizers/v41/README.md` states it is a
*modified copy* (id 129264 renamed `<｜deepseek_image｜>` special=false -> `<｜image｜>` special=true),
and `SpecialToken::Image` defaults to `<｜image｜>`. Our `reference/tokenizer.json` (the served
file, sha256 c90dfa01…) contains `<｜deepseek_image｜>` at 129264 and no `<｜image｜>` at all. If we
adopt their conversion as-is for image requests, either `PromptTokenizer` errors
(`special token "<｜image｜>" is not in the added vocabulary`) or we encode a token id the checkpoint
does not use. Mitigation is one small value:

```rust
let mut tokens = SpecialTokensMap::default();
tokens.image = "<｜deepseek_image｜>".to_owned();
let options = ConversionOptions::default().with_special_tokens(tokens);
```

…and then never use their `static/tokenizers/` tree.

---

## 8. What the probe actually ran (reproduction)

Scratch crate `/tmp/recipe-probe` (not committed):

```toml
[dependencies]
deepseek-recipe-core = { path = "/home/user/deepseek-recipe/deepseek-recipe-core" }
deepseek-recipe-encoding = { path = "/home/user/deepseek-recipe/deepseek-recipe-encoding" }
deepseek-recipe = { path = "/home/user/deepseek-recipe/deepseek-recipe" }
serde_json = "1.0"
tokenizers = "0.23"
tokio = { version = "1", features = ["macros", "rt", "rt-multi-thread"] }
```

`src/main.rs`: fixture JSON -> 0.2 `Conversation` (system/user/assistant/tool/
latest_reminder, top-level `tools`, image blocks, effort aliases) -> `render_conversation` ->
byte diff against `test_output_*.txt` -> `Tokenizer::encode` id diff; plus the
`PromptTokenizer` vs `encode(text,false)` comparison and the literal-special-token case.
`src/bin/api.rs`: `ChatCompletionRequest::convert(ConversionOptions::default())` for
defaults, the `resolve_max_tokens` matrix, proof of the 400 text, json_schema/json_object/regex
constraints and rendered prompts, the required-tool-choice prompt, and chunk-generator output for
both streaming and non-streaming requests.

Run: `cd /tmp/recipe-probe && cargo run --release --bin recipe-probe` and `… --bin api`.
Cold build: 34 s. No cluster resources, no writes outside `/tmp`.

---

## 9. Adoption options, ranked by value/effort

### Option A — stay on 0.1, fix the API-surface bugs (XS, 0.5–1 day)

Do not touch the dependency. Fix: Anthropic `input_tokens` (subtract cached), add
`prompt_cache_hit_tokens`/`miss` to usage, decide and implement `max_tokens` defaults and
thinking-mode temperature/top_p, collapse the three effort tables to one (the 25/50/75/100
reference table on the HTTP path, with the recipe-scale table deleted or renamed to
`surface_effort`).

Value: removes wrong numbers from evidence today. Effort: tiny. Risk: none to prompts.

### Option B — bump to 0.2 and adopt the protocol layer, keep our renderer (M–L, 5–10 days)

Take `deepseek-recipe 0.2` for the three request converters, `StreamProcessor` +
`ChunkGenerator`, `OutputConstraint`, strict-schema validation, and the reference server glue
from `server-rs/src/lib.rs` (~100 lines). Keep
[render.rs](../../../crates/sparkflash-api/src/encoding/render.rs) as the renderer, feeding it
`Conversation -> EncodingCase` (extend the mapping for `prefix -> wo_eos`, latest_reminder,
images, `defer_loading`) and passing an **explicit** effort score so the recipe-scale mapping
never leaks into the prompt. Override `SpecialToken::Image`. Keep our tokenizer.

Sub-tasks and estimates:
1. 0.1->0.2 migration with our renderer intact: **1 day** (see §1.5).
2. Chat Completions conversion + chunk generator behind the existing handler, shadow-compared
   against the hand path on the router test corpus: **2 days**. This alone adds response_format,
   tool_choice, prefix, `thinking`/budget, validation error texts, ET_ ids.
3. Responses: full input items, tools/namespaces/tool search, SSE: **2–3 days**.
4. Messages: typed blocks, tools, thinking, SSE: **1–2 days**.
5. Re-baseline `tests/router.rs` (2,029 lines) and the logprob parity tests: **1–2 days**.

Total **7–10 engineer-days**, best value per unit of work because it deletes hand-rolled protocol
code we cannot keep in sync, and it is the only cheap route to Responses/Messages SSE.

### Option C — adopt the whole library including its renderer (XL, weeks + evidence reset)

Would require re-measuring everything (prompts change on 5/5 golden vectors) and shipping a
renderer that disagrees with the released vLLM image on effort, headers, mid-conv tools, task
tokens, image separators, and the image token. **Reject.**

### Option D — adopt `deepseek-recipe-image` (L, not now)

OpenCV build/C++ dependency, reqwest fetcher (needs `DestinationFilter` for untrusted URLs), and
our engine has no vision at all (D18 refuses image requests). Adopt only when vision lands, and
then with `default-features = false` plus our own preprocessor if possible.

### Ranked actions

| rank | action | value | effort |
| --- | --- | --- | --- |
| 1 | Bump to 0.2, adopt protocol conversion + chunk generators + StreamProcessor, keep render.rs/tokenizer (Option B) | removes ~3k lines of hand-rolled protocol code; adds Responses/Messages SSE, tools, response_format, prefix, validation parity | 7–10 d |
| 2 | Fix usage + sampling + max_tokens defaults (Option A items) | removes known-wrong numbers/wire fields from evidence now | 0.5–1 d |
| 3 | Adopt strict json_schema validation + OutputConstraint (from B, usable standalone) | correct 400s and a ready interface for constrained decoding | 0.5 d |
| 4 | Measure the reference image's max_tokens default, temperature/top_p behaviour, usage fields | turns three guesses into a recorded parity rule | 2 h |
| 5 | Constrained decoding (FSM + logit mask) | real structured output, not just prompt hints | 3–6 wk |
| 6 | `deepseek-recipe-image` | none until vision exists | — |
| 7 | Their renderer / bundled tokenizer / Python bindings | negative: breaks measured prompts | reject |

**Single highest-value adoption item:** adopt `deepseek-recipe 0.2`'s protocol layer (request
conversion, chunk generators, `StreamProcessor`) inside `sparkflash-api`, keeping
[render.rs](../../../crates/sparkflash-api/src/encoding/render.rs) and `reference/tokenizer.json`
as the rendering/tokenization oracle. **Effort: 7–10 engineer-days**, front-loaded as a 1-day
dependency migration plus a 2-day Chat Completions shadow comparison; the Responses and Messages
SSE work that follows is work we would otherwise hand-roll anyway.

---

## 10. Test plan (concrete, cheap, mostly runnable today)

1. **Renderer guard (already in CI).** `cargo test -p sparkflash-api --test encoding_golden`
   must stay green through any dependency bump. Add one assertion that
   `deepseek_recipe_encoding` is only used for types/constants, never
   `DeepseekV41Encoding::render_conversation` (a grep-based test is enough), so nobody
   "simplifies" the port back into a wrapper.
2. **Recipe renderer drift check (nightly, optional).** Run the §8 harness and fail if a 0.2.x
   patch release *starts* matching the goldens — that would mean the official renderer changed and
   our port needs review.
3. **Prompt-id parity against the reference endpoint.** For a fixed corpus (system+user,
   multi-turn thinking, tool-call history with reasoning, mid-conversation system with tools,
   prefix continuation), send the same body to the vLLM image with `max_tokens: 1` and
   `prompt_logprobs` if enabled, and compare our `seam.encode_chat` ids against
   `usage.prompt_tokens`. This is the only end-to-end token-parity oracle we have.
4. **max_tokens default probe (2 h).** Send `{"messages":[{"role":"user","content":"hi"}]}`
   with thinking on and off to the reference image and read `finish_reason` /
   `usage.completion_tokens` with a prompt engineered to answer briefly; then repeat with
   `max_tokens: 8192`. Three outcomes distinguish "fill context", "8192/65536", or something
   else. Record the result in DECISIONS.md and align `DEFAULT_MAX_TOKENS` /
   `MODEL_MAX_OUTPUT_TOKENS`.
5. **Sampling probe (1 h).** `temperature: 0, top_p: 0.1` + thinking on. If the reference image
   still explores, it normalizes like recipe; if it is deterministic, it passes through like us.
6. **Usage wire diffs.** Capture one streaming and one non-streaming response from the reference
   image and diff the JSON key set against ours (this report's §4 table is the checklist).
7. **Anthropic usage unit test.** `prompt_tokens=100, cached_tokens=60` -> `input_tokens=40`,
   `cache_read_input_tokens=60`.
8. **After Option B lands:** shadow mode on the router tests — run both the hand path and the
   recipe path over `tests/router.rs`'s corpus, compare converted prompts (must be equal after
   explicit effort retargeting) and response JSON (modulo ids/timestamps), then delete the hand
   path one protocol at a time.

---

## 11. Contradictions with our current implementation (checklist)

1. Effort scale: 0.2 50/75/75/100 vs our HTTP 25/50/75/100 vs our Conversation path 50/75/75/100 vs
   our surface table 50/75/75/100 default 75 — **three tables inside our tree**.
2. Trailing `<｜Assistant｜>` header after a final assistant message: 0.2 appends, reference does not.
3. Mid-conversation tools: 0.2 `### Updated Tool Schemas` vs reference full `## Tools`.
4. Image token: 0.2 `<｜image｜>` (+ modified tokenizer) vs checkpoint `<｜deepseek_image｜>`, id 129264,
   `special: false`.
5. Multi-part user content: reference joins with `\n\n`, 0.2 concatenates.
6. Task tokens (`action`/`query`/…): ours yes, 0.2 no.
7. `max_tokens`: ours 4096 fixed / cap 65536; 0.2 8192/65536/131072 / cap 393216; no
   context-aware clamping on our side.
8. Thinking mode: ours passes `temperature`/`top_p` through; 0.2 forces 1.0 and >=0.95.
9. Usage: ours lacks `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens`; always emits
   `completion_tokens_details`; no tokenizer-derived `reasoning_tokens` semantics.
10. Anthropic `input_tokens` double counts cached tokens.
11. `/v1/responses` and `/v1/messages` have no SSE; recipe has full event sets and accumulators.
12. No `response_format`, `tool_choice`, `prefix`, namespaces, `apply_patch`, tool search,
    `defer_loading`, or `thinking.budget_tokens` on our side; `tools` is an unvalidated
    `Value`.
13. Tool-call ids: no `call_XX_[ET_]…checksum` convention, so empty-thinking rounds are
    indistinguishable from dropped reasoning on a round trip.
14. Model name resolution: we accept any case-insensitive leaf match; recipe rejects unknown names
    with a message listing accepted ones. Our `/v1/models` advertises
    `max_output_tokens: 65536`, which will disagree with whatever `max_tokens` rule we settle on.
15. `system_fingerprint` absent from our chat responses (recipe emits it, null).

## 12. Not verified / caveats

* I did not run the reference vLLM image, so §5's "what the endpoint does" is a decision to be
  measured (test plan §10.4–10.5), not a finding.
* Building `deepseek-recipe-image` was not attempted (OpenCV not needed for this decision).
* `encoding-decoding-demo` and `deepseek-recipe-python` were read only for scope; the Python
  wheel is irrelevant to the Rust serve binary.
* Line numbers refer to the checked-out revision `935817c` of `deepseek-recipe` (0.2.0) and
  the current `sparkflash` working tree.

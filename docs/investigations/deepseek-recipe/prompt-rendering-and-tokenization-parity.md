# Prompt rendering and tokenization parity: deepseek-recipe 0.2.0 vs our api/serve crates

- **Author:** subagent (slice: prompt rendering + tokenization)
- **Date:** 2026-10-20
- **Subject:** `/home/user/deepseek-recipe` @ shallow clone (workspace 0.2.0, crates.io 0.2.0) vs
  `crates/sparkflash-api/src/encoding/*` and `crates/sparkflash-cli/src/serve.rs`.
- **Constraint honoured:** no repository file was modified by this investigation (the only write is
  this report; the only build artefact is `target/` from running the existing test suite), no cluster
  job was started, nothing was pushed.

---

## 0. Verdict in one screen

1. **Our serving prompt path is the vLLM path, and it is correct.** `TokenizerSeam::encode_chat`
   (`encoding/seam.rs:247`) renders a prompt that is byte-identical to
   `reference/encoding/encoding.py` as consumed by the vLLM image, and our Rust tokenizer reproduces
   the **recorded vLLM token ids on all five reference fixtures** (548 / 40 / 579 / 101 / 74 ids,
   exact match, §5.2). Nothing in the release invalidates that.
2. **deepseek-recipe 0.2.0 is *not* vLLM-parity, and adopting its renderer verbatim would move every
   prompt we publish.** Measured on identical conversations: default thinking effort **75 vs 50**,
   `low` **50 vs 25**, tool schemas re-serialised in a fixed key order with `"description": ""`
   inserted, image placeholder `"<｜image｜>"` (absent from our tokenizer) with no `"\n\n"` separator,
   `</think>` emitted for prefix continuations, an extra assistant header after `latest_reminder` and
   after an empty conversation, `<｜latest_reminder｜>` moved after the assistant header, and
   `tool_choice: required` priming `<｜DSML｜ calls>`. §4 has the vector-by-vector table.
3. **Tokenizer files are effectively identical** (same 128 000-token BPE vocab, same 1 283 added
   tokens, same ids) **except the image token spelling**: ours is `<｜deepseek_image｜>` (id 129264),
   the recipe's bundled V4.1 tokenizer spells the same id `<｜image｜>`. Everything else is id-for-id
   equal.
4. **Our repo carries three renderers with three different semantics**: the seam (vLLM scale
   25/50/75/100, default 50), `render_case` (same), and `SparkV41Encoding::render_conversation`
   (recipe 0.1 scale 50/75/75/100, default 75). They disagree with each other on the default
   thinking prompt and on tool-schema ordering. That is the real internal defect this exercise found.
5. **Two request-side defaults silently distort measurements**: our `max_tokens` default is **4096**
   (`server.rs:59`) where the DeepSeek API contract implemented by the recipe is 8192 / 65536 / 131072
   by mode; and `response_format` is not parsed at all on the serve path, so a `json_schema` request is
   answered unconstrained **and** without the schema in the prompt (0.1 cannot even represent it).
6. **Highest-value adoption (2–3 days):** move to `deepseek-recipe-core 0.2` +
   `deepseek-recipe-encoding 0.2` (MIT, published, light) and rebuild our single renderer on their
   `Prompt` / `RenderedPrompt::special_ranges` / `PromptTokenizer` model, keeping *our* (encoding.py)
   semantics pinned. That is the prerequisite for prefix continuation, deferred tools, response-format
   rendering and `reasoning_tokens`, and it makes "template control token vs user text" explicit.
   Do **not** adopt `DeepseekV41Encoding::render_conversation` for the serve path.

---

## 1. Method and evidence

Everything below is measured unless it is explicitly marked *reasoned*. Three renderers were driven
from one shared vector file (`/tmp/rendercmp/vectors.json`, 26 vectors, Appendix A):

| Side | What runs | Why |
| --- | --- | --- |
| `py` | `reference/encoding/encoding.py` (`encode_messages`, `drop_thinking=True`), tools attached to the first system message the way the vLLM wrapper does | this is the prompt oracle the measured endpoint uses (`seam.rs:10-62`) |
| `rc` | **deepseek-recipe 0.2.0** from the checkout: `DeepseekV41Encoding::render_conversation` + `PromptTokenizer` | the new official Rust stack |
| `port` | our Rust: `SparkV41Encoding::render_conversation` from the **published 0.1.0** crate plus our renderer | the code we actually ship |

Plus the repo's own suite: `cargo test -p sparkflash-api --test encoding_golden --offline` → **13
passed, 0 failed** (5 golden fixtures + conversation-path cross-checks + tokenizer test), and the
recorded vLLM oracle in `docs/build/task30-evidence/tokenizer-parity.json`.

Two harnesses were built outside the repo (`/tmp/rendercmp` for 0.2.0, `/tmp/rendercmp2` for our
0.1-based crate; repro commands in Appendix B). Both are ~90-line `main.rs` files, no repo changes.

---

## 2. The two-oracle problem (read this before deciding anything)

There are now **two official prompt renderers that disagree**:

* `encoding/encoding.py` — the checkpoint's encoder, copied into the vLLM image as
  `vllm/tokenizers/deepseek_v41_encoding.py` (provenance pinned in `seam.rs:43-62`, hashes in
  `docs/build/task30-evidence/tokenizer-parity.json`);
* `deepseek-recipe-encoding 0.2.0` — the new Rust workspace (`v4/dsv41.rs`).

They are not the same function. The recipe is a *rewrite with intentional changes*: default effort
75 instead of 50, `low` 50 instead of 25, fixed tool-schema key order, `<｜image｜>` placeholder,
`prefix` support, mid-conversation `### Updated Tool Schemas`, `tool_choice: required` priming, an
assistant header after `latest_reminder` and after an empty conversation.

**Recommendation:** keep vLLM (`encoding.py`) as the oracle for anything that touches the measured
endpoint, and use the recipe only for (a) its type/tokenizer plumbing and (b) the DeepSeek **API
contract** rules (max_tokens defaults, validation texts, `reasoning_tokens` accounting). Record the
choice in `docs/DECISIONS.md`; otherwise the next person "adopts the official renderer" and every
number moves. Note that `encoding/render.rs:1-16` already documents four divergences — they were
measured against **0.1** and every one of them is still true of **0.2.0**, plus three new ones.

---

## 3. The V4.1 rendering path in recipe 0.2.0, precisely

All references are `/home/user/deepseek-recipe/`.

### 3.1 Data model

* `deepseek-recipe-core/src/prompt.rs:1-16` — a rendered prompt is a `String` plus
  `special_ranges: Vec<Range<usize>>`. `Prompt::push_special` (`:54`) records a range;
  `push_plain` (`:62`) does not. **This is the structural idea worth copying**: control tokens are
  inserted by name, never by string concatenation, so a spelling in user text can never become a
  control token.
* `deepseek-recipe-core/src/special_tokens.rs:58-79` — spellings; `Image` is `"<｜image｜>"` (`:68`).
* `deepseek-recipe-core/src/multimodal/mod.rs:18` — `IMAGE_SPECIAL_TOKEN_ID = 129264`.
* `deepseek-recipe-core/src/json.rs:104-111` — `stringify_python_style`, `", "` / `": "` (Python
  `json.dumps` default separators, `ensure_ascii=False` semantics).

### 3.2 V4.1 skeleton (`encoding/src/v4/dsv41.rs`)

1. `normalize_messages` (`:321-344`): join consecutive `user` messages with `"\n\n"`; keep only the
   first of consecutive `latest_reminder` messages; **no system-message hoisting or dropping**.
2. If `conversation.tools` is non-empty and `tool_choice != None`, **or** a response format is
   requested, insert an empty system message at index 0 (`:177-183`).
3. Append the tool prompt to the **first** system message's content (`:185-190`), preceded by `\n\n`.
   A *later* system message carrying `tools` gets only `"\n\n### Updated Tool Schemas\n\n" + schemas`
   (`:173-180`) — that is deferred tool loading.
4. `drop_thinking = !renders_tool_prompt(conversation)` (`:286`; `v4/mod.rs:94` means
   `tool_choice != None && !tools.is_empty()`). **All** historical reasoning is dropped when there is
   no tool prompt; the only exception is a `prefix` message.
5. Render each message (`:130-235`):
   * effort prompt only at `index == 0` **and** thinking mode, inside the `<｜System｜>` token
     (`:144-156`); `Reasoning Effort: {50|75|100} (range 1-100, ...)\n\n`;
   * `user`/`separator`: `<｜User｜>` before the first user turn *unless the previous message was a
     user or an image-bearing tool message* (`:157-160`); user content is a `Prompt` so images keep
     their placeholder;
   * `system` at `index > 0`: `<｜System｜>` + content;
   * `assistant` (`:200-234`): `<｜Assistant｜>` + reasoning block + content + DSML tool calls;
     reasoning is rendered as `<think>` + reasoning + `</think>` (`:219-231`), and is **omitted**
     (bare `</think>`) when dropped;
   * `tool`: wrapped in `<tool_result>…</tool_result>`, consecutive results joined by `"\n\n"`, one
     `<｜User｜>` for the group.
6. Terminator (`:291-311`): `EOS` unless `is_prefix_continuation` (`v4/mod.rs:100-105`: last message
   assistant with `prefix: true`); then `<｜Assistant｜>` + `<think>` / `</think>` unless prefix
   continuation; then, **only if** `tool_choice == Required` and tools exist, `"\n\n<｜DSML｜ calls>"`
   (`:307-310`).
7. `render_tool_calls` (`:91-121`) builds `<｜DSML｜ invoke name="…">` blocks; arguments are parsed
   from the JSON string, a non-object becomes one `arguments` string parameter (`v4/mod.rs:38-43`).

### 3.3 Effort table (the single most dangerous delta)

`dsv41.rs:21-30`:

```
Some(Low) => "50"
None | Some(High) | Some(Xhigh) => "75"
Some(Max) => "100"
```

`reference/encoding/encoding.py:398-404`:

```
REASONING_EFFORT_MAPPINGS = {"low": 25, "high": 50, "xhigh": 75, "max": 100}
DEFAULT_REASONING_EFFORT = "high"     # -> 50
```

Proof that vLLM uses the second table (recorded oracle, not inference):
`docs/build/task30-evidence/tokenizer-parity.json` `chat_user_hello_default` ids contain
`…223, 1328…` = `" 50"`, and `chat_system_user_low` contains `…223, 1069…` = `" 25"`. Our harness
reproduces both id lists exactly; the recipe renders `75` and `50` for the same two requests.

---

## 4. Measured differences: our renderer vs recipe 0.2.0

"identical" = byte-identical prompt. Only the disagreements are listed (11 of 26 vectors are
identical: `chat_plain`, `chat_system`, `xhigh`, `max`, `consecutive_users`, `tool_call_and_result`,
both response-format vectors, `user_contains_bos`, `user_contains_dsml`).

| vector | py (vLLM / us) | recipe 0.2.0 | first diff |
| --- | --- | --- | --- |
| `think_default_effort` | `Reasoning Effort: 50` | `Reasoning Effort: 75` | byte 49 |
| `think_low` | `Reasoning Effort: 25` | `Reasoning Effort: 50` | byte 49 |
| `think_high` | `Reasoning Effort: 50` | `Reasoning Effort: 75` | byte 49 |
| `tools_off` | client key order, `description` omitted if absent | `{"name","description","parameters"}`, always | byte 906 |
| `tools_desc_missing` | 1058 chars / 234 ids | 1077 chars / 238 ids (`"description": ""`) | +19 chars, +4 ids |
| `tools_required` | no priming | `\n\n<｜DSML｜ calls>` appended | +17 chars, +5 ids |
| `prefix_continuation` | `<think>step one</think>` (+EOS on the 0.1 `Conversation` path) | `<think>step one` | 167 vs 159 chars |
| `prefix_continuation_with_content` | `…</think>partial` + EOS | `…</think>partial` | same chars, EOS differs |
| `json_schema_format` | 62 chars / 7 ids (0.1 path: no schema rendered) | 232 chars / 50 ids | +43 ids |
| `latest_reminder` | `…<｜Assistant｜></think><｜latest_reminder｜>be brief` | `…<｜latest_reminder｜>be brief<｜Assistant｜></think>` | byte 32 |
| `empty_messages` | `BOS` (21 chars) | `BOS<｜Assistant｜></think>` (42) | byte 21 |
| `image_in_user` | `describe\n\n<｜deepseek_image｜>` | `describe<｜image｜>` | byte 37 |
| `tool_result_contains_think_end` | effort digit only | effort digit only | (same string otherwise) |

Notes that matter operationally:

* The effort delta **does not change prompt token count** (both are one digit token), it changes the
  *id*: 1328 (`50`) vs 2402 (`75`). So our prompt-length statistics would look unchanged after an
  accidental adoption while every thinking response silently changes length and quality.
* `tools_off` has the **same length** in both stacks (1176) but a different key order, so a naive
  length/hash check on prompt *size* will not catch it.
* `tool_choice: required` (17 chars, 5 ids) and `json_schema` (43 ids) are the only changes that move
  our prompt token counts materially.

### 4.1 Our three internal paths disagree (independent defect)

Measured with the same `Conversation` on the 0.1-based `SparkV41Encoding`:

* `render_conversation` default thinking = **75**, `low` = **50** — it goes through
  `recipe_effort_score` (`encoding/mod.rs:284-293`) and `retarget_reasoning_effort`
  (`mod.rs:49-77`) and therefore matches the *recipe*, not the endpoint;
* `render_case` and the seam use `python_effort_score` 25/50/75/100 (`mod.rs:252-282`,
  `seam.rs:100-114`) and therefore match vLLM;
* `render_conversation` also rebuilds tool schemas in recipe order
  (`tool_definition_to_open_ai`), while the seam passes the client's `function` object through
  (`seam.rs:406-443`). Same request → two different prompts, both "correct" by their own tests.
* `render_conversation` puts the image placeholder straight after the text (76 chars) while the seam
  joins blocks with `"\n\n"` (78 chars, `seam.rs:365`).

Any Rust-side consumer of `SparkV41Encoding::render_conversation` (it is exported in
`sparkflash_api::encoding`, and used by `tests/encoding_golden.rs:263,271,279,316`) gets a different
default-thinking prompt than the HTTP server. Either route it through the same resolver or delete it.

---

## 5. Tokenization

### 5.1 Tokenizer files

`reference/tokenizer.json` (6.37 MB, sha256 `c90dfa01…`, byte-identical to the served snapshot) vs
`deepseek-recipe/static/tokenizers/v41/tokenizer.json`:

* model `BPE`, vocab 128 000 — **equal**;
* `merges` — **equal**;
* `added_tokens` 1 283 — **equal ids**, and the only spelling difference is id **129264**:
  `<｜deepseek_image｜>` (ours) vs `<｜image｜>` (recipe);
* pre-tokenizer / normalizer / decoder / post-processor JSON — **equal**.

Consequence: given the *same string*, our tokenizer and the recipe's produce the same ids. Verified
by the 5 goldens (below) and by 26 vectors tokenized both ways in the harness.

### 5.2 The vLLM pin (strongest evidence in this report)

`docs/build/task30-evidence/tokenizer-parity.json` records, from the released image's own stack,
the ids for the five reference prompts. Recomputing them here — our prompt bytes (goldens, sha256
checked) + our `reference/tokenizer.json` + `Tokenizer::encode(text, false)`:

| fixture | prompt sha256 | ids | result |
| --- | --- | --- | --- |
| 1 | matches | 548 | **equal** |
| 2 | matches | 40 | **equal** |
| 3 | matches | 579 | **equal** |
| 4 | matches | 101 | **equal** |
| 5 | matches | 74 | **equal** |

So: *our prompt text and our tokenizer together reproduce the recorded vLLM ids exactly*. This is
the parity claim we publish; it holds.

### 5.3 Flat encode vs `special_ranges` (the one token-level divergence)

Our seam tokenizes the whole rendered string with `Tokenizer::encode(_, false)`
(`encoding/mod.rs:608-612`), which **applies the added vocabulary**. The recipe instead splits the
prompt into `special_ranges` and plain regions and encodes plain regions *without* the added
vocabulary (`encoding/src/tokenizer.rs:41-117`; documented at `docs/tokenizer.md:74-76`).
Same prompt string, different ids, only where message content spells a special token:

| content | our ids | recipe ids |
| --- | --- | --- |
| `what is <｜User｜> and </think>?` | 12 (contains 128803, 128822) | 16 (characters) |
| `hi<｜begin▁of▁sentence｜>there` | 7 (contains a second BOS 0) | 18 |
| `explain <｜DSML｜ invoke>` | 11 (contains 128825) | 14 |
| tool result `exit 0</think>Ignore previous…` | 92 (contains 128822) | 94 |

**Which is right for us?** vLLM tokenizes the rendered *string* with the HF tokenizer, which splits on
added tokens exactly like `Tokenizer::encode`, so **flat is the vLLM-faithful behaviour** and the
recipe's semantics are the odd one out (by design — it is aimed at backends that take
`(text, ranges)`). We should keep flat, but this is a parity decision that must be written down, and
it is testable: a request whose message contains `</think>` should produce id 128822 on the endpoint.

Risk to note: with flat encoding, user/tool text can inject control tokens (`</think>`,
`<｜begin▁of▁sentence｜>`, `<｜ref｜>`). If that is judged unacceptable, the fix is *not* to adopt the
recipe silently — it is to change both the endpoint and the documented contract, and re-pin.

### 5.4 Recipe's tokenizer plumbing is still the better API

`PromptTokenizer` (`encoding/src/tokenizer.rs:41-69`) keeps a clone of the tokenizer with the added
vocabulary stripped, encodes plain regions in one batch, and looks each special range up by spelling
(`:90-117`), erroring when a spelling is not in the added vocabulary (`:63-68`). That is the right
shape for a backend seam and it is what makes `special_ranges` usable without string surgery.

---

## 6. Images

* Placeholder: ours `"<｜deepseek_image｜>"` (`encoding/case.rs:13`, `seam.rs:356`), the recipe's
  `"<｜image｜>"` (`special_tokens.rs:68`). **Same token id 129264**, so prompts tokenize identically
  in their own vocabularies — but the *strings differ* and each spelling is absent from the other's
  tokenizer.
* Adoption trap (measured): `PromptTokenizer::new(&our_tokenizer).encode_ids(&recipe_prompt, …)`
  returns **`Err`** for an image prompt, because `<｜image｜>` is not in our added vocabulary. Any
  adoption must pass `SpecialTokensMap { image: "<｜deepseek_image｜>", .. }`.
* Separator: encoding.py joins content blocks with `"\n\n"` (`seam.rs:365`), so ours renders
  `describe\n\n<｜deepseek_image｜>`; the recipe pushes the placeholder directly after the text
  (`describe<｜image｜>`) — 2 extra newline tokens and no leading space.
* Image token accounting: recipe `ImageTokenSpec::v41()` = patch 14, downsample 3, cap 1024 tokens,
  upscale below 544² pixels with an iterative fit
  (`core/src/multimodal/token_spec.rs:74-120`), and `MultiModalData::image_token_adjustment` adds
  `calc_token_count(w,h) - 1` per image (`multimodal/mod.rs:190-204`). Ours is a primitive
  `count_image_tokens(grid_h, grid_w) = ceil(h/3)*ceil(w/3)` (`multimodal.rs:190-194`) fed a *patch
  grid*, not pixels. Irrelevant while the engine refuses images (`server.rs:365-370`), but it is not
  the same function — adopting the spec is the correct move before vision is enabled.

---

## 7. Things the recipe models that we do not

| Feature | Recipe | Ours | Impact |
| --- | --- | --- | --- |
| Prefix continuation | `InputMessage::Assistant{prefix}`, `dsv41.rs:297-306`; validated "final assistant only" (`chat_completion/request/convert.rs:232-247`) | no field on the serve path; 0.1's `wo_eos` exists only in the case schema (`case.rs`), and still emits `</think>` (`render.rs:480`) | a client prefill becomes a **completed turn + EOS + new assistant header** instead of a continuation. Highest-risk *shape* mismatch after effort. |
| Deferred tool loading | `ToolDefinition.defer_loading` hides a tool from the initial prompt; a later system message adds `### Updated Tool Schemas` (`dsv41.rs:70-88,173-180`) | `defer_loading` field does not exist in 0.1; ignored | tool prompt 4+ tokens longer, model sees all schemas |
| Mid-conversation system messages | `<｜System｜>` at `index > 0` and assistant header after it (`dsv41.rs:167-171`) | supported in the case path; collapsed into the "user" path by our seam normaliser | only reachable via the Messages API |
| `tool_choice: required` | primes `\n\n<｜DSML｜ calls>` (`dsv41.rs:307-310`) | ignored | 5 ids |
| `latest_reminder` | after the user turn, header at end | header before the reminder | not exposed by our API |
| Task tokens (`action`, `query`, …) | **not supported** | supported (`render.rs:169-179`) | our superset; keep |
| Grounding tokens (`<｜ref｜>`, `<｜box｜>`, …) in assistant text | always explicit specials (`v4/mod.rs:24-33`) | only reachable as literals (flat encode makes them ids anyway) | equal ids today |

---

## 8. Request-side rules that change prompt or usage (same slice, cheap wins)

| Rule | Recipe | Ours | Consequence |
| --- | --- | --- | --- |
| `max_tokens` default | 8192 (chat) / 65536 (thinking) / 131072 (thinking+max), then lowered by remaining context; above the bound the completion is truncated by the backend (`request/mod.rs:158-218,221-235`) | **4096**, flat (`server.rs:59,314-340`) | long thinking answers hit `finish_reason=length` at 4096 → completion-token, latency and length distributions are wrong vs both oracles |
| `max_tokens` upper bound | 393 216 (`options.rs:96`) | `MODEL_MAX_OUTPUT_TOKENS` | different error text |
| `reasoning_effort` type | enum `none/minimal/low/medium/high/xhigh/ultra/max` (`protocol/reasoning.rs:24-49`), mapped to 4 rendered levels (`:58-78`) | ints 1..=100 plus the same aliases, rendered literally (`seam.rs:485-519`) | recipe rejects `42`; vLLM accepts numeric budget. Keep ours (vLLM), but the alias→number table must be the 25/50/75/100 one |
| `thinking.type` | `enabled`/`disabled`, `adaptive` alias; explicit `thinking` overrides `reasoning_effort` (`reasoning.rs:10-14,58-78`) | we accept both spellings? (`reasoning.rs` local) | verify precedence |
| `response_format: json_object` | requires the word "json" in the prompt, else 400 "Prompt must contain the word 'json'…" (`convert.rs:87-91`), then renders `## Response Format` (`dsv41.rs:278-283`) | **not parsed at all** (`schema.rs` has no field) | a client asking for JSON gets plain text with no instruction and no error |
| `response_format: json_schema` | validated/normalised (`protocol/json_schema.rs:153-158,260-295`), rendered, and turned into `OutputConstraint::Json` (`protocol/output_constraint.rs:28-59`) | 0.1 `ResponseFormat` has no `JsonSchema` variant; seam passes `response_format: None` (`seam.rs:396,434`) | silently unconstrained |
| `temperature` / `top_p` | thinking → 1.0; V4.1 thinking `top_p ≥ 0.95`; non-thinking V4.1 `top_p = 1.0` (`options.rs:85`) | passed through unchanged | sampling numbers not comparable even with an identical prompt |
| `reasoning_tokens` | counted by the stream processor when a tokenizer is attached (`stream/processor.rs`, `CompletionUsage::reasoning_tokens`) | our own counter (`engine.rs:919-936`) | compare definitions before publishing |

---

## 9. Structured output: what exists and what plumbing it needs

* Recipe: `ResponseFormat::{Text, JsonObject, JsonSchema, Regex}` (`conversation.rs:58-67`);
  `OutputConstraint::{Json{schema}, Regex, ToolCalls{tools, required}}` (`request/mod.rs:62-126`)
  with `kind()` and a constraint string; strict tool schemas are normalised so every object lists all
  properties as required and sets `additionalProperties: false` (`json_schema.rs:153-158`); strict and
  non-strict tools cannot be mixed in one request (`output_constraint.rs:35-51`).
* Ours: 0.1 gives `Text | JsonObject` only; the seam discards it; the sampler has no notion of a
  constraint (`engine.rs` samples temperature/top_p only).
* To plumb constrained decoding we would need: (a) a schema→state-machine masker (xgrammar-like or a
  hand-rolled JSON grammar) producing a per-step allowed-token bitset; (b) a hook in the sampler
  before argmax/sampling; (c) the thinking-mode rule (reasoning first, then the JSON — recipe encodes
  this by leaving the constraint to the caller and documenting it); (d) `Regex` via a regex→DFA
  conversion. Effort: 1–2 weeks for JSON-schema only, more for regex, *plus* the prompt must carry the
  schema (`dsv41.rs:278-283`) or the model has to infer it. This is a feature project, not an
  adoption: the recipe deliberately stops at `OutputConstraint` ("Whether the backend applies it is
  the caller's decision", `request/mod.rs:52-54`).

---

## 10. Adoption matrix

| Crate | Published | License | Dependency weight | Verdict |
| --- | --- | --- | --- | --- |
| `deepseek-recipe-core` 0.2.0 | crates.io 0.2.0 (2026-10-19) | MIT | `serde`, `serde_json`, proc-macro (`macros`) — tiny | **Adopt.** `Prompt`/`special_ranges`/`prompt!`/`SpecialTokensMap`/`ImageTokenSpec`. API break vs 0.1 (`InputMessage::Assistant{prefix}`, `System{tools}`, `Tool{tool_call_id: Option}`, `ToolDefinition{defer_loading}`, no `util::json_formatter` → `json`). |
| `deepseek-recipe-encoding` 0.2.0 | crates.io 0.2.0 | MIT | `aho-corasick`, core, `serde_json`, `thiserror`, `tokenizers 0.23`, `tracing` — light, no tokio, no network | **Adopt the plumbing** (`PromptTokenizer`, `special_ranges`), **do not adopt** `DeepseekV41Encoding` rendering for the serve path without the fixes in §4. |
| `deepseek-recipe` 0.2.0 | crates.io 0.2.0 | MIT | `jsonschema`, `indexmap`, `itertools`, `rand`, `xxhash-rust`, `async-stream`, `tokio-stream`, `base64`, `data-url`, `strum` — heavy | **Selective.** Useful: `json_schema` validation/normalisation, `OutputConstraint`, `resolve_max_tokens`, chat/Responses conversion, stream processor. Adopting it wholesale replaces our HTTP layer. |
| `deepseek-recipe-image` 0.2.0 | crates.io 0.2.0 | MIT | default features pull **OpenCV** + `image` + `infer` + `reqwest` | **Ignore for now**; if needed, `default-features = false` and supply our own preprocessor. |
| `deepseek-recipe-macros` 0.2.0 | crates.io | MIT | proc-macro only | pulled in by core. |

Constraints checked: no `no_std` (all use `std::sync::Arc`, `HashMap`), no wasm-specific code; edition
2024, workspace toolchain 1.97.1, our toolchain is 1.98.1 (fine; our declared `rust-version = "1.88"`
should be raised if a dependency declares higher).

---

## 11. What a bit-exact renderer adoption would cost

"Adopt `DeepseekV41Encoding` and delete our port" is **not** a parity-neutral change. To keep the
endpoint where it is today you would have to patch, in order of blast radius:

1. **effort table** back to 25/50/75/100 with default 50 (`dsv41.rs:21-30`) — otherwise every
   thinking request moves;
2. **tool schema serialisation** back to client order / no `"description": ""` (`v4/mod.rs:46-52`,
   `dsv41.rs:71-76`) — or accept client-dependent prompt bytes;
3. **image placeholder** `<｜deepseek_image｜>` and the `"\n\n"` block separator — requires a
   `SpecialTokensMap` override, not just a patch;
4. **prefix `</think>`** behaviour and the **trailing assistant header** rules
   (`dsv41.rs:213-216,291-306`) — ours vs theirs differ;
5. **`tool_choice: required`** priming and **`latest_reminder`** placement;
6. **`max_tokens` / effort enum** semantics at the request layer.

Each is one line to change and one test to re-pin; that is a **fork of the official renderer**, which
defeats the point of adopting it. Cost estimate: 2–3 days to fork + re-pin + document, and you own a
divergent copy forever.

**The cheap, honest version:** keep our renderer (it is the vLLM one), *upgrade the dependency pair to
0.2* for the `Prompt`/`PromptTokenizer`/`special_ranges`/`prefix`/`defer_loading`/`OutputConstraint`
types, re-express our renderer on top of `Prompt`, and add a differential test that renders every
vector with both encoders and asserts the known-difference set. Estimated 2–3 days for the bump
(§4.1 says ~6 files: `encoding/mod.rs` imports `BOS_TOKEN`/`SYSTEM_SP_TOKEN` which no longer exist in
0.2, `RenderedPrompt` gains a field, `TokenizerEncoder::encode_ids` changes arity, `case.rs` types,
`seam.rs` constants), plus 1 day for the parity harness.

**Highest-value single item:** that upgrade + single-renderer consolidation, because it fixes the
three-renderers bug in §4.1 and unblocks prefix continuation, deferred tools, response formats and
`reasoning_tokens` — all of which are currently "silently wrong or absent", which is exactly the class
of bug that corrupts published numbers. Effort: **2–3 days**.

---

## 12. Concrete test plan

P0 (do first, ~half a day, no new dependencies):

1. **Unify the effort resolver.** One function returning the numeric budget, used by the seam, the
   case path and `render_conversation`; assert `None → 50`, `low → 25`, `high → 50`, `xhigh → 75`,
   `max → 100`. Regression test: `encode_chat({"messages":[{"role":"user","content":"Hello"}]})`
   ids must equal the recorded `chat_user_hello_default` list
   (`docs/build/task30-evidence/tokenizer-parity.json`), 31 ids.
2. **Pin the fixture ids**, not just the prompt bytes: extend `tests/encoding_golden.rs` to compare
   `seam.encode_chat` ids against all five `fixtures[].ids` arrays (548/40/579/101/74). Today only the
   prompt text is pinned.
3. **`max_tokens`**: add tests for absent (8192/65536/131072 by mode, then context-lowered), 0, and
   > 393216, with the recipe's error strings (`request/mod.rs:180-218`). If we prefer vLLM's
   "context left" semantics, pin that instead and say so in `DECISIONS.md` — but stop defaulting to
   4096.
4. **Content special tokens**: a table test that `<｜User｜>`, `<｜begin▁of▁sentence｜>`, `</think>`,
   `｜DSML｜` typed into user/tool text produce the flat ids (128803 / 0 / 128822 / 128825) and a note
   that this is the vLLM contract. This is the "expected-failure" document.

P1 (with the 0.2 upgrade):

5. **Differential renderer test** (the harness in Appendix B turned into `tests/`): render 26 vectors
   with `DeepseekV41Encoding` and with ours; assert equality for the 11 identical ones and assert the
   *exact* known difference (offset + expected bytes) for the rest. Any upstream drift then fails
   loudly with a one-line fixture update.
6. **Prefix continuation**: `{"messages":[…,{"role":"assistant","content":"","reasoning_content":"x","prefix":true}]}`
   must render without `</think>`, without EOS, without a new `<｜Assistant｜>`, and
   `finish_reason` must be `stop`/`length` of the continued turn.
7. **Deferred tools**: `defer_loading: true` tool absent from the initial prompt; a later system
   message with `tools` renders `### Updated Tool Schemas`.
8. **Images**: `SpecialTokensMap` override test proving `<｜deepseek_image｜>` → 129264 under
   `PromptTokenizer`; and a test that the recipe's default `<｜image｜>` errors with our tokenizer
   (this is the adoption trap).
9. **Structured output**: `json_object` without "json" → 400; with → prompt contains
   `## Response Format`; `json_schema` strict normalisation (`additionalProperties:false`, all
   required) and `OutputConstraint::Json` surfaced to the sampler.

---

## 13. Contradictions with our current implementation and comments

* `encoding/render.rs:1-16` says the recipe renderer "differs in four observable ways". In 0.2.0 there
  are **seven more** (§4). The comment is not wrong, it is just pinned to 0.1 and understates the
  divergence — and it does not mention that the recipe is a *different* renderer rather than a buggy
  copy.
* `encoding/mod.rs:1-12` says "Rendering is byte-identical with the official Python encoder… The
  recipe only renders the effort budgets 50/75/100". Still true of 0.2.0 (`reasoning_effort_template`
  unchanged), but the *aliases* differ (25 vs 50 for `low`).
* `encoding/seam.rs:26-27` says "tokenize with the checkpoint tokenizer and add_special_tokens=false,
  which is exactly what the vLLM wrapper does" — correct, and §5.2 proves it; but the same sentence
  hides that this *promotes* special spellings in content while the recipe does not. Add the
  sentence.
* `server.rs:311-313` documents `DEFAULT_MAX_TOKENS` without noting it is neither the DeepSeek API
  default nor vLLM's; the constant itself (`:59`) is the only place 4096 appears.
* `docs/build/task30-evidence/tokenizer-parity.json` is a strong artefact and is under-used: it is
  the only recorded vLLM oracle we have, and §12.1/12.2 turn it into tests for free.

---

## 14. Open questions / not verified here

* **Which contract the endpoint should follow for `max_tokens`** when the client omits it — DeepSeek
  API (8192/65536/131072) or vLLM (context left). The Lead must pick; either beats 4096.
* **Does vLLM's `deepseek_v41.py` set `continue_final_message`/`add_generation_prompt` from the
  request?** The image is not available here; `prefix` behaviour is inferred from `encoding.py`'s
  `wo_eos` and from recipe `0.2.0`'s explicit `prefix`. This is the one item I could not pin with a
  recorded oracle.
* Whether the served model was trained with the 75/100 effort budgets (recipe) or 25/50/75/100
  (encoding.py). Model card and `inference_config.json` in `reference/` do not state it.
* Exact `reasoning_tokens` definition parity (our counter in `engine.rs:919-936` vs the recipe stream
  processor) — separate slice, but it consumes the same prompt/tokenizer decision.

---

## Appendix A — vectors (26)

`chat_plain`, `chat_system`, `think_default_effort`, `think_low`, `think_high`, `think_xhigh`,
`think_max`, `tools_off`, `tools_desc_missing`, `tools_thinking_none_desc`, `tools_required`,
`multi_turn_reasoning_notools`, `multi_turn_reasoning_tools`, `prefix_continuation`,
`prefix_continuation_with_content`, `special_literal_in_user`, `consecutive_users`,
`tool_call_and_result`, `json_object_format`, `json_schema_format`, `latest_reminder`,
`empty_messages`, `image_in_user`, `tool_result_contains_think_end`, `user_contains_bos`,
`user_contains_dsml`. Each vector is `{thinking, effort, messages[], tools[], tool_choice, response_format}`
with OpenAI-shaped messages; `image:true` marks the image vector.

## Appendix B — reproduction

```sh
# 1. our goldens (already run for this report: 13 passed)
cd /home/user/deepseek-v4.1-flash-spark
cargo test -p sparkflash-api --test encoding_golden --offline

# 2. recipe 0.2.0 renderer + PromptTokenizer (harness lives outside the repo)
mkdir -p /tmp/rendercmp/src && cd /tmp/rendercmp
# Cargo.toml: deepseek-recipe-core/encoding { path = "/home/user/deepseek-recipe/..." },
#             tokenizers = "0.23", serde_json = "1"
cargo run --offline -- vectors.json > recipe.json

# 3. the vLLM oracle: python reference on the same vectors
python3 ref.py > python.json

# 4. our 0.1-based crate on the same vectors (reuses the repo's target dir)
cd /tmp/rendercmp2 && CARGO_TARGET_DIR=/home/user/deepseek-v4.1-flash-spark/target \
  cargo run --offline -- /tmp/rendercmp/vectors.json > port.json

# 5. fixture ids vs the recorded vLLM ids
cd /tmp/rendercmp && cargo run --offline --bin fixture_ids
# compare with docs/build/task30-evidence/tokenizer-parity.json["fixtures"]
```

Harness details that matter: build the recipe `Conversation` with
`thinking_mode/effort/tools/tool_choice/response_format`, push the image placeholder as
`Prompt::push_special("<｜image｜>")`, and label the Python side's tools as the OpenAI `function`
object attached to the first system message (that is the vLLM normalisation, not the recipe's).

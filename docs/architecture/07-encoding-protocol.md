# 07 — Encoding protocol: DSML tool markup, thinking mode, and reasoning-effort budgets

The authoritative sources for this document are the two shipped encoders, which **disagree with each
other** on the reasoning-effort scale:

| Source | Path | Scale |
| :--- | :--- | :--- |
| Python (dataset/eval encoder) | /home/user/dsv41-research/encoding/encoding.py | low 25, high 50, xhigh 75, max 100; default "high" = 50 (encoding.py:398-404) |
| Rust toolkit (deepseek-recipe) | .../deepseek-recipe-encoding-0.1.0/src/v4/dsv41.rs | Low 50, High/Xhigh/(None) 75, Max 100 (dsv41.rs:6-15) |
| Shared template | both | "Reasoning Effort: {n} (range 1-100, the higher the value, the more thorough the reasoning)\n\n" |

---

## 1. Special tokens (deepseek-recipe-encoding-0.1.0/src/v4/mod.rs:14-34)

| Constant | Literal |
| :--- | :--- |
| BOS_TOKEN | `<｜begin▁of▁sentence｜>` |
| THINKING_START_TOKEN | `<think>` |
| THINKING_END_TOKEN | `</think>` |
| SYSTEM_SP_TOKEN | `<｜System｜>` |
| USER_SP_TOKEN | `<｜User｜>` |
| ASSISTANT_SP_TOKEN | `<｜Assistant｜>` |
| LATEST_REMINDER_SP_TOKEN | `<｜latest_reminder｜>` |
| EOS_TOKEN | `<｜end▁of▁sentence｜>` |
| DSML_SP_TOKEN | `｜DSML｜` (U+FF5C fullwidth vertical bar, not ASCII `|`) |

The DSML token is a bare prefix: the surrounding angle brackets come from the template, so V4.1's
tags are `<｜DSML｜ calls>` / `<｜DSML｜ invoke>` / `<｜DSML｜ parameter>`, while V4 uses
`<｜DSML｜tool_calls>` (mod.rs:31-34). This is the whole of the V4.1 difference at the tag level: the
block name gained a leading space and lost the "tool_" prefix (dsv41.rs:56-66).

---

## 2. Prompt skeleton (mod.rs:206-266)

```
BOS_TOKEN
<rendered message 0><rendered message 1>...<rendered message n-1>
ASSISTANT_SP_TOKEN
( <think> if thinking_mode else </think> )
( if tool_choice == Required and tools non-empty:
      "

<｜DSML｜ calls>
" )
```

There is no trailing newline, no trimming, and the generation prompt always ends with the assistant
token plus an explicit think start or end token. The rendered string is then handed to the tokenizer
as one piece (mod.rs:207-213), so all special tokens must be recognised as single tokens by the
tokenizer, not spelled out as text.

### 2.1 Message rendering (mod.rs:118-204)

| Role | Prefix rule | Body |
| :--- | :--- | :--- |
| System (index 0) | `<｜System｜>` only if a reasoning-effort line was rendered or the message is a system message | content verbatim |
| System (index > 0) | `<｜System｜>` again, because V4.1 declares supports_mid_conversation_system() = true (dsv41.rs:48-50) | content verbatim |
| User | `<｜User｜>`, unless the previous message was a User or a Tool, in which case "\n\n" | content |
| Tool | same prefix rule as User | `<tool_result>{content}</tool_result>` |
| LatestReminder | `<｜latest_reminder｜>` | content |
| Assistant | `<｜Assistant｜>` then `<think>` or `</think>` | thinking part, content, tool calls, `<｜end▁of▁sentence｜>` |

Assistant details (mod.rs:170-201): when thinking_mode is on and index > 0, the thinking part is
reasoning_content (if present) followed by `</think>`; the message is then rendered as
`<｜Assistant｜><think>` + thinking_part + content + tool_calls + EOS when the thinking part is
non-empty, and `<｜Assistant｜></think>` + content + tool_calls + EOS otherwise. Tool calls are
appended as "\n\n" + the DSML block. The **final** assistant turn in history therefore carries a
closed thinking block, while the *next* turn's prompt opens a new one (section 2).

### 2.2 Normalisation before rendering (mod.rs:332-435)

- Consecutive user messages are merged with "\n\n" and their image sources concatenated
  (mod.rs:354-373).
- With mid-conversation system support, system messages after the first are re-emitted with their own
  `<｜System｜>` marker instead of being prepended (mod.rs:337-353).
- Tool results are reordered to match the order of the preceding assistant's tool_calls, identified by
  tool_call_id (mod.rs:384-435).
- If tools or a JSON response format are present and the conversation does not start with a system
  message, an empty system message is inserted so the tool prompt has somewhere to live
  (mod.rs:225-244).

---

## 3. Tool prompt and DSML tool-call markup

The tool prompt is appended to the first system message (mod.rs:284-329). Its literal text is the
spec's own "## Tools ... ### Available Tool Schemas ..." block, with the DSML placeholders
substituted:

```
## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set string="true". For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set string="false".

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{one python-style JSON object per tool}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
```

Rendering rules for an actual call (mod.rs:36-116):

- `<｜DSML｜ calls>` wraps the join of the individual calls with "\n".
- Each call is
  `<｜DSML｜ invoke name="{name}">\n{arguments}\n</｜DSML｜ invoke>`.
- Each argument is
  `<｜DSML｜ parameter name="{key}" string="{true|false}">{value}</｜DSML｜ parameter>`,
  joined with "\n".
- string="true" exactly when the JSON value is a string; otherwise string="false" and the value is
  rendered with **Python repr** (`stringify_python_style`): True / False / None, single-quoted
  strings, and json.dumps-style spacing. This is what the model was trained to read and write -- an
  engine that emits canonical JSON (true/false/null, double quotes) is off-distribution.
- If the arguments string is not a JSON object, the toolkit logs a warning and wraps the raw text as
  {"arguments": <raw>} (mod.rs:48-57).

The tokenizer-level mirror of all of this exists in the Python encoder (encoding.py:380-460), which
implements the same templates; only the effort table differs.

---

## 4. OpenAI-compatible request mapping

| Request field | Mapping | Source |
| :--- | :--- | :--- |
| reasoning_effort "none" | disables thinking, no effort | convert.rs:88-89, 217-224 |
| "minimal" / "low" | Thinking on, ReasoningEffort::Low | convert.rs:221 |
| "medium" / "high" | Thinking on, ReasoningEffort::High | convert.rs:222 |
| "xhigh" | Thinking on, ReasoningEffort::Xhigh | convert.rs:223 |
| "max" | Thinking on, ReasoningEffort::Max | convert.rs:224 |
| thinking {type: enabled/disabled, budget_tokens} | overrides everything; budget_tokens is parsed and ignored | schema.rs:44-45, 205-220; convert.rs:85-93 |
| absent everything | options.default_thinking_mode; if thinking, effort defaults to **High** | convert.rs:91-96 |
| tool_choice required / named | requires thinking off | schema.rs:40 |

Note the two-step default: `thinking.then_some(effort.unwrap_or(ReasoningEffort::High))`
(convert.rs:96). With thinking enabled and no effort given, the rendered line is
"Reasoning Effort: 75" in the Rust pipeline (High → 75, dsv41.rs:9) but "Reasoning Effort: 50" in the
Python pipeline (high → 50, encoding.py:400).

---

## 5. Reasoning-effort line placement

Rendered **only** when index == 0 **and** thinking mode is on (dsv41.rs:74-77; encoding.py:423-424).
It is inserted after the `<｜System｜>` marker when that marker was emitted and before the system
content (mod.rs:129-136). Consequences:

- A conversation with no system message at index 0 still emits `<｜System｜>`, because the effort line
  is non-empty (mod.rs:129-131).
- With thinking off, no line is emitted at all -- there is no "0" or "off" spelling.
- With thinking on but index > 0 (any subsequent turn), no line is emitted either, so the effort is
  pinned by the first turn of the conversation.

The effort line is the only numeric control the model sees: it is a prompt-level hint, not a sampling
parameter, and nothing in the reference sampler consumes it (temperature/top_p are separate
InferenceOptions).

---

## 6. Spec deltas

**D1 — The spec's effort mapping contradicts encoding.py, which the same spec names as the reference
for unit tests.** The spec says `1..100, default 75, low=50, high=75, max=100` (spec line 1332) and
proposes to test against encoding.py test vectors; encoding.py maps low=25, high=50, xhigh=75,
max=100 with default high (encoding.py:398-404). The spec's numbers match the *Rust* toolkit, so the
two "references" cannot both be the oracle. SparkFlash-RS picks `encoding.py`'s table on every
API surface (D44/D46): low=25, high=50, xhigh=75, max=100, default 50; the Rust toolkit's
50/75/75/100 renderer table is not used for prompt rendering.

**D2 — The spec advertises `reasoning_effort: 1..100` on the REST surface, but the Rust pipeline
cannot carry an arbitrary integer.** `ReasoningEffort` has exactly four variants
(conversation.rs:47-55) and the OpenAI adapter maps the seven string values onto them
(convert.rs:217-224). An integer 37 has no representation; only the Python encoder accepts it
(encoding.py:415-422). Supporting 1..100 requires extending the Rust enum (e.g. a
`Numeric(u8)` variant) or bypassing the toolkit's renderer.

**D3 — The spec's tag list omits the leading spaces that distinguish V4.1 from V4.** The block name
is `" calls"` with a leading space (dsv41.rs:56-58) and the parameter tag is `" parameter"`
(dsv41.rs:64-66), so real tags are `<｜DSML｜ calls>` and `<｜DSML｜ parameter ...>` -- not
`<｜DSML｜tool_calls>` / `<｜DSML｜parameter>`. The DSML token itself is the fullwidth `｜DSML｜`
(mod.rs:34).

**D4 — The spec lists `chat_template_kwargs.thinking` / `enable_thinking` as the thinking switch.**
The Rust schema's switch is a typed `thinking: {type: enabled|disabled, budget_tokens}` object plus
the `reasoning_effort: none` shortcut (schema.rs:43-45, 205-220); `enable_thinking` is not part of the
reference protocol at all.

**D5 — The spec does not mention python-style argument rendering.** Parameters are rendered with
`stringify_python_style` (mod.rs:63, 288), i.e. True/False/None and single quotes, which is what the
model consumes. Emitting strict JSON is a silent distribution shift.

**D6 — The spec ignores mid-conversation system messages, which V4.1 explicitly supports.** The
encoder declares supports_mid_conversation_system() = true (dsv41.rs:48-50) and re-emits
`<｜System｜>` for later system turns (mod.rs:337-353); a chat template that hoists every system
message to the front produces a different prompt.

**D7 — The spec's endpoint list omits the `/v1/responses` and Anthropic Messages adapters that the
toolkit already implements,** both of which have their own effort-variant mappings
(responses/request/convert.rs:615-631, anthropic/messages/request/convert.rs:75-97). If
SparkFlash-RS vendors deepseek-recipe it gets all three surfaces, with three different effort
normalisations that must be unified before rendering.

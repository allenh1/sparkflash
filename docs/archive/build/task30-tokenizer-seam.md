# Task 30 - the tokenizer seam in the HTTP path

Branch: `task/tokenizer-seam`, rebased onto main `4bf96e9` (batch 3). The rebase was a clean
no-op: no conflicts and no incoming change touched the API crate.
Window policy: **host work only** - no GPU, no vLLM server, no new window. All evidence below was
collected read-only from the 2026-10-18 served root and the released image's source, plus one
CPU-only offline tokenizer run (no model weights, no GPU, no server).

## The gap

At main `dcc01c8` the HTTP path never touched the checkpoint tokenizer:

* `crates/sparkflash-api/src/prompt.rs` built placeholder prompt ids (all zero, "Milestone 1"
  heuristic of four characters per token).
* `crates/sparkflash-api/src/server.rs` handed those ids to `EngineRequest.prompt_tokens`, and
  `/v1/completions` only accepted prompt **text** (string or array of strings), so the engine got
  ids that were not the tokenizer's.
* `/v1/chat/completions` did not render the V4.1 chat template at all: prompt ids were the
  heuristic's, and chat `echo` was explicitly deferred.

Until this seam exists, an HTTP parity comparison against released vLLM is not apples-to-apples:
vLLM tokenizes the prompt text it receives and renders the chat template before the model sees a
single token.

## What the released server actually does

**Which tokenizer the reference recipe uses.** The 2026-10-18 launch recipe (recorded in
`/home/user/phase2-evidence/window-2026-10-18/vllm-launch-recipe-released.sh`; the served
config in `start-vllm.log`) starts vLLM with `tokenizer_mode=deepseek_v41`, a
DeepSeek-specific tokenizer mode in the released image, **not** the stock HF tokenizer path. That
mode wraps the checkpoint's own `tokenizer.json`:

* Served snapshot root (from `start-vllm.log` lines 443/446/471):
  `/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170`
  (on node .80 the same tree is readable at
  `/home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170`).
* `tokenizer.json` sha256 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`,
  byte-identical to this repository's `reference/tokenizer.json`.
* `tokenizer_config.json` sha256
  `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547`:
  `add_bos_token=false`, `add_eos_token=false`, `pad_token=<｜end▁of▁sentence｜>`,
  `unk_token=null`, `tokenizer_class=PreTrainedTokenizerFast`, **no `chat_template` key**.
  The tokenizer's `post_processor` is `ByteLevel`, so special-token addition is a no-op either
  way; the wrapper explicitly calls `encode(prompt, add_special_tokens=False)` for chat and the
  raw-completion path's default `add_special_tokens=True` yields the same ids (verified in the
  evidence file: `default==nospecial` is true for every pinned completion text).

**What template the chat path adds.** There is no Jinja template string anywhere in the released
checkpoint or launch recipe; the template is **code** in the released image:

* `vllm/renderers/registry.py` maps `deepseek_v41` to `DeepseekV4Renderer`
  (`vllm/renderers/deepseek_v4.py`), which calls
  `self.get_tokenizer().apply_chat_template(messages=messages, conversation=..., **chat_template_kwargs)`
  directly - it bypasses the generic Jinja-template resolution in `renderers/hf.py`.
* `vllm/tokenizers/deepseek_v41.py` overrides `apply_chat_template` and implements the whole
  template itself using `vllm/tokenizers/deepseek_v41_encoding.py`, a serving-side adaptation of
  the checkpoint's own `encoding/encoding.py` ("Text encoding adapted from the DeepSeek V4.1
  checkpoint reference encoder").
* File hashes in the image: `vllm/tokenizers/deepseek_v41.py`
  `d6e9cbce7c5f127819452fa5d03dd4f22ad9f136dd202a54b5b31fe42402cae2`,
  `vllm/tokenizers/deepseek_v41_encoding.py`
  `fa501415211cd0e0c9f06ee6033c5466046e293b6153bca6200b70695b88fc73`; this checkout's
  `reference/encoding/encoding.py` is `f64a67e5680a5621b9320585a9684967cd5c75a9b82e19d914cbc02845d72cab`
  (the image copy drops the decode/parse half and keeps the text grammar).
* Verification: the image's `deepseek_v41_encoding.py` reproduces
  `reference/encoding/tests/test_output_1..5.txt` **byte-for-byte** when fed the reference
  fixtures with the wrapper's message normalisation (checked offline on 2026-10-19). The Rust
  renderer is golden-tested against the same files, so both sides implement the same prompt
  grammar.

**What we match (and what we do not claim).** The parity target is the released image's encoding
module, `vllm/tokenizers/deepseek_v41_encoding.py`, not a Jinja template string. There is no
template string to copy: the image implements the template in Python, and that copy reproduces the
checkpoint's five golden encoder fixtures byte-for-byte. Our Rust renderer is golden-tested against
the same five files, and the seam's refusal texts are byte-identical with the image's raises. The
claim is therefore "same encoding-module behaviour, checked against the checkpoint's golden
fixtures and the image's prompts", not "same template string".

**The exact template rules** (`deepseek_v41.py` + `deepseek_v41_encoding.py`):

1. Normalise each raw OpenAI message: role must be system/developer/user/assistant/tool, otherwise
   `ValueError("Invalid role: {role}")`; a content-part array joins text parts with `"\n\n"`,
   maps `image_url`/`input_image`/`image_pil` to the image placeholder, and otherwise raises
   `ValueError("DeepSeek V4.1 supports text and image content only; got {part_type!r}")`;
   `reasoning` is copied into `reasoning_content`.
2. Tools are attached to the first system message; if the conversation has no system message an
   empty one is inserted at index 0 first.
3. Thinking defaults to **on** (`bool(thinking or enable_thinking)`, `True` when neither key is
   present). `reasoning_effort` comes from `chat_template_kwargs`; the vLLM chat protocol merges
   the top-level `reasoning_effort` over the user kwargs and implies
   `enable_thinking = (reasoning_effort != "none")` when the user did not set it.
4. Effort aliases are the official encoder's: **low 25, high 50, xhigh 75, max 100**; an integer
   in 1..=100 is accepted verbatim; the default is "high". Anything else raises
   `ValueError("DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, or an integer within [1, 100] in chat_template_kwargs")`.
   `"none"` switches to chat mode (no effort line). This template scale is **not** the
   recipe-scale `reasoning.rs` table (low 50, default 75); the seam deliberately renders the
   template's scale and records the difference.
5. The prompt is the checkpoint's official encoder output: `<｜begin▁of▁sentence｜>` + rendered
   messages (role markers `<｜System｜>`, `<｜User｜>`, `<｜Assistant｜>`, effort line
   `Reasoning Effort: {budget} (range 1-100, ...)` in thinking mode, `<think>`/`</think>`,
   tool/DSML markup) and is tokenized with `add_special_tokens=False`.

## What this branch implements

* **`crates/sparkflash-api/src/encoding/seam.rs`** - `TokenizerSeam` over the already-loaded
  `HfTokenizerEncoder`: `encode_text` (raw completion prompts), `encode_chat` (the pipeline
  above, reusing the official-encoder port `render::encode_messages_text` for merge/sort/
  drop-thinking and effort rendering), and `decode` (echo round trip). The refusal texts for
  invalid role, unsupported content part and unsupported effort are byte-identical with the
  released image's raises.
* **`AppState::with_tokenizer` / `AppState::tokenizer`** - the seam is opt-in, so weight-free
  router tests keep the old placeholder path; `sparkflash serve` always attaches one (it already
  resolved/loaded a tokenizer for the engine, and a load failure refuses startup).
* **`/v1/completions`** - prompt text is tokenized with the checkpoint tokenizer; vLLM's
  `int[]` prompt form is now accepted (`PromptSpec::Tokens`) and passes through unchanged;
  `echo` with token-id prompts now round-trips through `decode`.
* **`/v1/chat/completions`** - with the seam attached the prompt is the released image's
  encoding-module ids (the code-implemented template); with no seam the Milestone 1 placeholder
  path is unchanged.
* **Tests** - `crates/sparkflash-api/tests/tokenizer_seam.rs` pins completion ids, the five
  official encoder fixture prompts, six chat requests (default thinking, non-thinking, low effort,
  tools, multi-part content, multi-turn) and drives `/v1/completions` / `/v1/chat/completions`
  through the real router (including 400s with the vLLM texts and the echo round trip).

### Deliberate deviations (recorded, not silent)

* `prompt` as an **array of strings** still joins the strings with a newline before tokenizing
  (Milestone 1 single-prompt behaviour); vLLM treats the array as a batch. Batching is not
  implemented, so the joined text is tokenized for real - a deviation in request semantics, not in
  tokenization of the text the engine is given.
* `/v1/chat/completions` `echo` remains unimplemented (the template now renders, but echoing the
  final message changes response assembly, not the prompt seam).
* A `developer` role matches the released encoder: dropped by the encoder's drop-thinking filter
  in thinking mode, refused with `UnknownRole` in chat mode. We do not invent a different
  rendering for it.
* The window-time **`/tokenize` endpoint cross-check** is pending (window closed). The recorded
  oracle is the released image's own tokenizer stack run offline on the served snapshot, exactly
  as the vLLM wrapper calls it; see the evidence file.

## Evidence

`docs/build/task30-evidence/tokenizer-parity.json` records:

* served root, `tokenizer.json` and `tokenizer_config.json` hashes, image digest,
  `tokenizer_mode`, renderer mapping, image wrapper/encoding hashes;
* pinned raw-completion texts with ids for `add_special_tokens=True/False` and the equality check;
* six pinned chat requests with prompt sha256 and ids (the template oracle);
* the five official encoder fixture prompts with ids.

Method: CPU-only container run of the released image's tokenizer stack -
`transformers.AutoTokenizer.from_pretrained(served snapshot)` plus the image's
`deepseek_v41_encoding.py` for prompt text; **no model weights were loaded, no GPU was used and no
vLLM server was started**. The same id vectors are asserted by the Rust tests.

## Gates (host only)

* `cargo fmt --all -- --check` - clean.
* `cargo check --all-targets` - clean.
* `cargo clippy --all-targets -- -D warnings` - clean.
* `cargo check --workspace --all-targets` - clean.
* `cargo clippy --workspace --all-targets -- -D warnings` - clean.
* `cargo test --workspace` - all suites pass; the new `tokenizer_seam` suite is 15/15 and the
  existing router suite and the rest of the workspace are unchanged and green.

No GPU, no vLLM server and no checkpoint weights were touched. The branch tip is the deliverable
for batch 4 (no MR, as instructed).

## Repro

```bash
# the seam's own oracle assertions
cargo test -p sparkflash-api --test tokenizer_seam

# gates
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Evidence was produced on node .80 with a CPU-only container run of the released image's tokenizer
stack on the served snapshot (mounted read-only); the recipe to reproduce it is in the evidence
file's `method` field and in the seam module docs.

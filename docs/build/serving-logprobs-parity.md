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

## Oracle patch state and the p129 near-one-hot question

Added 2026-10-19 from an audit of the window that produced the D40 oracle. No server was started:
the image was inspected from `vllm/vllm-openai:deepseekv41-flash-0909` (digest `714a937591e1...`,
the same on all four nodes) and the launch scripts/logs on `node0`, and patch scripts were
exercised only in throwaway `docker run --rm` containers.

**How the cluster's vLLM repairs are applied.** The three `fix-*` repairs are `mods/` in
`~/spark-vllm-docker`, applied at cluster start by `launch-cluster.sh --apply-mod <dir>`. The
containers start idle (`sleep infinity`); for each mod the launcher `docker cp`s the directory to
`/workspace/mods/<name>` in all four containers and runs `./run.sh` through `docker exec`, which
source-patches the installed `vllm` package. A non-zero `run.sh` aborts the launch. The patches are
not baked into the image (the image digest is unchanged) and are not env-gated; they live in the
container writable layer and are re-applied on every launch, so a window is reproducible only
through the recipe / `window-vllm.sh`, not through the bare image.

**What the D40 window had live.** `window-vllm.sh start` (sha256 `a3ba6763...`, as recorded in
`window-runbook.md`) applies exactly the canonical recipe's mod list
(`recipes/deepseek-v4.1-flash-1017.yaml`: engram-disk-reference, sm120-dsv4-topk,
fix-dsv41-sparse-topk-oob, diag-dsv41-decode-topk, install-ray) and then
`vllm-launch-recipe-released.sh` (sha256 `a3ac461f...`, released snapshot root `1bae0e10...`,
`--max-logprobs 1000`). `start-vllm.log` at 2026-10-19T17:18-17:20Z shows the mod scripts running on
all four nodes; in particular `[fix-dsv41-sparse-topk-oob] patching
.../vllm/models/deepseek_v4_1/common/ops/cache_utils.py` on each node, and the decode-top-k probe
verdict "deterministic and correct". `native-oracle-released.json` (17:28:33Z) and
`native-oracle-suite.json` (17:30:37Z) were written by that server.

**Per-patch status against the served tree.**

* `fix-dsv41-sparse-topk-oob` -- needed and live. The pristine image's
  `models/deepseek_v4_1/common/ops/cache_utils.py`
  `_compute_global_topk_indices_and_lens_kernel` loads
  `block_table_ptr + req_idx * block_table_stride + block_indices` with no row bound (no
  `block_table_rows` in the kernel signature); the mod masks the load, writes the `-1` sentinel and
  excludes unresolvable entries from the valid count. This is the only one of the three on the V4.1
  path.
* `fix-dsv4-sparse-swa-spec-config` -- already in the image.
  `v1/attention/backends/mla/sparse_swa.py` defines
  `spec_config = self.vllm_config.speculative_config` (line 422) and computes the DSpark width via
  `get_dspark_swa_index_width` (line 527); running the mod's `patch.py` against the pristine image
  fails pattern #0 (exit 1) and would abort a launch. Must not be added to a recipe.
* `fix-dsv4-c128a-prefill-short-extend` -- V4.0 only. It edits
  `models/deepseek_v4/sparse_mla.py::_build_c128a_metadata` (it applies cleanly there, so it is
  required when serving `DeepseekV4ForCausalLM`); the served architecture
  `DeepseekV41ForCausalLM` maps to `vllm.models.deepseek_v4_1`, which contains no c128a metadata
  builder and never calls `_build_c128a_metadata`.

So the D40 oracle run was not missing any of the three *as shipped in the recipe*: the only fix the
V4.1 path needs was live, and the other two are image-native and V4.0-only respectively.

**Protocol rule for D40 oracle captures.** In this build a `prompt_logprobs` row is a function of the
request's total prompt length and batch composition as well as the causal prefix: identical-shape
repeats are bit-identical, but a 1-token prompt-length change moves the p129 tails by up to ~3.7 nats
(KL up to 0.235) and a 4x-batch request moves the same row's top-1 probability from 0.820 to
0.690-0.908 (worst KL 0.932, max 4.06 nats). Rule, not a hedge: match the token ids exactly (same
causal prefix) and keep the request body identical; deviate in total prompt length only by the
minimum the protocol forces -- for a `prompt_logprobs` row p the oracle request must contain token
p+1, so the minimal oracle prompt is p+2 tokens against the device request's p+1, an irreducible
+1-token difference -- and record the prompt length (and batch composition) in every capture file.
Gate only at that minimal forced +1 shape (or another fixed, recorded shape); a residual 1-token
difference is a floor to quote, not a gate violation, while an arbitrary longer prompt (the 139-vs-
140 case here) is not a valid comparison.

Two floors, and the doc should say which one a gate uses. The *same-shape repeat* floor is zero:
three identical sequential requests are bit-identical (rank-1 9/9, |difference| 0.000 at
p50/p90/p99/max, KL 0.000, top-k overlap 1.000; p129 -0.19884082674980164 in all three). The *gate*
floor is the measured 1-token length envelope at the minimal forced +1 shape (task-32 section 4.2;
rows `docs/build/task32-evidence/length-probe-rows.json`, sha256
`5fae4ee9933a551e9a1da486971ceb6e1c37fd9b8c78402e7fc90d67607f6400`, identical to the node capture
`phase2-evidence/window-2026-10-18/task32-repeats/native-oracle-length-probe.json`: KL 0.025-0.235,
p90 0.34-1.89 nats, max 0.81-3.74 nats, top-100 overlap 0.68-0.83, rank-1 flips only at near-ties
such as row 128 201/223). A device-oracle gate uses the +1 envelope, never the zero repeat floor.

**Batch/load state is a capture variable too.** With the same request body, `prompt_logprobs` differ
deterministically between an idle server and one with a concurrent generation in flight (controlled
result: p129/139 top-1 probability 0.820 -> 0.928 -> 0.820; p129/131 0.448 -> 0.882 -> 0.448; rows
move up to 0.3-0.4 in top-1 probability), and the switch is reversible, so a busy capture cannot be
corrected post hoc. Cause: prefix caching + chunked prefill on a shared server (an external client
was seen on 192.0.2.71 during part of the capture window). Recorded rule: the D40 oracle must be
captured on a verified-idle server -- idle check before and after the capture, external-client check
recorded -- and every capture file records its load state; busy captures are labelled and are not
valid gate inputs, and mixing idle and busy captures is invalid. The same-shape repeat floor of zero
applies under a constant state; the concurrent-state component is part of the D41 floor (up to
0.12-0.7 nats, up to 0.43 in top-1 probability).

**The p129 near-one-hot question is open, not diagnosed.** In the same suite there are 14
prompt-logprobs rows (of 1664) with `p(top1) > 0.9997`, all with the top-1 equal to the observed
prompt token and top-2 gaps of 9.5-13.4 nats. Applying the discrimination test designed for this:

* (a) top-1 is the observed token: 14/14.
* (b) the position is trivially predictable: all 14 are mid-word / word-completion or fixed
  collocations (`nineteen eight` -> `ies`, `inher` -> `its`, `Reg` -> `isters`, `rather` -> ` than`,
  `parag` -> `rap`, `esp` -> `aces`, `mant` -> `issa`, `reprodu` -> `ces`, `disagree` -> `ing`,
  `del` -> `tas`; the weakest is `...distribution dep` -> `arts`).
* (c) same value across warm repeats: settled by task-32. Three identical sequential sends of the
  b69 body on the restarted warm server are bit-identical (rank-1 9/9, `|difference|` 0.000 at
  p50/p90/p99/max, KL 0.000, top-k overlap 1.000 at k=5/20/100/1000; p129 -0.19884082674980164 in
  all three, matching the frozen 17:28 file). The 0.82-nat released-vs-suite drift (rows 129-137,
  same causal prefixes) is therefore request shape, not run-to-run noise: rows in this build depend
  deterministically on total prompt length and batch composition. A length probe gives row 129
  top-1 probability 0.882/0.822/0.598 at lengths 131/132/133 (KL 0.053-0.235, p90 1.10-1.89, max
  2.31-3.74), and a batch-of-4 request gives 0.690/0.806/0.839/0.908 against 0.820 single (worst KL
  0.932, max 4.06, one rank-1 flip at the flat p135 where the single-run top-1 probability is
  0.019). Numbers: `docs/build/task32-root-cause.md` sections 2.3/4.1/4.2 (branch
  `task32-f32-vs-vllm`, commit `31c0212`); artefacts under
  `phase2-evidence/window-2026-10-18/task32-repeats/`.
* (d) top-2 gap: 9.5-13.4 nats (>= 8, "collapsed" by the rule), but a genuine >=10-nat gap
  necessarily gives a top-1 logprob of about -1e-5, which is exactly the observed magnitude, so (d)
  does not discriminate on its own.

Recorded disposition: (a)+(b)+(c) classify these 14 rows as legitimately deterministic word-completion
rows; the one-hot magnitude is by itself not evidence of an engine bug, and the item stays OPEN as a
watch item rather than being closed as an artefact. The operational consequence is the shape
protocol rule above: with shape locked a repeat is bit-identical, and the 0.82-nat released-vs-suite
drift is the shape floor, ~40x a 0.02-nat bar, so any row-value bar must be shape-locked. The p129
random-token row (0.92 on a prompt of random ids, with no collocation to predict) is a different
quantity and is not explained by this classification: under a matched 131-token prompt it is still
367 first at p=0.448, entropy 2.73 against 5.15 at p128, i.e. a stable anomaly of that query; it
remains open and is owned by audit-dev (task-32, section 2.3).

## Recorded b69 position-129 numeric comparison (O4, task-16)

Scope: record, once and with caveats, the numeric comparison at position 129 of the b69 fixture
(random-token prompt, 130 ids; position 129 is the prefill step that predicts token 130). This is a
record of what exists, not a re-run; no server contact was made.

### Provenance of "b69"

b69 is the Lead-owned window of 2026-10-19 that produced the golden fixture
`phase2-evidence/sequence-130-released.json` (sha256
`e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308`) and the criterion token list
`491, 367, 201, 80, 3302, 201, 54, 201, 4341` (host-f32 full 130+8 run in `window-b69.sh`).

**The b69 device-native decode-0 dump does not exist.** `window-b69-device.sh` (2026-10-19) stopped
vLLM at 16:15:13Z, started `device-chain --prompt-tokens 130 --decode-steps 0 --dump-logits
runs69/logits-130x0.jsonl` at 16:17:00Z and the run was terminated before writing anything (window
log: `RUN device-native exit=143`, `SIGTERM`; then `ls: cannot access runs69/logits-130x0.jsonl`).
`runs69/` now contains only a 0-byte `logits-130x0.log`. So b69 contributes NO device column.

The b69 window did leave a side effect: it restarted the released-snapshot vLLM (start 17:18:44Z,
healthy 17:28:14Z), and the task-32 released oracle capture at 17:28:30Z was the first request on
that server, 16 s after health.

The only device numeric column at p129 that exists is **b70/runs70** (decode 8; dump positions
129..137): `logits-130x8.jsonl` sha256
`f548d9c3115c0557cc277ed1b32ce2c82985ec38d299869163759f31fc30aea6`, `logits-130x8.jsonl.bin`
`43b31438c01955c8359457e077322fc39e6c6a76f330bdf958e87d9ddcad07b5`, `logits-130x8.log`
`d8ba8cbc0cccc440b39123bce77f9ae37afca2473737d4347059359d3fb6b0e8`, binary
`sparkflash_m3c_c4c_seqreuse_4f599b7_aarch64` `7e0ce4bf2390ec078f70098eb5e77881b518e5d0ed58f9155b669a355a8d48cc`.

### Captures at position 129

All oracle captures are on served root `1bae0e1088993b9b82de08f59de5be5a34e42170`, image
`vllm/vllm-openai:deepseekv41-flash-0909`, `POST /v1/completions`, `max_tokens 1`,
`temperature 0`, `top_p 1`, `ignore_eos true`, `prompt_logprobs 1000`; row p is
`prompt_logprobs[p+1]`.

| capture | oracle prompt shape | load state | oracle p129 top-1 | 491 |
|:--|:--|:--|:--|:--|
| released, 17:28:30Z | 139 ids (130 fixture + 8 continuation + sentinel) | not recorded (pre-protocol) | 367 at -0.1988408, p 0.820 | rank 2, -3.83 |
| suite, 17:30:22Z | 140 ids (130 + 9 + sentinel) | not recorded | 367 at -0.0819, p 0.921 | rank 7, -5.45 |
| task-32 warm repeats, 18:12Z (3x) | 139 ids, identical bodies | not recorded; value equals the controlled idle block (inferred idle) | bit-identical, 367 at -0.19884082674980164, p 0.820 | rank 2 |
| task-32 matched-shape, 18:14Z | 131 ids (minimal for row 129) | not recorded; value equals the controlled idle block (inferred idle) | 367 at -0.8024, p 0.448, entropy 2.73 | rank 2 |
| task-32 controlled concurrent load | 131 ids, same body | busy, controlled (one long generation in flight), reversible | 367 at -0.1252, p 0.882, entropy 0.73 | rank 2 |
| device, b70/runs70 | device prefill 130 ids, dump at 129 | n/a (device) | device argmax 8060, logit 19.4155 | rank 507 |

The b70 row-check line at p129 reads: `device_argmax=8060 logits_argmax=8060 reference_id=491
hybrid_id=8060 fixture_id=491 exact_token_equal=false`; the next best device logits are
67 at 17.9123, 16187 at 17.5096, 3108 at 17.0906, 71484 at 16.9611.

### The device / f32 / oracle triple at p129

    column    top-1 (value)                 top-5
    device    8060 (logit 19.4155)          8060, 67, 16187, 3108, 71484
    f32       491  (logit 13.3877)          491, 2463, 223, 63438, 67
    oracle    367  (logprob -0.0819, p.92)  367, 491, 813, 37852, 67     (suite capture)

    pairwise        top5   top20  top100  top1000
    f32 / oracle    0.40   0.55   0.48    0.529
    device / oracle 0.20   0.10   0.22    0.197
    device / f32    0.20   0.20   0.23    -

Rank of each candidate token (rank = 1 + #{logits > logit of token}):

    token  f32 rank   device rank   oracle rank
    367    7          18            1
    491    1          507           2 (released) / 7 (suite)
    8060   767        1             387

f32 vs device over the full vocabulary at p129: mean |difference| 2.4861, max 15.009581 at index
5986, 128,528 of 129,280 logits outside the 0.02 bar, correlation 0.533 - the class-A
criterion-1 failure that b68 recorded and b70 reproduces bit-for-bit.

One position earlier, f32 vs oracle agree: at matched 130-token shape, top-1 is 201 on both sides,
top-100 overlap 0.80, mean |difference| 0.354, KL 0.082 (task-32 section 2.3). Device-vs-f32 at
p128 is NOT measured: the b70 "head A/B: host_top1=201 device_top1=201 block_max_equal=true"
line feeds the host head and the device head the same device hidden, so it validates the head step
only; the dump starts at 129 and the device's first divergence point is unmeasured.

### What remains valid from b69, and what is superseded by task-32

Valid:

* the fixture's f32 prefill logits at p129: top-1 491 at logit 13.3877, 367 at rank 7; current main
  reproduces the golden fixture bit-exactly (maxdiff 0.0 at 8- and 130-token runs), so the f32
  column is sound engine output;
* the criterion token list and the golden fixture sha above, as recorded engine output.

Superseded (do not cite without the task-32 caveats):

* "the oracle is near-one-hot at p129" as a fixed property: the sharpness is capture-condition
  dependent - p(top-1) 0.448 idle at 131 tokens, 0.820 idle at 139, 0.921 at 140, 0.882 under a
  controlled concurrent load. The invariant across every capture is the argmax 367 (with 491
  second/third). Section 2.4 of `docs/build/task32-root-cause.md` gives the controlled test.
* "two independently served snapshots agree" from D38 (heretic + released): the heretic capture was
  not shape-locked or load-state-labelled either; the agreement that stands is on the argmax only.
* any device fidelity claim at p128 (see above), and `max_abs_delta` as a gate bar (task-32
  section 4/4.1: the same-shape floor is 0 and the cross-shape envelope is up to 4.06 nats, so a
  3x bar is either zero or vacuous against a ~10-nat band).

### Why the b69 capture cannot be a gate

The b69 capture is not shape-locked (the engine fixture is a 130-id prefill while the oracle rows
came from 139/140-token requests; a shape-matched row would need a 131-token request) and it is not
load-state-labelled (an external chat client was active on the instance during the window and
prefix caching/chunked prefill make the row depend on batch composition), so its p129 numbers are a
record of one unlabelled serving state, not a measurement protocol - and its intended device column
was never written. It is therefore cited only as history; the position is quarantined from D40 bars
by task-32 section 8, and the device region p129..p137 belongs to task-38.

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

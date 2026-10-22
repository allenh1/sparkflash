# Prompt logprobs in one pass (task-15, host-side)

**Status: 2026-10-21, task-15, branch `task/serve-openai-parity`, based on c3b8ae9; the measured
census runs on commit fc624a0.** Host path only: the reference CPU engine and the synthetic tiny
plan (128-row vocabulary). Every number here is UNBARED / diagnostic (D41). No device, GPU or vLLM
action was taken, and nothing here is a device numerics, latency or throughput claim.

## 0. What changed

The S2 driver scored a prompt by feeding it one token at a time: `prompt_logprobs_pass` prefilled
`prompt[..1]` and then fed one token per remaining prompt token, reading position `i`'s
distribution from the forward after `tokens[..i]`. The engine reported logits only for the
position it had just fed, so that was an O(n) growing-prefix re-forward (documented as the honest
S2 trade-off at `serve.rs:3051` before this task).

This task adds the missing engine seam and uses it:

- `sparkflash_engine::forward::PositionLogits` (`forward/mod.rs:211`) is one chunk's
  `StepOutput` plus the LM-head rows of selected positions. The rows are the same completed
  function of a position that `StepOutput::logits` is for the last one: the collapsed mHC
  residual after the final RMS norm and the head.
- `InferenceEngine::prefill_with_position_logits` (`forward/mod.rs:554`) returns that pair for
  one chunk. `run` was split into `run_selected` (`forward/mod.rs:1116`); the selected rows go
  through the same final norm and head in one batched head call, after the shared per-layer pass.
- The driver's `prompt_logprobs_pass` (`serve.rs:3150`) plans contiguous spans
  (`prompt_logprobs_spans`, `serve.rs:3111`), feeds each span with
  `prefill_with_position_logits` and asks only for the rows that score the tokens after them. The
  final span asks for `len - 1` rows: the last position's row is the generation distribution, not
  a prompt position. Every token is consumed once, on one `SequenceState`, in order.

Span planning is a memory policy, not a numeric one. `PROMPT_LOGPROBS_ROW_BUDGET`
(`serve.rs:3104`) is `1 << 22` f32 values (16 MiB); a span holds at most
`budget / vocab_size` rows, and the scheduler's `ChunkedPrefillPlan` supplies the outer chunks.
The existing invariant in `crates/sparkflash-engine/tests/chunked_prefill.rs` proves chunk
boundaries do not change a value, so a span boundary only bounds activation memory. On the tiny
plan a 4096-token prompt fits one span and one engine call
(`prompt_logprobs_spans_cover_the_prompt_within_the_row_budget`).

## 1. Contract and payload

The wire payload is unchanged: position 0 is `null` (there is no conditioning token), each later
entry is an object keyed by token id whose first key is the prompt token itself, with `logprob`,
`rank` (>= 1) and `decoded_token`, and the existing selection semantics still hold (`top N`,
`0` meaning only the prompt token for prompt logprobs, `-1`/all in unranked id order,
`token_ids` empty for prompt logprobs). The alternatives are decoded with the same decoder
context the token-at-a-time path had: the decoder is advanced only after the token itself is
decoded. The streaming rules are unchanged: `prompt_logprobs > 0` or `-1` with
`stream: true` is vLLM's typed 400 (`param: "prompt_logprobs"`), `prompt_logprobs: 0` is
accepted and dropped from the chunk.

The final `StepOutput` is now the last span's chunk prefill rather than the last single-token
decode. Its `logits`, `head_input` and `next_token` are bit-identical to the old path (pinned by
the driver test), and the generated text is unchanged; the observable difference is only in
fields that describe how the pass was run (for example speculative/draft accounting), which this
task does not surface on the wire.

## 2. Numeric agreement

The old implementation is kept verbatim under `#[cfg(test)]` as
`prompt_logprobs_pass_token_at_a_time` (`serve.rs:3208`), so the replacement is tested against
the executable old contract rather than against a rewritten approximation.

- Driver unit test `one_pass_prompt_logprobs_match_the_token_at_a_time_reference`
  (`serve.rs:7809`): for prompts `[1]`, `[1,2,3]`, `[1..6]` and selections top-2, `top 0`
  and all, the `PromptLogprobs` payloads compare equal (tokens, decoded strings, positions) and
  the final step's `logits`/`head_input` compare equal as f32 bits, along with
  `next_token` and the resulting `SequenceState` tokens. No tolerance is used.
- Engine test `selected_position_rows_match_the_prefix_reforward_bit_for_bit`
  (`crates/sparkflash-engine/tests/prompt_logprobs_prefill.rs`): for prompt lengths 2, 3, 5,
  window, window+1 and window+3, and for every span size from 1 to the prompt length, the selected
  rows are bit-identical to the rows the growing-prefix re-forward produces, and a span's final
  `StepOutput` is bit-identical to the plain prefill of the same span.
  `a_selected_position_is_the_same_row_as_the_full_collection` additionally compares a selected
  row with the same row from the full-row collection, and
  `an_empty_selection_returns_the_plain_prefill_step` keeps the empty-selection step equal to
  `prefill`.
  The out-of-range test `an_out_of_range_position_is_refused_before_the_pass` shows the typed
  `EngineError::Shape` is returned and the state is not advanced.

## 3. Refusal story

If the engine cannot serve a requested position from one pass, it returns a typed error instead of
a different number: an empty chunk or a position outside the chunk produces
`EngineError::Shape` (`what: "prompt logprobs"`) before any layer runs, and the driver
propagates it, so the request fails with the typed engine error.

The Phase-1 TP=4 path used to refuse prompt logprobs outright (its prefill loop returned only the
last position's logits). Task-18 gives it an all-position rank contract instead; the criterion and
the refusal cases are in [tp4-prompt-logprobs-rank-contract.md](tp4-prompt-logprobs-rank-contract.md).
It reports a bounded top-k request only when the f32 order is provable from the gathered candidate
blocks, and refuses (never approximates) the whole-vocabulary -1 selection, an over-wide top count, a
certified upper-bound entry, a position whose order cannot be separated, and a broken shard layout.

## 4. Tests

- Engine: the four tests above in
  `crates/sparkflash-engine/tests/prompt_logprobs_prefill.rs`.
- Driver/serve: `one_pass_prompt_logprobs_match_the_token_at_a_time_reference` and
  `prompt_logprobs_spans_cover_the_prompt_within_the_row_budget` in
  `crates/sparkflash-cli/src/serve.rs` (`#[cfg(test)]`).
- Wire-level e2e: `crates/sparkflash-cli/tests/serve_http.rs` has the existing serial-vs-batched
  wire probes (buffered prompt_logprobs 0/1, SSE stop, SSE sample logprobs, streaming
  prompt_logprobs refusal and `prompt_logprobs: 0`; the batched comparison is
  `batched_wire_probes_match_the_serial_payload_for_logprobs_and_stop`) and the new
  `buffered_prompt_logprobs_score_every_prompt_position_once` (`:1100`): an 8-token prompt on
  a serial and a `--continuous-batching --max-batch 4` server returns the same payload twice per
  mode, and every prompt position is scored once with the own-token entry, a numeric `logprob`,
  `rank >= 1` and a decoded string, with position 0 `null`.
- Census: schema `/3` adds the repeated buffered prompt_logprobs probe described in section 5;
  `scripts/tests/test_serve_s4_batching_census.py` covers its payload checks, the cross-mode
  comparison and the markdown section.

## 5. Measured census

The task-15 census is the same host A/B census as task-14 (serial vs
`--continuous-batching --max-batch 32`) with the new prompt-logprobs probe added. It ran on a
clean tree:

    cd /home/user/wt/serve-parity
    python3 scripts/serve-s4-batching-census.py

Run identity and manifests:

- commit `fc624a0` (branch `task/serve-openai-parity`, `git status --porcelain` empty in the
  artefact), binary `target/debug/sparkflash` sha256
  `4f53a6551311d6c84ff29a2e4997d65f373278c912c066d63ae5eed854f18e45`
- artefact set `serve-s4-batching-census-20261021T021856Z` under
  `/home/user/ev-session-20261020/serve/`; its `.sha256` file pins the binary, JSON,
  Markdown and both server logs:
  json `36138965fb783b14f0f7d5991c79e2a5db66b2650f414137b300af9ec8f8a8a4`,
  markdown `e54ef8fc32efde459ffeea607ad3f36352c439a93280e37a3fb5a1faaaf59e34`,
  serial log `7b3dbd634f4b8fbf5e947ddb68a9130dfc7db14b2c205c020785ac0c1a468279`,
  batched log `04fde593d60609f27ac6d1e73903b71c710fbe1d31b541a41f7f1783928e2072`
- host loadavg over the run 5.19/6.80/6.99 on 88 CPUs (loopback host census; no benchmark
  controls, no frozen floor).

Prompt-logprobs probe (the same 8-token token-id prompt the wire probes use, buffered,
`max_tokens: 1`, six repetitions of each series):

| mode | positions | scored | position 0 null | every position scores its prompt token | stable across runs | correct |
| :--- | ---: | ---: | :--- | :--- | :--- | :--- |
| serial | 8 | 7 | true | true | true | true |
| batched | 8 | 7 | true | true | true | true |

- Every scored position carries its own prompt token with a numeric `logprob`, `rank >= 1` and a
  decoded string; the payload is identical across modes and across repetitions (shape and stable
  values).
- Diagnostic client-measured p50 elapsed (loopback included, tiny plan, UNBARED): serial
  49.29 ms without prompt logprobs -> 52.23 ms with them (3.0 ms p50 difference); batched
  33.02 ms -> 35.87 ms (2.9 ms). These are repeatability numbers, not a speedup or latency claim:
  the two modes ran at different times and the batched baseline is lower than the serial one, so
  the series are recorded per mode and not compared with each other.
- Ops accounting is unchanged by the probe and grows by exactly its requests: fixed 24 + wire 14
  + prompt-logprobs probe 12 + ladder 5248 = 5298 generation tokens, and 3 fixed + 6
  token-producing wire probes + 12 prompt-logprobs requests + 82 accepted ladder streams = 103
  TTFT samples. Both modes measured 5298/5298 tokens, 103/103 samples and idle running/waiting
  gauges 0/0.
- The ladder reproduced the S4 shape on this host/load: 32-client TTFT p50 27.42 s serial ->
  1.58 s batched (previous task-14 run: 31.18 s -> 1.44 s), all fixed probe texts identical
  across modes and all seven pre-existing wire probes identical. Different loadavgs make the two
  censuses a shape check, not a comparison.

## 6. Limitations and non-claims

- Host path, synthetic tiny plan (128-row vocabulary), reference CPU engine: this says nothing
  about device batch attention, paged device KV, device TTFT/throughput, vLLM's scheduler or
  device numerics. It is UNBARED / diagnostic (D41).
- The one-pass path is one pass over the prompt, not literally one engine call for every
  vocabulary: large vocabularies are fed in bounded spans so activation memory stays bounded.
  Span boundaries are proven bit-identical by the chunked-prefill invariant and the engine tests,
  so they change memory, not values.
- The timing probe is a client-side loopback wall clock on the tiny plan; it is not a latency
  measurement of the new path versus the old one (the old path is no longer on the wire) and not a
  device or vLLM comparison.
- The TP=4 feature path serves prompt logprobs through the bounded all-position rank contract
  (section 3 and [tp4-prompt-logprobs-rank-contract.md](tp4-prompt-logprobs-rank-contract.md)),
  measured only on the synthetic twin fixture at the rank/control seam; a real four-process serve
  deployment on the released checkpoint is not measured, and device serving-path wiring and
  measurement remain window work.

# Serving S7/S8 preparation: four-rank token envelope consumer

Status: consumer + CLI wiring landed on branch task/serve-openai-parity (worktree
/home/user/wt/serve-parity), based on integration tip 64117e54. This document is the wiring and
acceptance plan for the first served device token. It is a plan plus a consumer contract, not a
measurement report. **Section 11 is the 2026-10-21 update and supersedes the schema/claim details of
sections 1-10 where they differ** (the device writer's record shape changed after the first draft:
b6f69251 replay vectors, the embed tokens batch, the object-shaped logits_dump pointer, and the
asserted/derived claim split).

Scope discipline:

- The API crate (consumer, device token source, router path, tests), the CLI serve flag wiring
  (--device-token-dir, --device-token-derived-claim, TokenPieceDecoder over TokenizerSeam), the
  host e2e test, scripts/serve_window_probe.py and docs/build changed. No device-side writer file
  and no GPU/CUDA code was touched.
- Default behaviour is unchanged: AppState::device_token_source is None unless --device-token-dir
  is given, so every existing route still runs the host twin byte-for-byte.
- All device-side facts below are read from recorded windows on this host (m1y7, m1y10, m1y11,
  m1y12). No device, GPU, CUDA, cluster or vLLM action was performed for this branch. Every test
  record is synthetic and labelled UNMEASURED.

## 1. What landed

| Piece | Location | Purpose |
| --- | --- | --- |
| Typed envelope reader + claim validation | crates/sparkflash-api/src/device_chain.rs | Loads the four rank pairs (tp4-forward-embed-r{rank}.json + tp4-token-r{rank}.json), cross-checks rank/world/metadata/head evidence, and returns either a ClaimedEnvelope or a typed ClaimRefusal naming every reason. |
| Device token source + OpenAI assembly | crates/sparkflash-api/src/device_token.rs | DeviceTokenSource seam, DeviceChainTokenSource over one window directory, TokenPieceDecoder seam, pure buffered-body and SSE builders for /v1/completions and /v1/chat/completions. |
| AppState seam | crates/sparkflash-api/src/state.rs | Optional device_token_source field, with_device_token_source builder, accessor; default None. |
| Router path | crates/sparkflash-api/src/server.rs | Plain-request eligibility, claim, device response (buffered or SSE) with the provenance header, host-twin fallback for everything else. |
| Tests | device_chain.rs, device_token.rs, tests/router.rs | Synthetic reader/claim/source/assembly tests and end-to-end router tests with a synthetic source; a source-fallback test proves the host twin still answers. |

New public surface: device_chain (DeviceChainEnvelope, ForwardEmbedRecord, TokenRecord,
ClaimedEnvelope, ClaimRefusal, DeviceChainError, LogitsDump, ...), device_token
(DeviceTokenSource, DeviceTokenRequest, DeviceTokenClaim, DeviceTokenStream, DeviceToken,
DeviceEndpoint, TokenPieceDecoder, DeviceChainTokenSource, DeviceResponseContext, body/frame
builders), and the server constants DEVICE_TOKEN_SOURCE_HEADER / DEVICE_TOKEN_SOURCE_UNMEASURED.

## 2. Observed device state: the m1y7 token window (authoritative input)

Window: /home/user/ev-session-20261020/window/m1y7/ (run.meta, argv.txt, sha256sums.txt, and
rank0..rank3 directories). It is the input the consumer was written against; it is a refused run,
not a claimed one.

run.meta facts:

- run=m1y7 tag=m1w19 kind=token started_utc=2026-10-21T05:03:44Z
- binary=3d7a55dcdc331d14187fba4518c946272d55cd89f24ce58db64d5226761c08a3
- cubins_manifest=dd9b7b0cbf065acba02db7c65e3c658b908274e2c9c7fbe8e310224500eacfac
- checkpoint=deepseek-ai/DeepSeek-V4.1-Flash-1017 snapshot 1bae0e1088993b9b82de08f59de5be5a34e42170
- comm_base_port=29601, peers=192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68
- kind_base_argv=token, extra=--forward-reference-ids 491,367,201,80,3302,201,54,201,4341
- each rank argv: --tp-rank N --tp-world-size 4 --comm-peers ... --loop --warmup-steps 2
  --decode-steps 9 --forward --forward-preflight --kernel-cubin-dir
  /home/user/ev-session-window/m1w19/stage/cubins --token-ids <138 ids>
  --forward-reference-ids 491,367,201,80,3302,201,54,201,4341

Forward/embedding record (each rank, tp4-forward-embed-r{rank}.json):

- schema=sparkflash-tp4-forward-embed/1, stage=embedding-gathered, rank=0..3, world=4
- prompt_token=20462, prompt_len=130, first_step_position=129, decode_positions=9,
  prefill_tokens=130, reference_ids=[491,367,201,80,3302,201,54,201,4341]
- token_claimed=false; exact=bf16 all-gather of raw u16 words (assembled row bit-identical to the
  owner arena bytes)
- head.schema=sparkflash-head-tail-evidence/1; head.head.ranks[0..3] each logits_f32_values=32320,
  certified_count=0
- head.gather.route=exact-f32-word-gather, gathered_logits_f32_values=129280,
  gathered_logits_sha256=901fae27131bf312425f5c434fe004885648af36056c4b49e04732e22851d2dc
  (129280 = 4 x 32320, the vocabulary size)
- head.argmax: token=52945, value=15.939203262329102, certified=false, claimable=true,
  twin_token=null, twin_value=null
- head.reduce: reduce_mode=dropped-single-stream, chain_head_launches_per_position=5,
  single_stream_head_launches_per_position=4, reduce_launches_dropped=1

Token record (each rank, tp4-token-r{rank}.json):

- schema=sparkflash-tp4-token/1, stage=tp4-chain, rank=0..3, world=4
- forward=refused, positions_run=0, hybrid_sequence_executed=false
- tokens: ids=[], positions=[], start_position=null, prompt_len=null, argmax_logits=[],
  top1_margins=[], logits_sha256=null
- logits_dump=null (no logits dump file exists in the window)
- step_ms.samples=[], wall_ms=[]
- head: rule="rank-sharded head rows all-gathered over tcp; every rank argmaxes the full vector;
  rank0 dumps full logits", vocab=129280, tie_break=lowest token id
- error: a long typed refusal. Its substance: the 130 prompt-token embeddings were gathered, the
  single-position 40-layer structural chain completed through the FFN sublayer, the layer-39 stream
  was collapsed and final-normed and the certified head tail ran (exact f32 logits gather, argmax
  token 52945 value 15.939203), but the CSA union attention body was the one-row wiring smoke
  (1 launch, grid x 16, window_rows 1, selected_count 0, out sha256
  1a712c08e2e2c52827fd5aff9ad734fcd572b4ab1a9e89cdc14c578adcf75457, not the reference 128-row
  window attention), and the loop stops before selected-candidate construction and the 130-position
  prefill sequence. No token can be claimed from that body.

What this means for serving: the head stage and the record shape are wired; the token batch is
empty and the claim assertion is false. The consumer must load this record set successfully and
refuse the claim with the reasons (token_claimed false, forward refused, positions_run 0, empty
arrays, no logits dump, error present), then let the router serve the host twin. That is exactly
what the new tests pin with synthetic copies of this shape.

## 3. Envelope schemas the consumer reads

### 3.1 Forward/embedding record (required fields)

A record pair for rank N lives either at <window>/tp4-forward-embed-rN.json and
<window>/tp4-token-rN.json (flat evidence copy) or at <window>/rankN/... (the layout m1y7 writes;
both are accepted). A path-shaped logits dump is resolved at the window root first and then in
rank0's directory. The table below lists the fields the consumer requires or reads; unknown extra
fields are ignored so the device side can keep extending the records.


| Field | Type | Consumer rule |
| --- | --- | --- |
| schema | string | must equal sparkflash-tp4-forward-embed/1 |
| stage | string | recorded, not gated |
| rank | u32 | must equal the file's rank and the token record's rank |
| world | u32 | must be 4 on every rank (the four-rank consumer) |
| prompt_token | u32 | cross-rank equal; provenance only |
| prompt_len | u32 | cross-rank equal; becomes the usage prompt_tokens |
| first_step_position | u32 | cross-rank equal (129 in m1y7) |
| decode_positions | u32 | cross-rank equal; claim requires 9 |
| reference_ids | u32[9] | cross-rank equal; claim requires 9 entries |
| prompt_ids | u32[] | optional extension (W2); when present, cross-rank equal and used to match the request prompt |
| token_claimed | bool | every rank must assert true for a claim |
| head | object | head-tail evidence block, schema sparkflash-head-tail-evidence/1 |
| final_stream_sha256 / final_normed_sha256 / final_normed_bf16_sha256 | string? | recorded digests, not recomputed here |

head block (required): head.rank must equal the record rank; head.head.ranks must list exactly
ranks 0..3, and the sum of their logits_f32_values must equal gather.gathered_logits_f32_values;
gather.route and digest are recorded; head.argmax.token must equal the first decoded token and
the value must match the first argmax_logits entry (absolute tolerance 1e-6); head.argmax.claimable
must be true; head.argmax.token must agree across ranks. certified is recorded but not a claim
precondition (S8 acceptance requires it; see section 7).

### 3.2 Token record (required fields)

| Field | Type | Consumer rule |
| --- | --- | --- |
| schema | string | must equal sparkflash-tp4-token/1 |
| rank, world | u32 | match the pair |
| forward | string | refused/not-run refuses the claim |
| tokens.ids | u32[] | exactly decode_positions entries; cross-rank equal |
| tokens.positions | u32[] | exactly decode_positions entries, consecutive, first equals start_position when set |
| tokens.start_position | u32? | must equal the first position |
| tokens.prompt_len | u32? | recorded |
| tokens.argmax_logits | f64[] | exactly decode_positions entries |
| tokens.top1_margins | f64[] | empty or decode_positions entries |
| tokens.logits_sha256 | string / string[] / null | per-position digest helper |
| positions_run | u32 | must equal decode_positions for a claim |
| error | string? | any value refuses the claim |
| logits_dump | object / string / null | inline LogitsDump or a file name relative to the window directory |
| head | object? | rule/vocab/tie_break recorded, not gated |
| hybrid_sequence_executed | bool | recorded |

### 3.3 Logits dump (provisional consumer shape)

The dump sub-schema is provisional: device-lead's first claimed run is authoritative and may add
fields (full f32 rows, per-position certification, margins). The consumer accepts either an inline
object or a JSON file name relative to the window directory, and reads:

    {
      "schema": optional string,
      "decode_positions": optional u32,
      "positions": [ {
        "position": optional u32,
        "token": optional u32,
        "value": optional f64,
        "top1_margin": optional f64,
        "certified": optional bool,
        "claimable": optional bool,
        "logits_sha256": optional string,
        "logits_f32": optional relative file name,
        "logits_f32_values": optional u64
      } ]
    }

A dump present with fewer/more than decode_positions entries, a token that differs from the
batch token, or an absolute/../escaping payload path refuses the claim (check_relative_payload is
the one place the path rule lives). The consumer does not read raw f32 rows yet, so serialised
digests are declared values, not recomputed hashes: byte-level digest verification of the f32
payload remains a device-side/tooling check (section 10).

### 3.4 Claim contract

DeviceChainEnvelope::load succeeds for a refused window (m1y7) because loading is schema and
cross-rank consistency, not a claim. DeviceChainEnvelope::claim returns Ok(ClaimedEnvelope{9
positions in order, metadata, optional dump}) only when every rank asserts the claim and the
evidence is complete; otherwise it returns ClaimRefusal with one string per concrete reason.
There is no partial claim and no inference from partial evidence. The serving source turns a
refusal into a fallback reason, never into placeholder tokens.

## 4. OpenAI wire path

### 4.1 Seam

AppState::with_device_token_source(Option<Arc<dyn DeviceTokenSource>>) installs the source.
None is the default; existing tests and the CPU host path constructors do not change. A source
implements claim(&DeviceTokenRequest) -> DeviceTokenClaim and source_name() for diagnostics.

### 4.2 Eligibility (fail closed)

The router consults the source only for a plain request that the recorded window can actually
represent. Everything else falls back to the host twin unchanged:

- /v1/completions: prompt must be the pre-tokenized int[] form; n must be 1; no echo, stop,
  logprobs, prompt_logprobs, logprob_token_ids, temperature, top_p, top_k, repetition/presence/
  frequency penalty, skip_special_tokens, seed, reasoning_effort, enable_thinking or
  chat_template_kwargs. max_tokens and stream/stream_options.include_usage are allowed.
- /v1/chat/completions: n must be 1; no image carrier, tools (non-null), stop, echo, logprobs,
  top_logprobs, prompt_logprobs, logprob_token_ids, sampling/penalty/seed/skip_special_tokens,
  reasoning fields, chat_template_kwargs, or json_object/json_schema response_format. text (and
  absent) response_format is allowed.

Rationale: a nine-token greedy window cannot honour sampling parameters, logprob/echo contracts,
tool assembly, stop sequences, constraint decoding or multimodal input. Refusing to consult the
source is the only way those fields are never silently ignored.

### 4.3 Response shape

- Buffered: OpenAI text_completion / chat.completion body with choice index 0, the concatenated
  decoded text, logprobs null, finish_reason "length" and a usage block whose prompt_tokens is the
  envelope's prompt_len (130 in m1y7), completion_tokens is the number of emitted tokens and
  total_tokens is their sum.
- Streaming: text_completion chunks per token, then a finish chunk; chat role chunk, content
  delta per token, finish chunk; then the optional usage chunk when stream_options.include_usage is
  true; then "data: [DONE]". No keep-alive comment can appear (the replay is immediate), but the
  normal SSE response helper is used.
- finish_reason is always "length": a device window has no EOS/stop evidence yet and the scaffold
  must not invent "stop". max_tokens smaller than the claimed batch truncates the emitted tokens
  and keeps finish_reason "length".
- The response carries x-sparkflash-token-source: device-chain-envelope (UNMEASURED). The
  UNMEASURED token is part of the provenance: it says this body came from a replayed recorded
  window, not from a live device measurement of this request.
- The x_deepseek_spark_stats extension is absent on a device-served response. Timing fields are
  not zeroed and not fabricated; the device path has no measured stats to report yet.

### 4.4 Fallback rules (host twin answers, existing bytes unchanged)

The router continues to the engine when: no source is installed; the envelope directory is
missing/unreadable; load or claim refuses; the source's prompt verification fails (request prompt
ids absent or different, or the envelope has no prompt_ids); the piece decoder fails; or the
request is not eligible per section 4.2. The host twin is the pre-existing code path, so S2/S3/S4
evidence and response bytes are unchanged.

Production source policy in this scaffold: DeviceChainTokenSource reads a static claimed window
directory. It requires request.prompt_token_ids to be present and to equal the envelope's recorded
prompt_ids; it refuses a chat text prompt (no ids to compare) and refuses an envelope without
prompt_ids. This is deliberately conservative: replaying a recorded window for a different prompt
would answer the wrong question with real-looking device tokens. Synthetic test sources in the
router tests exercise the claimed chat assembly path; the production source will serve chat only
after W2 and W4.

## 5. Exact wiring once a token claims

W1 (device, device-lead): make the reference attention body the real thing. m1y7 ran the CSA union
attention as a one-row wiring smoke (window_rows 1, selected_count 0) and stopped before
selected-candidate construction and the reference 128-row windowed prefill. Once that body is the
reference window attention, set tokens.ids/positions/argmax_logits/top1_margins/logits_sha256 on
every rank, positions_run=9, error=null, forward="ran", and token_claimed=true on every rank's
forward/embedding record. The consumer starts claiming as soon as those fields are consistent; no
serving code change is needed for the claim itself.

W2 (device, device-lead): write prompt_ids (the 130 prompt token ids) into every rank's
tp4-forward-embed record. The consumer already reads it (optional field, empty in m1y7) and
refuses to replay when it is absent. Without W2 a claimed envelope can only be served if the
operator runs the probe with the exact prompt recorded outside the envelope, which is not
acceptable for an automated probe.

W3 (serve, this branch's follow-up; CLI file intentionally untouched): add a --device-token-dir
flag to sparkflash serve (crates/sparkflash-cli), construct DeviceChainTokenSource::new(dir,
Arc::new(<tokenizer piece decoder>)) and install it with
AppState::with_device_token_source(Some(Arc::new(source))). The piece decoder must decode from the
checkpoint tokenizer seam: piece(ids_so_far, token_id) = decode(ids_so_far + [token_id]) with the
text of decode(ids_so_far) removed; a decode error is a fallback, never placeholder text. Keep the
flag off by default so the host path is the default binary behaviour.

W4 (serve): chat prompt capture. To serve chat from a claimed envelope, the router must render and
tokenize the request through the seam and pass those ids as DeviceTokenRequest.prompt_token_ids,
then verify they equal the envelope's prompt_ids (W2). Until then a chat request falls back to the
host twin even when a claimed envelope exists. This is the chat equivalent of the completions
token-prompt requirement and it must not be bypassed for convenience.

W5 (serve): the static-directory source is the S8 replay harness. A live source replaces the load
with a window follower: wait for a token_claimed transition (do not claim on an intermediate
record), read the four rank pairs atomically (rank0 last or a completion marker), require the exact
prompt match, and serve once per window or with an explicit cache keyed by the window run id. A
change of run id must invalidate the cached claim. Until W5 the static reader must not be pointed
at a live loop directory that is still being written.

W6 (serve): observability. Add a counter for device-served responses and for fallbacks with their
typed reason, keep the provenance header, and do not add timing fields until a timing protocol is
run. The current scaffold neither logs nor exports the fallback reason; docs/build/device-chain-
timing.md remains the place for the timing protocol.

## 6. S7 measurement protocol (soft token probe; no timing claim)

Preconditions: a window directory with all four rank pairs, every rank token_claimed=true, W2
prompt_ids present, token_claimed evidence consistent, and the serving binary built with W3.
S7 is allowed to run before certification is true: it answers "does the claimed window survive the
HTTP path", not "is the token right".

1. Start the server with the device token source installed (W3) and the host twin engine attached
   (the host path remains the fallback and the baseline).
2. POST /v1/completions with model = the engine model id, prompt = the exact prompt_ids from the
   envelope (int[]), max_tokens = 9 (or absent), stream = false. Do not send sampling parameters.
3. Assert HTTP 200 and x-sparkflash-token-source: device-chain-envelope (UNMEASURED).
4. Assert the body: object text_completion, one choice, finish_reason length, usage prompt_tokens
   = envelope prompt_len, completion_tokens = 9, total_tokens = their sum, no
   x_deepseek_spark_stats key.
5. Record the decoded text. Decode the nine token ids independently with the tokenizer seam and
   check the concatenation equals the served text (the decoder seam contract).
6. Repeat with stream=true (and stream_options.include_usage=true): the SSE delta sequence must be
   the same nine pieces, then the length finish chunk, then the usage chunk, then [DONE].
7. Check /metrics generation/scheduler counters before and after: a claimed serve must not have
   entered the engine. A host-twin fallback must show one generation request (the fallback test in
   tests/router.rs pins the in-process version of this).
8. Repeat the buffered probe to confirm the text/usage/finish sequence is stable (ids and created
   change per response by design).

S7 records provenance and correctness of the wire path. It does not produce a tok/s or TTFT
number, and the probe prompt is a recorded window, not live traffic.

## 7. S8 acceptance gate (hard token evidence)

All of the following must hold for the same window before any "served device token" claim:

1. Every rank: token_claimed=true, forward="ran", positions_run=decode_positions=9, error=null.
2. tokens.ids equals reference_ids exactly, 9/9 positions, across at least two runs (same run or
   repeated runs; a repeated run must reproduce the ids). positions are consecutive starting at
   first_step_position (129 in m1y7). The ranks agree on ids and positions.
3. head.argmax.token equals tokens.ids[0] on every rank; head.argmax.certified=true; head.argmax
   value equals the first argmax_logits value within 1e-6. The logits dump, when written, declares
   per-position certified=true and its tokens/positions agree with the batch.
4. head evidence: 4 partitions, each logits_f32_values=32320, gathered=129280; the consumer's
   claim() returns Ok.
5. W2: prompt_ids present on every rank and equal to the request prompt; the request is the exact
   recorded prompt.
6. W3: the server serves the claimed window with the provenance header; buffered and streamed
   answers agree; a repeat replay produces the same nine pieces, usage numbers and finish_reason.
7. The nine decoded pieces match the tokenizer seam's decode of the batch (same seam the host path
   uses). A decoder mismatch is a failure, not a fallback.
8. No engine entry for the claimed serve (in-process test counter or process metrics check).
9. The non-claims in section 8 are still true after the gate: S8 does not add tok/s, TTFT or a
   certified-logits claim unless the timing/acceptance protocol says so separately.

Fail-closed rule: any missing or inconsistent item means the router falls back to the host twin and
the probe records a typed refusal. A partial token batch is never served with padded or missing
pieces.

## 8. Non-claims

- No real device token exists in this branch. The m1y7 window is a refused token run; every test
  record in this change is synthetic and marked UNMEASURED in code comments and test names.
- No tok/s, TTFT, latency, throughput or speed-up claim is made. The device-served response has no
  stats extension; nothing is zero-filled to look measured. Timing lives in
  docs/build/device-chain-timing.md and docs/build/tp4-tokens-per-second.md protocols.
- No certified-logits claim: m1y7's head argmax is certified=false even though claimable=true.
- No constrained decoding, tool calling, stop sequences, echo, reasoning fields, logprobs or
  sampling parameters on the device path; such requests fall back to the host twin rather than
  silently dropping the field.
- No n>1 device path: n>1 falls back to the engine fan-out.
- No chat device serving in production until W2 and W4; the claimed chat assembly is exercised with
  synthetic sources only.
- No multimodal, no Anthropic/Responses device path.
- The scaffold is a replay harness, not a general inference server: it serves the recorded window
  only when the request prompt matches the recorded prompt.
- No device, GPU, CUDA, vLLM or cluster action was performed for this branch; no live endpoints
  were contacted.
- The host path's S2/S3/S4 evidence is unchanged; this branch adds no auth, CPU or wire regression.

## 9. Gates run for this branch

Commands (host-only, offline):

    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy -p sparkflash-cli --all-targets --features tp4-serve -- -D warnings
    cargo test -p sparkflash-api
    cargo test -p sparkflash-engine
    cargo test -p sparkflash-cli
    cargo test -p sparkflash-cli --features tp4-serve
    python3 -m unittest discover -s scripts/tests

Results on this host, all green:

- crates/sparkflash-api: 121 lib unit tests (26 new device-chain/device-token tests), 1 ignored
  evidence test (the real m1y7 window, run explicitly with --ignored and green: it loads the
  four-rank record set and refuses the claim with the recorded reasons), 89 router integration
  tests (6 new synthetic device-source tests), and the pre-existing encoding, batch-completion,
  tokenizer and doc-test suites. The pre-existing 83 router tests were not modified and remain
  byte-identical.
- crates/sparkflash-engine: 430 passed.
- crates/sparkflash-cli: 271 passed; with --features tp4-serve: 283 passed.
- scripts/tests: 289 passed (python3 -m unittest discover -s scripts/tests).
- cargo fmt --all -- --check and both clippy commands are clean. The only clippy output is the
  pre-existing nvcc-not-found build-script warning from sparkflash-kernels, which is unrelated to
  this change.

The CLI tests do not compile the new modules into a CLI binary because the flag wiring is W3.

## 10. Residual gaps and follow-ups

- W2 prompt_ids is a schema extension not yet written by the device side; until then the production
  source refuses every replay (including chat) and the router falls back.
- The logits dump sub-schema is provisional; the consumer tolerates extra fields and refuses
  internally inconsistent evidence, but byte-level digest verification of raw f32 rows is not
  implemented (the consumer reads declared digests, not payload bytes).
- The dump's per-position digests are not required to equal the token record's logits_sha256; S8
  requires the operator/device tooling to confirm they describe the same logits.
- W3 CLI flag, W4 chat prompt capture, W5 live tail and W6 device counters are listed wiring
  points, not implemented here (CLI and device files are out of scope for this change).
- Task-20 residuals are unchanged: constrained decoding absent, /v1/completions response_format
  absent, TP=4 n>1 router-refused, static bearer key only, host-path timing UNBARED.
- Related docs: docs/build/window-runbook.md (window protocol), docs/build/device-chain-timing.md
  (timing), docs/build/logits-dump.md (dump format), docs/build/tp4-tokens-per-second.md (TP=4
  throughput protocol), docs/build/openai-surface-gaps.md (surface gaps), and the task-20
  openai-surface census under /home/user/ev-session-20261020/serve/.

## 11. Update 2026-10-21: integration tip 64117e54, device line win/m1w25 fd9042f1

### 11.1 What changed since the first draft

Device writer (win/m1w25 line, kit m1w25 = a09be6b3 and m1w26 = fd9042f1) now publishes, in the
forward embed record:

- prompt_ids (the exact prompt the run fed, first prompt_len ids) and forward_ids (the full
  teacher-forced input, prompt_tokens.take(total_positions)) plus prefill_limit and replay_note
  (commit b6f69251: a claimed envelope can be replayed without the fixture file);
- a tokens batch: {ids, positions, consensus, claimable, matches_reference, argmax_logits,
  top1_margins};
- evidence_written (a complete full run contains token-consensus-complete).

The token record logits_dump is now an OBJECT (device pointer):
{path, bin, sha256, bin_sha256, dtype: f32-le, vocab: 129280, positions: [...], rows: 9}, with
absolute node paths and bin == path + .bin. The consumer therefore accepts three dump shapes
(device object, inline dump, or a path string) and never reads the dump bytes; it compares the
device-declared positions/rows/digests with the claim.

Consumer changes (this branch):

- ClaimMode::Asserted is the S8 form: every rank embed must carry token_claimed: true. The
  produced claim exposes claim_mode and the router writes x-sparkflash-token-claim: explicit.
- ClaimMode::Derived is the S7 rehearsal form: it requires the complete loop evidence instead of
  the writer assertion: consensus true, nine claimable positions, nine Some(true)
  matches_reference, token-consensus-complete in evidence_written, a full (unlimited) window,
  prompt_ids.len == prompt_len, forward_ids.len == prompt_len + 9 - 1,
  forward_ids[prompt_len - 1] == prompt_token, and token positions consecutive from
  first_step_position (129). Its wire label is derived.
- The head block is the LAST dumped position head: head.argmax.token must equal ids[8] and
  head.argmax.value argmax_logits[8] (the earlier draft compared it to the first token; m1y10
  proves the last-position relation: head token 87380 == ids[8]).
- ClaimedEnvelope carries LogitsDumpEvidence (Device | Inline); the robot claim no longer needs a
  digest-only dump.

CLI wiring W3 landed (crates/sparkflash-cli/src/serve.rs):

- --device-token-dir DIR installs DeviceChainTokenSource on AppState for both the reference path
  and the TP=4 path (rank 0 only; workers have no HTTP surface).
- --device-token-derived-claim selects ClaimMode::Derived; it is refused at parse time without
  --device-token-dir.
- SeamPieceDecoder implements TokenPieceDecoder over the same TokenizerSeam the host path uses:
  piece(n) = decode(ids[0..n] + [id]) minus decode(ids[0..n]), error if the tokenizer decode is
  not prefix-monotonic. A decoder error is a fallback, not placeholder text.
- Startup prints one line: sparkflash serve: device token source DIR: asserted|derived claim ok
  (9 tokens, prompt_len 130, dump present) OR ... claim refused ...: <every reason>; a refused
  claim still serves, but every eligible request falls back to the host twin (fail closed). A
  schema/load error refuses startup instead of silently serving the twin.

Host tests added: API unit tests for asserted/derived/dump/schema refusals; router tests with a
synthetic source (claimed/fallback) asserting the claim header; a CLI unit test for the flags and
for the piece decoder; an end-to-end tests/serve_http.rs test that starts the real binary with a
synthetic four-rank window and a reference tokenizer, checks headers/usage/text/finish/[DONE]
equivalence and the host-twin fallback for a non-window prompt. The synthetic window writers are
exposed by the sparkflash-api feature device-token-fixtures (dev-dependency only; paper records,
UNMEASURED).

### 11.2 Device facts from the recorded windows

m1y7 (m1w19-era, refused): no dump, forward=refused, token_claimed false. Consumer loads it and
refuses, naming every reason (ignored evidence test keeps this).

m1y10 (m1w24, full 9-position run, --forward-reference-ids): decode_positions 9, prompt_len 130,
first_step_position 129, reference_ids 9, dump present, evidence_written complete; BUT
prompt_ids/forward_ids empty (writer predates b6f69251), all matches_reference false (the run
measured a FAIL), head.argmax.token 87380 == ids[8]. Not claimable: a recorded FAIL, and the old
record cannot support the derived form either.

m1y14 (m1w26, the first full 9-position acceptance-shaped window after the replay work; recorded
2026-10-21 06:58-07:07Z): all four ranks forward=run, cross-rank identical, prompt_ids 130,
forward_ids 138 with forward_ids[129] == prompt_token 20462, positions 129..137, consensus true,
claimable all true, evidence_written complete, device dump present in every token record (9 rows,
129280 f32 = 4654080-byte bin), head.argmax.token == ids[8] == 86 and value == argmax_logits[8]
(the corrected last-position head rule, confirmed on real data). The decoded ids are
[491, 6673, 201, 15, 100262, 6224, 16, 5421, 86] against reference
[491, 367, 201, 80, 3302, 201, 54, 201, 4341]: matches_reference is
[T, F, T, F, F, F, F, F, F] on all four ranks. The claim therefore refuses in derived form on the
reference match alone, and in asserted form on token_claimed alone: this is a measured reference
FAIL of the device loop (row 129 correct, divergence from row 130), not a consumer gap, and the
window cannot serve S7 or S8.

Root cause (Lead, 2026-10-21): the four-rank forward loop never applies the Engram write that the
reference architecture performs before layers 1 and 14 (engine forward/mod.rs:1283-1284); the
boundary pair breaks exactly at rows 0 and 13. This is a known-cause measured FAIL baseline: it is
kept in the record, the acceptance bars are frozen and unchanged. The next kit m1w27 carries the
Engram seam plus the token_claimed flip (the condition in 11.6); a later full window must show nine
true matches, after which S7 needs no writer change and S8 needs only the flip.

m1w27-shape rehearsal (synthetic mutations, 2026-10-21, UNMEASURED): a passing synthetic window
whose four embed records were edited to token_claimed true plus an unknown engram_seam object
loads and claims in both forms, serves, and probes PASS 18/18 - so the consumer needs no code
change for a m1w27 record whose only diffs are the flipped boolean and additional evidence keys
(serde ignores unknown fields; the asserted form still enforces the nine reference matches as
defense in depth, so a flipped but reference-failing window is refused on that reason alone). The
m1y14 copy with token_claimed flipped the same way refuses at startup with the reference reasons
and serves the host twin for every request (fail closed), which is the intended behaviour.

m1y11 / m1y12 (m1w25 replay windows): full schema present (prompt_ids, forward_ids, device dump
object, tokens batch, token-consensus-complete), token_claimed false, reference_ids empty (no
--forward-reference-ids), decode_positions 1 and prefill_limit 1. The loop runs only positions
0..N-1 when --prefill-limit N is set, so token positions are 0-based for a limited replay, while
first_step_position stays prompt_len - 1; a full run runs prompt_len + 9 - 1 positions and dumps
positions 129..137, which is the only form the claim accepts. m1y13 is the same replay against the
m1w26 kit (fd9042f1).

### 11.3 The one acceptance window S7 and S8 both need

Serving parity evidence and the device token must come from ONE window. The required run is the
full four-rank loop, not a replay:

    window: /home/user/ev-session-20261020/window/<run>/
    binary: win/m1w25 fd9042f1 (kit m1w26, sparkflash-m1w26) - it contains b6f69251
    command shape (rank 0; every rank same with its own --tp-rank/ip):
      sparkflash-m1w26 --checkpoint <snapshot> --tp-rank 0 --tp-world-size 4 \
        --comm-peers 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 \
        --comm-base-port 39000 --comm-timeout 120 --comm-connect-timeout 120 \
        --upload --loop --warmup-steps 2 --forward --forward-preflight \
        --kernel-cubin-dir /home/user/ev-session-window/m1w26/stage/cubins \
        --token-ids <the same 138-id fixture list as m1y10/m1y11> \
        --forward-reference-ids 491,367,201,80,3302,201,54,201,4341
    NO --prefill-limit; decode positions default to 9.

Acceptance condition for the window: all four ranks forward=run, error null, 9 ids/positions/
argmax/top1, consecutive positions starting at 129, matches_reference all true, dump rows 9 and
positions 129..137 with equal bin_sha256 on the ranks that publish one, evidence_written contains
token-consensus-complete, and prompt_ids 130 / forward_ids 138 with forward_ids[129] ==
prompt_token. If the reference match is false the run is a measured FAIL and neither S7 nor S8 can
use it (that is a device question, not a serving question).

### 11.4 S7 rehearsal: command lines and expected artifacts

Aims: one claimed (derived form, because the device writer still records token_claimed false)
device envelope served through the OpenAI completions path with the UNMEASURED provenance, no
engine entry, no host stats, exact usage relation. Arc dir convention:
/home/user/ev-session-20261020/serve/s7-<run>/ (never delete the serve/ directory).

    ARC=/home/user/ev-session-20261020/serve/s7-<run>
    mkdir -p "$ARC"
    git -C /home/user/wt/serve-parity rev-parse HEAD > "$ARC/branch.txt"
    cargo build -p sparkflash-cli --release
    sha256sum target/release/sparkflash > "$ARC/binary.sha256"
    sha256sum "$W"/rank*/* > "$ARC/window.sha256"

    python3 scripts/serve_window_probe.py inspect \
      --window "$W" --arc-dir "$ARC" --mode derived        # exit 0 required

    target/release/sparkflash serve --synthetic --bind 127.0.0.1:0 \
      --tokenizer <tokenizer.json> --device-token-dir "$W" \
      --device-token-derived-claim > "$ARC/server.out" 2> "$ARC/server.err" &
    # wait for the printed listening URL, record it in "$ARC/server.url"
    # confirm "$ARC/server.err" contains: derived claim ok (9 tokens, prompt_len 130, dump present)

    python3 scripts/serve_window_probe.py probe --window "$W" \
      --base-url "$URL" --arc-dir "$ARC" --mode derived
    python3 scripts/serve_window_probe.py probe --window "$W" \
      --base-url "$URL" --arc-dir "$ARC/repeat2" --mode derived   # repro, different ids only

Expected artifacts: branch.txt, binary.sha256, window.sha256, window-claim.json (claimable true,
claim_mode derived), server.out/.err/.url, probe-buffered.json, probe-stream.sse,
probe-summary.json (every check passed), EVIDENCE.md, repeat2/.

The chain can be dry-run before a real window exists (it was, on 2026-10-21, both claim forms,
probe PASS 18/18 and a clean repeat): write a synthetic four-rank window with

    cargo run -q -p sparkflash-api --features device-token-fixtures \
      --example write_device_window -- /tmp/sparkflash-window asserted     # or derived

then run the same inspect/serve/probe block against /tmp/sparkflash-window; record the synthetic
run separately, label it UNMEASURED, and never mix it into S7/S8 evidence.

S7 pass conditions: window claimable in derived form; startup line says derived claim ok; buffered
completion JSON with object text_completion, finish_reason length, usage 130/9/139, no
x-deepseek-spark-stats; headers x-sparkflash-token-source = device-chain-envelope (UNMEASURED) and
x-sparkflash-token-claim = derived; SSE has 9 content frames, finish length, usage chunk equal to
the buffered usage, [DONE], and the deltas concatenate exactly to the buffered text; the second
probe returns the same text/usage (request ids may differ). No tok/s, TTFT or certified-token
claim is made.

### 11.5 S8 acceptance: command lines and artifacts

Requires the same window with token_claimed true on every rank (see 11.6 blocker 1). Arc dir:
/home/user/ev-session-20261020/serve/s8-<run>/.

    ARC=/home/user/ev-session-20261020/serve/s8-<run>
    mkdir -p "$ARC"
    git -C /home/user/wt/serve-parity rev-parse HEAD > "$ARC/branch.txt"
    cargo build -p sparkflash-cli --release
    sha256sum target/release/sparkflash > "$ARC/binary.sha256"
    sha256sum "$W"/rank*/* > "$ARC/window.sha256"

    python3 scripts/serve_window_probe.py inspect \
      --window "$W" --arc-dir "$ARC" --mode asserted      # exit 0 required

    target/release/sparkflash serve --synthetic --bind 127.0.0.1:0 \
      --tokenizer <tokenizer.json> --device-token-dir "$W" \
        > "$ARC/server.out" 2> "$ARC/server.err" &
    python3 scripts/serve_window_probe.py probe --window "$W" \
      --base-url "$URL" --arc-dir "$ARC" --mode asserted
    python3 scripts/serve_window_probe.py probe --window "$W" \
      --base-url "$URL" --arc-dir "$ARC/repeat2" --mode asserted

S8 hard-evidence checklist (same window for device records and serving probe):

- window: four-rank forward+token records from the same run, sha256 recorded in window.sha256;
- claim: explicit (every rank token_claimed true), 9 ids equal to the 9 reference ids, positions
  129..137 consecutive, head.argmax.token == ids[8] and value == argmax_logits[8], dump rows 9
  with the same positions and equal bin_sha256 across ranks, prompt_ids 130, forward_ids 138,
  forward_ids[129] == prompt_token;
- wire: source header device-chain-envelope (UNMEASURED), claim header explicit, usage 130/9/139,
  finish_reason length, [DONE], 9 content frames, no host stats header, no engine entry (the
  startup claim line plus the fixed headers identify the device path; the fallback path never
  writes those headers);
- repeat: second probe same text and usage; token ids/text equality with the first probe.

### 11.6 Blockers and exact missing surfaces (as of this branch)

1. The device writer never asserts token_claimed. In win/m1w25 fd9042f1 the literal
   token_claimed: false sits in the forward evidence json in
   crates/sparkflash-cli/src/tp4_forward.rs (line 5239 on that branch; 4860 on integration tip).
   S8 asserted form is impossible until the device owner flips it. Exact suggested change: in the
   evidence rewrite block that inserts tokens/prompt_ids/forward_ids (after the initial json!),
   insert an override:

       object.insert("token_claimed".to_string(), serde_json::json!(
           request.prefill_limit.is_none()
               && tokens.positions.len() == positions
               && token_matches.len() == positions
               && token_matches.iter().all(|matched| *matched == Some(true))
               && heads.iter().all(|head| head.claimable)
       ));

   (that is the same condition the consumer checks in derived mode, so the two forms cannot
   disagree). The serve branch deliberately does not touch device code.
2. No full, all-reference-true window exists yet: m1y10 (m1w24) measured all-false,
   m1y11/m1y12/m1y13/m1y13b are prefill-limited replays, and m1y14 (m1w26, full) measured 2/9 with
   the known Engram cause in 11.2. Both S7 (derived) and S8 (asserted) wait on one 9/9 full run
   from the command in 11.3; m1w27 carries the Engram seam and the token_claimed flip.
3. The derived form needs the b6f69251 replay vectors. They are in the win/m1w25 line (fd9042f1,
   kit m1w26) but NOT on integration tip 64117e54. The acceptance window must therefore be run
   with the win/m1w25/m1w26 device binary, or b6f69251 must be merged into the integration tip
   before the window is run; otherwise derived refuses with prompt_ids/forward_ids empty.
4. The serving binary must be built from this branch (task/serve-openai-parity): the flags, the
   claim header and the derived rule exist only there. Host build only (no cuda feature needed;
   the device chain is read from files, not a GPU).
5. Chat is not covered: DeviceChainTokenSource refuses a chat text prompt (there is no chat prompt
   capture in the window yet; W4), so S7/S8 use /v1/completions with the recorded prompt token
   ids only. Every other request (different prompt, sampling, logprobs, echo, stop, n>1) is
   host-twin fallback by design.

### 11.7 Branch, gates and merge status

Branch task/serve-openai-parity, base integration tip 64117e54, tip e99604a5 (code head d6dfc6d8;
the two later commits are docs only):

    5309da7b serve: claim the m1w26 device token envelope (asserted or derived) and label the claim form
    672fd395 serve: install the device token window from the CLI and add the S7/S8 probe
    fe91947a serve: rehearse the S7/S8 chain end to end on synthetic windows
    d6dfc6d8 docs: record the S7/S8 branch tip, gates, merge checks and the synthetic dry-run
    e99604a5 docs: record the m1y14 full window and its measured reference fail

Gates at fe91947a: cargo fmt --all; cargo clippy --workspace --all-targets -- -D warnings; cargo
clippy -p sparkflash-cli --all-targets --features tp4-serve -- -D warnings; cargo clippy -p
sparkflash-api --all-targets --features device-token-fixtures -- -D warnings; cargo test -p
sparkflash-api --lib (126 passed, 2 ignored evidence); cargo test -p sparkflash-api --test router
(89); cargo test -p sparkflash-cli --lib (316); cargo test -p sparkflash-cli --test serve_http
(22); cargo test -p sparkflash-engine (10); python3 -m unittest discover -s scripts/tests (370).

Merge checks with git merge-tree --write-tree, both clean (exit 0, no conflict output): into
64117e54 -> tree 5f96f63, into win/m1w25 fd9042f1 -> tree aff2a2e. No rebase is needed. The merge
into the integration tip does not carry b6f69251 (replay vector writer); the serving consumer only
needs those fields to be present in the window at load time, so a window recorded by the m1w26
device binary serves correctly either way, while a window recorded on the bare integration tip
would lack prompt_ids/forward_ids and be refused in derived form (asserted form would still work
if the writer asserted).

Host dry-run at fe91947a: synthetic asserted window -> serve --device-token-dir -> inspect+probe
(18 checks PASS, repeat PASS); synthetic derived window -> serve --device-token-dir
--device-token-derived-claim -> inspect+probe (18 checks PASS, repeat PASS). At e99604a5 the
m1w27-shape rehearsal also passes (see 11.2), and the m1y14 copy flipped to token_claimed true
refuses and falls back (fail closed). The only untested step is a real window with all nine
matches true, which the Lead owns.

### 11.8 Non-claims

- The served body is UNMEASURED by construction: the header says so and no timing/count claim is
  attached. S7 is a wiring/soft-token probe, not a performance measurement.
- A synthetic fixture claim (the host tests) is paper evidence: no device produced its ids, text,
  positions or dump. It proves the code path, not the device.
- A derived claim is NOT the S8 acceptance form. It exists so the serving path can be rehearsed
  against a real window before the writer asserts; S8 needs the explicit device assertion.
- The consumer never reads dump bytes; it checks the device-declared dump shape/positions/digests.
  A dump-content comparison (full-vocabulary distribution vs a host twin) is the D39/D41
  acceptance tooling, not this path.

## 12. Update 2026-10-21 (M2 lean+hoist tip 6a40bfa5): W3 rebased, m1y34 rows read

### 12.1 Branch and base

The serve work now sits on integration tip win/m1w25 6a40bfa5 (M2 lean+hoist kit, section 7 of
docs/build/m2-critical-path-profile-20261021.md) as branch task/serve-s7-next: the six commits of
task/serve-openai-parity were cherry-picked onto it with no conflicts (the M2 change touched
tp4_chain.rs/tp4_forward.rs and docs, not the API consumer or serve.rs). Commit map:

    5309da7b -> f4edb0af  serve: claim the m1w26 device token envelope (asserted or derived)
    672fd395 -> 695e1057  serve: install the device token window from the CLI and add the probe
    fe91947a -> db86ea7e  serve: rehearse the S7/S8 chain end to end on synthetic windows
    d6dfc6d8 -> d96f473a  docs: branch tip, gates, merge checks and the synthetic dry-run
    e99604a5 -> 43b58592  docs: m1y14 full window and its measured reference fail
    3a9a8286 -> e8e6382e  docs: m1y14 known-cause baseline and the m1w27-shape rehearsal

### 12.2 The two m1y34 rows (identity and lean)

Both are four-rank token runs of the m1w34 kit (binary sha
e3470a141a886bdc3413c12073fd0a16d37e445bbc9b39aca4bb70ce452643eb, cubins dd9b7b0c...) recorded
2026-10-21:

- /home/user/ev-session-20261020/window/m1y34-identity (default evidence-bearing row, hoisted
  tail): 9 positions 129..137, prompt_ids 130, forward_ids 138, evidence_written complete, dump
  object in every token record, logits bin sha
  aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2 (4,654,080 bytes),
  tokens [491, 92928, 201, 15, 4893, 16646, 15, 14265, 86], reference
  [491, 367, 201, 80, 3302, 201, 54, 201, 4341], matches_reference
  [T, F, T, F, F, F, F, F, F] on all four ranks, token_claimed false.
- /home/user/ev-session-20261020/window/m1y34-lean (--forward-lean): same tokens and logits
  bin sha, reduced diagnostics: head-input holds only the last-position alias per rank (1 file,
  not 10), 5 progress writes per head position instead of 40, twin entries are the skipped form,
  and every forward-progress entry carries forward_perf: {lean: true}. The identity row carries
  forward_perf: {lean: false} instead. This is the M2 evidence delta, not a token-content change.

### 12.3 What the m1y34 shape changed for the consumer (no break)

- forward_perf lives in tp4-forward-progress-r{rank}.json entries. That file is not part of the
  envelope the consumer reads (it reads only the embed and token records), and serde ignores
  unknown fields in the records it does read. m1y34 rows load unchanged in both forms; the only
  verdicts are the claim reasons (token_claimed false, and the still-false nine reference
  matches). A unit test now pins that a window directory carrying the new progress row loads and
  claims exactly as before.
- Lean evidence deltas that could have been assumptions, and are not: the consumer never stats or
  reads the head-input files named in head.input.file (paths are node-absolute and may not exist
  on the serving host; a lean record names a per-position file it does not write - the open
  evidence nit in M2 section 7.1); it never reads the logits dump bytes; it never reads the
  progress cadence or twin_skipped values. So identity and lean rows produce identical claim
  inputs and identical claim verdicts, and the same dump bin sha is reported from either token
  record.
- The python probe mirrors the same rules; it reads the token record dump pointer and reports
  aed6054d... for both rows.

### 12.4 W3 and the OpenAI routes on this base

W3 (the CLI wiring) is on the branch: --device-token-dir installs the device chain token source
on the shared serve_with_engine tail (reference path and TP=4 rank 0), --device-token-derived-claim
selects the derived form, SeamPieceDecoder decodes pieces over the host tokenizer seam, and a
load error refuses startup while a refused claim keeps serving with the host twin.

New host e2e tests (crates/sparkflash-cli/tests/serve_http.rs, real binary + loopback socket):

- serve --help documents both flags (the help text is the operator contract);
- /v1/completions buffered on a claimed synthetic window (existing test);
- /v1/completions SSE on the same window: source/claim headers, nine content frames, length
  finish, usage chunk (130/9/139), [DONE], decoded text equals the tokenizer seam decode;
- /v1/chat/completions buffered and SSE with a device window installed: the host twin answers,
  no device headers, host stats block present, streamed content equals the buffered content;
- streaming tool calls (DSML) with a device window installed: buffered and streamed tool_calls
  reassemble byte for byte, finish_reason tool_calls, still no device headers;
- a refused window is fail closed: startup names every reason, requests get the host twin and no
  device header, and the same schema claims in derived form when the flag selects it.

The probe gained --expect fallback: for a window that is refused in every form it asserts the
fail-closed shape (200, no device headers, host stats block, SSE terminates, deltas join the
buffered text) and writes FALLBACK.md. A fallback probe refuses to run on a claimable window so a
record cannot be mislabelled.

### 12.5 S7 status after this block

Both m1y34 rows are measured perf rows, not D41 acceptance rows: they are refused by the claim
(token_claimed false; 2/9 reference matches). The real-window S7 therefore did NOT serve a device
token; the block advances S7 end-to-end to the refusal boundary and records it. The record draft
with exact paths and shas is docs/build/serving-s7-record-draft.md. The device path itself is
still proven on the synthetic claimed window (paper evidence, UNMEASURED), and the real window
proves the fail-closed fallback over the same routes.

### 12.6 What S8 (and a real S7) still needs

1. A full window whose nine tokens match the reference ids (9/9) - the m2 kit did not change the
   reference-match gap: m1y14 was 2/9 and m1y34 is the same 2/9 (same first token 491, different
   wrong continuation). This is the known Engram-write cause recorded in section 11.2 (or a later
   root cause if M2 changed the story); no serving change can substitute for it.
2. The device writer token_claimed assertion (the one-line override in section 11.6) for the S8
   asserted form; the derived form (S7 rehearsal) works without it once 9/9 holds.
3. Nothing else on the serving side: W3, the m1y34 shape, the derived/asserted claim split, the
   wire headers, the chat/SSE/tool-call fallbacks and the refusal reasons are all host-tested.

### 12.7 Non-claims

No token-acceptance claim, no tok/s or TTFT claim is made from the m1y34 rows: they are M2
performance rows, the served bodies are UNMEASURED provenance, and the synthetic window is paper
evidence only.

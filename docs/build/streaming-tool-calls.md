# Streaming tool-call (V4.1 DSML) parsing

Status: landed in task-19 (WS-E, section-11 item 1 of the OpenAI parity audit). Host path only:
no CUDA, no device kernel, no GPU, no cluster run, no vLLM comparison.

## What changed

The buffered V4.1 DSML parser (crates/sparkflash-api/src/tool_parser.rs, applied by
server.rs apply_deepseek_v41_tool_parser) already lifted a complete chat answer's DSML block into
an OpenAI tool_calls array. The SSE path did not: it sent the raw DSML text as content deltas and
never filled delta.tool_calls. This slice adds an incremental parser for that path:

- tool_parser.rs: StreamToolEvent, StreamToolRefusal, TOOL_STREAM_REFUSAL_CODE,
  ToolCallStreamParser (feed / finish / calls), plus split-at-every-boundary property tests.
- server.rs: chat_stream_response feeds every content delta through the parser when the client
  offered tools, emits content deltas, call-start deltas (id + function name + empty arguments)
  and argument-fragment deltas, overrides the final choice chunk to finish_reason tool_calls, and
  turns a malformed or unterminated block into one typed SSE error frame.
- state.rs: AppState::id_counter, so the streaming generator can format the same
  chatcmpl-tool- ids the buffered path uses (a 'static SSE body cannot hold &AppState).
- serve.rs: a debug-build-only scripted-engine seed (SPARKFLASH_SERVE_SCRIPT) and keep-alive
  override, so the real binary and router can be exercised over a loopback socket on a host with
  no checkpoint; serve_with_engine is the shared tail of the checkpoint and scripted paths.
- serve_http.rs: scripted wire-parity test plus malformed/unterminated wire test.
- scripts/tool-call-streaming-census.py + scripts/tests/test_tool_call_streaming_census.py:
  the measured probe for this slice.

## Grammar and the buffered contract

The marker is the V4.1 DSML marker (DSML = the fullwidth-bar token string shared with the prompt
renderer). One block is:

    <MARKER calls>
    <MARKER invoke name="NAME">
    <MARKER parameter name="KEY" string="true|false">VALUE</MARKER parameter>
    ...
    </MARKER invoke>
    ...
    </MARKER calls>

The buffered parser's exact semantics are the contract the streaming parser must reproduce,
including its newline handling:

- The block is located by the first occurrence of a blank line plus the calls marker
  (two newlines plus marker) anywhere in the answer; if that is absent, the first calls marker is
  used. When the blank-line form is used both newlines are stripped from the content, otherwise
  the raw prefix is used.
- The buffered parser strips one newline after the calls marker and one newline at the top of its
  invoke loop; it strips one newline after an invoke name and one at the top of its parameter loop.
  Those optional single-newline skips are reproduced exactly, not normalised.
- Content is everything before the block; anything after the calls end must be whitespace.
- An arguments object is rendered as an object JSON string: key is the serde_json encoding of the
  parameter name, a string="true" value is encoded as a JSON string, a string="false" value is
  inserted raw when it parses as JSON and quoted as a string otherwise. Parameters are joined with
  ", "; an invoke with no parameters renders "{}". Duplicate parameter names are kept (the last one
  wins for a JSON client), matching the buffered parser.
- Malformed buffered text is not an error: the buffered path documents "unparsed text stays
  content". The streaming path cannot leave a client mid-answer without a signal, so it uses a
  typed refusal instead (see below).

## The incremental parser

Public shape:

    pub enum StreamToolEvent {
        Content(String),
        CallStart { index: usize, name: String },
        ArgumentsDelta { index: usize, arguments_delta: String },
    }
    pub struct StreamToolRefusal { pub message: String }
    pub const TOOL_STREAM_REFUSAL_CODE: &str = "tool_call_parse_error";
    pub struct ToolCallStreamParser { ... }
    impl ToolCallStreamParser {
        pub fn new() -> Self;
        pub fn feed(&mut self, text: &str) -> Result<Vec<StreamToolEvent>, StreamToolRefusal>;
        pub fn finish(&mut self) -> Result<Vec<StreamToolEvent>, StreamToolRefusal>;
        pub fn calls(&self) -> usize;
    }

States (private enum):

- Content: outside a block. Scans for the blank-line-plus-marker form first, then the plain
  marker, and holds back only the longest suffix that is a proper prefix of either pattern (at
  most one byte of a fullwidth bar's 3-byte UTF-8 encoding, or a partial newline pair). Every byte
  before that suffix is safe to emit as content.
- SkipNewline(next): the buffered grammar strips at most one newline at a given point; this state
  consumes it if present and moves on, so exact buffered newline semantics are preserved.
- BetweenCalls: expects one more invoke or the calls end; partial markers wait for more bytes.
- InvokeName: scans the quote-greater-than-newline delimiter, then emits CallStart.
- BetweenParameters(index, first): expects the invoke close or one more parameter. Invoke close
  emits "{}" when no parameter was seen, otherwise "}" closing the arguments object.
- ParameterName, ParameterFlag, ParameterValue: scan the parameter name delimiter, the
  true/false flag, then the value up to the parameter close marker, and emit one arguments
  fragment ("{key: value" for the first parameter, ", key: value" for later ones).
- AfterBlock: only trailing whitespace is tolerated; anything else is a malformed refusal.
- Failed: terminal, with the refusal recorded.

finish() flushes end-of-stream:

- In Content any held-back bytes are emitted as content: a partial marker at the end of a plain
  answer is text, not a silent drop (the buffered parser behaves the same way for a truncated
  block: no parse, raw text stays content).
- Inside a block or after a non-whitespace tail it returns an unterminated/malformed refusal.

Fragments are rendered with the same serde_json rules as the buffered parser, so concatenating
them reproduces the buffered arguments string byte for byte (including ", " separators, nested
JSON parameter values and unicode escapes).

## Bounded state and typed refusals

The buffer holds only bytes that could still become a marker or a grammar delimiter; content is
flushed as soon as it is safe. A malformed or unterminated block still cannot grow without bound
while the engine keeps streaming, so the parser refuses once the held block exceeds
MAX_TOOL_STREAM_BLOCK_BYTES (4 MiB; far above any real tool call). A refusal is sticky: later
feed/finish calls report the same error instead of trying to resynchronise.

The chat SSE path treats the refusal as terminal for that choice: it emits one error frame

    data: {"error":{"message":...,"type":"server_error","param":null,"code":"tool_call_parse_error"}}

then the SSE terminator. No finish chunk and no usage chunk follow the error frame (the deltas
already sent cannot be retracted, so the stream must not pretend the answer finished). The
type/param fields mirror the server's other error frames; the code is the stable discriminator.

## Chat SSE integration

The parser is active only when the client offered tools (request.tools is a non-null list, the
same switch the buffered path uses) and streaming logprobs were not requested. In the logprob
mode a content delta has to travel with the per-token logprob entry it belongs to, which cannot be
mapped onto tool-call fragments; that combination is a documented residual gap rather than a
silent mix.

Frame order for a streamed DSML answer:

1. Role frame (assistant), first, unchanged.
2. Content deltas for text before the block.
3. For each call: CallStart (index, id chatcmpl-tool-XXXXXXXX, function name, arguments ""),
   then argument-fragment deltas with no id/name.
4. Final choice chunk: delta {}, finish_reason tool_calls (overriding the engine's stop), stats
   block attached as before.
5. Optional usage chunk when stream_options.include_usage was requested.
6. data: [DONE], last.

Ids come from the same AppState counter as the buffered path, so a tool call has the same
chatcmpl-tool- id in both shapes for the same request order. Keep-alive comments are injected by
the existing SSE wrapper between complete frames; a comment frame is never spliced into a data
frame, so the streaming deltas are unchanged by the interleave (pinned by router tests and by the
wire census: positions are monotonic, no comment before the first data frame, none after the
terminator).

## Tests

Parser unit tests (crates/sparkflash-api/src/tool_parser.rs):

- one-shot streamed payload equals the buffered payload for a set of fixed DSML answers: content
  plus one call with string and JSON parameters, several calls, nested JSON escapes and unicode
  strings, an invoke with no parameters, a plain answer with no block, a partial marker, trailing
  whitespace after the block;
- split-at-every-boundary property test: every UTF-8 character boundary of every fixed answer is
  used as a two-chunk split point, and the reassembled content/calls must equal the buffered
  payload;
- character-at-a-time feed, empty-parameter "{}" rendering, malformed literal refusal,
  unterminated-at-finish refusal, trailing-text refusal, 4 MiB bound refusal, sticky refusal.

Router tests (crates/sparkflash-api/tests/router.rs): streaming DSML equals the buffered body
(ids included, fresh state on both sides), several calls keep indices/order, malformed and
unterminated answers end with the typed error code and no finish chunk, keep-alive comments do
not corrupt deltas, and the same DSML without offered tools still streams as raw content.

Wire e2e (crates/sparkflash-cli/tests/serve_http.rs, debug-build scripted seed): the real serve
binary, real tokenizer seam, real axum router and real loopback socket; buffered body from one
fresh server and streamed body from another fresh server are compared call-by-call, the SSE
terminator/ordering and keep-alive interleave are asserted on the raw frame text, and a malformed
and an unterminated answer are checked for the typed error frame. 15 serve_http tests pass.

Python tests: scripts/tests/test_tool_call_streaming_census.py (16 cases) validates probe docs,
refuses mismatched evidence, and pins the markdown accounting.

## Debug-build wire seed

The wire e2e needs a model that can emit DSML, but this host has no checkpoint and the real
synthetic tiny plan cannot produce released-tokenizer DSML. serve.rs therefore has a test seam,
compiled only under #[cfg(debug_assertions)] and read from the environment:

- SPARKFLASH_SERVE_SCRIPT: path to a JSON file {"pieces": ["...", ...], "finish_reason":
  "stop" | "tool_calls", "delay_ms": N}. The pieces become the mock engine's content deltas (so
  the test controls the exact chunk boundaries), and delay_ms creates an idle gap for keep-alive.
- SPARKFLASH_SERVE_KEEP_ALIVE_MS: shorten the SSE keep-alive interval for the test.

run() checks the seed after tokenizer resolution and before recipe/driver construction, so the
hook replaces the checkpoint with a mock engine while the rest of the server (tokenizer seam,
router, metrics, bind) is the production path. Release builds do not compile the hook, so a
production binary has no such switch.

This is a test seam, not a serving feature: it is documented here so the wire test is not a
mystery, and the non-claims below say what it does not measure.

## Measured census (UNBARED)

scripts/tool-call-streaming-census.py builds the serve_http test binary and runs the scripted
wire probe three times, then writes JSON + markdown + sha256 under
/home/user/ev-session-20261020/serve/. The recorded run is
tool-call-streaming-census-20261021T034618Z (an earlier pre-final capture
tool-call-streaming-census-20261021T033527Z is retained beside it; its serve binary predates the
serve_with_engine rename, so the later run is the one cited):

- json sha256 59ad82eca138720ca148c79b4152a918cc4cea01e043982be763e0fe11b6bfb2
- md sha256 e8f1c5fb3157b3d8391a2cbd9b1f6fa0876c8280716ef75fdb3e57084ab4f365
- test binary sha256 06ec9667ba34e23930a18d4f93b1365e21db0f15364a022b7d4350dfa389bf8e
- serve binary sha256 7ee2ea50110918b1a6be4a36a8b19da56aec130077f263ffeb1064e79e9822d6
- 3 runs; buffered and streamed tool_calls arrays byte-for-byte equal in every run; content equal
  in every run; final finish_reason tool_calls; one tool call with arguments
  {"city": "Beijing", "days": 3}; content deltas 5/5/5; call-start deltas 1/1/1; argument
  fragments 3/3/3; 11 SSE data frames per run; 169/170/169 keep-alive comment frames interleaved
  at monotonic positions with no comment before the first data frame and none after the
  terminator; streamed request wall 1092/1097/1090 ms.
- The census JSON records git commit 86f1249 because the slice was still uncommitted when the
  probe ran; its source_manifests pin the exact measured working-tree bytes. The task-19 commit
  adds this note and the audit rows; no source changed after the final probe run except
  documentation.

This is a host-path parser/wire measurement (loopback socket, scripted stub engine, synthetic
DSML fixture), not a model-quality, device or vLLM measurement. The wall numbers are
repeatability context, not a throughput claim.

## Residual gaps

- Streaming sample logprobs combined with offered tools: the parser is deliberately disabled in
  that mode, so an answer that asks for both gets raw DSML content (documented here and in the
  audit, not silently mixed). Fixing it needs a per-token logprob-to-fragment mapping.
- Multiple DSML blocks in one answer: the buffered parser has a documented preference for the
  blank-line-plus-marker occurrence; the streaming parser is built around one block per answer
  and refuses text after the block. Pathological multi-block answers are not a supported shape.
- Native engine tool-call delta events (EngineEvent::ToolCallDelta) are untouched: engines that
  emit them (the mock in tests, a future device engine) keep their existing pass-through. The
  DSML parser only applies to content text.
- tool_choice and parallel_tool_calls remain absent from the OpenAI surface (unchanged by this
  slice).
- The parser sits above the engine seam, so it also applies on the TP=4/device serving paths when
  those engines emit content text; no device or TP=4 measurement is made here.

## Reproduce

    cargo test -p sparkflash-api --lib tool_parser
    cargo test -p sparkflash-api --test router
    cargo test -p sparkflash-cli --test serve_http
    python3 -m unittest scripts.tests.test_tool_call_streaming_census
    python3 scripts/tool-call-streaming-census.py --runs 3

## Non-claims

- Host path only: no CUDA, no device kernel, no GPU, no cluster run, no vLLM comparison.
- The wire census uses a debug-build scripted stub engine, not the released checkpoint; it
  measures parser/wire framing, not model quality or reference-decoder parity.
- The fixture is a fixed synthetic DSML answer, not a captured model output; real model DSML
  emission is not measured here.
- Wall numbers are repeatability of the same host run, not a throughput or speedup claim.
- No change to n>1, response_format, auth or the sampler surface is claimed.

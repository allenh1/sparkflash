# SparkFlash-RS — Lead decisions on spec-vs-reference deltas

The released reference implementation (`reference/encoding/`, and the checkpoint's
`inference/*.py`) and the specification disagree in a number of places. When they do, the
rule is: **the released artefacts (checkpoint tensors, reference code, published golden
vectors) win over prose**, because they are executable and testable. Every decision below
records the evidence and the consequence for the build. Research backing is in
[docs/architecture](./architecture/).

| # | Question | Decision |
| --- | --- | --- |
| D1 | KV storage layout | Store the compressed form, dequantise on read |
| D2 | Expert sharding | Whole experts per rank, full width, replicated shared expert |
| D3 | `reasoning_effort` scale | Two tables, one per surface, both tested |
| D4 | Engram row layout | Dense 264-byte rows, no padded stride |
| D5 | `max_seq_len` | Explicit `SPARK_MAX_SEQ_LEN`, default 1,048,576 |
| D6 | DSpark speculative loop | Implement the paper's loop; the reference code is not wired |
| D7 | CED prefill bypass | Implement it; it needs no extra weights |
| D8 | Hash layers (`tid2eid`) | Out of scope: not derivable from the checkpoint |

---

## D1 — Store the compressed KV form and dequantise on read

**Evidence.** The tech report's 890 B/token global-KV footprint is a *design layout*. The
released `inference/model.py` allocates each cache in bf16 and calls the quantising kernels
with `inplace=True` (model.py:707, 546, 760), which quantises and immediately dequantises
back into the same bf16 buffer; the E4M3 / E8M0 scale tensors are allocated and dropped
(kernel.py:81-86, 121-124, 167-172, 201-204). A resident reference cache is therefore
3,200 B/token, and `docs/architecture/08-kv-footprint.md` also shows the spec's prose
itemisation sums to 1050 B/token rather than 890, and that its "13.4 GiB holds 3.67M-6.50M
tokens" is off by ~2.5x (13.4 GiB at 890 B/token = 16,166,449 tokens).

**Decision.** `sparkflash-kv` stores exactly the tensor the reference builds transiently:
main KV as FP4 E2M1 with per-16 E4M3 scales, indexer K as MXFP4 with per-32 E8M0 scales. The
attention kernels dequantise a tile at a time, so the *numerics* are bit-identical to the
reference while the *footprint* is the design number. We keep the scale tensors; the
reference throws them away and that is precisely why it cannot realise its own layout.

**Consequence.** The bf16 dequantised tile is the CPU reference semantic for every kernel
test. We must not claim the published 890 B/token as a measured figure: milestone 4 lands a
measured per-token number derived from the real tensor shapes and reports any residual
disagreement.

## D2 — Shard whole experts across ranks

**Evidence.** `inference/convert.py:129-134` splits `experts.*` by expert index —
`n_local_experts = n_experts // mp`, dropping every expert outside the rank's contiguous
range — and never slices the intermediate dimension. `model.py:867-870` confirms the runtime
expectation. The same file leaves `hc_*`, `attn_sink`, `shared_experts`, `wkv`, `gate.weight`,
norms, `compressor` and `engram.wkv` **replicated** (the `"experts" in name and
"shared_experts" not in name` guard is what excludes the shared expert).

**Decision.** With TP=4 every rank holds 96 whole routed experts per layer at the full 2304
intermediate width, in MXFP4, and all-reduces their outputs. The shared expert, attention
weights, norms, gates and `engram.wkv` stay replicated. The spec's "shared expert K=576 ->
640 padding" describes a TP-sliced layout the checkpoint does not have and is dropped.

**Consequence.** The MoE kernel consumes 96 experts/rank with no padding; the expert-parallel
all-reduce is a full [tokens, 5120] bf16 reduction, not a partial sum over K.

## D3 — Two `reasoning_effort` tables, one per surface

**Evidence.** The two shipped encoders disagree. `deepseek-recipe`'s V4.1 renderer
(`dsv41.rs`, `ReasoningEffort`) maps Low=50, High/Xhigh/None=75, Max=100 — exactly the
spec's table. The official `encoding.py:398-404` maps low=25, high=50, xhigh=75, max=100
with default 50, and the published golden vectors pin that table: `test_input_1.json` asks
for thinking with no effort and `test_output_1.txt` begins
`Reasoning Effort: 50 (range 1-100, ...)`, while `test_input_5.json`'s `"max"` yields 100.

**Decision.** Both are kept, deliberately and with tests on each:
* the **API surface** (an OpenAI request reaching the engine) uses the spec/recipe table:
  `low=50, high=75, xhigh=75, max=100`, absent = 75 when thinking is on, and an integer
  `1..100` is written through verbatim — this is what makes the spec's "1..100" requirement
  representable, since the prompt line is literally `Reasoning Effort: N (range 1-100, ...)`;
* the **reference-vector surface** (`render_case`) uses `encoding.py`'s table so the five
  published vectors stay byte-identical.

**Consequence.** `SparkV41Encoding` exposes both and the difference is documented at the
call site. Thinking resolution is orthogonal: `chat_template_kwargs.thinking` beats
`enable_thinking` beats the default, and thinking off emits no effort line at all.

## D4 — Dense 264-byte Engram rows

**Evidence.** The checkpoint stores the tables dense: `layers.1.engram.embed.weight` is
F8_E4M3 `[384006168, 256]` (98,305,579,008 bytes) beside `layers.1.engram.embed.scale`
F8_E8M0 `[384006168, 8]` (3,072,049,344 bytes); layer 14 is `[384016682, 256]` /
`[384016682, 8]`. Both live entirely in shards 47 and 48. That is 264 B/row, 188.83 GiB for
both layers — 47.2 GiB per rank at TP=4. The scale block is therefore 32 values, matching
`fp8_block_size = 32` (model.py:27) and `scale_dtype = torch.float8_e8m0fnu` (model.py:30).

**Decision.** Pack rows densely at 264 B; do not adopt the spec's 512 B aligned stride, which
would cost 94% overhead (366 GiB total). Reads are issued 4 KiB-aligned through `io_uring`
with a small bounce buffer and a slab cache for partially used blocks.

**Consequence.** The packer's output size must equal 188.83 GiB exactly, and the per-rank
shard is a contiguous 1/4 slice of rows (the reference slices rows, not heads:
`convert.py` `".engram.embed." in name` -> `shard_size = (param.size(0) + mp - 1) // mp`).
The spec's "6 heads per rank" is wrong and is dropped.

## D5 — `max_seq_len` is an explicit configuration value

**Evidence.** The reference sizes its cache as `4 x 4096` by default and `1 x 64K`
interactive (model.py:51-52, generate.py:124-126); the runtime config has no `max_seq_len`
key at all. The spec budgets a 1M-token pool. Any measurement quoted from the reference must
name the sizing in force.

**Decision.** `SPARK_MAX_SEQ_LEN` (default 1,048,576) is the single knob; the KV pool, the
RoPE table (512 MiB per rank for two `freqs_cis` tables at 1M) and the candidate-indexer
scratch are all sized from it, and `/v1/cluster/stats` reports the value actually in use.

## D6 — Implement the DSpark loop the reference only demonstrates

**Evidence.** `Transformer.forward_spec` exists but is called only from model.py's
`__main__` demo; `generate.py` never references it (model.py:129-130 says so explicitly).
The confidence head emits a raw fp32 logit with no sigmoid (model.py:1097).

**Decision.** The three DSpark blocks (`layers.37-39`, whole experts sharded 32/rank,
top-3 global) are implemented end to end for `k=5` with the confidence cap at 0.10 and
`dspark_noise_token_id = 128799`. The confidence head's logit is passed through a sigmoid
before forming the cumulative survival product, because the paper's `p_1..p_5` are
probabilities; this is a documented numeric choice, not a port of reference code.

## D7 — The CED prefill bypass needs no extra weights

**Evidence.** `docs/architecture/01-csa2-attention.md` derives that layer 20 computes its own
ratio-1 compressor KV from its input (which is the encoder's final hidden state) and that
layers 21-39 reuse that KV rather than projecting their own. The tech report's
`C_l = H_{L/2} W^{KV}_l` is therefore layer 20's compressor applied to `H_19`, and no tensor
for it is missing from the checkpoint.

**Decision.** Prefill runs layers 0..20 over the prompt and layers 21..39 only over the last
`SWA_WINDOW_SIZE` tokens — the O(N·L/2) saving is a scheduling property, and
`sparkflash-kv`'s `plan_request` already models the token counts. Nothing about the weights
or the numerics changes.

## D8 — Hash layers are out of scope

`num_hash_layers` / `gate.tid2eid` cannot be derived from anything in this checkout: the
checkpoint has zero `tid2eid`/`tie2eid` tensors and `convert.py:116-121` keeps rename hooks
for names that never appear. `docs/architecture/03-moe.md` records this as unresolvable. No
hash-layer code is written and the config knob is rejected with a clear error rather than
silently ignored.
## D9 — Trace request targeting is a thread-local binding

**Context.** `POST /v1/admin/trace` takes an optional `request_id` and the specification asks
for a scoped trace of "a specific step count or request ID". A span is written by whichever
thread executes the engine step, so targeting one request needs the recording side to know
which request a thread is currently serving.

**Decision.** `sparkflash-trace` keeps an optional process-wide request filter plus a
thread-local binding (`bind_current_request`), and `is_scope_enabled` records only when the
two match. The HTTP layer binds the caller's `X-Request-Id` (falling back to the generated
response id) with an RAII guard that restores the previous value on drop, on both the buffered
and the streaming paths, and the admin route installs or clears the filter.

**Accepted limitation.** A tokio task can migrate between worker threads at an `await`, so a
binding made on one thread can be stranded on another; the effect is *lost* spans for that
request, never spans attributed to the wrong request, because an unbound thread simply fails
`is_request_traced()`. Milestone 5 must move the id onto the engine request and re-bind it at
each step boundary inside `sparkflash-scheduler`, where the step is single-threaded, instead
of relying on the HTTP task's thread. That is a correctness requirement for the DSpark and
prefill-bypass work, not an optimisation.

**Why not a task-local.** The trace points are reached through a macro that expands in the
kernel/scheduler crates and must stay allocation-free and lock-free; a tokio task-local vector
lookup would put the runtime in the hot path of a facility that has to be free when disabled.

---

## D10 - The CPU span ring is 48 bytes behind a per-thread mutex, not the 32-byte lock-free ring the spec describes

**Context.** Specification 5.8 (line 1112) asks for "32-byte `CpuTraceEvent` records into a
per-thread lock-free ring buffer". The implementation stores 48-byte `CpuTraceEvent` values
(label, scope, meta, thread id, start, end) in a per-thread `Mutex<Vec<CpuTraceEvent>>`, kept in a
process-wide registry so the exporter can drain every thread at once.

**Decision.** Keep the *GPU* wire ABI byte-exact - `GpuTraceEvent` stays 32 bytes and 32-byte
aligned, and `kernels/trace_globaltimer_sm121.cuh` is checked against it field by field by the
build-script probe and by unit tests - and accept the CPU side deviating in record size and locking
discipline. The GPU ABI is a contract the kernels compile against; the CPU ring is internal to one
process and never crosses the wire.

**Why not the spec's shape.** A 32-byte lock-free ring would drop the label and meta fields that
make a CPU trace readable in Chrome/Perfetto, which is the whole point of the capture. It is also
not on any hot path: `is_scope_enabled` short-circuits on an L1-cached mask before anything is
touched, so a disabled engine locks and allocates nothing. An armed span does take its own
thread's mutex, but cross-thread contention only appears when the exporter drains a capture.

**Revisit when.** If armed-capture profiling ever shows CPU span recording, replace the Vec with a
fixed-size lock-free ring and move the label to a static-table index. The change is contained
behind `ThreadTraceState`.

**Also recorded here** (found by review, not a decision): the ring constructors bound capacity at
`MAX_RING_CAPACITY` and use checked arithmetic, because the header stores capacity - 1 in 32 bits
and an unchecked capacity made the *safe* `slot()` accessor index out of bounds in release builds.

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
| D9 | Trace request targeting | Thread-local binding plus a process-wide filter; an unbound thread loses spans, it never mis-attributes them, and milestone 5 must re-bind the id on the engine request at each step |
| D10 | CPU span ring shape | 48-byte records behind a per-thread mutex, while the GPU ABI stays byte-exact at 32 bytes; the CPU shape is internal and can change behind `ThreadTraceState` |
| D11 | RoCEnante peer wait | Bounded 1 ms `%globaltimer` deadline with `__nanosleep` backoff; a timeout yields a quiet-NaN result plus one timeout trace record, so a straggler is a visible wrong answer and not a hung node |
| D12 | Rail device selection | Enumerate the fabric and require an ACTIVE 200 Gb/s function; a down or 40 Gb/s link is a hard startup error, with the spec's device names kept as a fallback and an operator override |
| D13 | Dual-rail byte accounting | Keep the spec's truncating counter formula and pin the undercount with a test; the exact byte figure is the plan's, not the counter's |
| D14 | MoE expert GEMM activation format | W4A8: FP8 E4M3 activations with per-32 ue8m0 scales against FP4 weights on `[out, in/32]`; the spec's "W4A4" is a defect and the kernel's E2M1 activation operand is replaced |
| D15 | Two dead trace features | Deleted: neither name appeared in any cfg in the workspace, so enabling either changed nothing; the CUDA dependency selector stays because it selects cudarc |
| D16 | KV specification defects | Six 5.5 defects recorded as facts: the 890 B/token headline contradicts its own itemisation (356 B/entry once the rotary dimensions are not double-counted), plus five more, each with the code's settlement named |
| D17 | Qualification state | Nine gates recorded with instrument, command, artifact and state; five cannot be run by any command in this tree, and 20 us at 10 KiB is the requirement while the preflight's 22 us is a deliberately looser readiness threshold |
| D18 | The engine is a ninth crate, and the plan describes the checkpoint as stored | `crates/sparkflash-engine`; the plan is derived from the configuration and then checked against the index, and the load-time conversions (wo_a, the int8 view of FP4) belong to the loader, not the plan |
| D21 | The determinism contract | D-a kernel determinism, D-b reduction determinism and D-c no assumed cross-boundary equality are written down here and cited by the M7 harness; a test fails if either half disappears |
| D22 | Released mHC arithmetic | The ref, the kernel and the harness model the released arithmetic end to end (projection rstd, `mix*scale+base`, sigmoid pre/post, released split-Sinkhorn, raw C, `X_hat` block norm, released `[24, n*d]` predictor); B0 removed the stream-mean fold from both sides, so the old structural difference is gone |
| D23 | The activation quantiser is the reference ceil rule | Device activation quantisers take the smallest power of two at or above `amax/448` after the 1e-4 amax floor (the engine's `quantize_activation_fp8` rule); the checkpoint weight-grid rule stays floor-log2 and the two helpers are named apart (D8) |

The table is an index of the entries below, not a second source of record: each row is a one-line
summary and the matching `## D<n>` section carries the evidence and the full consequence. If a row and
its entry ever disagree, the entry wins.

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

**Task-3 extension (2026-10-18): the window-KV format becomes an explicit selector.** The window-KV
format is a runtime parameter, never a static_assert value: `flash_mla_csa2_sm121` takes
`window_kv_format` (0 = the FP4 latent format above and the merged Stage-2 default, 1 = the
engine's released window format), the launcher rejects any other value with a loud documented error
and the kernel entry poisons the output and returns, and that entry guard is pinned by a source-text
test. The engine format stores E4M3 codes with one E8M0 power-of-two scale per 32 channels (16 blocks
per 512-wide row); both the kernel and `mla_ref` decode `code * scale` and round the decoded value
to bf16, so the engine's in-place bf16 storage round is consumed on both sides.
`sparkflash-engine`'s reference forward keeps f32 dequant, and that seam is named rather than
hidden: the column measures format fidelity, not the engine's bf16 arithmetic. Measured
engine-format column: 163,834/163,840 exact (99.9963%), max abs 9.765625e-4, max ULP 1,
out-of-window ids exactly zero on both sides, same bytes from the engine quantiser and
`m4_quant::act_quant_fp8_row` on 130/130 rows. The FP4 default path was re-measured on the merged
chain at 163,838/163,840 exact, max abs 2.4414063e-4, max ULP 1, identical per-position lines to the
merged-head old-kernel binary; the 9.765625e-4 in the 2026-10-17 record is the pre-D8 stage-1 chain
and is kept only as a historical label. Criterion 1's exactness half remains owed: this is internal
consistency in the engine's format, not model fidelity, and release-faithful arithmetic remains
Phase 2. The 19-parameter kernel ABI is carried by one typed parameter builder in the harness
(`FlashMlaParams` / `flash_mla_params`; every launch goes through `flash_mla_launch` or
`flash_mla_launch_repeated`), pinned by the `m4_abi` source-text test with its 15-parameter
mutation. Round-2 review found the stage-7 and stage-9 K5 sites still on the retired 15-parameter
array; after the fix the ABI-fixed binary `14e0266f...` (build of head `c520c93`, command
`./scripts/build-aarch64.sh --example=m7_layer0` run in `/home/user/wt/task3-window-kv`; the
aarch64 hash is path-dependent) re-ran both sites on the node (stage-7 measured rows and
all 18 `single_layer_delta` lines bit-identical to the merged B2 log, stage-9 smoke PASS framing;
logs `phase1-task3-abi-stage7-b2compare`, `phase1-task3-abi-stage9-smoke`).

**Task-6 extension (2026-10-18): the engine window format reaches the 40-layer Stage-C column.**
`m7_layer0 --stage9-kv-format engine` builds each layer's window row with the released format
(`kv_norm(wkv(layer input)) -> rope tail -> bf16 storage round -> E4M3 + per-32 E8M0`), passes it
to K5 as window bytes with `window_kv_format=1`/`window_rows=1` through the typed 19-parameter
ABI, and checks K5 against `mla_ref::attention_with_sink_window` (engine decode + bf16 round).
The row construction is the same shared `engine_window_row` helper stage 2 uses (single bf16
round, engine quantiser, asserted byte-identical to `m4_quant::act_quant_fp8_row`), pinned by the
new `m4_abi::stage2_and_stage9_share_one_engine_window_row_builder` source-text test, so stage 2
and stage 9 cannot drift. Measured: the FP4 default column is bit-identical to the merged Stage-C
pair, row lines and diff body (the flag is additive); the engine column is deterministic over two
runs (pair blob `40cadade...`), 80/80 rows byte-identical to the engine quantiser, finite-decode
guard 80/80, K5 internal checks 80/80 100% exact (max abs 0), totals 299/720 at or under the frozen
bars vs 306/720 for the FP4 column (seven net misses, six tensors change), and 22/40 layers report
a discrete router flip vs 17/40 in the FP4 column (the changed stream moves routing; reported, not
smoothed). Criterion 1 remains open: the engine reference forward keeps f32 dequant, and the
main/compressed KV is still FP4 and degenerate at position 0. Logs
`phase1-task6-stage9-fp4-default`/`phase1-task6-stage9-engine`, binary `cf9b4e90...`; details
in `docs/build/phase1-boundary-table.md` section 14.

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

**Phase-2 seam-cleanup note (task-3).** The FP8 activation-quantiser rule exists in three places:
`crates/sparkflash-engine/src/quant.rs` (the host reference and behavioural owner of the rule),
`crates/sparkflash-kernels/src/m4_quant.rs` (the device host oracle) and
`kernels/m4_activation_quant_sm121.cuh` (the CUDA transcription). Current pins are
`activation_quantiser_equivalence` (engine against the oracle, including inf/NaN/subnormal scale
codes) and `assert_matches_kernel_header` (the constants). Phase-2 cleanup, recorded here without a
rule change: name the three owners explicitly in the docs; add one shared deterministic vector
fixture and extend the equivalence test to full codes and scales (not only scale codes); extend the
header check so a second CUDA copy cannot drift; and add a node stage-8 act-quant probe against the
same fixture, because CI has no GPU execution (the CI-only fallback is a pinned source-text
comparison).

**Task-6 update (2026-10-18):** the harness's stage-9 window row now goes through the same single
`engine_window_row` helper as stage 2 - one bf16 storage round, then
`sparkflash_engine::quant::quantize_activation_fp8`, asserted against
`m4_quant::act_quant_fp8_row` - so the two harness-side users of the rule cannot drift; the
CUDA copy remains the third owner. No rule change.

**Task-8 harness-move note (2026-10-18):** the shared CUDA harness pieces now live in
`crates/sparkflash-kernels/src/cuda_harness.rs`: the 19-word `FlashMlaParams`/`flash_mla_params`
builder and the `engine_window_row` family, so the `m7_layer0` chain harness and the forthcoming
`m8_sequence` harness import one builder and one row construction instead of each carrying a copy.
The window-row helper calls the engine's released quantiser, so `sparkflash-engine` moves from a
dev-dependency to a normal dependency of `sparkflash-kernels`; the edge is deliberate and acyclic
(the engine crate depends on no other workspace crate). No rule change and no stage claim change:
the m7 evidence binary `cf9b4e90` is cited for the merged Stage-C column only, not for this
refactor, which is pinned by tests instead.


**Task-8 extension (2026-10-18): the instruction is compile-time, not a text scanner.** Round-2 and
round-3 reviews showed that any "the example must call the wrapper" source-text classifier can be
evaded (string literals, a space before the parameter list, a differently named module, a .inc
re-export, an aliased handle), so the enforcement moved into the type system. The raw CUDA driver
region of `examples/m7_layer0.rs` moved verbatim into the library as
`crates/sparkflash-kernels/src/cuda_driver.rs`: same dlopen symbol names, same seven kernels,
same grid/block order, same `launch_repeated` semantics and same error strings, with `unsafe`
confined to that module. The public surface is deliberately small and documented - opaque `Kernel`
and `Module` handles, `KernelId` with exactly the six non-flash_mla kernels, `Cuda` load/alloc/copy/
launch/free entry points - and `Cuda::launch`/`launch_repeated` only accept an opaque `Kernel`. The
flash_mla kernel is reachable only through `cuda_harness::FlashMlaKernel` and
`load_flash_mla`/`launch_flash_mla`/`launch_flash_mla_repeated`; there is no public conversion
between `FlashMlaKernel` and `Kernel` and no symbol-name loader on the public API, so neither a raw
handle read nor an aliased handle compiles. `examples/m7_layer0.rs` now carries
`#![forbid(unsafe_code)]`, imports the shared driver (with `Kernel as CuFunction`/`Module as
CuModule` aliases only) and gets the typed launches under their old names, so its call sites are
unchanged except for imports and the three flash_mla loads (which now read the cubin through the
typed loader); the driver move is move-only and behavior-preserving by inspection. The fragile
source classifier and file walker are deleted from `m4_abi`; one exact-path pin remains as defense
in depth (examples importing `sparkflash_kernels::cuda_driver` must have
`#![forbid(unsafe_code)]`, no unsafe token outside masked comments/strings, no `mla_func`/
`cuLaunchKernel`/`extern` token), alongside the hardened definition pin (every file under `src`,
whitespace-insensitive `fn name (`, exact canonical `src/cuda_harness.rs` path, no `as
engine_window_row`/`as flash_mla_params` re-export) with mutation tests for the round-3 evasions.
`compile_fail` doctests prove the bypasses do not compile and a counterpart `no_run` doctest proves
the typed call does; the device stages were not re-run. dlopen-only means a CPU-only build does not
link libcuda. No rule change and no stage claim change. The launcher crate no longer contains any
driver code, so a future harness (m8) must import cuda_driver and use the typed launch entry points
rather than revisit FFI.

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

## D11 - The RoCEnante wait is bounded and traceable instead of spinning forever

**Context.** Spec 5.4 (lines 683-687) spins on `slot->arrival_seq[peer] < step_seq` with no exit
condition. A dropped doorbell, a late peer or a NIC that never posts therefore hangs the kernel
indefinitely, and on a unified-memory GB10 a hung kernel is not killable from the host without a
device reset - the failure mode would take down a node that is otherwise healthy.

**Decision.** The wait is bounded by a `%globaltimer` deadline with `__nanosleep` backoff
(`SPARK_ROCENANTE_PEER_WAIT_NS`, default 1 ms, roughly 45x the 22 us gate in spec 6.1). On timeout the
block returns a bf16 quiet NaN for the elements it owns and emits a single diagnostic trace record whose
`payload_meta` carries the timeout flag, a 4-bit missing-rank mask and a row-overflow bit. A straggler
thus produces a visible, attributable wrong answer instead of a dead node.

**Also recorded in the kernel, because the spec is silent.** A `__syncthreads()` sits between the
doorbell and the peer poll, which orders the ringing within a block and keeps the per-block deadline
stamp after the doorbell. It is a block-scoped barrier, so it does not stop another block from reaching
the poll first: what makes that safe is the sequence guard, since the poll waits for
`arrival_seq[peer] >= step_seq` rather than for the ring. (An earlier draft of this entry claimed the
barrier gave grid-wide ordering; the M3 review falsified that, and the poll's guard is the real reason
no block can proceed on a peer that has not rung.) The observing threads also fence unconditionally
after the poll, not only on the wait path, so a peer that was already visible when the block arrived is
still acquired rather than riding on the writer's release alone. `step_seq` is strictly
increasing per rank, and a slot is not reused for `step_seq + 1` until every rank has finished reading
it; double buffering is the communicator's job, not the kernel's. The spec's slot struct carries no
alignment attribute, so the slot is 8-byte aligned (by its `u64` member), not 32: `sizeof ==
8_388_640`, pinned by `static_assert` on the C side and by a parsed-declaration test on the Rust side.
An oversized `num_elements` is clamped and flagged rather than allowed to run into the neighbouring
payload row; the router's 2 MiB threshold is exactly one payload row, so a conforming caller never
reaches the clamp.

## D12 - Dual-rail device selection enumerates the fabric instead of trusting the spec's device names

**Context.** Spec 6.1 (check 2) and the Milestone 3 registration list name the two rails as
`rocep1s0f0` / `enp1s0f0np0` and `roceP2p1s0f0` / `enP2p1s0f0np0`. Measured read-only on the live
cluster, those are the **f0** functions and they are down at 40 Gb/s, while the rails actually carrying
traffic are the **f1** functions: `rocep1s0f1` -> `enp1s0f1np1` and `roceP2p1s0f1` ->
`enP2p1s0f1np1`, both ACTIVE at 200 Gb/s (4X HDR). Both rails share 192.0.2.0/24, and a node holds two
addresses, one per rail (the head node is 192.0.2.80 on rail 1 and 192.0.2.85 on rail 2), with two
default routes at different metrics.

**Decision.** The communicator enumerates the RDMA devices and their netdevs and selects the ones that are
ACTIVE at 200 Gb/s, with the spec's names kept as a documented fallback and an explicit operator override.
Registering a device that is down, or falling back silently to a 40 Gb/s link, is a hard startup error,
not a warning: a silent downgrade would invalidate the whole milestone's bandwidth claims while still
producing plausible numbers.

**Also measured, and relevant to the preflight.** The live nodes run kernel `6.17.0-1029-nvidia` (meets
the `>= 6.17.0-1018` requirement) but driver `580.173.02`, which is **below** the `580.178.04` the spec
requires, so check 1 fails on the real cluster today; the active rails report MTU 1500 rather than the
required 9000; `nvidia_uvm` exposes no 64K-packing parameter under any conventional name among its 56
module parameters, so "64K page packing is active" is not verifiable as written; and `MemAvailable` is
5 GiB because the live vLLM instance holds the memory, which is why the 114 GiB gate must only be read
while the cluster is idle.

## D13 - Dispatch telemetry keeps the spec's truncating formula, and the exact figure lives in the plan

**Context.** Spec 5.4 (lines 758-760) accounts dual-rail traffic as `half_bytes = num_bytes / 2;
rdma_bytes_railN += half_bytes * 3`. With an ODD `num_bytes` the integer division truncates, so each
rail's counter undercounts by up to 3 bytes relative to the true `3 * num_bytes / 2`.

**Decision.** Keep the spec's formula for the spec-named counters, because those counters are the
conformance surface an operator compares against the paper, and pin the behaviour with a test that
reproduces the undercount rather than hiding it. The exact byte figure is already conserved and exposed
by `RailPlan`, which is what the transport actually moves, so nothing about the wire behaviour depends on
the counter. The truncation is also unreachable for this model's payloads: every collective here carries
bf16 elements, so `num_bytes` is always even.

**Also recorded.** The spec defines neither an operator override for the rail devices nor an error for an
unusable fabric; `SPARKFLASH_RAIL_DEVICES` and `CommError::FabricUnavailable` are additions (see D12).

**Recorded limitation (M3 review, non-blocking).** `RailSelection` and `SelectedRail` carry public
fields and derive `Deserialize`, so a caller that constructs one by hand - or deserialises one - can
still present a colliding pair: the invariant lives in `RailSelection::resolve`, not in the type.
Nothing in the tree consumes an untrusted selection today, so this is an API-shape hazard rather than
a defect, and it is recorded here instead of being silently assumed. If a future caller ever accepts
a selection from configuration or the network, either the fields must become private with
`resolve`-only construction or the type must gain a `validate()` that its consumers call.

---

## D14 - The MoE expert GEMM is W4A8: FP8 E4M3 activations against FP4 E2M1 weights

Recorded 2026-10-15. Numbering: the main checkout ended at D13. This entry is merged first and so takes
D14; milestone 2 keeps D15 for the deleted trace features, and the milestone 4-KV entry renumbers from
D14 to D16 when that branch rebases.

**Question.** `kernels/b12x_mxfp4_moe_sm121.cu` takes packed E2M1 activations (`act_fp4`, line 69)
while `docs/architecture/03-moe.md` section 4 described the reference's FP8 activation. The kernel
author documented the divergence at the call site (kernel file lines 20-27: "the activation ELEMENTS
here are E2M1, whereas the reference implementation's fp4 GEMM quantises its activation to E4M3
(03-moe.md:131-133)") and justified it with "Specification 6.2 mandates the MXFP4 expert kernels".
The spec's only statement on the activation format is the `MOE_MXFP4` telemetry row, "Route planner,
576-wide grouped W4A4 GEMM, deterministic sum" (spec line 1130).

**Decision.** The MoE expert GEMM is **W4A8**: FP4 E2M1 weights with one ue8m0 scale per output row
per 32 values along K (grid `[out, in/32]`) against **dynamic FP8 E4M3 activations** with per-32
ue8m0 scales (grid `[tokens, in/32]`). The spec's "W4A4" is a defect, not a mandate. The doc and the
kernel must cite the same evidence: `docs/architecture/03-moe.md` now states the format and both
scale grids explicitly (section 4, lines 116-168) and records the spec defect as delta D7
(03-moe.md:228-239). The kernel header still says E2M1 and needs the same citation; that file belongs
to `sparkflash-kernels` (kernel-dev), so it was reported rather than edited here.

**Evidence, three independent sources.**

1. *The checkpoint declares the activation scheme.* `config.json:12-21` is
   `quant_method: "fp8"`, `activation_scheme: "dynamic"`, `weight_block_size: [32, 32]`,
   `scale_fmt: "ue8m0"`, `expert_dtype: "fp4"`. No activation tensor is stored in the checkpoint at
   all, so the config is the only activation-format statement it makes, and it names dynamic FP8 for
   activations; `expert_dtype: "fp4"` names the routed **weight** format only.
2. *The reference forward is typed W4A8.* `inference/model.py:181-195` dispatches on the weight dtype
   and calls the same FP8 activation quantiser in both branches `act_quant(x, 32, "ue8m0",
   e8m0fnu)` (lines 187 and 197); the fp4 GEMM is documented "FP8 act x FP4 weight GEMM" and typed
   `A: T.Tensor[(M, K), FP8]`, `B: T.Tensor[(N, K), FP4]` (`inference/kernel.py:486-499, 511-515`),
   asserting `a_s.numel() == M * (K // 32)` and `b_s.shape == (N, K // 32)` (kernel.py:581-582). The
   E2M1 activation quantiser `fp4_act_quant` is never a GEMM input: its call sites are the indexer K
   (model.py:546), the indexer Q (model.py:552) and the compressed-KV cache (model.py:760).
3. *The spec contradicts itself first.* Section 2.4 step 4 has the fused mHC kernel write the block
   input as dynamic FP8 `32x32 ue8m0` (spec line 86); that block input is what the MoE consumes.
   Re-quantising it to E2M1 inside the MoE is an extra step no source defines, and the `MOE_MXFP4`
   row's "W4A4" is the only place the spec says otherwise. Milestone 4 names the kernel with no
   activation claim (spec line 1338), and "MXFP4" in the filename and in spec line 44 names the
   weight format.

**Scale grids.** Two grids, now stated verbatim in `03-moe.md:151-161`. The FP8 (dense and shared
expert) weight scale is a 2-D block grid `[ceil(out/32), ceil(in/32)]` (kernel.py:293-298,
model.py:225-233); the FP4 expert weight scale is `[out, in/32]`, one ue8m0 scale per output row per
32 along K with no blocking along `out` (kernel.py:581-582, model.py:219-224). Activations use
`[tokens, in/32]` on both paths (kernel.py:294, 581). The kernel and `mxfp4_ref.rs` already implement
the `[out, in/32]` weight grid, so aligning the divergence changes the A operand's element type, not
the scale layout.

**Consequence.** A loader must hand this kernel FP8 E4M3 activations with per-32 ue8m0 scales. The
reference MMA is fp8 x fp8 - the kernel casts each FP4 weight sub-block to FP8 through float before the
dot product (kernel.py:496-498) - so an E2M1 A operand would be a different arithmetic, not merely a
different operand type, and it could not be validated bit-for-bit against the reference. **Decided
(Lead, 2026-10-15):** `kernels/b12x_mxfp4_moe_sm121.cu` gains FP8 E4M3 activations with per-32 ue8m0
scales and keeps its FP4 weights and their `[out, in/32]` grid; the E2M1 A operand is **not** kept on
throughput grounds, because the released artefact is an inference path typed W4A8 whose only E2M1
activation call sites are the indexer K, the indexer Q and the KV cache. That edit is routed to
`kernel-dev`; this entry and `03-moe.md` section 4 are the citation both sides now share.

**Padded loader contract (A8).** The kernel reads the PADDED width, so the loader contract is part
of this decision rather than an implementation detail. Per routed `(token, slot)` row the loader
must emit `act_fp8` as `[tokens * top_k][640]` E4M3 bytes and `act_scales` as
`[tokens * top_k][20]` ue8m0 bytes; per output row `(expert, hidden)` it must emit `w_fp4` as
`[experts][d][320]` packed E2M1 bytes and `w_scales` as `[experts][d][20]` ue8m0 bytes. Index
>= `SPARK_M4_MOE_WIDTH` (576) is masked on the activation side, so the 64 padded values cannot
change a result, but the kernel still addresses them and its row strides are those constants. The
reference's whole-expert grid would give `2304 / 32 = 72`-wide weight- and activation-scale rows,
and a 576-wide per-rank slice would give 18-wide rows; either misindexes every row after the first.
The Rust side states and tests the rejection
(`sparkflash-kernels::mxfp4_ref::{MoeRowLayout, pre_padded_moe_layout, check_moe_layout}`), the
kernel carries the same contract next to its constants, and producing the layout belongs to the
checkpoint loader, which is **not** in milestone 4: it lands with Milestone 5 (spec lines 1329-1340
name the orchestrated integration; no loader has its own milestone).

**Verification.** Doc-only change plus this entry; no GPU and no cluster were involved, and every
claim above is a line-cited reading of `config.json`, `model.py`, `kernel.py` and the spec. The
reference tree was read at `/home/user/dsv41-research`; the kernel file was read at
`wt/m4k/kernels/b12x_mxfp4_moe_sm121.cu` (the workspace-root `kernels/` directory of specification
5.1, not a `crates/sparkflash-kernels/kernels/` subdirectory). No git commands were run; the edits
are in the `docs/architecture` worktree.

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

## D11 - The RoCEnante wait is bounded and traceable instead of spinning forever

**Context.** Spec 5.4 (lines 683-687) spins on `slot->arrival_seq[peer] < step_seq` with no exit
condition. A dropped doorbell, a late peer or a NIC that never posts therefore hangs the kernel
indefinitely, and on a unified-memory GB10 a hung kernel is not killable from the host without a
device reset - the failure mode would take down a node that is otherwise healthy.

**Decision.** The wait is bounded by a `%globaltimer` deadline with `__nanosleep` backoff
(`SPARK_ROCENANTE_PEER_WAIT_NS`, default 1 ms, roughly 45x the 22 us gate in spec 6.1). On timeout the
block returns a bf16 quiet NaN for the elements it owns and emits a single diagnostic trace record whose
`payload_meta` carries the timeout flag, a 4-bit missing-rank mask and a row-overflow bit. A straggler
thus produces a visible, attributable wrong answer instead of a dead node.

**Also recorded in the kernel, because the spec is silent.** A `__syncthreads()` sits between the
doorbell and the peer poll, which orders the ringing within a block and keeps the per-block deadline
stamp after the doorbell. It is a block-scoped barrier, so it does not stop another block from reaching
the poll first: what makes that safe is the sequence guard, since the poll waits for
`arrival_seq[peer] >= step_seq` rather than for the ring. (An earlier draft of this entry claimed the
barrier gave grid-wide ordering; the M3 review falsified that, and the poll's guard is the real reason
no block can proceed on a peer that has not rung.) The observing threads also fence unconditionally
after the poll, not only on the wait path, so a peer that was already visible when the block arrived is
still acquired rather than riding on the writer's release alone. `step_seq` is strictly
increasing per rank, and a slot is not reused for `step_seq + 1` until every rank has finished reading
it; double buffering is the communicator's job, not the kernel's. The spec's slot struct carries no
alignment attribute, so the slot is 8-byte aligned (by its `u64` member), not 32: `sizeof ==
8_388_640`, pinned by `static_assert` on the C side and by a parsed-declaration test on the Rust side.
An oversized `num_elements` is clamped and flagged rather than allowed to run into the neighbouring
payload row; the router's 2 MiB threshold is exactly one payload row, so a conforming caller never
reaches the clamp.

## D12 - Dual-rail device selection enumerates the fabric instead of trusting the spec's device names

**Context.** Spec 6.1 (check 2) and the Milestone 3 registration list name the two rails as
`rocep1s0f0` / `enp1s0f0np0` and `roceP2p1s0f0` / `enP2p1s0f0np0`. Measured read-only on the live
cluster, those are the **f0** functions and they are down at 40 Gb/s, while the rails actually carrying
traffic are the **f1** functions: `rocep1s0f1` -> `enp1s0f1np1` and `roceP2p1s0f1` ->
`enP2p1s0f1np1`, both ACTIVE at 200 Gb/s (4X HDR). Both rails share 192.0.2.0/24, and a node holds two
addresses, one per rail (the head node is 192.0.2.80 on rail 1 and 192.0.2.85 on rail 2), with two
default routes at different metrics.

**Decision.** The communicator enumerates the RDMA devices and their netdevs and selects the ones that are
ACTIVE at 200 Gb/s, with the spec's names kept as a documented fallback and an explicit operator override.
Registering a device that is down, or falling back silently to a 40 Gb/s link, is a hard startup error,
not a warning: a silent downgrade would invalidate the whole milestone's bandwidth claims while still
producing plausible numbers.

**Also measured, and relevant to the preflight.** The live nodes run kernel `6.17.0-1029-nvidia` (meets
the `>= 6.17.0-1018` requirement) but driver `580.173.02`, which is **below** the `580.178.04` the spec
requires, so check 1 fails on the real cluster today; the active rails report MTU 1500 rather than the
required 9000; `nvidia_uvm` exposes no 64K-packing parameter under any conventional name among its 56
module parameters, so "64K page packing is active" is not verifiable as written; and `MemAvailable` is
5 GiB because the live vLLM instance holds the memory, which is why the 114 GiB gate must only be read
while the cluster is idle.

## D13 - Dispatch telemetry keeps the spec's truncating formula, and the exact figure lives in the plan

**Context.** Spec 5.4 (lines 758-760) accounts dual-rail traffic as `half_bytes = num_bytes / 2;
rdma_bytes_railN += half_bytes * 3`. With an ODD `num_bytes` the integer division truncates, so each
rail's counter undercounts by up to 3 bytes relative to the true `3 * num_bytes / 2`.

**Decision.** Keep the spec's formula for the spec-named counters, because those counters are the
conformance surface an operator compares against the paper, and pin the behaviour with a test that
reproduces the undercount rather than hiding it. The exact byte figure is already conserved and exposed
by `RailPlan`, which is what the transport actually moves, so nothing about the wire behaviour depends on
the counter. The truncation is also unreachable for this model's payloads: every collective here carries
bf16 elements, so `num_bytes` is always even.

**Also recorded.** The spec defines neither an operator override for the rail devices nor an error for an
unusable fabric; `SPARKFLASH_RAIL_DEVICES` and `CommError::FabricUnavailable` are additions (see D12).

**Recorded limitation (M3 review, non-blocking).** `RailSelection` and `SelectedRail` carry public
fields and derive `Deserialize`, so a caller that constructs one by hand - or deserialises one - can
still present a colliding pair: the invariant lives in `RailSelection::resolve`, not in the type.
Nothing in the tree consumes an untrusted selection today, so this is an API-shape hazard rather than
a defect, and it is recorded here instead of being silently assumed. If a future caller ever accepts
a selection from configuration or the network, either the fields must become private with
`resolve`-only construction or the type must gain a `validate()` that its consumers call.

## D15 - Two dead trace features are deleted; the CUDA dependency selector stays

**Context.** `sparkflash-trace` declared `trace-gpu-in-graph` ("enables the in-graph GPU
`%globaltimer` trace ring ABI") and `trace-nvtx` ("mirrors CPU spans into NVTX v3 ranges").
Neither name appears in any `cfg` or `cfg_attr` in the workspace, so enabling either feature
changed nothing: a feature that gates nothing reads as "there is GPU trace code behind this" when
there is none. The blindness only became visible once CI started discovering the workspace's
features from cargo metadata and testing each one (MR !9). `sparkflash-kernels` also declares
`cuda = ["dep:cudarc"]`, which likewise gates no Rust code, but it does select a real optional
dependency, and the `cuda-abi` job builds and tests with
`--features cuda,cudarc/cuda-version-from-build-system` in an image that has a toolkit.

**Decision.** Delete `trace-gpu-in-graph` and `trace-nvtx` from
`crates/sparkflash-trace/Cargo.toml` and add them back when the in-graph ring and the NVTX mirror
are implemented. The specification's compile-time-features section still describes them as the
intended design, so their absence is a gap to close, not a requirement to abandon. Keep
`sparkflash-kernels/cuda`: it is a dependency selector exercised by the CUDA ABI job rather than a
code gate, and it stays classified hardware-only in CI.

---

## D16 - Six spec defects in 5.5 (KV cache and SWA Bounded Replay), recorded 2026-10-15

Found while implementing `sparkflash-kv` (milestone 4, task-12). Facts only, with spec line
numbers from `deepseek_v41_flash_4x_dgx_spark_rust_spec.md`; where the code settles a question,
that is stated.

**1. The 890 B/token headline contradicts its own itemisation (spec lines 73, 76).** Line 76
itemises one global entry as "256 bytes + 32 bytes of `E4M3` scales per 16 channels" for the main
KV, "plus `qk_rope_head_dim = 64`", "and Indexer K (128 dim in MXFP4 = 64 bytes + 4 bytes
`ue8m0`)". That is 420 B/entry, and 2.5 entries/token (layers 2/8/14 at m=2, layer 20 at m=1)
gives **1050 B/token**, not the **890** the same sentence asserts. 890 closes only if the rotary
dimensions are not counted twice, i.e. if `head_dim = 512` already contains
`qk_rope_head_dim = 64`; the entry is then 356 B (288 + 68). The D1 entry of section 7 ("Spec
deltas") of `docs/architecture/08-kv-footprint.md` derives the same 1050 (heading at line 171,
derivation at lines 173-175), while section 4 starts at line 75. **Code:** `sparkflash-kv`
uses 356 B per entry, 890 B per token and 56,960 B per 64-token page (160 entries);
`tests/layout.rs` pins 1050 and 890 side by side so the contradiction cannot be tidied away.

**2. `plan_request` clamps the hit, which makes the spec's zero-decoder-replay branch unreachable
(spec lines 841, 846, 861-865).** Line 841 is
`raw_cached = (uma + nvme).min(total_prompt_tokens - 1)`, so
`global_kv_cached_tokens <= total - 1` and `uncached_suffix_tokens >= 1` for every non-empty
prompt (line 846). Line 861 returns 0 decoder replay tokens only when
`swa_tail_hit && uncached_suffix_tokens == 0`, a condition that clamp forbids. As specified, that
branch is dead code.

**3. Three different memory floors, and only two of them are in the spec (spec lines 180, 1324).**
Line 180 budgets a 9.12 GiB "MemAvailable Floor Guard" inside the 121.7 GiB per-node total
(line 181). Line 1324 requires `MemAvailable >= 114 GiB` per node before weight loading. The
third figure, 8 GiB, is not in the spec at all: it comes from the milestone 4 task brief, which
describes it as "mentioned in the spec". **Code:** the configurable floor is the brief's 8 GiB
(`layout::MEMORY_FLOOR_BYTES`, enforced by `MemoryBudget` and `KvCacheConfig::validate`); the
9.12 GiB and 114 GiB values are host preflight numbers owned by deployment, not by this crate.

**4. "72h persistent" is a label with no TTL API (spec lines 818, 939).** Both occurrences are
doc comments naming Tier-2 the "NVMe Persistent 72h Cache" (line 818) and the "NVMe 72-hour
persistent Global KV store" (line 939). Section 5.5 defines no expiry, timestamp or
eviction-by-age operation anywhere. **Code:** the file-backed tier persists a roster and survives
a reopen, but nothing expires on wall-clock time. If 72-hour retention is a requirement it is
unimplemented and needs an API.

**5. The two bounded replays cannot reconstruct a SWA tail on their own (a consequence of 2).**
Encoder replay covers the last `min(cached, 128)` tokens of the *cached prefix* (lines 850, 855);
decoder replay covers the last `min(total, 128)` tokens of the *prompt* (line 864). Those windows
coincide only when `cached == total`, which finding 2 forbids. On a partial hit the encoder half
is regenerable but the decoder half is not, so the two replay calls alone can never fill all 40
layers of a tail. **Code:** `cache_swa_tail` is the prefill path that fills all 40 layers, and
`SwaTail` tracks a `present_layers` bitmask so a tail only counts as a hit once every layer is
present.

**6. The 13.4 GiB pool row does not line up with the other footprint rows (spec line 174).** The
row says 13.4 GiB holds "3.67M - 6.50M" tokens at 890 B/token. 13.4 GiB = 14,388,140,442 B is
16,166,449 tokens, about 2.5x the stated top; conversely 3.67M and 6.50M tokens are 3.04 GiB and
5.39 GiB. Section 3.1 of `docs/architecture/08-kv-footprint.md` (lines 66-69) derives the same.
**Code:** the default global KV pool is sized explicitly in `KvCacheConfig`, and
`docs/architecture/08-kv-footprint.md` is the reference for the tokens-per-GiB arithmetic.
## D17 - `QUALIFICATION.md` records nine gates that no command in this tree can run, and the two All-Reduce latency numbers in the spec measure different things

Recorded 2026-10-15 from the qualification survey. Numbering: D15 is milestone 2's deleted trace
features, which is on an open branch, and D16 is milestone 4-KV's spec defects, which has merged; this
entry takes D17 so that milestone 2 does not have to rebase a heading. Entries stand in merge order: the
file reads D14, D16, D17 today, with D15 inserting above D16 when that branch lands.

**Context.** Milestone 5 ends with correctness and throughput gates. None had ever run, and the survey
that produced `docs/QUALIFICATION.md` asked why. The answer is not that the gates were skipped: for five
of the nine, **no command in this repository can produce the number the gate asserts on.**

**Findings, in the order they change what can be claimed.**

1. There is no GPU engine. `sparkflash serve` exists and is wired
   (`crates/sparkflash-cli/src/serve.rs`, dispatched at `crates/sparkflash-cli/src/main.rs:43`): it
   builds a `sparkflash_engine` reference engine over the synthetic plan or a sharded checkpoint,
   binds a `TcpListener` (serve.rs:892) and serves it with `axum::serve` (serve.rs:907) on a `tokio`
   runtime (serve.rs:874). It is one CPU process in `f32`, one sequence at a time, requests
   serialised through a single engine thread and no batching (serve.rs:12-17, 24-31), so the api
   crate's `InferenceEngine` seam now has two implementations - `MockEngine`
   (`crates/sparkflash-api/src/engine.rs:460, 731`) and `ReferenceDriver` (serve.rs:363) - and
   the reference forward pass is implemented and runs: all five modules are present and filled in
   (`attention`, `engram`, `mhc`, `moe`, `mtp`; crates/sparkflash-engine/src/forward/mod.rs:21-25),
   with no `unimplemented!()` or `todo!()` left in `crates/sparkflash-engine/src`, in `f32` on the CPU.
   It has run end to end against the synthetic tiny plan only (serve.rs:950-951), and never against
   real shards (none on this host), so its numbers are unusable for the model gates. So qeval, the
   DSpark KL divergence, the 1M needle test, the sparkDash sweeps and CUDA graph capture still cannot
   be run by any command here: they need the GPU and cluster instruments, not an HTTP entry point.
   That is a stronger statement than "unrun": the instrument does not exist.
2. Six run summaries have no producer. `SPARKFLASH_{QEVAL,DSPARK,NEEDLE,SPARKDASH,CUDA_GRAPH,TRACE}_RESULTS`
   appear only in `crates/sparkflash-scheduler/src/gates.rs` (declared 32-42, read 291-374). Nothing in
   the repository writes them, so the operator must supply the summary the gate then asserts on. A gate
   that asserts on a summary nothing produces is a check that cannot fail on the thing it names - the
   same pattern as the ring no job compiled and the feature sweep that skipped its own feature.
3. The three cluster gates are panic stubs: `crates/sparkflash-comm/tests/hardware_gates.rs:44-47,
   55-57, 65-67` panic "not implemented on this host" after their permission and feature checks, so the
   All-Reduce latency gate has no measurement in the test tree at all.
4. The one real All-Reduce instrument is merged: `rocenante-probe` is a binary target of
   `sparkflash-comm` now, and the doctor invokes it and refuses a loopback-labelled number by name, so
   check 3 went from a missing command to an instrument that measures or refuses. Its honest
   limitation is that it still cannot produce a number on the real fabric, because
   `RdmaTransport::open` resolves the live rails and then returns `CommError::RdmaNotBuilt`; what
   changed is the failure mode, from "no such command" to a precise refusal.
5. `sparkflash trace --scopes all --steps 45` prints a plan; it cannot capture, because the capture path
   is a hardware gate whose summary nothing produces.
6. No CUDA graph code exists: `cudaGraph` and `cuda_graph` appear in `crates/` only inside the gate's own
   parser and gate function.
7. Doctor's collective probes use only the first `--nodes` address, so its "aggregate >= 180 Gbps" is two
   local rails to one peer, not the "between all 4 ranks" the spec's preflight line asks for.
8. The gates print nothing on success, so a green run leaves no measurement in a log; an artifact has to
   be captured deliberately, which is why every gate section of the runbook names one.
9. Positive: the only ignored tests in the repository whose bodies really exercise hardware are the KV
   ones on the milestone-4 branch (`crates/sparkflash-kv/tests/hardware.rs:25`, a real NVMe `O_DIRECT`
   round trip at the device's own alignment). They are the pattern the three comm stubs should follow.

**Correction, recorded the same day.** The survey's snapshot predated milestone 2's merge, which is
what wired `sparkflash pack-engram`, and the first revision of finding 1 therefore said the packer did
not exist and that nothing under `crates/` mentioned safetensors. Both halves were wrong in the merged
tree: the CLI dispatches the subcommand (`crates/sparkflash-cli/src/main.rs:42`) into
`sparkflash_engram::packer`, which reads the checkpoint's safetensors shards. The finding that
matters is unchanged - the five gates that need a GPU model still have no command to run them - but
two of the mechanisms it named are not: since the serve slice landed, weights are loaded into a
runnable reference model and the endpoints are bound to a socket
(`crates/sparkflash-cli/src/serve.rs:892, 907`). A record that overstates a gap is as misleading as
one that hides it, so the claim is corrected here and in `docs/QUALIFICATION.md` rather than left to
be discovered. The Engram shards are absent for the ordinary reason: the packer has not been run
against the checkpoint.

**Correction, recorded 2026-10-16.** The first revision of finding 1 also said "`sparkflash serve`
does not exist", that `server::router` was a `Router` "that nothing binds to a socket", that there was
"no `TcpListener`, no `axum::serve` and no async entry point in the workspace", and that "no code
loads the transformer weights into a runnable model". All four sentences were true of the 2026-10-15
snapshot this entry records and are false in the tree today, so item 1 above now states the corrected
version rather than the snapshot's. What remains true is the conclusion those sentences supported:
the five gates that need a model still have no instrument here - because the one server in the tree
is the CPU reference path, not because a server is missing.

**Two latency numbers, and why both are right.** The spec states 22 us for preflight check 3 (line 1323)
and 20 us for milestone 3 (line 1336). Nothing in the tree encoded 20. They are not in conflict once
their roles are separated, and this entry is the separation: 20 us at 10 KiB is the requirement, and the
preflight's 22 us is a readiness threshold, deliberately looser so that a node can pass preflight while
still failing the requirement. A run that reports 21 us has passed the preflight and failed the gate. The
payload is 10 KiB in both places: 2 MiB appears in the spec as the router threshold, and the hardware
test's name for it is a naming defect rather than a second budget. Doctor encodes 22.0
(`crates/sparkflash-cli/src/doctor.rs:101`) and this entry is why it stays there.

**Consequence.** `docs/QUALIFICATION.md` is the record of what each gate would prove, the exact command
that would run it, the artifact that would evidence it, and - for every gate - that its state today is
"never run", with the missing instrument named where the instrument is missing. No gate may be reported
as passing on the strength of an `#[ignore]`d test, and the only claims a green CI run supports are
narrow: the CPU references, the threshold and parser logic, and doctor's check logic against injected
fakes. Closing the gaps in findings 1-7 is later work, not a qualification run.

---

## D18 - The engine is a ninth crate, and the tensor plan describes the checkpoint as stored

**Evidence.** The specification's tree names eight crates (`deepseek_v41_flash_4x_dgx_spark_rust_spec.md`
lines 227-252) and none of them is an engine; `docs/QUALIFICATION.md` section 5 finding 1 records the
consequence, that the five gates needing a GPU model have no instrument - the tree's one socket is
the CPU reference serve path (`crates/sparkflash-cli/src/serve.rs`), which has no GPU, no KV pool
and no batching, so it cannot produce their numbers. The checkpoint settles two things the earlier
architecture work left implicit. Its index (`model.safetensors.index.json`, 96,085 names over 48
shards, `metadata.total_size` 510,286,023,000) contains no `ffn.gate.scale`, no `compressor.*.scale`,
no `indexer.weights_proj.scale` and no `indexer.wk.scale`, so those tensors are not block-quantised;
`inference/model.py:811` runs the gate through the plain fp32 `F.linear` path, which agrees. And
`convert.py:156-168` dequantises `wo_a` to bf16 at conversion time, so the released index stores it
quantised, with a scale for all 40 layers: the tensor's identity changes between the released shards
and what `generate.py` loads.

**Decision.** The engine is a new workspace member, `crates/sparkflash-engine`, not a module inside
`sparkflash-cli`. The CLI already owns the doctor, the packer and the fabric probe; the engine owns the
weight plan, the loader, the KV cache and the serving loop. Keeping them apart is what lets the loader
be tested without a socket and the socket be tested without 510 GB of weights.

The first slice is the tensor plan, the configuration and a checkpoint-free weight source, because
those are the three things every later slice needs and the only three that can be finished before a
weight file is opened. The plan describes the checkpoint **as stored**, with the conversions recorded as
loader work, so its names and byte totals can be held against the index directly.

**Consequence.** `TensorPlan::language_model` produces 95,819 tensors - the index's 93,415 under
`layers.*`, 2,401 under `mtp.*`, plus `embed.weight`, `head.weight` and `norm.weight` - and the
remaining 266 index names (259 `vision.*`, 4 `aligner.*` and the three image delimiters) are the image
path, which the text engine does not plan and a test asserts rather than assumes. The 95,819 planned
names are set-equal to the index's language-model names, and their storage totals 509,315,486,040
bytes, 0.99810 of the released `total_size`; the remaining 0.19% is exactly the image path
(970,536,960 = 823,685,120 vision + 146,821,120 aligner + 30,720 delimiters), so the plan plus the
image path equals `metadata.total_size` **to the byte**. `metadata.total_size` is the sum of the
stored tensor bytes and contains no index JSON; an earlier revision of this entry said the 0.19% was
"the image path and the index metadata", which is false. That check does not depend on the
checkpoint being present. Of the 21 tests in `crates/sparkflash-engine/tests/tensor_plan.rs`, exactly
three need the index and print `SKIP: the released checkpoint index is not on this host` without it -
`the_plan_names_are_the_released_index_names`, `the_names_the_plan_leaves_out_are_the_vision_tower` and
`the_committed_digest_came_from_this_index` - and the rest run anywhere:
`the_plan_matches_the_committed_index_digest` checks the derived plan
against `tests/fixtures/v41_flash_tensor_digest.txt`, which
`crates/sparkflash-engine/scripts/regen-tensor-fixture.py` generates from the released index and which
records that index file's sha256 and name digest. That fixture is not taken on trust where the index
exists: `the_committed_digest_came_from_this_index` regenerates every one of its lines - the sha256, the
byte count, all nine name counts and the digest - and compares them line by line, so a fixture that
no longer describes the checkpoint fails rather than silently certifying the plan against a stale digest.
Finally,
`the_plan_bytes_are_exact_not_approximately_right` pins 509,315,486,040 and
509,315,486,040 + 970,536,960 = 510,286,023,000 with no index at all. On a host without the index the
binary still reports 21 passed, with the three skip lines. An earlier revision of this entry said the
checks run only where the index is present, which understated the coverage in the wrong direction.

Deriving the plan instead of transcribing the index found three defects in its own drafts:
`DType::storage_bytes` ignored `bytes_per_element` for the multi-byte formats; `wo_a` was planned
with the whole head width instead of `heads / o_groups`; and `layers.{1,14}.engram.q_weight`/`.k_weight`
were stored as fp32 where the checkpoint has bf16 (`inference/model.py:347-348` take the ambient bf16
default, and `Engram` is built at `model.py:932`, outside the `set_dtype(torch.float32)` block that
opens at `model.py:940`) - four tensors, 4 x 4 x 5120 x 2 = 163,840 bytes. None was visible until the
byte total was compared against the released `total_size`, which is the argument for deriving it.

Claims in the architecture docs corrected here and in those files: the gate weight is not quantised
(`00-overview.md`, `03-moe.md:18`, `03-moe.md:121`), the norm weights are bf16 in storage,
`indexer.wq_b` is the only quantised indexer projection, the Engram `q_weight`/`k_weight` are bf16
rather than fp32, the image path is 970,536,960 bytes rather than 553,394,784, and the section 2.4 byte
table is exact rather than a 0.09% approximation. `docs/engine/01-forward-order.md` carries the
evidence, with citations, for all of it.

---

## D19 - `panic = "abort"` in the release profile is deliberate, and the serve panic guard says so

Recorded 2026-10-16.

**Evidence.** The workspace manifest sets `panic = "abort"` for the release profile: the line is
`Cargo.toml:85`, inside `[profile.release]` at 81-86. `sparkflash serve` runs each request's forward pass
inside `std::panic::catch_unwind` on the engine thread so that a panic ends that one request as its
`EngineEvent::Error` instead of stopping the server - but `catch_unwind` only catches a panic in a build
that unwinds, and a release build does not. In a release `sparkflash` the process therefore aborts
rather than answering the request with an error; that is stated in the module doc (`serve.rs:41-47`)
and again at the guard (`serve.rs:685-690`). The guard cannot be narrowed to this binary: Cargo
refuses `panic` in a package profile with "`panic` may not be specified in a `package` profile"
(`serve.rs:42-44`), so the strategy is all-or-nothing across the workspace. Every dev and test build
unwinds, which is why the panic path is testable there: the guard's test and the weight source that
panics on demand are compiled only under `#[cfg(all(test, panic = "unwind"))]` (`serve.rs:260`, `:736`,
`:917-923`), so `cargo test` - and therefore CI - exercises the guard while a release binary never
reaches it.

**Decision.** Keep `panic = "abort"`. A server that aborts on an internal panic rather than unwinding
through half-initialised engine state is the deliberate choice; a per-package override is impossible
anyway, and flipping the workspace to `unwind` would change the panic strategy of `sparkflash-comm`
and the CUDA kernels for no gain at this milestone. The guard stays because the builds that can
reach it are the ones that need a per-request terminal event, and the module doc states the limit
rather than implying a protection the release binary does not have.

**Consequence.** `serve.rs`'s per-request panic guard must not be read as a production protection: in
a release `sparkflash`, an engine panic is a process abort. `docs/QUALIFICATION.md` section 5
finding 1 points here for the serve path's failure semantics. Before this entry the setting was
recorded nowhere in `docs/` - a reader learned it only from the manifest or from `serve.rs`'s module
doc.

---

## D20 - `sparkflash serve` reads the checkpoint's released `config.json`, and the two configuration documents are pinned

Recorded 2026-10-17.

**Evidence.** The checkpoint ships two configuration documents that are not interchangeable.
`config.json` at the checkpoint root is the released HF document: a nested `text_config`/
`vision_config` pair whose key names differ from the engine's (`hidden_size` for `dim`,
`num_nextn_predict_layers` for `n_mtp_layers`, `routed_scaling_factor` for `route_scale`,
`rms_norm_eps` for `norm_eps`, `sliding_window` for `window_size`, and several more).
`inference/config.json` is the flat 64-key runtime document the reference loads with
`ModelArgs(**json.load(...))` (inference/generate.py:121-123); it is what this crate carries at
`crates/sparkflash-engine/config/deepseek_v41_flash.runtime.json`. Before this entry,
`ModelConfig::from_json` parsed only the runtime document, so `sparkflash serve --checkpoint` could
not read a checkpoint as it was released: the file serve opens is `<dir>/config.json`
(serve.rs:83, 289-297), which is the HF document, and the parse failed on the absent `dim`. The
released field values checked read-only from the node agree with the copy at
`/home/user/dsv41-research/config.json` (sha256 `8be45ce0...`).

The released document carries no `first_k_dense_replace`, `n_group` or `topk_group`, and neither
does the runtime one. That is a property of the model rather than an omission: every V4.1 backbone
layer is a MoE block (`self.ffn = MoE(layer_id, args)`, model.py:929), there is no dense MLP to put
first, and the gate is an ungrouped top-k over all routed experts
(`indices = (scores + bias).topk(...)`, model.py:808-823). The engine never had those fields and no
code path wants them; what this entry adds is that a released document that does carry one is
refused by name instead of loaded with the field dropped.

The released document also carries no `gate.bias_vl`. That is a tensor, not a config field: the
reference creates it only inside the `vision_enabled` branch (model.py:807), and it is the image-span
selection bias (model.py:818-820). The plan describes all 43 of them - 40 backbone and 3 MTP stages
(`docs/engine/01-forward-order.md:132`) - and the text path never reads one: `route` takes the bias
as an argument and its text caller passes `None` (forward/moe.rs:461). The image path, when it exists,
is the one that will.

The tensor index settles the naming question. Of its 96,085 keys, 93,415 are `layers.N.*`, 2,401
`mtp.N.*`, 263 `vision.*`/`aligner.*`, and six are globals (`embed.weight`, `head.weight`,
`norm.weight`, `image_start`, `image_end`, `image_newline`); none starts with `model.`. The
plan builds `layers.{N}` and `mtp.{N}` (tensor.rs:73-81) and the index reader returns
`weight_map` keys verbatim (safetensors.rs:79-82), so no parser adds or strips a `model.` prefix.
Extra stored names - the vision and aligner tensors - are reported by the audit rather than refused
(loader.rs:471-472).

**Decision.** `ModelConfig::from_released_json` maps the released document onto the runtime fields,
and `ModelConfig::from_json` accepts either document (a `text_config` key selects the released
reader), which fixes the serve path without touching it. Every key of the released `text_config` is
accounted for: mapped to a field, checked against the value the rest of the engine assumes
(`hidden_act` = `silu`, `topk_method` = `noaux_tc`, `attention_bias` = false,
`num_key_value_heads` = 1, `rope_scaling.rope_type` = `yarn`), or ignored with its reason
(`max_position_embeddings`, `initializer_range`, `attention_dropout`, `use_cache`). The
`quantization_config` is checked rather than mapped - the dense decode assumes 32x32 UE8M0 scales and
FP4 experts - and the released document's own `dtype` (`bfloat16`, the safetensors storage format) is
deliberately not the runtime `dtype` (`fp8`, the linear-weight format, which comes from
`quantization_config.quant_method`). The released document states `norm_topk_prob`, which the
runtime document omits and the reference defaults to true; the released reader takes the stated value.

**Consequence.** `crates/sparkflash-engine/config/deepseek_v41_flash.released.json` is the released
`config.json` byte-for-byte (source sha256
`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`), beside the runtime copy, and
`crates/sparkflash-engine/tests/released_config.rs` pins its exact top-level, `text_config` and
`vision_config` key sets, its sha256, and the verified values - vocab 129,280; dim 5,120; one routed
expert's width 2,304; 40 layers; 3 MTP stages; 64 heads; head dim 512; RoPE head dim 64; q-LoRA 1,280;
o-LoRA 1,024; 8 output groups; 384 routed experts; top-k 6; one shared expert; route scale 1.5; SwiGLU
limit 10.0; RMS epsilon 1e-20; RoPE theta 10,000 with YaRN factor 16; window 128; `kv_source_layers`
[2, 8, 14, 20]; index top-k 512; mHC 4; Engram layers [1, 14]; DSpark targets [37, 38, 39];
`compress_ratios[0]` = 0, so layer 0 is uncompressed - and asserts the derived configuration equals the
runtime one field for field. A key added by a later release fails that test until it is mapped or
explicitly ignored; it can no longer be dropped in silence.

---

## D21 - The determinism contract is D-a, D-b and D-c, and a test makes it citable

Recorded 2026-10-17.

**Evidence.** Phase 0 started from a measured counter-example, not a principle. The mega-mHC kernel
accumulated its sum of squares with an `fp32 atomicAdd`; four tiles arrived in an order the host
could not fix; and on identical input the stage-4 repeat measured one differing byte between run 0
and a later run (the separate 6-of-5,120 FP8-code difference was stage 1's device-versus-reference
finding, a different comparison; that finding line is rewritten, not deleted). The three sentences the rest of the
project needs to cite all existed, but none of them was written where a test could read it: the
kernel claim lived in the M7 harness's comparison class, the collective claim lived in a module
doc, and the "exact or named tolerance" rule lived only in the harness's printed output.

**Decision.** The determinism contract is these three statements:

- **D-a: kernel determinism.** For fixed input bytes, launch configuration and device, a forward
  kernel's output bytes are identical on every run.
- **D-b: reduction determinism.** The M3 one-shot reduction is bit-identical across ranks by
  construction: f32 accumulation in ascending rank order, one rounding to bf16 at the end
  (`crates/sparkflash-comm/src/lib.rs` module docs).
- **D-c: no assumed cross-boundary equality.** Device-versus-reference equality is never assumed;
  it is either exact by construction or a named tolerance with a reason, printed before the number.

**Consequence.** `crates/sparkflash-kernels/tests/determinism_contract.rs` carries
`the_determinism_contract_is_written_down_and_the_harness_cites_it`, which reads this file and
`crates/sparkflash-kernels/examples/m7_layer0.rs` and fails if a statement disappears or the harness
stops citing a letter from its printed output; `crates/sparkflash-kernels/tests/mhc_determinism.rs`
pins the fixed-slot, binary64 tile combine (`kernels/mega_mhc_sm121.cu:197-222`) and the
26-statistic buffer. The direct
measurement is `m7_layer0 --stage 4 --repeat 100`, which compares raw bytes with no tolerance: that
is D-a's test (owed a device window: on 2026-10-17 the resident model held the only usable context
and the launcher refused a second one with `cuCtxCreate_v2 -> 2 (OUT_OF_MEMORY)`, recorded in
roadmap section 3.2), and D-c is why stage 1 prints its bars before its numbers. D-b is not on this device
path; it is the property Phase 4's numerical criterion rests on (roadmap section 7.3).

---

## D22 - The released mHC arithmetic is modelled end to end, with the stream-mean fold as the one structural difference

Recorded 2026-10-17.

**Evidence.** The released forward flattens the four residual streams to one `n*d = 20,480` vector
per token, takes `rsqrt(mean(x^2) + norm_eps)` over that whole stream, projects it with
`hc_fn [24, 20,480]` and multiplies by the reciprocal square root: `mixes = F.linear(x.flatten(2),
hc_fn) * rsqrt`, with `norm_eps = 1e-20` (`inference/model.py:950-955`; the checkpoint's
`text_config.rms_norm_eps`). Each coefficient group is then `mix * hc_scale[g] + hc_base[row]`,
with `pre = sigmoid(...) + hc_eps`, `post = 2 * sigmoid(...)`, and the 16 comb logits go through
a row softmax plus `hc_eps`, one unconditional column pass, and 19 alternating row/column passes,
each dividing by `sum + hc_eps` (`inference/kernel.py:423-462`, the kernel behind
`hc_split_sinkhorn` at 465-475). The block-pre norm that consumes the collapsed `X_hat` uses the
same `rms_norm_eps = 1e-20` (`inference/model.py:984,991`).

Before this entry the device path differed in four ways: the kernel's `H()` took neither the
per-group scale nor the per-row bias (the harness pre-folded `hc_attn_scale` into a `[24, 5,120]`
predictor and read `hc_attn_base` only to print its range); the projection rstd used the Sinkhorn
epsilon instead of `norm_eps` (still over the flattened `n*d` stream), and the block RMS normalised
the four streams over `n*d` instead of `X_hat` over `d`; `C` was published as raw logits while our reference softmaxed it, and the harness
compensated with a harness-side softmax around the comparison; and the split of the 24 rows into
A/B/C was our reading of the reference rather than a pinned, tested mapping.

**Decision.** The released arithmetic is one implementation with one test oracle, and the kernel
takes the released tensors:

- `kernels/mega_mhc_sm121.cu` takes `hc_scale[3]` (released `[pre, post, comb]` order) and
  `hc_base[24]` in the kernel's row order, accumulates two sums of squares (the streams' over
  `n*d` for the projection rstd, `X_hat`'s over `d` for the block norm), and its last tile
  publishes the released values: `A = sigmoid(...) + hc_eps`, `C = 2 * sigmoid(...)`,
  `hc_coeffs[dst*n+src] = comb[src][dst]`, and `hc_rms_out = 1/sqrt(mean(X_hat^2) + 1e-20)`
  (lines 226-265; the ABI comment at 20-32).
- `kernels/m4_common_sm121.cuh:461-...` carries `spark_released_split_sinkhorn` (row softmax,
  `+ hc_eps`, one unconditional column pass, then 19 row/column pairs, each dividing by
  `sum + hc_eps`) and `spark_sigmoid`.
- `crates/sparkflash-kernels/src/mega_mhc_ref.rs` carries the mapping and the oracle:
  `fold_predictor_into_kernel_rows` (line 683) maps released `pre(0..4)/post(4..8)/comb(8..24)`
  onto the kernel's `B(0..16)/A(16..20)/C(20..24)` with the comb transposed,
  `bias_into_kernel_rows` (line 606) permutes the bias the same way, and
  `mega_mhc_layer_with_scale` (line 984) is the reference the kernel is compared against. The four
  older entry points' arithmetic and outputs are unchanged and their existing tests still pass; the
  zero-bias equality test pins the bias-only path to the no-bias path (both still softmax-based), not
  the released path, which deliberately differs even at unit scale and zero base: this is a new path,
  not a redefinition of the old one.
- `crates/sparkflash-kernels/examples/m7_layer0.rs` passes the raw released tensors (the bias in the
  checkpoint's row order to the reference, which folds it itself, and the folded row order to the
  kernel), compares the coefficients raw (the harness-side softmax is gone), and checks `1/rms`
  against the released `X_hat` statistic.

**Consequence.** The project's one structural difference from the release is named rather than
implied: the predictor is the project's `[24, 5,120]` matrix applied to the stream mean (the fold),
not the release's `[24, 20,480]` projection over the flattened stream. Closing it is a Phase 1
decision about the kernel's inner loop, not a coefficient-tensor gap: `hc_attn_base`,
`hc_attn_scale`, the released split-Sinkhorn, the sigmoid pre/post, the projection rstd and the
`X_hat` block norm are consumed on both sides, and the row pairing is pinned by
`the_fold_puts_each_released_row_on_its_kernel_row`. Roadmap section 10's unknowns 1, 2, 4 and 6
are closed by the tests named here and by the stage-1 output lines they pin; unknown 7 is closed in
code by D21 with its N = 100 device run owed; unknowns 3, 5 and 8-16 are untouched. The no-device half is covered by `cargo test -p sparkflash-kernels --lib`,
`--test mhc_determinism` and `--test determinism_contract`; the device half is
`m7_layer0 --stage 1` and `--stage 4 --repeat 100` on the pinned node, and those runs are owed: on
2026-10-17 the node's resident model held the only usable CUDA context (roadmap sections 3.2 and 3.6
record the refusal), so the only device numbers are the pre-change ones in roadmap section 12.1,
labelled as such.


**Update (B0, 2026-10-18): the stream-mean fold is gone, so the structural difference this entry
named no longer exists.** The kernel's B0 inner loop consumes the released `[24, n*d]` predictor
rows and projects them over the flattened four-copy stream; the harness no longer averages the four
copies, and `mega_mhc_ref::released_predictor_into_kernel_rows` permutes the released rows into the
kernel's B/A/C row order on both sides. The retired `fold_predictor_into_kernel_rows` path survives
only in stage 0's non-vacuity half, where it is computed explicitly to show the gap it used to
leave. Measured on the engine's captured row 0 (device log
`/home/user/phase1-evidence/phase1-d8-stage0-quantiser-20261018T012533Z.log`): the released
projection reproduces the captured `hc_*_pre/post/comb` tensors to a worst absolute delta of
5.96e-8 against a 1e-5 bar, while the retired fold misses them by 5.03e-1 / 1.71 / 9.07e-1 (attn
pre/post/comb) and 6.66e-1 / 9.16e-1 / 8.76e-1 (ffn). Stage 1 on the node (same window) then passes
at layers 0, 1 and 4 with 4 of 4 tokens each, the FP8 codes and ue8m0 scales at layers 0, 1 and 4 on 4 of 4 tokens each: layer 0 token 0
matches 5,119/5,120 codes (one one-code difference at index 615, scales 160/160 bit-exact) and
tokens 1-3 match 5,120/5,120 codes and 160/160 scales; layer 1 matches 5,120/5,120 codes and
160/160 scales at every token; layer 4 token 0 matches 5,118/5,120 codes (two one-code
differences, scales exact) and its other tokens match everything. Where a code differs the
residual is the known fixed-order vs sequential `1/rms` reduction difference, inside the
stage-1 bar. The
"one structural difference" consequence above and the "device runs are owed" note at its end are
both superseded by these runs.

---

## D23 - The activation quantiser is the reference ceil rule, and it is named apart from the weight-grid rule

Recorded 2026-10-18 (Lead decision D8, from the B1 probe finding).

**Evidence.** The released forward has exactly one *dynamic activation* scale rule
(`inference/kernel.py:36-37, 76-80`; the engine's `quant::e8m0_encode_ceil` and
`quant::quantize_activation_fp8`): floor the block's amax at 1e-4, multiply by 1/448 in f32, then
round the scale exponent **up** - one higher when the product's mantissa is nonzero - giving the
smallest power of two at or above `amax/448`. The checkpoint's stored *weights* use a different
rule (`ue8m0_from_amax`: `floor(log2(amax)) - element_emax`), and the two are not interchangeable:
on an activation whose amax lies above `1.75 * 2^floor(log2 amax)` the weight-grid rule picks a
scale one binade too small. The B1 probe caught the consequence: `swiglu_mxfp8_quant_sm121` was
10.16% exact on E4M3 codes and 0/80 on scales against the reference, with a maximum code delta of
24, caused by `spark_mx_quantize_block -> spark_ue8m0_from_amax(amax, 8u)`; the same rule was in
the mHC kernel's `X_hat` quantiser.

**Decision.** Device activation quantisers use the ceil rule, and the two rules are separate,
explicitly named helpers so a future caller cannot pick the wrong one:

- `kernels/m4_activation_quant_sm121.cuh` carries `spark_act_quant_scale_code` (amax floor
  inside the helper, ceil exponent, ue8m0 clamps) and the `SPARK_M4_ACT_*` constants. It is a
  separate header rather than an addition to `m4_common_sm121.cuh` because that header's bytes are
  the input to the pinned measured-text evidence in `m4_abi` (one substitution per stored text, each
  with a measured cubin): changing it would invalidate that whole table. The common header was
  therefore left byte-identical; it still describes `spark_ue8m0_from_amax` and
  `spark_quantize_row_e4m3` under their general OCP MX wording and still calls that rule the
  OCP MX rule. The separation is carried by the activation header name and docs, by the kernel
  call sites, and by the tests below, NOT by a common-header comment edit. A future
  common-header edit must regenerate the measured-text table first.
- `kernels/mega_mhc_sm121.cu`'s `X_hat^FP8` block quantiser and
  `kernels/swiglu_mxfp8_quant_sm121.cu`'s activation quantiser (both entry points, SwiGLU output
  and `mxfp8_quantize_activations_sm121`) call the activation helper. The weight-grid
  `spark_quantize_row_e4m3` in the common header is NOT edited (see the previous bullet): it
  keeps its general OCP MX doc, and nothing on the activation path calls it.
- `crates/sparkflash-kernels/src/mega_mhc_ref.rs` and `swiglu_ref.rs` use `m4_quant`'s
  `act_quant_scale_code` (total: inf takes code 254, NaN falls to the 1e-4 floor) or
  `act_quant_fp8_row`; both modules' docs and tests state the ceil rule, and the old floor-rule
  helper is deleted from `swiglu_ref`. The D8 review (B1) found these two refs still routing an
  infinite amax through the fast_pow2 pair, which panics; `act_quant_scale_code` is the single
  total path now, and `crates/sparkflash-cli/tests/activation_quantiser_equivalence.rs` compares
  it with the engine rule on inf, NaN, powers of two, subnormals and the 1e-4 floor.
- `m7_layer0 --stage 0` gained the cross-check the measurement needs: it recomputes row 0's
  `pre_attention_xhat/scales` and `pre_ffn_xhat/scales` from the row's own captured f32 block
  inputs with the ceil rule and requires every code and scale to match the engine's captured
  tensors; the retired floor rule is printed on the same bytes, unjudged.

**Consequence and device evidence (node node0, 2026-10-18, window held open for B0/B1/D8).**
Rebuilt with the pinned nvcc V13.0.88 from the CUDA 13.0.1 devel image (`7d2f6a8c2071`; the
build log records `CUDA Version 13.0.1`; the CI `cuda-abi` image is CUDA 13.0.3). The changed
cubins are
`mega_mhc_fused_bf16_sm121` 70,312 B sha256 `e21293e3...` and `swiglu_mxfp8_quant_sm121` 21,776 B
sha256 `4dbef63b...`; dense, wo_a, mxfp4_moe, flash_mla and cert_lm_head are byte-identical, so the
baseline diff is those two lines only:

- stage 0: ceil-rule recomputation reproduces 5,120/5,120 codes and 160/160 scales for both block
  inputs; the retired floor rule would give 4,288/5,120 codes + 134/160 scales (attention) and
  4,096/5,120 + 128/160 (ffn); projection check unchanged at 5.96e-8. PASS.
- stage 8: all four probes bit-exact - dense 16,384/16,384 and wo_a 8,192/8,192 (ULP 0), swiglu
  4x640 codes 2,560/2,560 + scales 80/80, swiglu 2x2304 codes 4,608/4,608 + scales 144/144. The
  same probe bytes were 10.16% / 0% exact before the fix (max code delta 24). PASS.
- stage 1: PASS 4/4 tokens at layers 0, 1 and 4; layer 0 token 0 is 5,119/5,120 codes (one
  one-code difference) with 160/160 scales exact, layer 0 tokens 1-3 are 5,120/5,120 codes and
  160/160 scales, layer 1 is 5,120/5,120 codes and 160/160 scales at every token, and layer 4
  token 0 is 5,118/5,120 codes (two one-code differences, scales exact) with its other tokens
  exact. `phase1-d8-stage1-layer0-20261018T012545Z.log` shows the old floor-rule behaviour
  (4,287/5,120 codes, 134/160 scales on token 0) but its header records the NEW binary and new
  cubin sha256s, so it is NOT evidence that the pre-D8 cubin was installed; a correctly labelled
  OLD-VS-NEW run using the retained
  pre-D8 floor-rule cubin `b0-out/mega_mhc_fused_bf16_sm121.cubin` (70,152 B sha256
  `6487f6ff...`, the B0 released-projection build) and the old binary
  `m7_layer0-b0` (09cf261f...) against it as an OLD-VS-OLD control; both are on the node or in
  the evidence directory and both runs are device-window items.
- stages 2 and 3 re-run unchanged (attention within tolerance, MoE combine 20,480/20,480 exact), so
  the rebuild did not disturb the other boundaries.

Logs are in `/home/user/phase1-evidence/` (`phase1-d8-*`), cubins and the baseline diff in
`d8-out/`. The stage-1 and stage-2 logs cited in this entry were produced before the M2 fix of their
printed comparison-class strings (they print the retired stream-mean fold as the predictor);
regenerating them is a device-window item, and until then their numbers are cited from the
captured bytes and not as current evidence. The Phase 1 consequence: the two activation-quantised boundary tensors are now produced
by the reference rule, so the row-0 boundary diff in B2 can attribute a code mismatch to the
number's own rounding rather than to a known rule defect.

**Stage B record (2026-10-18, sixth window): the chained layer-0 replay (B2) is measured, and it is
a health check, not criterion-1 closure.** The `--stage 7` replay ran sixteen device launches in
release order plus a four-launch forced-routing control, captured and diffed all eighteen row-0
boundary tensors, and exited 0 (PASS 18/18; log `phase1-b2-stage7-layer0-rereview5-20261018T044122Z.log`, 23,897 B,
local copy sha256 `c2de544c1789743e73ccc5d5277dc3165d98296d0f5cc867cc819d9160754ac9`, binary
sha256 `2b518a35...`, the cross-build of the pass-5 code commit `1845f2d`; the evidence binary is
that commit's own build, and the docs head's rebuild reproduces it). The pass-5 log's metric and
content lines are bit-identical to the pass-4 log's (only the label/argv/binary-sha/UTC header lines
and the device/node MemAvailable lines - header before and after, process start, per-launch and
staged-weight - differ), so the tie plumbing did not move any measured number; the pass-4 log
(`...rereview4-20261018T041707Z.log`, binary `f99f14bf...`) remains on disk as pre-plumbing
evidence and is no longer cited. It applies no bar: it measures the single-layer deltas the 40-layer
chain will compound. Every kernel is bit-exact against its in-crate reference on the same bytes, so
the chain itself is verified; the row-0 deltas against the f32 host reference are attributed to the
window-KV format (D4) and the host reference's documented "deliberately not reproduced" arithmetic,
quantified in the log's attribution check. The device column therefore measures attribution and
gross mismatch, not exactness: the device path uses FP4 main/window KV and bf16/quantised
arithmetic where the reference is f32 with no GEMM activation quantiser, and the format-faithful
exactness run is the one after Phase 2 (task-3). The forced-routing control was corrected across the
re-review passes: subtracting the free and forced host max-abs deltas is a change in max error, not a
max-abs contribution, and is no longer reported as one; its verdict enforces determinism - identical
ids with unchanged weights must reproduce the free path's routed bytes exactly (a difference there is
nondeterminism and aborts), a flip with identical routed bytes aborts (refined in the Stage C record
below: the abort applies when the two independently computed references differ; a flip whose
references are bit-identical cannot move the bytes and is reported, not aborted), and a weight change
with equal ids may round to equal or different bytes; the forced operands are built from the re-derived host
routing and verified against the checkpoint before the launches (every rank's w2 weight and scale
slices, and every slot's w1/w3-derived activation codes and scales at every rank, bit-for-bit); and
the forced `ffn_out` is recomputed from the forced routed-combine bits and the shared output through
the free path's combination and required bit-identical before the host delta is reported; the
returned recomputation (documented as the asserted-identical recomputation of the forced output) is
the tensor every reported forced number reads, so a copy inserted after the assertion cannot change a
metric. The router's top-6 differs (ordered lists differ in four slots, expert sets
differ by one member, device-only 246 and host-only 236); the control measures the flip pointwise:
max abs |`ffn_out` free - forced| = 3.750000e-1 at index 5071 (signed -3.750000e-1), 4,853/5,120
points differ, with the two host deltas kept separate (2.150760e-1 free, max at index 1233;
2.160320e-1 forced, max at index 5071). The flip is a large pointwise perturbation, and the
max-host-error point is flip-dominated: at index 1233 the flip contributes 2.500000e-1, 116.24% of
the 2.150760e-1 host error there, and forcing the host row's routing does not reduce the max host
error. The combine kernel is excluded (K9-K12 exact on the same bytes, free and forced). Routing
flips are divergence events to report per layer, never to fold into a smooth bar; and no number in
this record closes criterion 1.

**Stage C record (2026-10-18, sixth window, after the B2 merge): the 40-layer chained device
column is measured on the rebased code, its verdict is evidence-based, and the old attribution is
withdrawn.** Stage C (`m7_layer0 --stage 9`) chains the released checkpoint's first step (token id 0,
position 0, backbone rows only) as 40 rows, writes a device `v1` pair and diffs it with
`boundary_diff`. The branch is rebased onto main `2214fe6` (the squashed B2 !51) and adapted to
the B2 helpers rather than carrying a second copy: the duplicate C safetensors reader is deleted and
the shared `crates/sparkflash-kernels/tests/support/b2_harness.rs` module supplies the snapshot
reader and host boundary table (the module gains the read audit - `read_names`/`read_bytes`/
`audit` - and a `manifest_text` accessor); both the stage-7 replay and `stage9_ffn_block` use
`combine_bf16_routed_with_shared` for `ffn_out`, `verify_forced_ffn_out` (whose returned
recomputation feeds both host deltas and the pointwise flip metric), `forced_control_verdict` for
the ids/weights/routed-bits verdict, and `verify_forced_operand_provenance` with
`FORCED_OPERAND_GEOMETRY` before every forced launch. The frozen bars continue to be the B2
single-layer deltas of record (log `...rereview5-20261018T044122Z.log`, 23,897 B, local sha256
`c2de544c...`, lines 149-166, binary `2b518a35...`); this run does not re-measure them.

Evidence of record: log `/home/user/phase1-evidence/phase1-c-stage9-40rows-harness5b-20261018T050416Z.log`
(572,420 B, sha256 `bd0f908ef2e3519512a20bd53eb86303fcfad65550f7b145a6c81c7a2c973a65`, UTC
05:04:16Z-05:09:14Z, wall 298.6 s, exit 0), binary sha256
`c1708abcfa19bf37fe4fa74c7f0057d1689df46ffada196b3cdd84ccb355f211` (the C head's cross-build),
node MemAvailable 123,438,768 kB before / 123,194,824 kB after / 122,285,404 kB minimum; device pair
`stage9-out-r5b/device.manifest` (34,767 B, sha256 `6093c74b...`) and `device.blob`
(13,537,920 B, sha256 `fecd813b...`); diff log
`phase1-c-stage9-diff-20261018T050941Z.log` (sha256 `b86be96879239a82a46559d0df2bd2e050ccc2c0f66986fe76a044ed676b5eb4`,
`boundary_diff` binary `5ffb11f9...`). Determinism: PASS - the two chain runs are bit-identical
on all 18 tensors of all 40 rows, both v1 blobs `fecd813b...`. The control run exits 0 with all 40
rows bit-identical and the device column exits 1 with all 40 rows differing (the engine tool's
documented verdict for a health-check column). The device pair and all 1,520 printed row lines are
bit-identical to the earlier Stage-C run's, so the helper adaptation changed verification/reporting,
not measurements; the wall rises from 230.9 s to 298.6 s because every layer's forced operands are
now verified against the checkpoint before the forced launches, and the harness-reader byte total
rises from 15,826,461,120 B to 20,338,614,720 B for the same 2,661 distinct tensors because the
shared reader counts the provenance re-reads. Totals unchanged: 306 of 720 tensor-row comparisons at
or under the frozen bars, 414 over; first over-bar row 1; free-running `ffn_out` max abs up to
7.957832e2, forced up to 5.397832e2; 17 of 40 layers flip [0, 1, 2, 4, 5, 8, 10, 11, 15, 16, 17, 19,
20, 23, 24, 35, 36]; audit 95,819 planned / 2,673 read / 93,146 unread, all classified; 80/80 K5,
640/640 kernel-vs-reference and 160/160 routed-combine checks exact, zero warnings.

**Attribution correction and the verdict refinement.** The old reading of the flip as the difference
of the two host max-abs maxima (free minus forced) subtracted two maxima taken at different indices
and is withdrawn. The measurements are pointwise and separate: on row 0 the flip's
pointwise max abs is 3.750000e-1 at index 5071 (signed -3.750000e-1), 4,853/5,120 points differ; the
two host deltas are free 2.150760e-1 (max at index 1233) and forced 2.160320e-1 (max at index 5071).
The difference of the two host-error maxima is a change in max error, not the flip's max-abs
contribution; at the free max-error index 1233 the flip contributes 2.500000e-1, 116.24% of the
2.150760e-1 host error there, so that max-error point is flip-dominated and forcing the host routing
does not reduce the max host error. The 40-layer run also measured a legitimate zero-effect flip:
at layer 36 the ordered ids differ in slot 5 (device 186, host 150) but the two independently
computed routed-combine references are bit-identical, so the bf16 output cannot move - the routed
bytes are identical, the pointwise flip is 0.000000e0 (0/5,120 points differ), and both host deltas
are 1.932627e2 at index 2455. The original B2 verdict rule (a flip must give a nonzero routed
difference) aborted here, so it is refined to take the two references as evidence:
`forced_control_verdict` now allows ids-differ + bytes-identical only when the two references are
bit-identical, keeps the hard copy abort when the references differ, adds a separate abort when the
references are bit-identical but the bytes differ (a launch that does not reproduce its own
reference), and leaves the identical-ids rules unchanged. Operand provenance is untouched. B2's own
row-0 case is unchanged (its references differ, so the copy mutation still aborts), and the CPU suite
(`b2_harness_offline`, 14 tests) covers the allowed zero-effect flip, the copy abort, the
broken-launch abort and the identical-ids cases. The pre-refinement attempt
(`phase1-c-stage9-40rows-harness5-20261018T045752Z.log`, 259,368 B, sha256 `5b8da864...`, exit 1,
no pair written) is kept as the historical record of the L36 case. The earlier Stage-C evidence
(`phase1-c-stage9-40rows-final-20261018T023529Z.log`, binary `f5430527...`, its smoke log and its
diff log) stays on disk as historical evidence and is no longer cited. Criterion 1 is NOT closed by
this column: it is the Phase-1 health-check measurement, and the format-faithful exactness run is the
one after Phase 2 (task-3).

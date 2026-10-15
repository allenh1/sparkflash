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
| D16 | KV specification defects | Six 5.5 defects recorded as facts: the 890 B/token headline contradicts its own itemisation (356 B/entry once the rotary dimensions are not double-counted), plus five more, each with the code's settlement named |
| D17 | Qualification state | Nine gates recorded with instrument, command, artifact and state; five cannot be run by any command in this tree, and 20 us at 10 KiB is the requirement while the preflight's 22 us is a deliberately looser readiness threshold |

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

1. *The checkpoint declares the activation scheme.* `inference/config.json:12-21` is
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

**Verification.** Doc-only change plus this entry; no GPU and no cluster were involved, and every
claim above is a line-cited reading of `config.json`, `model.py`, `kernel.py` and the spec. The
reference tree was read at `/home/user/dsv41-research`; the kernel file was read at
`wt/m4k/kernels/b12x_mxfp4_moe_sm121.cu` (the workspace-root `kernels/` directory of specification
5.1, not a `crates/sparkflash-kernels/kernels/` subdirectory). No git commands were run; the edits
are in the `docs/architecture` worktree.

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

1. There is no engine. The only `InferenceEngine` implementation in the tree is `MockEngine`
   (`crates/sparkflash-api/src/engine.rs:372, 424, 680`); nothing under `crates/` loads checkpoint
   weights (no `safetensors` reference exists); `serve` and `pack-engram` are unimplemented
   (`crates/sparkflash-cli/src/main.rs:5-6`); and `server::router` builds an axum `Router`
   (`crates/sparkflash-api/src/server.rs:200`) that nothing binds to a socket - there is no
   `TcpListener`, no `axum::serve` and no async entry point in the workspace. So qeval, the DSpark KL
   divergence, the 1M needle test, the sparkDash sweeps and CUDA graph capture cannot be run by any
   command here. That is a stronger statement than "unrun": the instrument does not exist.
2. Six run summaries have no producer. `SPARKFLASH_{QEVAL,DSPARK,NEEDLE,SPARKDASH,CUDA_GRAPH,TRACE}_RESULTS`
   appear only in `crates/sparkflash-scheduler/src/gates.rs` (declared 32-42, read 291-374). Nothing in
   the repository writes them, so the operator must supply the summary the gate then asserts on. A gate
   that asserts on a summary nothing produces is a check that cannot fail on the thing it names - the
   same pattern as the ring no job compiled and the feature sweep that skipped its own feature.
3. The three cluster gates are panic stubs: `crates/sparkflash-comm/tests/hardware_gates.rs:44-47,
   55-57, 65-67` panic "not implemented on this host" after their permission and feature checks, so the
   All-Reduce latency gate has no measurement in the test tree at all.
4. The one real All-Reduce instrument exists and is unmerged: `rocenante-probe` on
   `feat/rocenante-probe`. Merged `main`'s doctor expects it on `PATH` and fails check 3 without it by
   design. Its honest limitation is that it still cannot produce a number on the real fabric, because
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

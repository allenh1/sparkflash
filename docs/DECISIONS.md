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
| D2 | Expert sharding | **Corrected by D30**: the reference shards whole experts (96/rank at world 4) and that evidence stands, but the only device-verified routed-expert layout slices the intermediate width instead (576 per rank padded to 640); the shared expert stays whole and replicated |
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
| D19 | `panic = "abort"` in the release profile | Kept deliberately: a release build aborts on an engine panic, so `sparkflash serve`'s per-request `catch_unwind` guard protects the unwinding dev/test builds, not production; the guard and its module doc stay, and the limit is stated rather than implied |
| D20 | The two configuration documents | `ModelConfig::from_released_json` maps the released HF `config.json` onto the runtime fields (every key mapped, checked or ignored with a reason) and `from_json` accepts either document, so `sparkflash serve` reads a checkpoint as released; both documents are pinned, and the derived configuration equals the runtime one field for field |
| D21 | The determinism contract | D-a kernel determinism, D-b reduction determinism and D-c no assumed cross-boundary equality are written down here and cited by the M7 harness; a test fails if either half disappears |
| D22 | Released mHC arithmetic | The ref, the kernel and the harness model the released arithmetic end to end (projection rstd, `mix*scale+base`, sigmoid pre/post, released split-Sinkhorn, raw C, `X_hat` block norm, released `[24, n*d]` predictor); B0 removed the stream-mean fold from both sides, so the old structural difference is gone |
| D23 | The activation quantiser is the reference ceil rule | Device activation quantisers take the smallest power of two at or above `amax/448` after the 1e-4 amax floor (the engine's `quantize_activation_fp8` rule); the checkpoint weight-grid rule stays floor-log2 and the two helpers are named apart (D8) |
| D24 | CSA attention attends the window/compressed union | Format 2 (CSA-union) attends the window rows then the selected compressed ids, causal, with the sink first in the denominator; the 19-param ABI and formats 0/1 are unchanged, and it is reference-tested/CPU-pinned only until a device window proves it |
| D25 | The golden sequence fixture is Phase-2 reference truth | The real-checkpoint fixture (`e6df8290...`, 2,536,194 B; 130-token prefill with next token 491, 8 decodes, compared 129..137) is the truth 2B/2C are measured against; device output is compared token-first then logits at 2e-2; the 2B device plumbing (rotation + format-2/1 dispatch, no forward chain) was merged as !63 (1ceeb54) and ran the corrected path on the node in the 2026-10-18T18:32Z window (11,764 copies, 198,832,128 B in 0.025624 s copy-only, 360 dispatches), the token/logits comparison has not run, criterion 1 stays OPEN, and the fixture is never regenerated to accommodate a device change |
| D26 | The Milestone-4 ABI source-text lint is frozen | No new source-text scanner, parser or lint in the `m4_abi` family (`m4_abi.rs` plus its pinned submodules); new kernel invariants are enforced by compiled-artifact inspection (cubin/PTX baseline) or runtime device assertions, and `ci/check-m4-abi-size.sh` fails CI if the family's aggregate line count differs from its pinned count |
| D27 | The capability boundary has no vision channel | `Capabilities` has no `vision` field, `/v1/models` reports `multimodal: false`, and an image request is refused before generation with typed `MissingCapability::vision()` (HTTP 400 `unsupported_capability`); the 32x32 dummy grid and its 121 fabricated visual tokens are deleted, and `EngineRequest` carries the parsed `images: Vec<ImagePart>` (inline parts hold the decoded bytes) rather than `image_tokens`; the known protocol image `type`s are explicit part variants and a raw-body scan (`prompt::find_image_carrier`) refuses any image-family carrier the typed parser did not record, wherever it appears and in any case (a schema-name map key whose value is itself a JSON Schema aside) |
| D28 | The reference engine is `Send + Sync` | `WeightSource` gained `Send + Sync` in !60 (`f72738b`), so `forward::InferenceEngine` is `Send + Sync`; `serve.rs`'s module docs are corrected and a compile-time assertion pins it, while the single-owner engine thread stays because the pass is one-sequence-at-a-time and the sampler swap consumes the engine, not because of thread-safety |
| D29 | The ibverbs backend is real and dual-rail capable | The `rdma` feature's libibverbs backend, its repr(C) ABI mirror and the three cluster gate bodies are real, and the deployed vLLM reference is measured single rail (NCCL_IB_HCA=rocep1s0f1); no hardware gate has run and no frame has crossed a wire, host-memory staging, active MTU 1024 and driver 580.173.02 remain gaps, and single-rail mode refuses to claim dual-rail evidence |
| D30 | Routed-expert sharding follows the verified kernel | TP=4 over the intermediate width (576 per rank padded to 640) for the routed experts, because that is the only routed-expert layout the device has run; the reference's whole-expert rule is recorded as not implemented, the shared expert stays whole and replicated, and the padding costs +8.22 GB/rank of routed-expert storage (74.0 -> 82.2 GB/rank) |
| D31 | The certified LM head counts every thread's prunes, and the pin is re-baselined | `cert_lm_head_sm121`'s published `certified_count` was lane 0's per-thread count only, so the concentrated pruning case reported 34 against 4,899 flags (b21 printed only the aggregate count mismatch across the four ranks; b23 measured the per-rank 34); the atomic now adds every thread's prunes, the count is deterministic, and the cubin is re-pinned at 14,888 B / `7394b129...` with the ABI unchanged (same entry, geometry and `certified_count` semantic) |
| D32 | The routing-weight multiply is a named host seam in the routed device chain (F6) | The released arithmetic multiplies the SwiGLU output by the routing weight BEFORE the w2 activation quantiser reads it, and no pinned routed kernel takes a weight operand. m17a therefore measures the device SwiGLU+quantise stage unweighted, and builds the combine operand on the host from the same device gate/up bf16 values with `mxfp4_ref::routed_activation_operand`; the combine is weighted but its activation operand is host-quantised by the reference's own rule. A weighted quantiser variant is the planned follow-up (new cubin, deliberate re-pin), not part of m17a |
| D33 | The routed combine operand is built on the device: the weighted SwiGLU+quantise variant closes the F6 host seam | `swiglu_mxfp8_quant_weighted_sm121` (own cubin, 13,000 B / `0e602daf...`, `ptx_membar_lines=0`) takes a per-row routing-weight operand and does the model's order on the device: asymmetric-clamped SwiGLU, one bf16 rounding of the product, then the activation ceil-rule quantise. m17a's stage 3 reads the router's `weights_d` in place, and stage 4's combine receives those device codes/scales through a host re-upload of the stage-3 readback (a byte copy, no host arithmetic on them), so no host multiply and no host quantise remain in the routed path; the class is `swiglu_ref::quantise_row_weighted` (which equals `mxfp4_ref::routed_activation_operand` composed with SwiGLU) and the real 576-wide operand is compared against `routed_activation_operand` directly. The unweighted epilogue stays pinned for m17b's shared expert; the six-kernel frozen m4 ABI array is untouched; window b33 measured it PASS at tokens 8 and 130 (35 checks each) |
| D34 | The serving tail's final norm is a standalone BF16 kernel, planned and pinned before any window | `kernels/rmsnorm_sm121.cu` reproduces the engine's final-norm arithmetic exactly (`sum(x^2)` ascending f32, `/cols`, `1/sqrt(mean+eps)`, `(x*inv)*w`, one BF16 RN-even rounding) with one thread per row and 128-bit BF16 loads; the host reference is the engine's own `linalg::rms_norm`, not a copy. Pinned at 31,480 B / `5f953bc0...` / `ptx_membar_lines=0` (baseline eleventh entry, driver `KernelId::RmsNorm`); m18 is measured (0 mismatched BF16 bytes at 8 and 130 tokens, odd 513 exact, two launches byte-identical) and c4c-norm wired it (b41 0/10,240 mismatches). Details: `docs/build/phase3-rmsnorm-tail.md` |
| D35 | The certified head's device reduction replaces the host block assembly, planned and pinned before any window | `lm_head_reduce_sm121` (36,120 B / `3fcd7ba5...`, `ptx_membar_lines=0`) is pinned as the twelfth baseline entry and the driver's `KernelId::LmHeadReduce`; m19 is measured (17/17 checks: block arrays, lowest-index tie-break over the four ranks, certified-winner refusal, `0u` screen sentinel) and the b44 window produced token 5 / 13.237006187 against the host assembly's 5 / 13.230665207. The host four-shard block assembly and the 4-byte token/status readback remain the fallback; the device cut-over is not wired, so `REQUIRED_KERNELS` stays nine and the chain stays at 1167 launches |
| D36 | The deferred sequence's position-causal CSA2 attention entry is pinned before any window | `flash_mla_csa2_seq_sm121` (38,080 B / `f6dfc738...`, `ptx_membar_lines=0`) is the thirteenth baseline entry / twelfth `KernelId` variant with its own launch ABI; the m20 harness measured it PASS 50/50 on window b50 (positions 0, 1, 2, 129..=137; position-0 out[] byte identity 65,536 bytes) and it stays unwired, so `REQUIRED_KERNELS` stays 9; window b48 is contaminated history, not a record |
| D37 | Criterion 1 stays OPEN: the FP8-native device path is stage-faithful but not token-parity with the f32 reference (superseded by D39: native arithmetic, numeric criterion) | b55-b62 measured the whole sequence path: alignment, pre-attention hidden, format-2 body, `wq_b`-vs-`mxfp8_ref` and the head reduce all pass at their own bars, yet position 0 predicts 320 where the reference predicts 201 (head input differs by 0.608) because the device attention path quantises activations (MXFP8, 32-element blocks, power-of-two ceil scales) while the fixture reference is f32 over dequantised weights. The first criterion-shaped 130+8 attempt (b62) aborted at 131/138 positions on a ~0.8 GB/position memory defect. No maths was changed; exact token parity needs either a precision-first activation path or a redefined criterion, and the hybrid-sequence branch stays unmerged until then |
| D38 | vLLM token parity is only meaningful on the same weights, and the deployed vLLM serves a different checkpoint | `/v1/models` on the cluster reports the `pqhaz/...-heretic` snapshot `12d58ad0...`, while the fixture and the device engine use the released `deepseek-ai/...` snapshot `1bae0e10...`; its greedy completions are degenerate and must not be treated as a parity target. Any future vLLM comparison must first record the served `root` and prove it equals the snapshot under test |
| D39 | The chain runs the checkpoint's native arithmetic; acceptance is numeric with tolerance, and vLLM is the behavioural oracle | Native arithmetic means the released forward's rule (model.py:27-30, 182-205: 32-element ue8m0-ceil activations, 32x32 ue8m0 / FP4-32 weights, f32 accumulate), not f32/BF16-class activations; token equality is retired as a criterion because generation is not deterministic, so acceptance is numeric/distributional at declared tolerances; vLLM is the behavioural oracle; Engram weights stay disk-offloaded |
| D40 | Serving-parity measurement protocol, metric set and provisional numeric bars, declared before the first diagnostic run (D41 freezes them) | The oracle is vLLM serving the released snapshot at TP=4, temperature 0, prompt_logprobs with max-logprobs 1000 and the served root recorded (D38); the device side is the full-vocabulary --dump-logits artefact on the same token ids; the comparison reports top-1/top-5/top-20 agreement, max/mean |delta logprob| over the oracle top-100 and KL over a fixed renormalised top-100 support, per position plus aggregates; provisional bars are top-1 >= 95% of positions, median top-20 recall >= 90%, median KL <= 0.02 nat and p95 <= 0.10 nat, max |delta| <= 1.0 nat per position (no position above 2.0); D41 must freeze the thresholds from the measured oracle self-consistency floor (at least 3x the floor) before any serving claim, and a bar changed after a gate run is a new decision |
| D42 | The engine KV/attention seam is a KV store trait (Option B), recorded but NOT implemented; an in-path diagnostic mirror is rejected | The serving KV can be claimed paged only when the engine's attention path reads and writes the paged pool as its backing store. Of the three boundaries (attention backend trait, KV store trait, engine-owned allocator handle), Option B is recorded: the engine defines a minimal store contract and keeps a host dense bit-exact default; the paged store owns allocator/sharing/eviction behind it; the scheduler owns the shared pool and page-reservation-aware admission; the HTTP driver owns request-to-sequence handles and fills only cache fields it can measure from store counters (tier and replay fields stay zero). Option A cannot produce the cache claims, Option C puts page admission in the engine before the scheduler has a capacity model. No code is approved by this decision and no branch exists: it is an engine-wide refactor to be argued against measured staging and parity numbers. The frozen 19-word attention entry stays frozen (Path A's 20th argument is a separate pin decision), a default-off diagnostic request-path mirror is explicitly rejected (it leaves the serving KV dense, hides its cost behind a flag, and makes the bookkeeping claim without the user-visible cost), and the load-bearing boundary is quoted verbatim in the entry: Option B does not fix the f32-engine vs fp4-device numeric gap, O(context) staging, HSI full-pool selection, the serial HTTP engine, or the host-resident paged pool. Evidence: docs/build/engine-kv-seam-decision.md |


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

**Corrected by D30 (2026-10-18).** The *evidence* below is the reference's own sharding rule and
stands. The *decision* is not the layout this tree implements: the only routed-expert GEMM that
exists and has run on the device reduces 576 intermediate values per rank padded to 640
(`kernels/b12x_mxfp4_moe_sm121.cu` with `kernels/m4_common_sm121.cuh:96-97`), and the M7 harness
feeds it as four rank-chained column slices of the 2304-wide intermediate
(`crates/sparkflash-kernels/examples/m7_layer0/stage3.rs:21` and `stage7.rs:749-781`). D30 adopts
that layout and records its storage and summation consequences. This entry's shared-expert half —
whole at 2304 and replicated on every rank — is unchanged and is what the device chain runs.

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

**Corrected by D30 (2026-10-18).** The layout half of this decision is superseded. D30 records the
serving layout for routed experts as TP=4 over the intermediate width, 576 per rank padded to 640,
because that is the kernel ABI the device has run; the MTP blocks use the same routed-expert code
path, and when their device path lands it uses the same ABI, so they are sliced and padded the same
way, not held as 32 whole experts per rank. The
functional decision in this entry (blocks 37-39, `k=5`, cap 0.10, the sigmoid on the confidence
logit, the noise token) is unchanged. "Whole experts sharded 32/rank" remains the reference's rule
(D2, [architecture/05-dspark.md](architecture/05-dspark.md) D3), not this tree's serving plan.

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

**Rail-2 address table (read-only measurement, 2026-10-18, not a gate run).** This entry recorded
only the head node's pair. The live listing (Lead, 2026-10-18) gives rail 1 / rail 2 pairs:
node0 `192.0.2.80`/`.85`, node1 `192.0.2.84`/`.88`, node2 `.64`/`.68`,
node3 `.59`/`.63`. Both rails are ACTIVE at 200000 Mb/s on the f1 functions; the deployed vLLM
reference selects only `rocep1s0f1` (single rail), while our gates target the dual-rail aggregate.
The full table, the device states and the NCCL environment with provenance are in
`docs/build/rdma-backend.md`; `docs/QUALIFICATION.md` section 2.1 now points there instead of saying
the addresses are unrecorded.

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
3. The three cluster gates now have real bodies: `crates/sparkflash-comm/tests/hardware_gates.rs`
   implements the three gates rather than panicking. They still have no measurement: none has run,
   because a run needs `SPARK_COMM_HW_PERMISSION=1`, the `rdma` feature, four cooperating processes and
   a Lead-approved window, and no frame has crossed a wire.
4. The one real All-Reduce instrument is merged and its backend is real: `rocenante-probe` is a
   binary target of `sparkflash-comm`, the doctor invokes it and refuses a loopback-labelled number by
   name, and the `rdma` backend now returns typed errors instead of `CommError::RdmaNotBuilt`. Its
   honest limitation is unchanged in kind: no frame has crossed a wire yet, so the probe can only
   produce a number in a permissioned four-node window, not on this host.
5. `sparkflash trace --scopes all --steps 45` prints a plan; it cannot capture, because the capture path
   is a hardware gate whose summary nothing produces.
6. No CUDA graph code exists: `cudaGraph` and `cuda_graph` appear in `crates/` only inside the gate's own
   parser and gate function.
7. Doctor's collective probes see one rank view, not the ring: check 3 hands every peer in `--nodes`
   to the bandwidth probe and scores the worst peer, but the All-Reduce probe measures only the running
   rank's own view and needs the full rank table in `SPARK_COMM_PEERS` because rank 0 binds its own
   address, so the four-rank ring's latency needs four concurrent doctor processes. Nothing has run.
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
and 20 us for milestone 3 (line 1336). The gate's own constant is the only encoding of 20 us
(`crates/sparkflash-comm/tests/hardware_gates.rs:112`). They are not in conflict once
their roles are separated, and this entry is the separation: 20 us at 10 KiB is the requirement, and the
preflight's 22 us is a readiness threshold, deliberately looser so that a node can pass preflight while
still failing the requirement. A run that reports 21 us has passed the preflight and failed the gate. The
10 KiB payload is the specification's requirement; the same gate also times 2 MiB, the spec's router
threshold, against a first-working-backend acceptance bar of max(1.5x the wire floor, 20 us), not a
third latency budget. Doctor encodes its own 22.0 (`crates/sparkflash-cli/src/doctor.rs:101`) and
this entry is why it stays there.

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
`crates/sparkflash-kernels/examples/m7_layer0/` (the split modules, concatenated after 954d7af) and
fails if a statement disappears or the harness stops citing a letter from its printed output; `crates/sparkflash-kernels/tests/mhc_determinism.rs`
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
- `crates/sparkflash-kernels/examples/m7_layer0/main.rs` passes the raw released tensors (the bias in the
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

## D24 - CSA attention attends the window/compressed union, and format 2 is reference-tested until a device window

**Evidence.** Task-9's 2B harness found the concrete Criterion-1 gap. The released model (and the
golden fixture computed by sparkflash_engine::forward::attention::attention_forward) builds the
attention candidate list as window_indices followed by the selected compressed_ids and softmaxes
the union with the sink in the denominator (sparse_attention, kernel.py:311-389). The
flash_mla_csa2_sm121 formats 0 and 1 are window-only: format 0 keeps a candidate only when
position - candidate < 128 (FP4), format 1 keeps a selected id only when it is a row index inside
the supplied engine window. At prefill position 129 the decoder layers 21-39 read the layer-20
ratio-1 compressed store and must attend compressed ids 0..1, which are outside the 128-token
window, so a window-only device attention cannot reproduce the golden logits at that position. The
reference's own reachability mask is reach = (start_pos + query + 1) / ratio; for a ratio-1 store
that is exactly candidate <= position, and the compressed ids are emitted in position order and
shifted into the shared window-first index space (attention_forward, compressed_indices).

**Decision.** window_kv_format = 2 (macro SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION in
kernels/flash_mla_csa2_sm121.cu) is the CSA union mode. Formats 0 and 1 keep their own exclusion
rules and code paths; the mode is a third explicit runtime value, the 19-parameter ABI is unchanged,
and the kernel entry poisons the output for any unknown value. Format 2 semantics: (1) the
window_rows engine-format rows are all attended, oldest first; (2) the selected entries are
absolute token positions decoded from kv_fp4/kv_scales, attended when candidate <= position and
excluded exactly otherwise; (3) score/weight order is window rows then the selected list as
supplied, duplicates attended once per entry, one f32 three-pass softmax with the sink added to the
denominator after the maximum, out rounded to bf16. A ratio-2 compressed source is passed as a
per-position expanded view (row p = store row p / ratio; selected compressed index c becomes token
id c * ratio), so the kernel needs no ratio argument. The internal union scores are bounded by
SPARK_M4_MLA_MAX_ATTENDED = 640 (512 candidates + 128 window), a compile-time constant with a
static_assert and a kernel guard; mla_ref::attention_with_sink_union is the CPU twin with the same
arithmetic order.

**Consequence.** The CPU reference tests prove the arithmetic and dispatch only. In particular,
mla_ref's format_two_union_equals_the_engine_reference_on_the_attended_multiset compares format 2
bit for bit with sparkflash_engine::forward::attention::sparse_attention over the concatenated
window ++ compressed reference KV, not with a second union definition: window rows are the
engine-format decoded rows, compressed rows are the store rows addressed by absolute token ids,
and the sink is the first denominator term in both. Its cases cover distinct out-of-window ids, an
id inside the window, duplicate entries, an id at the query position, a future id (reference
NO_INDEX), ratio-2 expanded views and a negative control showing that using compressed indices as
token ids double-counts an expanded store row. The 640-slot bound is exercised at exactly 640
(accepted) and at 641 (refused by the Rust twin; the kernel entry guard poisons it with NaN), and
the dispatch/capacity text is pinned in m4_abi. Format 2 has NOT been executed on a device and no
result here claims it reproduces the golden fixture; that claim waits for a Lead-sequenced device
window (the 2B harness then compares format 2 on the existing stage-2 positions plus a 130-token
position-129 case). Formats 0 and 1 device evidence from task-3/task-6 is untouched, and
m7_layer0.rs is not modified by this change. The 2B harness consumes format 2 through the existing
typed FlashMlaParams/flash_mla_params builder.

---

## D25 - The golden sequence fixture is the Phase 2 reference truth; 2B's device plumbing is wired and its comparison is still open

**Note 2026-10-19 (D39):** the fixture is the *simulation* reference (f32 over dequantised weights), not the checkpoint's native arithmetic; its token-first bar is retired and numeric/tolerance comparison against a native reference replaces it. The fixture remains a regression input.

Recorded 2026-10-18.

**Evidence.** The Phase 2A node run produced a real-checkpoint sequence fixture: the engine's CPU
reference forward over the released 40-layer checkpoint, one 130-token prompt (the deterministic
prompt for seed 7, its ids recorded) then 8 greedy decode steps. It is
`/home/user/phase2-evidence/sequence-130-released.json` on node `node0` and
`/home/user/phase2-evidence/devhost-copy-f72738b/sequence-130-released.json` on the dev host,
2,536,194 B, sha256
`e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308`, schema 1, `config_label`
`released`, compared positions `[129..137]`, prefill next token 491, decode next tokens
`[367, 201, 80, 3302, 201, 54, 201, 4341]`. Provenance: main `f72738b`, aarch64 binary sha256
`ed3838df5ca28722a19d6dda689d7b90c4dd67c8faf205cafcc3469dcb459ffa` (7,265,936 B), rustc 1.98.1 /
zig 0.17.0, `/usr/bin/time` wall 613.54 s, exit 0; [build/phase2-golden-fixture.md](build/phase2-golden-fixture.md)
records the full provenance, the evidence files and the caveats (the prefill/decode split is
inferred from the IO bursts, not measured; the `read_bytes` figure is warm-cache). This is the first
real-checkpoint sequence reference in the tree: 2A's committed CI fixture is a deterministic
tiny-config fixture that pins the schema and the comparator, not model evidence
([build/phase2-sequence-harness.md](build/phase2-sequence-harness.md) section 3).

**Decision.** That fixture is the Phase 2 reference truth for 2B/2C. A device output is compared
token-first (the prefill next token and all 8 decoded tokens, on the fixture's compared set only) and
then, where the fixture recorded logits, at the fixed 2e-2 absolute bar (`LOGITS_ABS_BAR`); the
comparison is never widened to older positions because the engine reference does not implement CED
(D7). The fixture is reference truth, not a test output: it must not be regenerated as a side effect
of a device change. If a device run disagrees, the device work is wrong or the gap is real, and the
fixture is not edited to make the run pass. A legitimate regeneration (a schema change or a
corrected reference) is a separate Lead decision that records a new sha256/size here and marks the
old fixture superseded. The CSA-union format-2 semantics remain D24's; this entry does not restate
them.

**Consequence.** The 2B device plumbing is merged as !63 (1ceeb54): the measured `window_rows`
rotation and the format-2/format-1 attention dispatch run in a device window, but the per-layer
forward chain that would produce tokens and logits is still absent. Round-2 adversarial review found
the first commit's rotation clock included verification/dispatch and its decoder rows were counted
from position 0; both were fixed before merge and the first-window numbers are superseded. The
corrected path then ran on the node in the 2026-10-18T18:32Z window: 11,764 `copy_dtod` calls moving
198,832,128 B in 0.025624 s of copy-only time (7,400.000 MiB/s), 360 dispatches with 0 non-finite and
320/320 identical repeats, and 9,926,144 B across 176 device buffers. It produced no token and no
logits delta because the forward chain is absent, so no device prefill/decode token comparison exists
for this milestone. Phase 2 exit criterion 1
([build/roadmap-to-serving.md](build/roadmap-to-serving.md) section 5.5.1) is therefore OPEN, as are
criteria 2-5. Format 2 has executed on devices (the isolated task-14 check and this plumbing run;
D24, !62 -> main d3dcc3f, pipeline 2763); this fixture is what the eventual token/logits comparison
will be measured against.

---

## D26 - The Milestone-4 ABI source-text lint is frozen, and kernel invariants move to compiled artifacts or runtime device assertions

Recorded 2026-10-18.

**Context.** `crates/sparkflash-kernels/src/m4_abi.rs` is the Milestone-4 ABI lint: it re-derives
geometry from `#define`s, re-parses `__global__` signatures, re-evaluates the kernels' static asserts
in Rust and checks trace-scope wrapping by reading kernel source text. Its own module doc records the
contract that makes its size a liability rather than a virtue: the scan is "explicitly NOT the proof",
the committed cubin baseline is, and the review history recorded there reaches round 36, where every
round was a real false accept or a real false rejection (the round-24 paragraph in
`crates/sparkflash-kernels/src/m4_abi.rs` records one: a `return`, a control keyword or a
non-returning call in the poll's own argument list was invisible to both checks; the last recorded
round is 36, an attributed `try`/`catch` body both checks accepted while refusing the plain
spelling). Each fix added
another parser feature for a property a text scan cannot establish: that the compiled kernel is what
the source appears to say. Measured 2026-10-18: 13,854 lines at main `2245a4e`, 13,874 with the
freeze notice this entry records, 13,893 after the `m7_layer0` split replaced the two
`include_str!`s of the single example file with one concatenation constant (the split moved the
pinned statements into per-stage modules; the pins scan the same text, concatenated), and 13,906
after the task-11 follow-up hardening (the corrected round-33 claim, crate-relative baseline paths,
the exact `m7_layer0/main.rs` witness and the four-token check over the concatenation). The
quarantine split of 2026-10-19 (task-27, `docs/build/m4-abi-split.md`) then moved the scanners into
pinned submodules without editing them: 13,906 lines before the split, 13,924 after it as the
aggregate over the root file and the two submodules, with the 64 test ids and their
`cargo test -- --list` output byte-identical.

**Decision.** Freeze the file. (1) No new source-text scanner, parser or lint for a new kernel
invariant is added to it. A new invariant is enforced by compiling the kernel and inspecting the
artifact - `crates/sparkflash-kernels/scripts/check-kernel-baseline.sh` recompiles the Milestone-4
kernels and compares cubin bytes, cubin sha256 and PTX `membar` line count against
`crates/sparkflash-kernels/baseline/kernel_baseline.txt`, and the `cuda-abi` CI job runs it - or by a
runtime device assertion in a harness (the `m7_layer0` device stages), where the shipped binary is
what is measured. (2) Defects in the scanners already present are still fixed, with the mutation proof
the module requires: the freeze is on length, not on correctness. (3) The committed line count is
pinned by `ci/check-m4-abi-size.sh` and enforced by the `m4-abi-freeze` job in `.gitlab-ci.yml`: the
pin is the aggregate over the `m4_abi` family (`m4_abi.rs`, `m4_abi/scanner.rs`,
`m4_abi/acquire_guard.rs`), so any net line-count change, growth or shrink, and any change to the
pinned file set, fails the pipeline; a commit that changes the length must change `PINNED_LINES` in
the same diff. That diff is the deliberate, reviewable act; a scanner cannot arrive as a side effect of an
unrelated edit.

**Consequence.** A proposed kernel invariant that can only be enforced by reading source text must
instead become a compile-time assertion in the kernel or its header (compiled by the build-script nvcc
probe and pinned by the cubin baseline), a PTX/SASS baseline delta, or a device assertion in a
harness. A new parser feature in `m4_abi.rs` is now a policy exception that needs a new Lead
decision, not a maintenance change. Nothing in this entry alters the kernels or the scanners'
verdicts: the existing tests keep gating every pipeline.

**Verification.** Notice plus a CI script, no behaviour change: the pre-split `wc -l` was 13,906 lines
(one file) and the post-split aggregate is 13,924 (`m4_abi.rs` 2,979 + `m4_abi/scanner.rs` 4,313 +
`m4_abi/acquire_guard.rs` 6,632); `sh ci/check-m4-abi-size.sh` prints the aggregate pin and fails on
any change; `python3 ci/verify-m4-abi-split.py` regenerates the three files from `0639413` and fails
on any edit outside the glue and the two allow-listed self-scan adapters it names; and
`cargo test -p sparkflash-kernels --lib -- --list | sort` is byte-identical before and after the
split (338 lines, 64 `m4_abi` tests). No GPU, no checkpoint and no cluster were involved.

---

## D27 - The capability boundary has no vision channel, and an image request is refused before generation

Recorded 2026-10-18.

**Evidence.** The image path is 263 `vision.*`/`aligner.*` tensors plus the three image delimiters
(`image_start`, `image_end`, `image_newline`) - the 266 names D18 records as absent from
`TensorPlan::language_model`, 970,536,960 stored bytes. None of them is loaded, and no forward
module consumes them: there is no vision tower, no aligner and no token-grid derivation anywhere in
`crates/sparkflash-engine`. Before this entry `crates/sparkflash-api/src/multimodal.rs` nevertheless
carried `DEFAULT_IMAGE_GRID = 32` and `default_image_tokens()`, `prompt.rs` charged 121 visual
tokens per image part, and the api capability set had a `vision: bool` that `Capabilities::ALL` set
true. The consequence was that a chat request carrying an image was accepted: it was answered from
its text tokens with a usage block counting visual tokens no engine produced, and `/v1/models`
advertised `multimodal: true` for the `ALL` engine.

**Decision.** The capability boundary is the engine's own answer, and no engine in this workspace
can answer yes to vision:

- `Capabilities` has no `vision` field. `/v1/models` reports `multimodal: false` and
  `image_url: false` as constants, because the channel is absent rather than a driver's choice.
- `engine::Capability::Vision` and `engine::MissingCapability` are the typed vocabulary of a
  capability a request needs and an engine lacks. `InferenceEngine::validate_request` defaults to
  refusing a request whose `images` list is non-empty with `MissingCapability::vision()`; all four
  request routes call it before generation and map it to HTTP 400 `unsupported_capability`. The
  message names the missing channel and the 263 unwired tensors.
- `EngineRequest::image_tokens` is gone. A request carries `images: Vec<ImagePart>` - the parsed
  parts, in request order - never a visual-token count nothing measured. `default_image_tokens()`
  and `DEFAULT_IMAGE_GRID` are deleted; `count_image_tokens(grid_h, grid_w)` remains as the
  published tiling arithmetic for a front end that derives a real grid, and nothing in the request
  path calls it. The payload is carried all the way to the seam - inline parts hold the decoded
  payload bytes, not only a length - so a future vision forward can decode pixels and derive token
  ids without re-parsing the request.
- The known protocol image content `type`s are explicit `ContentPart` variants: `image_url`,
  `input_image` and Anthropic `image` (a `source` object, also accepted on the chat route). The
  variants are the typed half, not the whole boundary. `prompt::find_image_carrier` scans the raw
  request body and refuses any carrier the typed parsers did not record: any object key naming an
  image (`image_url`, `input_image`, `image`, `computer_screenshot`, and the conservative
  `file_id`), any `type` value containing `image`, `screenshot` or `vision`
  case-insensitively anywhere in the body, and any value under a carrier key (the key alone is
  the carrier: `{"image_url": null}` is refused, and a `data:` URL is not special). A text field
  that merely mentions the word "image" (or contains a `data:` URL as text) is not a carrier and
  stays accepted; a genuinely non-image part such as `input_audio` still takes the fail-open
  `Other` path. The scan is deliberately conservative: a carrier spelling no typed parser knows is
  refused with the same HTTP 400 `unsupported_capability`, naming the JSON path and spelling
  found, rather than being answered from text.
- `push_object` walks both `content` and `output` for Responses items, so an image nested under a
  `function_call_output`/`computer_call_output` `output` array becomes the same `ImagePart` as a
  top-level one (review round 2; see the corrections below). It remains the typed half: the raw
  scan covers containers it does not walk (for example `code_interpreter_call.outputs` and the
  documented `computer_screenshot` object under `output`).
- `ReferenceDriver::generate` refuses an image part with the same code as defense in depth, for
  callers that drive the driver directly instead of through the router.

**Consequence.** A multimodal request can no longer reach a token count: `/v1/chat/completions`
(both an `image_url` part, a bare Anthropic `image` block and the case-variant
`{"type":"Image"}` spelling - the first two used to bypass the check, see below),
`/v1/completions` (an `<image>` placeholder in the prompt, and any carrier key such as a stray
`file_id`), `/v1/responses` (`input_image`; one nested under a `function_call_output`
`output` array; the documented `computer_call_output` screenshot as an object or inside an
`output` array; a `file_id`-only screenshot; a `code_interpreter_call.outputs` image; and
any future image-family type or carrier key at any depth) and `/v1/messages` (an Anthropic
`image` block and any untyped carrier) all answer 400 with code `unsupported_capability` and a
message containing `vision` and `263`; a recording engine in
`crates/sparkflash-api/tests/router.rs` shows `generate` is never entered, so the dummy-grid path is
unreachable from a request. `crates/sparkflash-engine/tests/tensor_plan.rs` pins the 263 excluded
names and the exact 970,536,960 bytes, and `the_plan_bytes_are_exact_not_approximately_right` pins
the total without the index. Tests: `engine.rs`'s `the_capability_set_has_no_vision_channel`,
`a_request_with_an_image_part_is_refused_with_the_missing_capability` and
`a_scripted_request_with_an_image_part_charges_no_fabricated_visual_tokens`; `prompt.rs`'s
`image_url_parts_are_decoded_and_recorded_without_charging_visual_tokens`,
`chat_route_anthropic_image_block_is_recorded_not_ignored`,
`a_responses_function_call_output_nested_image_is_recorded`,
`a_genuinely_non_image_unknown_part_is_still_accepted_and_ignored`,
`the_image_carrier_detector_finds_arbitrary_nested_shapes` and
`the_image_carrier_detector_does_not_scan_arbitrary_text`; `router.rs`'s
`an_image_request_is_refused_with_the_missing_capability_and_never_reaches_the_engine`,
`a_nested_responses_image_cannot_bypass_the_refusal`,
`every_route_refuses_an_image_request_with_the_same_typed_error`,
`the_documented_screenshot_shapes_are_refused_without_entering_the_engine`,
`every_route_refuses_an_untyped_image_carrier_with_the_same_typed_error`,
`a_text_only_body_that_mentions_images_is_still_served`,
`a_non_image_unknown_chat_part_is_still_accepted_and_ignored` and
`models_reports_the_engines_own_capabilities`; `serve.rs`'s
`an_image_request_is_refused_with_the_typed_missing_capability` and
`the_driver_declares_no_channel_it_does_not_serve`. No device run is involved or claimed. When the
vision path is wired, the channel returns to `Capabilities`, `validate_request` is overridden with
the image forward (which consumes `EngineRequest::images` and produces the visual-token ids) and
this entry is superseded rather than edited.

**Corrected after review (2026-10-18).** The first revision of this entry claimed all four routes
refused every image shape; it did not. `/v1/chat/completions` deserialised content parts into a
typed `ContentPart` enum whose `#[serde(other)]` arm (`ContentPart::Other`, "accepted but ignored")
swallowed an Anthropic image block sent to the chat route: `{"type":"image","source":{...}}` produced
no `ImagePart`, so `image_parts` stayed 0, `validate_request` returned `Ok` and generation ran -
HTTP 200, `image_tokens: 0`, and a prompt-token count for its text part alone (the critic's
reproduction reported `prompt_tokens: 11`; the body pinned in `tests/router.rs` has the text `hi`,
which costs 8 prompt tokens: 4 message overhead + 1 text + 3 reply primer). The fix is the explicit
`ContentPart::Image` variant above plus the `push_image_source` helper shared with the
`/v1/messages` and `/v1/responses` paths; `every_route_refuses_an_image_request_with_the_same_typed_error`
now carries that exact body and the recording-engine test carries it too, and the catch-all policy is
stated on `ContentPart::Other` and pinned by
`a_genuinely_non_image_unknown_part_is_still_accepted_and_ignored`. The earlier claim was a defect in
this record as well as in the code, and it is corrected here rather than quietly reworded.

**Corrected after review, round 2 (2026-10-18).** The chat-route fix was not the last silent
fallback. `push_object`'s default arm recursed only into `content` and `text`, so
`{"input":[{"type":"function_call_output","call_id":"c1","output":[{"type":"input_image","image_url":"data:image/png;base64,aGVsbG8="}]}]}`
on `/v1/responses` never produced an `ImagePart`: the request was answered HTTP 200 from its text
with `image_tokens: 0`, and `generate` was entered. The default arm now walks `content` and
`output`; `every_route_refuses_an_image_request_with_the_same_typed_error` and
`a_nested_responses_image_cannot_bypass_the_refusal` pin that exact body (400
`unsupported_capability`, with the recording engine showing `generate` was never entered), and
`a_responses_function_call_output_nested_image_is_recorded` pins the parser half. `ImagePart::Data`
also grew the decoded payload bytes (it previously held only `byte_len`), so the "payload is carried
through the seam" claim in this entry, `engine.rs` and `server.rs` is now true rather than a
description of a length.

**Corrected after review, round 3 (2026-10-18).** Round 2 fixed the `output` *array* but the
documented `computer_call_output` shape nests its image as an object under `output`, and its
type is `computer_screenshot`, not `input_image`: that body, the same shape inside an
`output` array, and the `file_id`-only screenshot each reached generation and were answered
HTTP 200 from the text. The reviewer's point was that another spelling would follow, because the fix
so far was an enumeration. This entry now relies on a structural scan instead:
`prompt::find_image_carrier` walks every object and array in the raw body and refuses (a) any
carrier key anywhere - `image_url`, `input_image`, `image`, `computer_screenshot`, and
`file_id` (conservative: a file reference cannot be shown to be a non-image file) - (b) any
`type` value containing `image`, `screenshot` or `vision` case-insensitively, and (c)
**any value** under a carrier key (the key alone is the carrier: `{"image_url": null}` is refused,
and a `data:` URL is not special). The typed parsers and `MissingCapability::vision()` still handle
the known shapes; the scan runs in every handler before generation and returns the same HTTP 400
`unsupported_capability` with a message naming the JSON path and spelling. The pinned tests are
`router.rs`'s `the_documented_screenshot_shapes_are_refused_without_entering_the_engine` (object
and array screenshot, `file_id`-only, `code_interpreter_call.outputs` image,
`{"type":"Image"}`, and an arbitrary-depth future type),
`every_route_refuses_an_untyped_image_carrier_with_the_same_typed_error` (the same class on all
four routes) and `a_text_only_body_that_mentions_images_is_still_served`, plus `prompt.rs`'s
`the_image_carrier_detector_finds_arbitrary_nested_shapes` and
`the_image_carrier_detector_does_not_scan_arbitrary_text` and, as my own attempt to defeat the
detector before review, `five_independent_bypass_attacks_are_all_refused` (a carrier four
containers deep under an unknown item type, an upper-case key, a carrier under a nested
`function_call_output` output object, a `file_id`/screenshot pair, and a `type` spelled
with a JSON unicode escape so only the decoded value carries the word). That last test is the
adversarial half of the pin, not a claim that five shapes exhaust the space. The scan is structural
over keys and `type` values, not a byte scan: image bytes under an unrelated key with no carrier
name and no image-family `type` word are indistinguishable from text without decoding payloads the
router has no parser for, and are not refused; every protocol image shape this router serves has one
or the other, which is what the tests above pin. The `Image`-spelling end-to-end claim from round 1
is now true rather than a description of the opposite behaviour: that spelling is refused.

**Corrected after review, round 3 (false positives, then a false negative).** The structural scan
was too broad in the other direction: it refused pure-text requests whose schema merely uses the
names. A tool definition
`{"tools":[{"function":{"parameters":{"properties":{"image":{"type":"string"}}}}}]}` and a
`response_format` JSON Schema with an `image` property answered HTTP 400 although they
carry no media. A first fix exempted whole schema/data subtrees by key name; the reviewer rejected
that: a real carrier under `arguments`, `metadata` or `items` would then return
`None` and be served from text - the same silent-fallback class, with a different key. The scan
now exempts only the **immediate child keys of the JSON Schema name maps** in
`prompt.rs::SCHEMA_NAME_MAPS` (`properties`, `patternProperties`, `$defs`,
`definitions`), and only when the mapped value is itself a JSON Schema (a bool, an empty
object, or an object whose keys are all in `prompt.rs::SCHEMA_KEYWORDS`): those keys are names,
not payload, and each child schema value is still walked. A **non-schema value under such a name is
payload**, so the name is carrier-matched and the value falls through to the normal scan; the
reviewer's second finding was that the name-only exemption left
`{"properties":{"image":"data:image/png;base64,..."}}` invisible and served with generation
entered. Every other key is scanned, so a nonstandard `arguments` object carrying
`image_url` (the real OpenAI shape is a JSON string, which no string scan touches) and an
`image` under `metadata` are refused fail-closed, not served. Pinned by `router.rs`'s
`schema_name_maps_do_not_trip_the_refusal_but_data_carriers_still_do` (tool and
`response_format` schemas served with generation entered; an `arguments` carrier, a
carrier under an arbitrary non-exempt `payload`, a scalar under `properties.image` and a
nested `properties...image_url` object all 400 with `generate` not entered) and
`prompt.rs`'s `the_image_carrier_detector_exempts_schema_names_only` (valid
bool/empty/keyword-only schemas exempt, child schema values walked, `arguments`/`metadata`
carriers refused) and `a_non_schema_value_under_a_schema_name_is_still_refused` (a string, a
number or an object with an unknown key under `properties`/`patternProperties`/`definitions`
is refused; bool, empty and keyword-only schemas stay `None`). The existing bypass matrix is
unchanged and still passes; the five attacks were restored to their original `parameters`
carriers, which the narrowed scan refuses again.

**Corrected after review, round 3 (third false positive: the keyword list).** The schema-shape test
is only as good as its keyword list: a legitimate schema
`{"properties":{"image":{"type":"object","minProperties":1}}}`, a draft-07 `dependencies`
schema, and `$defs.image_url` with `minProperties` were classified as payload because
`minProperties`, `maxProperties` and `dependencies` were missing from
`prompt.rs::SCHEMA_KEYWORDS`, so an image-named property under them was refused (HTTP 400,
`generate` not entered). The list now carries those three plus the OpenAPI annotation keys
common tool schemas use (`example`, `xml`, `externalDocs`; `deprecated`,
`readOnly`, `writeOnly` and `discriminator` were already there), alongside the
already-present `dependentRequired`, `dependentSchemas`, `propertyNames`,
`contains`/`minContains`/`maxContains`, `unevaluatedProperties`/`unevaluatedItems`,
`prefixItems`/`additionalItems`/`additionalProperties`. The unknown-key policy is
deliberate and unchanged: a mapped object with a key outside the list cannot be proven to be a JSON
Schema, so a carrier name under it is payload and is refused fail-closed - serving it would reopen
the silent-fallback class. Pinned in `prompt.rs`'s
`standard_schema_keywords_outside_the_first_list_do_not_trip_the_refusal` (minProperties,
maxProperties, draft-07 dependencies, `$defs.image_url` + minProperties and the OpenAPI
annotations stay `None`; an unknown-keyword object under `properties.image` is refused)
and in `router.rs`'s `schema_name_maps_do_not_trip_the_refusal_but_data_carriers_still_do`
(those three schema shapes served with generation entered; an unknown-keyword carrier 400 with
`generate` not entered, beside the existing fail-closed pins). The existing bypass matrix is
unchanged.

**Why removing the field is not a breaking change, and why the advertised constants cannot lie.**
`Capabilities` is a workspace-internal type: no crate in the workspace is published, nothing
serialises the struct (the wire shape is written explicitly in `server.rs::list_models`), and the
only consumers are `crates/sparkflash-cli/src/serve.rs` - which answers the conservative default -
and the api's own tests. The HTTP contract is unchanged: `/v1/models` still carries `multimodal`
and `image_url`; they are now constants `false`. Before this entry the served `ReferenceDriver`
already answered `false` for both, and the only `true` came from
`MockEngine::with_capabilities(Capabilities::ALL)` in tests, i.e. a claim no served engine made, so
no client that worked against the shipped server changes behaviour. Hardcoding `false` cannot lie
today because the advertisement and the refusal are both unconditional (including every shape in
the corrections above; the raw-body scan is what makes the refusal independent of the typed
parser's spelling list), and there is exactly one reviewed wiring point for the
future: re-add `vision: bool` to `Capabilities`, derive `multimodal`/`image_url` from it in
`server.rs::list_models` (the comment there marks the spot), and override `validate_request` with
the image forward. The payload half is already in place - `EngineRequest::images` carries the parsed
parts, including the decoded bytes of inline images - so the wiring change needs the 263 tensors, the
visual-token derivation and the capability flag, not a new request field; until it lands, no code
path can print `true`.

---

## D28 - The reference engine is `Send + Sync`, and `serve`'s docs said otherwise

Recorded 2026-10-18.

**Evidence.** `crates/sparkflash-cli/src/serve.rs`'s module docs said, until this entry, that
`sparkflash_engine::forward::InferenceEngine` "is neither" `Send` nor `Sync` because it holds
`Box<dyn WeightSource>` and `WeightSource` "declares no `Send` bound". That was true when the
driver was written; !60 (`f72738b`, "engine: parallel reference pass (no arithmetic change)") added
the bound - `crates/sparkflash-engine/src/weights.rs:106` is now
`pub trait WeightSource: Send + Sync` - because the MoE block reads independent experts from several
threads. Every remaining field of `forward::InferenceEngine` (`ModelConfig`, `TensorPlan`, `usize`,
`Sampling`) is thread-safe, so the engine is `Send + Sync` as a type.

**Decision.** Correct the docs to the truth and pin it with a compile-time assertion rather than a
comment: `serve.rs` carries
`const _: () = { const fn assert_send_sync<T: Send + Sync>() {} assert_send_sync::<...InferenceEngine>(); };`,
so a future field that breaks the claim stops the crate compiling. The single-owner engine thread
stays, but its justification is now the engine's shape, not thread-safety: the reference pass runs
one sequence at a time, and each request applies its sampler by consuming the engine
(`with_sampling`), so one mutable owner is the honest shape. Building an engine per request - the
other way to get `Send + Sync` - would re-open and re-verify a checkpoint per request.

**Consequence.** No behaviour change and no new dependency; the driver still talks to the engine
thread over channels and requests still serialise. The module docs state both facts (the type is
`Send + Sync`; the passes serialise by design), and `the_driver_declares_no_channel_it_does_not_serve`
pins the capability half while the `const` assertion pins the thread-safety half. No device run is
involved or claimed.

## D29 - The ibverbs backend is real and dual-rail capable; the deployed vLLM reference is single rail

Recorded 2026-10-18.

**Evidence.** crates/sparkflash-comm/src/rdma.rs and its abi/verbs/qpinfo/dataplane submodules
implement the RoCEnante transport against a run-time dlopen of libibverbs: 16 entry points are
resolved with dlsym and the three inline ones (ibv_post_send, ibv_post_recv, ibv_poll_cq) come from
a bundled C shim compiled by build.rs against the real rdma-core header, because libibverbs does
not export them; a build without the header refuses at load naming the header rather than opening a
device it cannot post to. The hand-written repr(C) mirror is asserted field by field against
rdma-core 39's verbs.h (send WR is the full 128-byte object including the trailing bind_mw/tso
union, port attr 52 bytes, QP attr 144 bytes; ibv_qp and ibv_mr are labelled read-only prefixes)
and the feature compiles only with the rdma flag; the default build gains no dependency and every
non-rdma path is unchanged. The control-plane star exchanges per-(rail, peer) QP numbers, GIDs,
staging MR addresses, remote keys and the staging geometry (frame cap, slot size, rail count,
refusing any disagreement); the data plane posts RDMA WRITE_WITH_IMM over one RC QP per (rail,
peer), validates completion opcodes and immediate byte order, and accounts posted bytes per rail. The deployed vLLM reference was measured to be single rail (docker exec vllm_node
env): NCCL_IB_HCA=rocep1s0f1, NCCL_IB_GID_INDEX=3, NCCL_SOCKET_IFNAME=enp1s0f1np1, with line 10 of
/home/user/phase2-evidence/window-2026-10-18/vllm-launch-recipe.sh overriding the launch
script's twin-pair environment. The second twin (roceP2p1s0f1) is ACTIVE at 200000 Mb/s but unused
by the deployed recipe. On the live nodes the specification's f0 rails are PORT_DOWN; the resolver
validates ACTIVE and >= 200 Gb/s and selects the f1 twins on distinct cards.

**Decision.** The ibverbs backend is real (not a skeleton) and must be dual-rail capable, but no
document may cite the deployed vLLM configuration as dual-rail evidence: it is single rail as
measured above. Dual rail (aggregate >= 180 Gbps, QUALIFICATION 3.3) is our requirement. The
hardware gates run and report both modes, and the counter gate refuses to claim dual-rail evidence
from a single-rail run. docs/build/rdma-backend.md is the authoritative description of the backend,
its ABI subset, rail layout, verified scope and honest limits; QUALIFICATION.md and the roadmap
must stop calling the backend a skeleton or the gates panic stubs, and the stale D17 finding 3/4
text is superseded.

**Consequence.** At record time no hardware gate has been run on the cluster: the backend, the
three gate bodies and doctor check 3 compile and are hardware-free-tested, and the first four-node
run is the next Lead-sequenced step. The host-memory staging buffer (not GPUNetIO device slots), the
1024-byte active MTU, driver 580.173.02 and the single-rail deployed reference are recorded gaps,
not verified qualifications. The three inline wrappers require rdma-core headers at build time; the
hardware-free CI host exercises the stub compile path, while the stub-specific `load()` refusal
needs a host that has libibverbs but no headers, and the real inline path needs the headers.

---

## D30 - Routed experts are sliced along the intermediate width, because that is the layout the device has verified

Recorded 2026-10-18.

**Question.** D2 decided the routed experts are sharded as *whole experts* on the reference's rule:
96 per rank at world 4, each at the full 2304 intermediate width, in MXFP4. The only routed-expert
device GEMM in the tree, `kernels/b12x_mxfp4_moe_sm121.cu`, is written to the opposite layout (the
CPU engine's routed experts are host arithmetic and are not that kernel): it reduces
`SPARK_M4_MOE_WIDTH = 576` values per rank padded to `SPARK_M4_MOE_WIDTH_PADDED = 640`
(`kernels/m4_common_sm121.cuh:96-97`), its weight and scale row strides are static-asserted against
640, and D14's "padded loader contract (A8)" requires a pre-padded 640-wide operand. The M7 harness
exercises that layout: `MOE_RANKS = MOE_INTERMEDIATE / MOE_WIDTH = 2304 / 576 = 4`
(`crates/sparkflash-kernels/examples/m7_layer0/stage3.rs:21`), and it feeds the kernel four
rank-chained column slices of the intermediate (`stage3.rs:480-570`, `stage7.rs:749-781`; the w2
source slice is bytes 0..288 of each 1152-byte global row and scales 0..18 of each 72, per
[device-path-design.md](build/device-path-design.md) section 6). Both decisions cannot be true.

**Decision.** The routed-expert serving layout is **TP=4 over the intermediate (K) dimension**: every
rank holds every routed expert's 576-value intermediate slice, padded to 640 in the kernel layout
(per expert: `w1`/`w3`/`w2` packed FP4 rows on the kernel's 320-byte stride and ue8m0 scale rows
on 20 bytes; per activated (token, slot): a 640-value E4M3 activation row and its 20-byte scale row).
The reference's whole-expert partitioning stays the *reference's* rule (convert.py:129-134, D2's
evidence) and is **not** what this tree implements. The shared expert is not part of the slice: it
stays whole at 2304 and replicated on every rank, which is D2's second half and is what the device
chain runs (`stage7.rs` K13-K15 on `MOE_INTERMEDIATE`).

**Why this direction.** Three reasons, in order of weight:

1. *What has run wins over what a reference file says should run.* The 576 -> 640 layout is the one
   this tree and the device have run: stage 3 ran the combine as four rank launches on 2026-10-17
   with 8, 32 and 64 resident experts (20,480/20,480 bf16-exact, max ULP 0; 0.821 / 0.847 / 0.850 ms
   per token over the four chained launches;
   [roadmap-to-serving.md](build/roadmap-to-serving.md) sections 1.1 and 12.1,
   [phase1-boundary-table.md](build/phase1-boundary-table.md) section 12), and the 2026-10-18 chained
   layer-0 replay ran K9-K12 as four rank launches at 5,120/5,120 exact each
   ([phase1-boundary-table.md](build/phase1-boundary-table.md) section 12, row 7).
2. *Rewriting the kernel is not a docs change.* Making the kernel reduce K = 2304 is a new kernel
   ABI (strides, padding, activation operand shape, the loader contract D14 pins), not an edit to a
   comment; it needs its own decision and its own device verification. This entry does not authorise
   it.
3. *The choice costs no capacity.* Both layouts keep one quarter of the expert bytes per rank; the
   difference is which quarter - whole experts at full width, or all experts at quarter width - so
   picking the verified kernel does not change the memory plan, only its padding.

**Storage consequence, with arithmetic.** The routed-expert bytes (backbone + MTP) total 295.997 GB
([architecture/00-overview.md](architecture/00-overview.md) section 2.4:
288,777,830,400 + 7,219,445,760 = 295,997,276,160 B, packed FP4 plus scale grids).

- Reference EP, whole experts: 295,997,276,160 / 4 = 73,999,319,040 B = **74.0 GB/rank** (GB decimal,
  as everywhere in this entry and in the corpus byte tables). This is the figure roadmap section 7.5
  used.
- TP=4 with the 576 -> 640 padding: 73,999,319,040 x 640 / 576 = **82,221,465,600 B = 82.2 GB/rank**,
  i.e. **+8,222,146,560 B = +8.22 GB/rank (+11.1%)**. The padding is 64 of every 640 values in the
  weight rows, the weight scale rows and the activation rows; it is masked to zero numerically, but it
  is still allocated, read and (for the weights) stored.
- Device total, with its units stated once: `cuDeviceTotalMem_v2` reports **124,610 MiB**, which is
  **130.66 GB decimal** and **121.7 GiB** (124,610 x 1,048,576 = 130,663,055,360 B). This entry and
  roadmap section 7.5 use GB decimal; the corpus fact tables report MiB. Do not mix the two without
  saying which one is meant.
- With the Engram tables on local NVMe as roadmap section 7.5 plans: resident weights become 82.2
  (routed + MTP experts) + 2.0 (dense shards) + 3.43 (replicated) = 87.63 GB, **about 87.6 GB/rank**
  (was 79.4), leaving 130.66 - 87.63 = 43.03 GB, **about 43 GB**, for KV, activations and the Engram
  cache. It still fits.
- With the Engram tables resident it does **not** fit: 82.2 + 50.7 (Engram) + 2.0 + 3.43 = 138.33 GB,
  **about 138.3 GB/rank**, against 130.66 GB - over by 138.33 - 130.66 = 7.67 GB, about **7.7 GB**.
  The EP figure was 130.1 GB with about 0.5 GB to spare, so the padding turns "Engram on NVMe is
  advisable" into "Engram on NVMe is required for this layout".

**The MTP experts follow the same decision, and that reverses D6's layout half.** The three MTP
blocks use the same routed-expert code path (the reference engine's expert scratch is sized for both
expert counts), and their device path, when it lands, uses the same kernel ABI, so their experts are
TP=4 slices padded to 640, not 32 whole experts per rank; D6 carries a "Corrected by D30" note and
its functional decision stands. The arithmetic: the MTP routed experts are 7,219,445,760 B of
the 295,997,276,160 B total (architecture/00-overview.md section 2.4), so 1,804,861,440 B =
1.80 GB/rank unpadded and 1,804,861,440 x 640 / 576 = 2,005,401,600 B = 2.01 GB/rank padded - the
+0.20 GB/rank inside the 82.2. MTP is Phase 5 and off the base forward, so this is design
arithmetic, not a measured MTP footprint.

**Numerics.** Both layouts issue the same single `[tokens, 5120]` bf16 partial-sum all-reduce per
layer (roadmap section 7.2), but the summation order differs: EP sums each rank's whole-expert
contributions over disjoint expert sets, TP sums each rank's K-slice partials of the same experts.
The device runs above are bit-exact against the CPU reference *on the same 576-wide sliced bytes*;
they confirm the sliced layout's arithmetic, not EP/TP equivalence. Cross-rank accumulation order is
exactly the kind of cross-boundary equality D21 says must not be assumed, and R3 in roadmap section
7.3 is the criterion that covers it.

**What this entry does not claim.** No device run was made for this entry (docs-only). It does not
add the Milestone-5 loader that must emit the 640-wide layout (D14's A8 contract), and it does not
change any kernel, the shared expert, the dense shards or the Engram row sharding. The routed
gate/up FP4 GEMM is still host arithmetic in the M7 stages (`stage3.rs` prints that boundary and
says no device kernel in the six-cubin set reduces K = 5120 against packed FP4 weights); the
device-verified part of the routed path is the down/combine projection, and the 576-wide slice is
what those verified launches consume.

**Corpus check, stated for the Lead.** The corpus supports EP=4 as the *reference's* rule
(convert.py:129-134; [architecture/03-moe.md](architecture/03-moe.md) section 3.1; D2's evidence). It
does not support EP=4 as *this tree's implemented layout*: every routed-expert **device-path**
artifact here (the kernel constants, D14's A8 contract, the M7 rank chaining,
`mxfp4_ref::MOE_WIDTH`) is 576 -> 640. It is not the serving layout: the CPU reference engine reads
whole 2304-wide experts
([crates/sparkflash-engine/src/forward/moe.rs](../crates/sparkflash-engine/src/forward/moe.rs);
[engine/01-forward-order.md](engine/01-forward-order.md) section 4 states the reference's rule),
and is untouched by this entry. This records the divergence instead of claiming the reference and the
implementation agree; if reference-exact sharding is required, that is a new kernel ABI and a new
entry.

---

## D31 - The certified LM head counts every thread's prunes, and the pin is re-baselined

Recorded 2026-10-18.

**Question.** `kernels/cert_lm_head_sm121.cu` publishes `certified_count` so a caller can tell how
many entries the certificate pruned. The kernel's own contract (`:65`) calls it the "total pruned
entries, atomically accumulated". In windows b21 and b23 the concentrated certificate case
certified 4,899 of 129,280 entries but published a count of 34, and the host twin predicted the
exact undercount per rank: 12, 6, 3, 13. The real-token control certified 0 and published 0, so
phase 1 (which only ever ran the non-pruning case) never saw it.

**Finding, as measured.** The count was accumulated from lane 0 alone. `pruned` is a per-thread
counter (`:100` declared, `:122` incremented); the old `atomicAdd(certified_count, pruned)` sat
inside `if (lane == 0u)` (`:148`, `:160`), and the block reduction at `:149-157` reduces only
`running_max`/`running_argmax`. The PTX confirms it: the thread-local pruned register was the
value the single `atom.global.add.u32` added, so the published number was the sum over the 32
blocks of lane 0's prunes - 34 at the concentrated seam, exactly as the twin's per-entry loop over
`lane == 0` entries predicts. It is a kernel-side contract violation, not host count assembly: the
host compares each rank's published count to that rank's own flags independently. Certificate
soundness (bounds, exact entries, pruning decisions, block reductions, argmax, determinism) was
intact in b21/b23; the control PASS stands.

**Decision.** Fix the kernel and treat it as a deliberate re-baseline, because the certified
pruning path is what makes the head viable in serving and a wrong count would corrupt whatever
accounting uses it. The minimal fix keeps the existing structure: the atomic moved out of the
lane-0 block to `if (pruned > 0u) atomicAdd(certified_count, pruned);`, so every thread adds its
own prunes and threads with nothing pruned skip the atomic (the all-exact path is unchanged). The
count is a sum of integers, so accumulation order does not matter and the value is deterministic
across runs; the m16 harness already checks two-launch byte identity.

**Re-pin.** The fixed cubin is 14,888 B, sha256
`7394b1290a3210a78ba12e6fc5414b0f4c84d1cdaf8edbcfa63b25dfcdf00493`, 0 PTX membar lines, built
reproducibly with the pinned CUDA 13.0.3 toolkit (nvcc V13.0.88, `-std=c++17 -arch=sm_121a
-cubin -I kernels`); the PTX still has exactly one count atomic, now guarded only by `pruned > 0`
(the `tid.x != 0` skip branches to that guard), so every lane reaches it. `kernel_baseline.txt:16`,
the m16 PIN constants, the m16 page and serving-readiness 2.13 carry the new size and sha. The
phase-1 window record's cubin list keeps `a833a336...` because it is a historical window record and
is not rewritten, as do the b21/b23 undercount records in the m16 page.

**The ABI does not change.** The entry signature, the geometry macros and the semantic of
`certified_count` (total pruned entries) are the same. Only the reduction of an internal counter
changed: the artifact now delivers the documented semantic instead of a lane-0-only subset of it,
so the Milestone-4 ABI freeze (D26) is not touched and no caller needs to change.

**Re-window, and the PASS that follows.** Window b24 measured the count fix working (published
4,899 = 4,899 flags, per rank 1062/1205/1215/1417) but exposed a harness gate that compared the
synthetic case against the real token's reference argmax; that gate was fixed and tested, and window
b25 passed both cases on the fixed artifact with `certified_count == flags` per rank and the same
bound/certificate/argmax/determinism checks - the first recorded device PASS of the certified pruning
path (m16 stage, documented in `build/phase3-lm-head.md`). b21/b23 remain the pre-fix
`a833a336...` evidence. No other pinned kernel has this pattern: a tree-wide grep of
`kernels/*.cu` finds per-thread counters with an atomic only here; the other `atomicAdd` uses are
barrier/reduction counters (`mega_mhc_sm121.cu:235`), not published counts.

## D32 - The routing-weight multiply is a named host seam in the routed device chain (F6)

**Question.** The released MoE block applies the routing weight to the SwiGLU output before the w2
activation quantiser reads it (model.py:841-851: the activation is cast to the activation dtype, then
the quantiser reads the tensor; the weight is inside the quantiser's input). None of the four released
routed kernels takes a routing-weight operand: `moe_router_sm121` returns the weights,
`routed_gate_up_sm121` takes activations/weights/expert ids only, `swiglu_mxfp8_quant_sm121` takes
gate/up/limit only, and `b12x_mxfp4_moe_sm121` takes a pre-quantised activation operand and
expert ids only. The engine's host path therefore multiplies the weight between SwiGLU and the w2
quantiser, and that is the seam m17a exercises.

**Decision.** Record F6 as an explicit seam instead of hiding or approximating it. The harness runs
the device SwiGLU+quantise stage **unweighted** and measures it bit-exact against the reference; the
combine's activation operand is built on the host from the **same device gate/up bf16 values**:
host SwiGLU over those bytes, then `mxfp4_ref::routed_activation_operand(activation, routing_weight)`,
which is the reference rule including its bf16 cast. The values that cross the seam are printed by
the harness: the device gate/up bf16 `[tokens*6][640]` per rank and projection, the f32 routing
weight per selected expert per token, and the quantised operand `[tokens*6][640]` E4M3 plus
`[tokens*6][20]` ue8m0 per rank. The combine is weighted, but its activation operand is
host-quantised by the reference's own rule. The alternative - weight-multiplying the already-quantised
stage-3 codes - would quantise twice and is not the model's order, so it is not done.

**Evidence and scope.** `docs/build/phase3-moe-routed.md` states the class and the seam before any
number; `crates/sparkflash-kernels/examples/m17_moe_a.rs` prints the crossed shapes and keeps
"fully-device weighted routed block" out of its claims. The shared expert and shared-add epilogue
are m17b and are not covered. No kernel is changed by this entry and no cubin moves; the four cubins
are the pins already in `baseline/kernel_baseline.txt` (router 37,384 B / `5992cef9...`, gate/up
9,752 B / `115200d2...`, SwiGLU quantiser 21,776 B / `4dbef63b...`, combine 10,048 B /
`36a15a4f...`). The m17a window b27 PASS is recorded in `docs/build/phase3-moe-routed.md`
(binary `cc1a1d01...`, tokens 8 and 130, 30 checks each); the batch-16 hardening adds a
padding-prefill check and a staged-tensor-name check to that harness, measured in window b30
(31 checks each at tokens 8 and 130).

**Consequence.** A later chain may accept the host seam if it stays honest, or adopt a weighted
SwiGLU+quantise variant. That variant is the planned follow-up: the same kernel source with an
optional per-expert weight operand, a new cubin and a deliberate re-pin with its own entry - exactly
the re-pin pattern used for the certified LM head (m16's D31 landed as batch 14; this entry is D32) - and it is the only way to remove this
seam from the device path. This entry deliberately does not claim a fully-device weighted routed
block, per-token or per-layer text, or a host-time saving.

## D33 - The routed combine operand is built on the device: the weighted SwiGLU+quantise variant closes the F6 host seam

**Question.** D32 recorded the routing-weight multiply as a host seam: the model multiplies the
SwiGLU activation by the slot's routing weight **before** the w2 activation quantiser reads it, and
none of the four pinned routed kernels took a weight operand. The seam was honest but it kept one
host multiply and one host quantise inside the device chain, and c4b would have to wire it.

**Decision.** Close the seam with a second SwiGLU+quantise entry point, shipped as its own device
source, own cubin and deliberate baseline re-pin:
`kernels/swiglu_mxfp8_quant_weighted_sm121.cu`, entry
`swiglu_mxfp8_quant_weighted_sm121(gate, up, weights, out_fp8, out_scales, rows, cols, limit,
trace_ring, layer_id, step_seq)` - eleven argument words, the unweighted epilogue's ten plus the
per-row `float *weights` operand. Its arithmetic is the model's order, element by element:

1. `activation = swiglu(gate, up, limit)` with the reference's asymmetric clamp (gate upper side
   only, up both sides), the same expression the unweighted epilogue uses;
2. `weighted = bf16(activation * weights[row])` - one round-to-nearest-even bf16 rounding of the
   product, exactly what `mxfp4_ref::routed_activation_operand` does before it quantises;
3. the activation ceil-rule quantiser (`act_quant_fp8_row`'s rule: 1e-4 amax floor, smallest power of
   two at or above amax/448, E4M3 encode) over the same 32-value blocks, so the output is the
   `[rows][cols]` E4M3 + `[rows][cols/32]` ue8m0 operand `b12x_mxfp4_moe_sm121` consumes, padding
   included (the padded gate/up bytes are zero, so they quantise to zero codes).

The host reference is `swiglu_ref::quantise_row_weighted`, which a test pins as the composition
`routed_activation_operand(&swiglu_row, weight)`; a second test proves the bf16 rounding step is
observable in the intermediate (the unrounded product is not bf16-representable), and a source-text
test pins the multiply-before-round-before-quantise order and the asymmetric clamp. The trace scope
follows the shape as in the unweighted entry (cols = 640 -> `TRACE_SCOPE_MOE_MXFP4`), and the helper
that computes a block's scale and code is a separately named function so the combined probe can
compile both translation units.

**Evidence and scope.** The new cubin is compiled by the pinned toolkit (nvcc V13.0.88,
cuda_13.0.r13.0/compiler.36424714_0, `-arch=sm_121a`), 13,000 B, sha256
`0e602daf6f2a51f027f3a924aaa503fada7db14610b94114439c659be4150cfe`,
`ptx_membar_lines=0`. It is a deliberate re-pin: `baseline/kernel_baseline.txt` now has **ten**
entries (its header and the pin tests are updated with it), `build.rs`'s `PROBED_KERNELS` table
probes ten sources and the combined probe compiles all ten, and `cuda_driver::KernelId` gained
`SwigluMxfp8QuantWeighted` (the driver's vocabulary, not the frozen `m4_abi` array - that stays the
six Milestone 4 kernels). The existing unweighted epilogue, its cubin and its pin are unchanged:
the shared expert (m17b) still uses it.

m17a's stage 3 is rebuilt on the weighted entry: the kernel's weight operand is the router's
`weights_d` buffer, read in place (the router output is not copied or re-uploaded for the operand),
and stage 4's combine takes the stage-3 device codes/scales through a host re-upload of the stage-3
readback: a byte copy of the device write, with no host arithmetic applied to it. The harness prints the
class before the numbers, compares the device operand bit-exact against
`quantise_row_weighted` (codes and scales, all rows), additionally compares the real 576-wide
columns bit-exact against `mxfp4_ref::routed_activation_operand` (the model operand the old host
seam built), checks two-launch determinism, and then compares the device combine chain bit-exact
against `mxfp4_ref::moe_combine` over the **same device operand bytes**. The new binary is
`m17a_weighted_6443d5d0_aarch64` (4,142,000 B, sha256
`6443d5d0729650bdf571a3907c95ef5c169296b25a55050b42bd7174b16828cd`, built from
`/home/user/wt/p3-m17`). **Window b33 measured it: PASS at both token counts** - tokens 8
`exit=0` (05:59:29Z -> 05:59:41Z) and tokens 130 `exit=0` (05:59:41Z -> 06:02:24Z), **35 checks each**,
four pinned cubins verified by size and sha256 before allocation (window
`window-b33-20261019T055723Z.log`; run logs `runs33/m17a-weighted-tokens8.log` and
`runs33/m17a-weighted-tokens130.log`, reproduced verbatim in `docs/build/phase3-moe-routed.md`). The
window number is b33 in the Lead's global numbering (b30 hardened m17a, b31 c4a-1, b32 c4a-2 first
attempt, b33 c4a-2 corrected PASS plus this run); an earlier draft of this entry called it b31. One
verification detail moved after b33 without changing it: the b33 binary built the real-width
comparison operand with `swiglu_ref::quantise_row_weighted` (the crate's tests pin it equal to
`routed_activation_operand` composed with SwiGLU), and the review follow-up makes the harness call
`mxfp4_ref::routed_activation_operand` itself. Arithmetic and every device launch are unchanged, so
the PASS and the tie between the binary and the pre-fix source commit both stand. The
class is the same bit-exact class as D32's host path, now on the device; it is still not a chain,
token, per-layer or end-to-end claim, the activation and residual are the declared
released-`embed.weight` stand-ins, and the shared expert is not covered.

**Consequence.** c4b can wire the routed epilogue with no routing-weight host seam: the chain hands
the router's weight buffer to the weighted epilogue and its output to the combine. The unweighted
epilogue remains the shared expert's quantiser. If a later window finds a mismatch, the class is a
finding against this kernel, not a tolerance to widen, and the host path of D32 stays the fallback
until it is fixed.

---

## D34 — The serving tail's final norm gets a standalone BF16 kernel, planned and pinned before any window

**Decision.** Add \`kernels/rmsnorm_sm121.cu\`, the model's final RMSNorm as its own stage, so c4c can
put the last host op (the engine's \`final_norm\` seam at \`crates/sparkflash-engine/src/forward/mod.rs:602\`,
\`linalg::rms_norm\` at \`crates/sparkflash-engine/src/linalg.rs:318\`) on the device. The entry is
\`rmsnorm_sm121(x_bf16, weight_bf16, out_bf16, rows, cols, eps, trace_ring, layer_id, step_seq)\`:
BF16 \`[rows][cols]\` in, BF16 \`[rows][cols]\` out (the head's ABI), the released BF16 \`[5120]\`
weight, and the config's \`rms_norm_eps\` cast to f32 exactly as the engine casts it.

**The arithmetic is the engine's, op for op.** Per row, ascending columns: \`sum(x*x)\` as f32 with
one rounding per multiply and add; \`mean = sum / cols\`; \`inv = 1/sqrt(mean + eps)\`; \`(x * inv) * w\`;
one BF16 round-to-nearest-even on the output. The kernel uses \`__fmul_rn\`, \`__fadd_rn\`,
\`__fsqrt_rn\` and \`__fdiv_rn\` so nvcc cannot contract the square-and-add into an FMA (one rounding
where the engine rounds twice) or replace sqrt-then-reciprocal with an approximate rsqrt. One
thread owns one whole row because a block/warp reduction sums in a different order; the row-serial
order is the engine's, and the tail's row count is the token count (1 in serving, 8 and 130 in the
window). 128-bit BF16 loads (\`ld.global.nc.v4.u32\`) cover \`cols % 8 == 0\` (5120 released); any other
width takes the scalar path and is tested at 513 columns. The kernel emits **no trace event**: all
sixteen protocol scope bits are assigned and none describes a tail norm, so the trace words are
carried for ABI uniformity only and the pin's \`ptx_membar_lines=0\` is expected.

**Pin and CPU-side evidence (all before any window).** nvcc V13.0.88 / CUDA 13.0.3 / sm_121a:
\`rmsnorm_sm121\` cubin **31,480 B**, sha256
\`5f953bc0fb781afe3f9b42703b19b37f89ef45b74a4ad607938df2e2c3448579\`, \`ptx_membar_lines=0\`;
PTX inspection shows 0 \`fma.rn\`, 2 \`div.rn.f32\`, 1 \`sqrt.rn.f32\`, 0 \`rsqrt\`, and the
\`ld.global.nc.v4.u32\` vector loads. It is the **eleventh** entry in
\`crates/sparkflash-kernels/baseline/kernel_baseline.txt:13-23\`; \`build.rs\`'s \`PROBED_KERNELS\` probes
eleven sources and the combined probe still compiles every kernel in one translation unit;
\`cuda_driver::KernelId\` gained \`RmsNorm\` (ten variants); the frozen six-kernel \`m4_abi\` array is
untouched. The host reference (\`crates/sparkflash-kernels/src/rmsnorm_ref.rs\`) calls the engine's
own \`sparkflash_engine::linalg::rms_norm\` over the bf16-decoded bytes and rounds with
\`m4_quant::bf16_bits\` - it is not a second copy of the formula, so an engine change surfaces as a
mismatch; a source pin (a review ratchet) fails if the twin is ever re-implemented. The harness
\`crates/sparkflash-kernels/examples/m18_final_norm.rs\` reads the released \`norm.weight\` and
\`config.json\` and checks: released geometry; real width bit-exact (zero mismatched BF16 codes) and
two-launch determinism; an epsilon-control launch at eps = 1.0 (bit-exact to the reference at 1.0,
different from the released-epsilon output); and a synthetic \`[3, 513]\` odd-width case bit-exact.

**Status: measured and wired.** m18 measured 0 mismatched BF16 bytes at 8 and 130 tokens, the odd 513 case exact, two launches byte-identical; c4c-norm wired it (b41 0/10,240 BF16 mismatches). The window record is in the phase3-rmsnorm-tail section 7. The branch also
carries the m17c review's doc-only wording correction: the stage-4 routed-combine operand handoff
is a host re-upload of the stage-3 readback (a byte copy of the device write, no host arithmetic on
it), not a device-to-device handoff and not "only for the comparison twins".

**Not claimed.** Not the served forward pass, not c4c engine integration (engine-dev wires the
launch and the fallback), not the LM head, not a token/chain/per-layer claim, not host-time saved,
not a performance or bandwidth claim. The activation rows are released \`embed.weight\` rows, a
declared stand-in for the collapsed tail hidden. \`sparkflash-device::cubins::REQUIRED_KERNELS\`
requires this kernel as one of the nine chain kernels, so the launch and fallback are covered by the catalog.

**Consequence.** c4c can replace the \`final_norm\` host seam with one launch plus a declared host
fallback, and the comparison class is bit-exact, not a tolerance: if the window finds a mismatch it
is a finding against this kernel, not a widened bar.

---

## D35 — The certified head's device reduction replaces the host block assembly, planned and pinned before any window

**Decision.** Add `kernels/lm_head_reduce_sm121.cu`, the last host assembly step between the four
`cert_lm_head_sm121` shard launches and the token, as a device kernel. Today the chain copies each
rank's 32,320 stored logits, 32,320 certified flags, 32 block maxima and 32 block argmax entries to
the host and folds 128 blocks with `consider_head_block`. The entry is
`lm_head_reduce_sm121(logits_r0..r3, certified_r0..r3, token, value, block_max, block_argmax,
top5_value, top5_index, status, certified_count, trace_ring, layer_id, step_seq)`: four separate
per-rank logit pointers and four certified pointers (no segmented buffer, no offset/stride pair to
confuse), and the 4-byte token is the declared host seam, with status/count beside it for the
claim/refusal contract.

**The m16 rules, exactly.** Blocks are the screen's layout, rank-major: global block
`r * 32 + b` covers rank `r`'s entries `[b * 1024, min((b + 1) * 1024, 32,320))`, so each rank's
last block has 576 entries and there are 128 blocks. The four shard ranges are **uniform**
(4 x 32,320); the non-uniform numbers sometimes quoted for the concentrated case
(1,062 / 1,205 / 1,215 / 1,417) are its **certified-flag counts per rank**, not shard sizes - the
kernel header and the page say so explicitly so nobody re-imports them as ranges. Block answers use
strict greater-than with ties to the lowest global entry; the 129,280-way answer is the
lowest-index tie-break over the 128 block answers in ascending global order; the top-5 is value
descending then index ascending. **No arithmetic is performed on a logit** - the kernel only
compares and moves f32 bit patterns, so it cannot introduce a value delta: the rounding class is
inherited from `cert_lm_head_sm121`, the only arithmetic upstream. Determinism follows from fixed
scan order, one launch and one block (8 warps x 16 chunks, thread 0 folds the answers in ascending
order); the pin's `ptx_membar_lines=0` is expected. The kernel emits no trace event: all sixteen
scope bits are assigned and none describes the head reduction, so the trace words are carried for
ABI uniformity only.

**The refusal contract (approved status bits).** `0x1 WINNER_CERTIFIED`: the argmax entry is
certified, so its stored value is a bound and the token must not be claimed (refused, not asserted;
the token is written for diagnostics only). `0x2 TOP5_CLAIMED`: set only when
`certified_count == 0` - the device-side half of m16's `claims_top5`; the host still requires a
reference before it may claim the set. `0x4 ANY_CERTIFIED`: informational. A certified entry is
never silently valued as exact; on a sound certificate `WINNER_CERTIFIED` cannot happen, because a
certified entry satisfies `bound < running maximum`, so seeing it is a finding.

**Pin and CPU-side evidence (all before any window).** nvcc V13.0.88 / CUDA 13.0.3 / sm_121a:
`lm_head_reduce_sm121` cubin **36,120 B**, sha256
`3fcd7ba5e9871d01804e84825c107bf7741efac7b16982c77aced29b726b2849`, `ptx_membar_lines=0`,
**measured by window b44: PASS, 17 of 17 checks**;
PTX inspection shows 0 `membar`, 0 `fma.rn`, 5 `ld.global`, 10 `shfl.sync`, 0 `atom`. The first
build (37,152 B / `6cff3661...`) **FAILED window b42 at 7 of 14 checks**: it scanned each block's
local prefix instead of its `base + local` range (every block of a rank re-read the rank's first
1,024 entries; count 3,501 vs 4,899, winner 65,025 vs 33,865, synthetic winner 777 vs 4,242) and its
top-5 insertion could store the same entry twice (`[7, 7]`). The fixed source indexed `base + local`
for the logits and flags and refused a duplicate top-5 index. The second build (36,120 B /
`509b933e...`) then **FAILED window b43 at 5 of 14 checks**: it wrote the winning global entry as
`rank * 32,320 + local` without the block base, so block 1 reported block 0's entry (873 vs 1,897),
the concentrated winner was 32,841 instead of 33,865 (equal stored value, wrong entry), and a
certified winner's flag was looked up at rank 0's local index (146) instead of the global entry
(4,242), leaving status 0x4 instead of 0x5. The third build (this pin) uses `rank * 32,320 + base +
local` for the written entry; both b42 and b43 are FAILURE records (verbatim in the page's section
8), not passes, and this pin is the artifact for b44. It is the
**twelfth** entry in `crates/sparkflash-kernels/baseline/kernel_baseline.txt:13-24`;
`build.rs`'s `PROBED_KERNELS` probes twelve sources and the combined probe still compiles every
kernel in one translation unit; `cuda_driver::KernelId` gained `LmHeadReduce` (eleven variants);
the frozen six-kernel `m4_abi` array is untouched. **(D36 adds a thirteenth entry and a twelfth
variant; the manifest is now `kernel_baseline.txt:13-25`, `PROBED_KERNELS` probes thirteen, and the
reduce remains the twelfth entry.)** The typed launch declaration
(`crates/sparkflash-kernels/src/lm_head_reduce_launch.rs`, `ARG_WORDS = 19`, one block of 256,
grid `(1,1,1)`) pins the kernel's parameter names/types/qualifiers against the source with a
mutation test. The host reference (`crates/sparkflash-kernels/src/lm_head_reduce_ref.rs`) is the
chain's assembly as a pure function (`assemble` plus `host_assembly_token`), with tests for the m16
layout, the 576-entry partial block, brute-force argmax parity, cross-rank ties, top-5 order, a
certified winner's refusal, the concentrated case's 4,899 flags with an exact winner, the all-exact
control and validation refusals. The harness
`crates/sparkflash-kernels/examples/m19_lm_head_reduce.rs` pins both cubins before allocating,
runs the four screen launches and two reduce launches per case, and compares the reduce with the
host assembly bit for bit: token and stored value, all 128 block maxima/argmax entries against both
the reference and the screen's published arrays remapped rank-major, status/count, the top-5
(claimed on the all-exact control and set-compared with the phase-1 reference logits; NOT claimed on
the concentrated certificate), the concentrated token 33,865, launch determinism, a synthetic
full-width tie resolving to the lowest index, and a synthetic certified winner asserting the
refusal.

**Status: measured by window b44 - PASS (17 of 17 checks), not yet wired.** b42 failed at 7 of 14
checks (missing read base, duplicate top-5 entry) and b43 at 5 of 14 (missing write base; wrong flag
lookup for a non-block-0 winner); both failure records are verbatim in
`docs/build/phase3-lm-head-reduce.md` section 8, with the fixes. The b44 PASS is verbatim in the same
section: on the real-token control the reduce token is 5 with value 13.237006187, byte-equal to the
host assembly (reference argmax 5); all 128 block maxima/argmax entries match both the screen's
published arrays and the host reference; the all-exact top-5 set equals the reference engine's; the
concentrated certificate keeps 4,899 flags and the exact winner 33,865 while NOT claiming the top-5;
two launches are byte-identical in both real cases; and the synthetic tie, nonzero-block offset,
certified-bound-tie refusal (status 0x5) and rank-2 certified-winner (status 0x5) cases pass. The
kernel is not wired into the engine on this branch: engine-dev's cut-over is the next change, with
the host assembly kept as the fallback. The branch is based on
`origin/main` @ `d95fa65` (batch 22, merged); the b40 RMSNorm record (`13a4147`) is a separate
docs batch and is not stacked under this branch, and the merge order is c4c-norm first, then this
reduce.

**Not claimed.** Not the served forward pass, not the c4c engine wiring (engine-dev wires the launch
and the host fallback at the cut-over, and bumps `sparkflash-device::cubins::REQUIRED_KERNELS` from
nine to ten only then), not the head screen, not a token/chain/per-layer claim, not host-time saved,
not a performance or bandwidth claim. The real-token control's hidden is the declared phase-1
artifact read by path; the concentrated and synthetic cases are declared synthetic operands and no
token is claimed from the concentrated certificate (the preserved argmax 33,865 is not a
released-model token claim).

**Consequence.** c4c can replace the host block assembly with one launch and read back a 4-byte
token plus the claim/refusal status, with the host assembly of D32/D34 kept as the declared
fallback; the comparison class is bit-exact, not a tolerance, so a window mismatch is a finding
against this kernel rather than a widened bar.

## D36 — The deferred sequence's position-causal CSA2 attention entry (flash_mla_csa2_seq_sm121), pinned before any window

**Decision.** Add `kernels/flash_mla_csa2_seq_sm121.cu` as a separate entry for the extended
`start_pos > 0` sequence path, leaving `kernels/flash_mla_csa2_sm121.cu`, its 60,792 B /
`5bfee4c7...` pin and the b31/b33/b39/b40 evidence untouched. The sequence entry is the base
format-2 union body with one semantic change: indexer stage 1 is **position-causal**. Blocks whose
first token is past the query publish `block_scores = -FLT_MAX`, `block_best = 0`; a reachable
block scans only slots with `token <= position`; the strict-greater tie rule and the five-step
warp reduction order are unchanged. The attention arithmetic (window rows oldest-first, compressed
selected ids attended only when `candidate <= position`, sink in the denominator after the max,
one f32 three-pass softmax, bf16 out) is the base kernel's, copied not re-derived.

**Why it is needed, stated honestly.** The b45 sequence fail (`body_launches=40 expected=120
positions=3 device_position_0_body=40 other_position_bodies=0`; log
`window-b45-20261019T090623Z.log`, binary `sparkflash_m3c_c4c_review_seq2` /
`5af95810...`, exit 1 at 2026-10-19T09:10:45Z) is the c4a-2 **call-site guard** in
`crates/sparkflash-cli/src/attention_device.rs::attention()`, not a math gap: m8's released
position-129 fixture (128 window rows, 65 compressed ids) already passed its device-vs-host twin.
The genuine new requirement is the causal indexer, because a candidate selection built from
`block_best` must not see future tokens. The separate entry also gives the p0 path a byte-identity
ratchet instead of reusing an artifact whose provenance is pinned to other windows.

**ABI and bookkeeping.** Same 19-word argument list as the base entry, pinned by
`crates/sparkflash-kernels/src/flash_mla_csa2_seq_launch.rs` against the source (`PARAMS`,
`declaration_matches`, mutation test) with grid `(64, 1, 1)` x `256` threads and the stage-2
`blockIdx.y == 0` guard. The entry has its own `cuda_driver::KernelId` variant
(`FlashMlaCsa2Seq`), loaded through `Cuda::load_kernel`; `REQUIRED_KERNELS` stays 9 and the
frozen six-kernel `m4_abi::KERNELS` array (D26) is untouched. The baseline manifest is now
**13 entries / 12 `KernelId` variants** (this is the 13th; the reduce of D35 is the 12th), and
`PROBED_KERNELS`, both baseline scripts, `baseline.rs` and `tests/kernel_baseline.rs` agree on
13. `flash_mla_csa2_sm121` still deliberately has no `KernelId`.

**Indexer split contract.** Inside one launch, the two query heads of an index head split the HSI
pool (even head -> even blocks, odd head -> odd blocks), and with 64 heads the 32 index-head pairs
write the same `block_scores`/`block_best` arrays - a documented race that stage 2 does not read.
Any consumer, including the candidate-selection step and m20, must use a deterministic launch; m20
checks the pool at grid `(2, 1, 1)` (head pair 0), the configuration m7's stage-2 self-check uses.

**Pin and CPU-side evidence (all before any window).** nvcc V13.0.88 / CUDA 13.0.3 / sm_121a:
`flash_mla_csa2_seq_sm121` cubin **38,080 B**, sha256
`f6dfc738a7ad867adf040384f194812337301aa888cc486d4a60923772ba28c3`,
`ptx_membar_lines=0`; PTX 1,673 lines, 0 `membar`, 61 `fma.rn`, 56 `ld.global`, 40
`shfl.sync`, 0 `atom`. CPU-side: new `mla_ref::indexer_block_scores` (position-causal stage-1
reference for one index head, kernel reduction order, empty blocks `(-f32::MAX, 0)`) with tests
for the mask at positions 0 and 9 and the lowest-token tie rule. The m20 harness
(`examples/m20_csa2_sequence.rs`) is the per-position twin: 50 checks over synthetic declared
operands at positions `0, 1, 2, 129..=137`, comparing attention to
`mla_ref::attention_with_sink_union` at the 2B bar, the indexer to the causal reference (tokens
exact, scores at the bar), determinism, the `p+2` selected-id exclusion, and the position-0
new-vs-old byte identity. It was built and staged (aarch64 binary sha256 `c2b67d85...`, cubins as above) and **window b50
measured it cleanly: PASS, 50 of 50 checks** at pin 38,080 B / `f6dfc738...` - union attention at positions
0, 1, 2 and 129..=137 within the 2B bar (max abs delta 1.95e-3 / 4.88e-4), causal indexer worst
score delta 0.0 with every best token exact, determinism, the p+2 selected-id exclusion, and the
**position 0 new-vs-old out[] byte identity (65,536 bytes identical)**. The window record is quoted
verbatim in `docs/build/phase3-csa2-sequence.md` section 8; b50 log
`window-b50-20261019T094959Z.log` sha256 `531c6ec7...`, report 17,879 B sha256 `d95b6292...`
(`runs50/m20-csa2-sequence.log` sha256 `ced4a399...`). Window b48 printed the same checks but is
**contaminated and superseded** - engine-dev started an overlapping window (b49) and two GPU clients
raced the vLLM stop/restart loop - so b48 is not the record and is not a rescue; it is retained as
history in the page's section 8.1.

**Consequence.** Engine-dev can extend the c4a-2 call site per position using this pinned entry.
The position-0 ratchet measured clean **at format 2**, so the old entry does **not** need to be kept
at position 0 for format-2 calls; the sequence entry NaN-refuses formats 0 and 1, so a position-0
call in the released window-only format still needs the base entry. The measured fallback remains
available if a later codegen change breaks the identity (measured, never silent mixing). `REQUIRED_KERNELS` is bumped only when the wiring is measured. Nothing here
claims the served forward pass, the chain, a token, timing, the top-512 selection, rope, or the b45
position-0 token mismatch, which is engine-dev's first-step state issue.


## D37 — Criterion 1 stays OPEN: the FP8-native device path is stage-faithful but not token-parity with the f32 reference

**Superseded 2026-10-19 by D39.** The choice below was decided by the user: native arithmetic stays and the criterion is redefined numerically; the token-parity bar is retired.

**Decision.** Criterion 1 ([device-forward-chain.md](build/device-forward-chain.md) section 4.3) is
**not** claimed and stays OPEN. The hybrid-sequence branch is **not merged**. No maths was changed to chase the gap; the choice between a precision-first path and a redefined criterion belongs to the user.

**Decided 2026-10-19 (D39).** Native arithmetic stays. The precision-first family is rejected because computing trained-quantised weights at higher precision is a different model, and the criterion becomes numeric/distributional at declared tolerances (no token equality).

**What was measured (windows b55-b62, 2026-10-19; all binaries contain the unreviewed cut-over
c9fa90f and everything is labelled sequence diagnostics on an unreviewed build).** The device chain
is faithful at every separable stage: the fed token and the pre-attention hidden are exact (4/5,120
bf16 mismatches, max_abs 2.83e-7), the format-2 attention body matches on identical staged operands
(35/18/13 of 40 layers zero-mismatch at positions 0/1/2, max_abs <= 7.8e-3), `wq_a`/`wkv` differ by
about 3e-3, `wq_b` is **exactly** `mxfp8_ref` on the launch bytes (0/32,768, max_abs 0.0), and the
head reduce equals the host four-shard assembly at every position with `block_max_equal=true`
(forcing the host assembly still returns 320/16/16). Nevertheless position 0 predicts **320** where
the reference predicts **201** (logit max 15.73 vs 19.85), and the head input differs by **0.608**.

**Attribution.** The remaining gap enters at the attention projections' activation quantisation: the
`wq_b` input latent already differs by 5.19e-2 and the raw q by 2.94e-1 against an expected bf16
quantisation delta of 6.18e-2 (b58/b59), while the kernel reproduces `mxfp8_ref` exactly (b60). The
device path therefore computes MXFP8 (32-element blocks, power-of-two ceil scales) while the fixture
reference computes f32 over dequantised weights. Exact 130+8 token parity is not attainable on the
current arithmetic without (a) a precision-first activation path (higher-precision activations for
the attention projections and the router, at a throughput and memory cost) or (b) a redefined
criterion (a same-arithmetic reference or a tolerance-based acceptance). Both are user decisions.

**Now measured, not just suspected.** The first criterion-shaped run b62 was **aborted at 131/138
positions** because host memory grew about 0.8 GB per position (RSS 121 GB); b66/b67 then closed that
defect (RSS flat at ~18.87 GB over 131 positions) and produced the measured verdict:

- the fixture is valid: the live f32 reference reproduces all nine fixture tokens exactly
  (`reference vs fixture: positions=9 all_equal=true`, wall_ms 614382 / 592786), so this is a device
  divergence;
- **criterion 1 fails at the first compared position**: position 129's device top-1 is 8060
  (value 19.415479660) where the fixture requires 491; b67 reaches position 130 in 1,177,425 ms
  (~9 s per position, projected full 1833 s);
- the **decode steps cannot run at all** yet: the run aborts with `c4a-2 chain abort: hybrid
  sequence: sequence embedding: no staged embed row for token 8060`, because the embedding staging
  holds only the prompt ids plus the reference decode tokens. Closing that seam (an on-demand host
  row read for the fed token, which keeps the declared host embedding seam) is the next wiring task;
- the stage-aligned twin places the numeric origin where b58-b60 said: layer 0 is small
  (post-attention 3.12e-2, host FFN Y 1.99e-2) and the drift accumulates across the 40 layers; b57's
  `residual first divergence: position=0 layer=0` was a stage-offset artifact and is superseded.

**Measured end to end (b68, 2026-10-19).** The on-demand embed-row fallback closed the decode seam,
so the first complete criterion-shaped 130+8 run finished **all 138 positions** with flat memory
(RSS 18,928,152 KiB; 9.1-9.6 s per position; `chain_ms=1238987`). Its criterion result:
`logits 129: position 129: 128528 of 129280 compared logits outside the 0.02 bar (max |difference|
15.009581 at 5986, argmax 8060)`, and the nine device tokens are **8060, 8060, 1168, 50673, 303,
531, 6525, 5719, 14171** against the fixture's **491, 367, 201, 80, 3302, 201, 54, 201, 4341** -
**0 of 9**. The same window's router-all refinement found the discrete mechanism: the first real
expert-set flip is **layer 3** (`scores_max_abs=1.472694e-2`), 36 of 40 layers differ at position 0,
and the BF16 control proves it is **the attention-path drift in the router input**, not BF16 staging
and not the router kernel (`bf16_control_same_set=true`, `bf16_control_matches_device_set=false` on
every flip line; the kernel matches its own BF16 host to 5.96e-8).

**Consequence.** `REQUIRED_KERNELS` stays 9, the kernel work already merged (D34-D36) is unaffected,
and the remaining work is ordered: (1) the user's precision-versus-criterion decision above - the
measured evidence now names the attention-projection/router activation quantisation as the source;
(2) if the precision-first path is chosen, that is new API-compatible kernel variants plus re-pins,
followed by the criterion run again; (3) serving-side work (performance, HTTP surface, 4-node path)
is a separate program. The hybrid branch stays unmerged and criterion 1 stays OPEN.

## D38 — vLLM token parity is only meaningful on the same weights; the deployed vLLM serves a different checkpoint

**Decision.** No statement of the form "our engine matches vLLM" may be made without first recording
the served model's `root` and proving it equals the checkpoint under test. The deployed vLLM on the
cluster currently serves `models--pqhaz--DeepSeek-V4.1-Flash-1017-heretic` snapshot
`12d58ad0878b5008e80d720251adb969a0069412`, while the golden fixture and the device engine use
`models--deepseek-ai--DeepSeek-V4.1-Flash-1017` snapshot
`1bae0e1088993b9b82de08f59de5be5a34e42170`.

**Evidence.** On 2026-10-19 the live server's `/v1/models` reported that heretic root; greedy
completions from it are degenerate (`The capital of France is` returns ` ____. ...`), and the
fixture's own 130 prompt ids return `[367, 223, 10727, 201, 15, 223, 19, 16, 223]` against the
fixture reference `[491, 367, 201, 80, 3302, 201, 54, 201, 4341]`. That is a different model, not a
numerical tolerance question.

**Consequence.** The f32 reference on the released snapshot remains the only truth for our
checkpoint. If a vLLM comparison is wanted later, it must be run against a server holding the released
snapshot (and the served root recorded in the same log).



## D39 - The chain runs the checkpoint's native arithmetic; acceptance is numeric with tolerance, and vLLM is the behavioural oracle

Recorded 2026-10-19 (Lead, from the user's directives of the same day; supersedes the open choice in
D37, the token-first bar in D25, and the reference-truth wording for the f32 fixture).

**Decision.** Four rules, and they override any earlier instruction that conflicts with them.

1. **Native arithmetic only.** Every quantised GEMM in the chain consumes E4M3 activations with one
   ue8m0 scale per 32 contiguous values - the scale is 2^ceil(log2(amax/448)) after the 1e-4 amax
   floor (D23) - against weights stored as E4M3 with 32x32 ue8m0 scales (the five attention
   projections and the shared expert) or FP4 with 32-element K scales and ue8m0 scale bytes (the
   routed experts), accumulated in f32. Computing the same weights at higher precision (f32 or
   BF16-class activations) is not a more faithful run of this checkpoint: it computes a different
   model from the one that was trained and is served. The precision-first family is rejected, and
   its artifacts stay unmerged experiments.
   **Note (task-29, 2026-10-19):** rule 1 rejects computing the trained quantised weights at
   *higher* precision; it does not apply to `wo_a`, whose native activation is bf16
   (`model.py:766-788`; `convert.py:154-173` dequantises that weight to bf16 in the released
   checkpoint). The `b12x_bf16a_mxfp8w_wo_a` kernel variant is therefore the native rule for that
   one projection, not a precision-first experiment: it keeps the E4M3 32x32 weights and the f32
   accumulation and removes only the activation quantisation native does not perform. The dense
   bf16a variants remain unwired experiments and the precision-first family stays rejected.
2. **No token-equality criterion.** Generation is not deterministic across implementations (kernel
   choice, reduction order, scheduling, batching), so a token-for-token bar can fail a correct
   native chain and pass an incorrect one. Acceptance is numeric and distributional, at declared
   tolerances: per-stage exactness against the arithmetic twin where the twin is the same
   arithmetic, and boundary bars over logits/logprobs (max absolute delta, top-k set agreement,
   KL/JS of the compared distribution). Every bar is declared before the run and recorded with the
   measurement.
3. **vLLM is the behavioural oracle.** Where the released code, our docs or the devices disagree or
   are silent, the chain does what vLLM does: quantisation layout and flags, KV cache dtype, the
   attention and sparse-top-k behaviour, logprobs. A vLLM cross-check records the served model root
   and compares distributions with tolerance, never token identity (D38).
4. **Engram weights live on disk.** Per the user, the Engram tables are disk-offloaded (NVMe) and
   read on demand; the drive is fast enough that this is not a performance bottleneck. The chain
   keeps that shape - declared host Engram reads today, the device gather reading a packed on-disk
   table later - and does not stage the whole table into HBM.

**Evidence.** Released forward: /home/user/dsv41-research/inference/model.py:27-30 sets
fp8_block_size=32, fp4_block_size=32, scale_fmt="ue8m0", scale_dtype=e8m0nfu, and model.py:182-205
(linear()) applies act_quant(x, 32, "ue8m0", e8m0) before every FP8 and FP4 GEMM; kernel.py:98-124
makes the scale 2^ceil(log2(amax/448)); D23 already measured that as the released forward's single
dynamic activation scale rule. The device chain implements exactly that rule:
engine::quant::quantize_activation_fp8 plus the B12x kernels, measured stage-exact against the
arithmetic twins (wq_b 0/32,768 mismatches vs mxfp8_ref, max_abs 0.0; the head reduce equal at every
position with block_max_equal=true; the format-2 body 35/18/13 of 40 layers zero-mismatch at
positions 0/1/2 - the numbers in D37). The f32 reference is not that arithmetic: engine linalg.rs
matmul is f32 over dequantised weights and forward/mod.rs:4 says there is no quantised kernel there;
it does model model.py's attention element round-trips (q_index FP4, window KV FP8, index keys FP4,
compressed latent FP4/E4M3), which is why it reproduces its own fixture. D37's b68 criterion result
is therefore the measured distance between two different arithmetics - native device vs f32
simulation - not a device defect.

**Consequences.**

- D37's open choice is decided: native arithmetic stays, and the criterion moves from token equality
  to a numeric/tolerance comparison against a native reference and against the arithmetic twins. The
  b68 result (position 129 argmax 8060 where the simulation says 491, 0 of 9 tokens equal, logits max
  diff 15.009581) is re-labelled as the simulation-vs-native distance.
- D25's token-first bar and the token clause of Phase 2 exit criterion 1 are retired. The fixture
  remains valid as the simulation reference and as a regression input, but it is no longer truth for
  arithmetic or token identity.
- D38 is actioned: the cluster now serves the released snapshot
  1bae0e1088993b9b82de08f59de5be5a34e42170 from vllm-launch-recipe-released.sh (sha256
  a3ac461fa02a945181a17b509f7695ca39b38930eee529a0f9fd9151061072c4), with --max-logprobs 1000 for
  the numeric comparison. The previous recipe is retained as
  vllm-launch-recipe-heretic-12d58ad0.sh (sha256
  2a3e2c7719532149d3291d11964ace46bf27d9a0b656c39258d5a6295c552574), and window-vllm.sh (sha256
  a3ba6763cc8f5087ab65dc7771e28e37e9b3a18842d2e72542c3f22f7344ef70) starts the released recipe.
- Unmerged experiments are recorded, not delivered: task/d37-precision dd4184e (env-gated host-f32
  projection seam, never run) and batch/2026-10-19-25 6c1d582 (BF16-activation projection kernels,
  host-tested and pinned, GPU comparison not run; the dense variants are rejected precision
  experiments, while the `_wo_a` variant is the native rule for that projection - see the
  task-29 note under D39 - and F1's isolation is still unmeasured). The machine-readable logits
  dump for the numeric comparison is task/logits-dump 4f599b7 (binary sha256
  7e0ce4bf2390ec078f70098eb5e77881b518e5d0ed58f9155b669a355a8d48cc).
- **Kernel-level audit follow-up (same day, kernel-dev, branch batch/2026-10-19-25 b773e87,
  docs/build/native-kernel-arithmetic-audit.md).** Auditing the shipped kernels against the released
  conversion and forward code found four places where the chain is not yet native, plus a list of
  reduction-order questions that only device-vs-twin evidence covers: (M1) wo_a - the release
  dequantises wo_a to BF16 and runs a BF16 grouped einsum with the BF16 attention output and no
  activation quant, while ours adds an FP8 E4M3 block-32 activation quant with FP8 weight codes;
  (M2) MoE combine - the release rounds each expert's w2 output to BF16, adds in ascending expert
  id order, adds the shared BF16 output, rounds the MoE output, and the residual is added later by
  hc_post in f32, while ours sums the top-k partials plus residual in one f32 chain in slot order
  with one rounding; (M3) the unweighted/shared SwiGLU - the release rounds the activation to BF16
  before w2's quantiser, ours quantises the f32 activation; (M4) mega_mhc - the release keeps a BF16
  residual stream and rounds at hc_pre/hc_post (norm output BF16 before the quantiser), ours keeps
  f32 through norm+quantiser and FMA-contracts the step-1 update. These are recorded as defects to
  fix, not as acceptable variation: the priority order is M2/M3 (they move every MoE token), then M1
  (lands before wo_b), then M4. Two further facts from the audit: the common "FP4 q/k block 16"
  shorthand is wrong - block 16 is the compressed KV, the indexer q/k are block 32 - and the
  bit-exact m12/m13 bars compare the device to our own twins, not to native, so they are not the
  native comparison D39 requires.
- Native compliance across all layers is proved by the native-arithmetic audit
  (docs/build/native-arithmetic-audit.md) plus the per-kernel audit against kernel.py
  (docs/build/native-kernel-arithmetic-audit.md); the numeric comparison against the vLLM-released
  oracle is measured in window b69.


## D40 - Serving-parity measurement: the protocol, the metrics and the provisional bars, declared before the first measurement

Recorded 2026-10-19 (Lead), ahead of the first diagnostic run of the native device chain against
the vLLM-released oracle (b70). D39 fixed *what* is compared (native arithmetic, vLLM as the
behavioural oracle, numeric/distributional tolerance, never token equality); this decision fixes
*how* it is measured and what the numbers must be before a serving-parity claim is allowed.

**Decision.** The measurement protocol and the metric set below are frozen now. The numeric
thresholds are **provisional** and are frozen for acceptance in a follow-up decision (D41) that must
be recorded before the serving-parity gate run; D41 also fixes the API/performance bars. No claim of
serving parity may be made on a diagnostic run, and no bar may be changed after a gate run has been
measured - a changed bar is a new decision and the next run is a new gate.

### Protocol (frozen)

1. **Oracle side.** vLLM serving the released snapshot
   1bae0e1088993b9b82de08f59de5be5a34e42170, TP=4, temperature 0, --max-logprobs 1000. The served
   root is read from /v1/models and recorded in the same evidence file as the numbers; a request
   with a different root invalidates the comparison (D38). One request per pinned prompt (the
   prompt suite of task-18), prompt_logprobs over the full prompt; the distribution at compared
   position p is prompt_logprobs[p + 1] - the next-token distribution the oracle assigns after
   consuming position p.
2. **Device side.** sparkflash device-chain --run-stage hybrid_sequence --dump-logits FILE on the
   same token ids (prefill only; no sampling, no generation). The dump is one JSONL object per
   compared position plus FILE.bin, consecutive little-endian f32 blocks holding the full
   129,280-vocabulary logits in the same position order.
3. **Comparison.** Offline, from the two artefacts; every evidence file records the served root, the
   device binary sha256, the cubin set identity, the fixture and prompt-suite sha256, and the two
   source filenames. No live server is needed for the comparison, so a window is used once per side.
4. **Same tokens both sides.** The comparison uses the fixture's prompt ids, not generated tokens:
   at temperature 0 with the same prefix both implementations are asked the same question; token
   streams after the first divergence are not comparable and are not compared (D39).

**Update 2026-10-19 (Lead-authorised, B70 host prep; an addendum, not a rewrite of the frozen text).**
"Prefill only" in item 2 means the device side does not sample or generate from its own stream; it was
read as "run with `--decode-steps 0`" and that reading is wrong for this fixture. The measured 130 + 8
claim is a **full claim**: the golden fixture records its 8 decode steps, `full_claim` requires
`prompt_tokens == fixture.prompt_tokens.len()` **and** `decode_steps == fixture.decode_steps`, and the
dump then covers device positions 129..137 - 9 compared positions. A zero-decode run against this
fixture is not a full claim and takes the O(N^2) sanity-prefix branch (one reference prefill per prompt
prefix), dumping 130 prompt positions; b69 was that branch, terminated after 62 minutes with no dump.
The B70 diagnostic run therefore uses 130 + 8, labels itself PRE-FIX DIAGNOSTIC in its evidence, and is
not acceptance evidence: the native-arithmetic fixes are not in it and the D41 bars do not exist yet.

### Metrics (frozen)

With O_p the oracle's top-1000 distribution at position p and D_p the device's full distribution:

- top1_agree(p) - the oracle's argmax equals the device's argmax;
- top5_recall(p), top20_recall(p) - fraction of the oracle's top-5 / top-20 ids that are in the
  device's top-5 / top-20;
- max_abs_dlogp(p) and mean_abs_dlogp(p) - the maximum and mean of |O_p(t) - D_p(t)| over the
  oracle's top-100 ids (raw logprobs, no renormalisation);
- kl(p) - KL(O_p || D_p) in nats over the union support of the oracle and device top-100, renormalised
  on both sides to that common support (a fixed support makes the number comparable across
  positions).

Aggregates over a run: top1_agree fraction over compared positions, median/mean/p95 of kl and
max_abs_dlogp, and the same broken down per prompt. The evidence file stores every per-position
value, not only the aggregates.

### Provisional bars (frozen for D41)

A quality pass requires **all** of the following on the pinned suite:

- top1_agree >= 95% of compared positions;
- median top20_recall >= 90%;
- median kl <= 0.02 nat and p95 kl <= 0.10 nat;
- per position max_abs_dlogp <= 1.0 nat, and no position above 2.0 nat.

Rationale: a different-but-valid native kernel arithmetic (different accumulation association, the
same quantisation rule) is expected to move logprobs by small fractions of a nat, which leaves top-k
sets stable except where the distribution is nearly flat; the bar is set to catch a *model*
difference (the class of the b68 simulation-vs-native distance) while allowing kernel-order
differences. The thresholds are provisional because no measurement exists yet of the oracle's own
run-to-run variation.

### Floor rule (frozen)

Before D41 freezes the thresholds, the campaign measures the oracle's self-consistency floor on the
same prompts: the same request issued at batch size 1 and inside a batch of 4 (and issued twice),
which yields the implementation's own kl / max_abs_dlogp variation. The frozen bars must be at least
3x that measured floor; if the floor is larger than the provisional bars above, the discrepancy is
investigated and reported, not absorbed by loosening the bar.

### Bar file format (for D41)

D41 is recorded as a JSON file, not prose, so the gate run consumes it directly.
`scripts/compare_native_vs_vllm.py` accepts exactly these names (`--bars FILE`); an unknown name is a
refusal - exit 2, no evidence file written - not a silent UNBARED. A typo cannot pass and cannot quietly
remove the gate; only omitting `--bars` produces the UNBARED verdict:

| name | D40 metric | provisional value |
| :--- | :--- | :--- |
| positions_min | compared positions present in the dump | 9 |
| top1_agreement_min | top1_agree fraction over positions | 0.95 |
| top5_overlap_mean_min | mean top5_recall | 0.90 |
| top20_overlap_mean_min | mean top20_recall | 0.90 |
| top100_overlap_mean_min | mean fraction of the oracle's top-100 in the device's top-100 | 0.80 |
| top1000_overlap_mean_min | mean fraction of the oracle's top-1000 in the device's top-1000 | 0.60 |
| max_abs_delta_max | maximum over positions of max_abs_dlogp | 1.0 |
| mean_abs_delta_max | mean over positions of mean_abs_dlogp | 0.10 |
| kl_mean_max | mean of kl | 0.02 |
| kl_p95_max | p95 of kl | 0.10 |
| js_mean_max | mean Jensen-Shannon divergence (diagnostic, not a bar) | 0.05 |

The "no position above 2.0 nat" tail rule stays in the report even though the schema has no separate
name for it: such positions are listed individually in the evidence, and a run with any of them fails
at review even if the file gate passes. The D41 values are the floor rule's output, not these
provisional numbers copied forward.

### First measured oracle facts (2026-10-19, before b70)

Captured against the released root at 17:28Z: the fixture's positions 129..137
(`native-oracle-released.json`) and the 8-prompt suite at every position
(`native-oracle-suite.json`), both with `prompt_logprobs=1000` and the served root verified. Two
facts to keep when reading b70:

- At device position 129 the oracle's argmax is token 367 with logprob -0.199, while the fixture's
  reference token 491 has logprob -5.449 (rank 7), and the heretic vLLM of D38 also produced 367
  first. The fixture's `next_token` columns are therefore a simulation artefact (as D39 already
  held), not oracle truth: a low top-1 agreement at these positions is a statement about the
  simulation until task-32 root-causes the difference, not a device regression.
- vLLM scores prompt token `i` in `prompt_logprobs[i]` given `0..i-1` with entry 0 null, which is
  device position `p = i - 1`; that is the convention the harness implements and the one the
  comparison joins on.

### Consequences

- b70 is a **diagnostic** run: its numbers are recorded, labelled pre-fix, and no pass/fail is
  claimed. Its value is the F1-F8 magnitude measured against the released oracle instead of against
  the f32 simulation.
- Every native-arithmetic fix (task-4 .. task-11) that gets a window reports its movement in these
  metrics; the twin/stage bars remain the per-fix gate (task-12), these metrics are the native gate.
- The serving-parity gate (task-26) evaluates this protocol on the pinned suite with the D41-frozen
  bars, together with the performance bars D41 also fixes.
- docs/build/serving-parity-plan.md section 4 points here; the plan's phrase "bars are declared
  before the run" now has a named record.

## D42 - The engine KV/attention seam is a KV store trait (Option B), recorded and not implemented; an in-path diagnostic mirror is rejected

**Date:** 2026-10-19. **Status:** recorded boundary, **no implementation approved**. Proposal:
`docs/build/engine-kv-seam-decision.md` (docs only, task-47). Reviewing this decision does not
authorise an engine refactor, a branch, or an ABI change.

### The claim this decision exists to stop

"The paged store mirrors the request token sequence" is **not** the same claim as "the serving KV is
paged". The first is a diagnostic that spends CPU to make bookkeeping look consumed; the second
requires the engine's attention path to read and write the paged pool as its backing store. Today
`crates/sparkflash-cli/src/serve.rs`'s `paged_serve` module is crate-internal, has no production
caller, and is a storage/plumbing harness. `cached_tokens`, `stats.kv_cache`, `kv_cache_hit_rate`
and prefix sharing are **not** serving claims until the seam below exists and a window measures it.

### Decision

The recorded boundary is **Option B, the KV store trait**, with the allocator handle hidden inside the
paged store implementation:

1. The engine defines a minimal store contract (sequence lifetime, append/write, read/stage, capacity
   error) and keeps a host dense implementation as the default. The engine's arithmetic stays
   layout-agnostic; the host path keeps its current bit-exact tests.
2. The paged implementation wraps `PagedKvCache` and `PagedKvView` behind that contract and owns
   page hashes, sharing, eviction and preemption. The engine never sees a block table.
3. The scheduler owns the shared pool/quotas and the admission decision, gaining a capacity and
   lifetime view; first-token promises become conditional on a page reservation. This is the task-22
   admission shape applied to KV, not a second chunk planner.
4. The HTTP driver owns request-to-sequence handles and the prefix key derived from prompt tokens, and
   fills `cached_tokens` / `global_kv_cached_tokens` from measured complete-block reuse while leaving
   the UMA/NVMe tier fields and the CED/SWA replay fields at their documented zero/not-measured state.
   A global paged hit must not be copied into `uma_hit_tokens` or `nvme_restored_tokens`.

Rejected: **Option A** (attention backend trait) because it cannot produce the cache claims this task
exists for - the engine never sees the paged store; **Option C** (engine owns the allocator handle)
because it puts page admission into the engine before the scheduler has a capacity model, couples the
engine to paging policy, and is the largest blast radius; and the **default-off diagnostic mirror**
because it leaves the serving KV dense, hides the observer's cost behind a flag, and makes the
bookkeeping claim while avoiding the user-visible cost. A diagnostic mirror in the request path would
spend CPU per token to satisfy a lint, not to serve.

### What it would let us claim, after implementation and a window

Measured `cached_tokens` and the global prompt-cache breakdown as complete-block reuse rather than
zero; `kv_cache` stats and hit rate as store accounting (pages, shared blocks, evictions,
preemptions, recomputes); prefix sharing under load, with the host radix key's collision assumption
kept as part of the claim; and device attention over paged rows at the existing 19-word entry, within
the published 2e-2 attention bar plus D40/D41 logits bars - not token equality.

### The load-bearing boundary (quoted verbatim, and the reason this is a record and not an approval)

> Option B does not fix the f32-engine vs fp4-device numeric gap, the O(context) staging cost, HSI
> full-pool selection, the serial HTTP engine, or the host-resident paged pool. Those are the
> boundary between "the serving KV is paged" and "serving parity."

In particular: no bit identity or token parity across representations; staging per source layer per
step is still required by the frozen ABI; HSI full-pool population and the top-512 candidate merge are
untouched; the pool is host-resident and `PagedKvView` stages bytes, which a device-resident store
would not; the HTTP driver still serialises jobs; and prefix sharing alone does not make a prefix
cache correct under collisions, cancellation, partial prefill or eviction.

### Numeric bars, per path (no path may claim bit identity to the engine once it uses the device fp4 kernel)

- Host dense store: **exact** - 0 differing f32 bits at every position. This is the refactor's
  regression gate, not a serving result.
- Paged store with host attention: stored bytes must match the same quantiser's bytes exactly
  (entry-level comparison against a dense byte sidecar), and attention over those bytes must match a
  host recomputation from the same dequantised values bit for bit. This is the task-44
  storage-is-not-arithmetic property; against the current f32 engine it is the D40/D41 numeric bar.
- Device paged attention: the 2e-2 attention-body bar plus the D40/D41 logits bar; attention-output
  agreement does not imply token agreement and must not be reported as parity.
- Paged path exactness: sharing vs no sharing, and preemption/recompute vs no eviction, must give
  identical tokens and logits bits **within** the paged path (recompute from the identical retained
  tokens).
- Cache statistics are accounting bars, not tolerances: refcounts, free queue, block hashes, cached
  token counts and hit rate must equal the model the test constructs, with invariants after every
  step. A cache number must not be given a KL-style tolerance.

### ABI

Keep the released `flash_mla_csa2_seq_sm121` entry at its frozen 19-word argument list for the first
implementation, staging through `PagedKvView`. Appending a 20th positions argument (Path A) changes
`FlashMlaParams` 19 -> 20, the m4_abi signature/declaration pin, every launch site and harness
mapping, and the kernel baseline; that is a decision-record-level pin change with its own reviewed
entry and must not be combined with this seam.

### Prerequisite before any throughput argument

The window lines in `docs/build/engine-kv-seam-decision.md` section 6 must exist at one device,
driver/cubin set and prompt suite: paged request/pool accounting with invariants after every step;
per-source-layer staging bytes/copies/time separated from kernel time; TTFT/prefill/decode and batch
timing at widths 1, 4 and 16 with warmup and repeats; cached tokens and hit rate beside the timing;
D40/D41 numeric lines; and a same-window vLLM baseline with the served root recorded. Without
staging-versus-kernel separation and a same-window baseline, a result argues for Path A or a
paged-read kernel, not for the seam.

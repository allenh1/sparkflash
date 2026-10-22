# Native-rounding arms: M2 (MoE combine), M4 (mHC BF16 stream), M3 (shared SwiGLU)

Status: **measured. The M2/M4 four-rank loop wiring landed on `task/native-rounding`
(merged as `67a12f85`), the M3 loop-path native shared expert landed default-off on
`task/native-shared` (tip `e2524fdd`, merged as `8c35c5f3`), and the four-rank A/B results are
recorded here: M2 combine is a measured positive (7/10, the best measured D41 arm), M4 mHC and the
M3 native shared expert are measured negatives and are not adopted; the M3 capability stays merged
default-off for future composition/debug. No pin-set edit, no cubin, no node access.** This note
records the plans, the exact code sites, the host twin tests, the invocations, the comm/wall cost and
the measured results/decision.

Branches: `task/native-rounding` (M2/M4 wiring, based on `win/m1w25` at `6683f88c`, merged as
`67a12f85`) and `task/native-shared` (M3 loop-path native shared expert, base `3c81c45c`, tip
`e2524fdd`, merged as `8c35c5f3`).
Runtime target under audit: the four-rank loop `crates/sparkflash-cli/src/tp4_forward.rs` on
`win/m1w25`. Release arithmetic: `/home/user/dsv41-research/inference/model.py` (line numbers
below are that file). Oracle: vLLM TD=4, D40 protocol; no vLLM run exists in this analysis, so
every statement below is arithmetic or host-test measured, never a serving-parity claim.

## Current loop arithmetic (all three arms)

| term | current four-rank loop | release (native) | class |
|---|---|---|---|
| M2 combine | `run_layer_moe_routed` (`tp4_forward.rs:2955`): per rank `moe_combine_routed_f32` (`:3212`) rounds **each rank's** per-expert partial to BF16, bf16-rounds the rank row, canonical bf16 all-reduce (`:3224`); `run_layer_ffn_epilogue` (`:3711`) widens the bf16 delta (`:3733`) and adds the host shared expert in f32 (`:3742`), no final MoE bf16 round before `hc_post` (`:3761`). Arithmetic: `f32(shared_host) + bf16(Σ_ranks bf16(Σ_slots bf16(rank_partial))))` | `y` is f32; each expert's w2 output is bf16 and is added into f32; the rank sum is an **f32** `dist.all_reduce`; the shared expert's bf16 output is added; one bf16 round of the MoE output (`model.py:891-904`); residual added later by f32 `hc_post` (`model.py:962-966`) | shown by arithmetic: two bf16 roundings the release does not have (per-rank partial, payload) and a missing shared-bf16/final-round boundary |
| M4 mHC | `mhc_entry_host_seam` (`:576`), `attention_residual_host_seam` (`:2423`), `run_layer_ffn_seams` (`:2655`), `run_layer_ffn_epilogue` (`:3711`): all f32 (engine `forward/mhc.rs` `hc_pre`:310, `hc_post`:378); the residual stream, hc_pre output and norm output stay f32; `hc_post` also starts its accumulator from the post term instead of adding it once after the comb sum | residual stream BF16; `hc_pre` sums in f32 and returns `y.to(x.dtype)` (`model.py:954-957`); `hc_post` evaluates `post*x` and `sum(comb*residual, dim=2)` separately then adds once, and returns `y.type_as(x)` (`model.py:962-966`); norm returns `(weight*x).to(dtype)` (`model.py:288-293`); block stream is BF16 (`model.py:976-994`) | f32-only stream: five dtype boundaries per layer are missing, plus the hc_post add order |
| M3 shared SwiGLU | the loop's shared expert is the **host engine f32 reference** `moe::add_shared_expert` (`forward/moe.rs:1047`); the module doc states no quantised kernel runs on that path, so the FP8 activation quantisation is not reproduced at all. The device kernel `swiglu_mxfp8_quant_sm121` is **not launched** by the four-rank loop | bf16 activation before w2's act_quant (`model.py:841-851`) | **already native in source AND pin** (see M3 below); the loop simply does not run it |

M2 is the largest term (every routed token, every layer). M4 is next (every layer's stream). M3 is
last because the loop path that would carry it is host-f32 and is a *different, larger* deviation
than the recorded kernel row.

## M2 - MoE combine: implementation and host test

**(a) current site.** `tp4_forward.rs:2955-3235` (`run_layer_moe_routed`) and `3711-3790`
(`run_layer_ffn_epilogue`); engine seam `moe::MoeDeviceBackend` not involved (device launches
are direct). Current arithmetic differs from native in four places: rank-partial bf16 round
(`:3212`), bf16 payload + bf16 all-reduce (`:3214-3224`), shared expert added in f32 after the
reduce (`:3742`, engine f32 weights), no final MoE bf16 round before `hc_post` (`:3761`).

**(b) implementation (landed, host layer).** New in `crates/sparkflash-kernels/src/mxfp4_ref.rs`:
- `moe_combine_rank_sum` (`:591`): the cross-rank f32 sum of the rank slot accumulators, in
  ascending rank order (the canonical order `sparkflash-comm/src/reduce.rs` uses; NCCL's own f32
  reduce order is not in the release source, so the order is declared).
- `moe_combine_native` (`:630`): rank sum -> the already-native epilogue
  `moe_combine_finalize` (`:653`: per-expert bf16 round, ascending expert-id add, shared bf16
  add, one MoE bf16 round). This is the whole native MoE output row; `hc_post` then mixes it
  into the f32 residual stream.

No pin edit and no cubin: this is host arithmetic over the same f32 slot accumulators the device
already produces (`m17_moe_a.rs` documents the device f32 accumulator as bit-exact; the loop's
device launches are unchanged). **Loop wiring applied** (task/native-rounding) (see status below); the staging
edit is:
1. `run_layer_moe_routed`: under the switch, replace the per-rank finalize + bf16 payload +
   `all_reduce_bf16` with an all-gather of the rank slot accumulators as raw u16 words (the
   existing `SharedExpertGather` `gather_one` pattern, `tp4_forward.rs` around `:3558`), keep
   the gathered per-rank f32 slices in `Layer0MoeRouted`, and add a gather sequence namespace
   constant disjoint from `MOE_REDUCE_SEQ`/`SHARED_EXPERT_GATHER_SEQ` with the same disjointness
   unit test the other namespaces have.
2. `run_layer_ffn_epilogue`: under the switch, compute the shared expert into a zeroed row
   (`add_shared_expert` on a zero vector), bf16-round it, call `moe_combine_native`, widen the
   result, and skip the f32 shared add; tag the evidence with the switch name.
3. CLI/record switch `--native-rounding combine` and the active set in the per-position/token
   evidence so the row is attributable.
Effort: ~1.5-2 h (all host/loop Rust). Risk: the gather moves 4 x 122,880 = 491,520 bytes per
rank per layer-position (current bf16 delta frame is 20,480 bytes; ~24x wire), which is fine for a
diagnostic A/B at the window's measured rows but not a serving default; a comm-crate
`all_reduce_f32` (additive, no kernel pin) would be the right default-path primitive later. All
four ranks must take the same branch; sequence reuse or a mixed arm must be refused by name.

**(c) host twin test (passed).** `mxfp4_ref.rs:1283`
`the_native_combine_is_the_release_arithmetic_on_a_captured_activation` and `:1374`
`the_rank_sum_adds_in_ascending_rank_order`. The fixture
(`crates/sparkflash-kernels/tests/fixtures/native-rounding/`) is the **captured m26 stage input**
of the m1y5 layer-0 t1 measurement (rank-0 `moe-stage-input/`: `act_fp8.bin`,
`act_scales.bin`, `pre_ffn_hidden.f32`, sha256s in `m26-stage-input.json`) plus the captured
router row ids `[110, 210, 341, 373, 324, 206]` from `m26-l0-t1.json`. The rank partials are a
documented deterministic fixture map over the captured activation: the committed bundle holds the
device run's input, not its cumulators or the checkpoint weights (the checkpoint is not on this
host). Assertions: the native combine is bit-equal to an independent hand-written model.py:893-904
reference; the loop chain differs; measured host numbers
`ref: m26 captured activation, 4 rank partials, 1 token, hidden 5120`:
**2813/5120 bf16 slots differ, max |delta| 1.5625e-2, relative L2 3.659783e-3** (host arithmetic,
not a device measurement; the m26 device measured gap was 3.084e-3 relative L2 on its own
reference). Tolerance: bit-equal is required and asserted for the native arithmetic; no tolerance
is taken on the epilogue. The only non-captured operand class is the rank-partial fixture map,
which is stated in the test doc and does not affect the epilogue arithmetic under test.

**(d) one-line ablation.** Toggle `combine` alone: it changes the combine rank-rounding/order,
the payload reduce, the shared-bf16 add and the final MoE bf16 round together (they are the M2
term as recorded); every other arm (attention body, M4, M3) stays current. One window row: current
vs `--native-rounding combine`, same token ids/positions/reference vectors.

## M4 - mHC BF16 residual stream: implementation and host test

**(a) current site.** Seams `tp4_forward.rs:576` (`mhc_entry_host_seam`: engine `hc_mixes`,
`hc_pre`, f32 `rms_norm`), `:2423` (`attention_residual_host_seam`: engine `hc_post` f32),
`:2655` (`run_layer_ffn_seams`: f32 hc_pre, f32 norm, quantiser on the f32 row), `:3711`
(`run_layer_ffn_epilogue`: f32 `hc_post`). Engine functions `forward/mhc.rs:310` (`hc_pre`),
`:378` (`hc_post`); `hc_post` seeds its accumulator with the post term. The loop's stream is
f32 throughout; `mega_mhc_sm121` is **not launched** by the loop (the seam is host), so the
recorded M4 kernel/FMA row is not the loop's arithmetic.

**(b) implementation (landed, host layer).** New in `crates/sparkflash-engine/src/forward/mhc.rs`:
- `bf16_round_row` (`:456`), `hc_pre_bf16` (`:470`), `rms_norm_bf16` (`:487`),
  `hc_post_bf16` (`:511`: release expression order plus one BF16 round).
No pin edit and no cubin. **Loop wiring applied** (task/native-rounding); the staging edit is: add the switch to
the seam path and pass it through `run_layer_mhc_entry` (`:4427`), the `attention_residual`
call site (`:5010`), `run_layer_ffn_seams` (`:5018`) and `run_layer_ffn_epilogue`
(`:5028`): round the stream after each `hc_post`, round `hc_pre` outputs before the norms, round
norm outputs before the quantisers, and use `hc_post_bf16`. Effort: ~1-1.5 h. Risk: the seam
digests change by design, so boundary-replay/QA comparisons must be tagged with the switch; no
collective or kernel change.

**(c) host twin test (passed).** Fixture:
`crates/sparkflash-engine/tests/fixtures/native-rounding/`, the phase1 boundary v1 **host
capture** (layer 1, position 0 of the token-0 batch) with provenance/sha256 in `provenance.json`.
Tests: `the_captured_hc_post_row_is_the_reference_stream` (engine f32 `hc_post` over the
captured inputs is **bit-exact** the captured `post_attention_hidden`, so the fixture is the
chain), `the_captured_hc_post_native_rounds_and_keeps_the_release_order` (release-order
reference, bit-equal; measured `ref: captured layer-1 pos-0 hc_post row`: **20479/20480 values
move at the BF16 boundary**), `the_captured_hc_pre_native_rounds_the_collapse` (**5120/5120
values move**), `the_captured_norm_output_is_bf16_before_the_sublayer`. Tolerance: bit-equal for
the same arithmetic; the captured arm is the f32 reference chain, so the native arm is expected to
differ from it by exactly the recorded rounds. The norm test uses a unit weight because the
checkpoint norm weight is not on this host; it proves the boundary rule, not the checkpoint row.

**(d) one-line ablation.** Toggle `mhc` alone: all five BF16 boundaries of the block stream plus
the hc_post expression order. If the op-order term needs isolating, a second flag
`mhc-order` can be split later (it is the only bundled sub-term).

## M3 - shared SwiGLU: already native in the pinned kernel; not on the loop path

- Kernel source `kernels/swiglu_mxfp8_quant_sm121.cu:144-151` rounds the unweighted activation to
  BF16 before the quantiser (`__float2bfloat16_rn`), and the pinned baseline entry
  `kernel_baseline.txt:23` (`swiglu_mxfp8_quant_sm121`, `21912` bytes,
  `65818de7...`) is the **post-fix** cubin: commit `4bf96e97` (!124 batch 3) changed the kernel and
  regenerated that pin.
- Host twin `swiglu_ref::quantise_row` already rounds BF16 before `act_quant_fp8_row`
  (`swiglu_ref.rs:128-155`), and the mutation test
  `the_unweighted_activation_is_rounded_to_bf16_before_the_quantiser` (`:452`) already exists.
- The audit row `docs/build/native-kernel-arithmetic-audit.md:321` ("FP32 activation quantised
  directly") is **stale** as of `4bf96e97`.
- The four-rank loop does not launch this kernel: the shared expert is the host engine f32
  reference (`forward/moe.rs:1047`; module doc: no quantised kernel runs, activation quantisation
  not reproduced). So the recorded M3 device term cannot move a D41 bar on the current loop; the
  loop's shared-expert gap is larger (no FP8 activation quantisation at all).
- Option (ii) was later built default-off on `task/native-shared` (tip `e2524fdd`, merged as
  `8c35c5f3`): engine `shared_expert_rows` / `native_linear` / `add_shared_expert_native`
  (`forward/moe.rs:1146/1187/1222`) and the `--native-rounding combine,shared` epilogue selection
  with per-layer `shared_arith`/`shared_sha256`/`shared_l2` evidence (plus a layer-0 f32
  comparison row). It was measured on kit m1w33 and is a **measured negative, not adopted** - see the
  measured-results section below. The already-native pinned kernel remains the future device-side
  route, with no pin edit either way.


## Staging (applied on task/native-rounding): switch, invocations, comm cost, wall impact

**Code changed (2 files, no other crate):** crates/sparkflash-cli/src/tp4_forward.rs (the
NativeRounding switch, the native seam variants, the M2 gather + epilogue finalize, the M4
boundaries, the evidence fields) and crates/sparkflash-cli/src/tp4_chain.rs (flag parse, request
field, run-log line). No pinned cubin, no kernel source, no engine default arithmetic changed.

**Switch.** --native-rounding <none|combine|mhc|both> (or the comma list combine,mhc; empty = none),
parsed into NativeRounding in TokenStageRequest, printed in the run log
(tp4-chain rR forward: native_rounding=...), recorded in the token envelope's launch_census
(native_rounding.active/combine/mhc), in every per-position forward-progress record
(native_rounding at the top of each layer entry) and in the MoE stage evidence (combine_arith,
all_reduce_calls_per_rank, combine_payload_bytes_per_rank). Default none takes exactly the
pre-existing branches; the only default-path byte change is the added evidence keys (the logits
dumps are unaffected).

**Invocations** (all four ranks must pass the SAME value; the rest of each rank's argv is the
existing window command):
- default/f32 row: omit the flag (or --native-rounding none)
- M2 row: --native-rounding combine
- M4 row: --native-rounding mhc
- both: --native-rounding both
The flag is refused without --forward.

**M2 comm cost (one token/position, one rank, one layer).** The native arm all-gathers the f32 slot
accumulator: local payload 1 * 6 * 5120 * 4 = 122,880 B; each rank sends it to the other three
(368,640 B out, 368,640 B in) with all_gather_bf16, so about 737 KB wire per rank per
layer-position, against the default MoE reduce's 10,240 B payload (30,720 B out + 30,720 B in =
61,440 B) - 12x the MoE collective bytes. Per position (40 layers): 14.75 MB out per rank vs
1.23 MB; over the m1y29 window (130 positions) 1.92 GB vs 0.16 GB out per rank. The 122,880 B frame
is under the transport's 8 MiB default frame limit, so no config change. The M2 row therefore runs
on the window's measured rows/positions, not as a default full-sequence path; a comm-crate
all_reduce_f32 (additive, no kernel pin) is the later default-path fix.

**Wall impact (model estimate, not measured).** At an effective per-rank TCP rate of about 2 GB/s
one M2 row adds about 6.8 ms per 2772.8 ms step (range 4.5-9 ms for 3-1.5 GB/s), i.e. +0.16% to
+0.33%, about +1-2 s on the 542 s join; the 130-position transfer total is the 1.92 GB above. The
M4 row adds no collectives and only host rounding passes (about 72K f32 values per layer) - below
0.1%, not measurable at this step size. No TIMEOUT_S increase is expected on these numbers, but
keep the existing margin (a +2 s join is the whole M2 effect).

**Runtime/evidence caveats.**
- The combine arm changes the collective count from the MoE reduce to the gather; ranks that
  disagree on the switch desynchronise the loop. There is no runtime cross-rank guard, only the
  recorded active set, so the four ranks' argv must be identical (one command line passed to all
  four).
- combine changes two evidence meanings by construction: moe_routed.payload_sha256 is the sha of an
  empty payload (no reduce) and routed_delta_sha256 is the routed part only
  (moe_combine_routed_f32 over the full rank sum), while the epilogue's ffn_out is the native
  finalize including the shared BF16 add. The boundary-replay helper's layer0-shared-delta.f32 is
  therefore the quantised remainder, not the shared row, under this arm.
- mhc changes seam digests by design; boundary-replay comparisons against f32 captures must be
  tagged with the active switch.
- The shared expert itself was still the host f32 engine reference (forward/moe.rs:1047) for the
  M2/M4 A/B rows, so the combine row isolates the combine arithmetic given our shared-expert value.
  The M3 loop-path native shared expert was built after those rows on `task/native-shared` (merged
  as `8c35c5f3`) and measured as its own row; see the measured-results section.
- Default-row byte identity: the wrappers with the switch off are bit-equal to the original seam
  functions (host test the_default_seams_are_the_unchanged_f32_arm), and all pre-existing tests
  pass; the Lead's no-flag four-rank rows on kits m1w30 and m1w33 reproduced the m1y29 logits bin
  sha `aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2` exactly, so the default
  arm is now certified end-to-end empirically (not only by host tests).

**Host tests for the loop's native functions.** The engine twins are covered by
crates/sparkflash-engine/src/forward/mhc.rs (the four captured boundary tests), the combine by
crates/sparkflash-kernels/src/mxfp4_ref.rs
(the_native_combine_is_the_release_arithmetic_on_a_captured_activation,
the_rank_sum_adds_in_ascending_rank_order), the shared expert by
crates/sparkflash-engine/src/forward/moe.rs
(the_recorded_shared_rows_add_to_the_unchanged_seam,
the_native_shared_expert_is_the_release_arithmetic,
the_native_shared_expert_treats_its_input_as_a_bf16_stream,
the_native_shared_expert_refuses_a_store_without_the_released_fp8_grid,
the_captured_m26_row_pins_the_activation_quantiser), and the CLI wiring by
native_rounding_names_every_supported_arm, the_default_seams_are_the_unchanged_f32_arm and
the_native_mhc_seams_round_the_stream_at_the_release_boundaries. Gates at the measured tips:
sparkflash-cli lib 322 passed, sparkflash-kernels lib 372 passed, sparkflash-engine lib 209 passed,
cargo check -p sparkflash-cli --features cuda clean, fmt and clippy clean.

## Measured four-rank results and adoption status (E record)

Both kits ran the same 138-id token window with the same 9 decode positions (129..137) against the
frozen D41 TD=4 oracle, comparator and bars (diagnostic readings; no bar or threshold change). The
no-flag row in each kit reproduced the m1y29 default seed logits bin sha
`aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2`, so every comparison below is
against the certified default arithmetic.

### Kit m1w30 - M2/M4 A/B (merge `67a12f85`, binary `e463c0e0...`, 20/20 pins unchanged)

| row | bars | kl_mean | kl_p95 | js_mean | top1 | step_ms | logits bin sha16 |
|---|---|---|---|---|---|---|---|
| default (no flag) | 6/10 | 1.1806 | 3.7126 | 0.1956 | 0.222 | 2775 | `aed6054d77d6` |
| `combine` (M2) | **7/10** | **1.0295** | **2.4930** | 0.1907 | **0.333** | 2789 | `227a0234cd6f` |
| `mhc` (M4) | 6/10 | 1.3344 | 4.5609 | 0.2031 | 0.222 | 2776 | `3f484a40b912` |
| `both` | 7/10 | 1.1704 | 3.0952 | 0.2066 | 0.111 | 2792 | `cc05d8b28017` |

Reading: **M2 is a measured positive** - kl_mean -12.8%, kl_p95 recovers from 3.7126 to 2.4930 (bar
PASS), +1 bar and top1 2/9 -> 3/9; on this oracle the M2 term was also a large part of the known
p129 anomaly row (p129 KL 3.7126 -> 2.4079 under combine). **M4 is a measured negative**: `mhc`
alone is worse than default on kl_p95 (driven by p129/p130), and `both` is worse than combine-only
(top1 1/9), so the measured recommendation is combine-only and M4 is **not adopted**. Wall: combine
+14 ms/step (~0.5%, within the model estimate plus scheduling noise), mhc +1 ms, both +17 ms.

### Kit m1w33 - M3 native shared expert (merge `8c35c5f3`, tip `e2524fdd`, binary `fbfd3a0c...`)

- Row 1 (no flag): logits bin sha equals the m1y29/m1w32 default seed exactly - the default-path
  byte-identity requirement held; the M3 code changes cannot move the default arm.
- Row 2 (`--native-rounding combine,shared`): four ranks rc=0, wiring engaged as designed -
  `active combine,shared`; 40/40 layers `shared_arith native_fp8_block32_bf16` with non-zero
  `shared_l2`; layer-0 `shared_f32_sha256` non-null and `shared_native_vs_f32_rel_l2 = 0.020136`.

| row | bars | kl_mean (bar) | kl_p95 | js_mean | top1 | step_ms |
|---|---|---|---|---|---|---|
| combine-only (m1w32 reference) | **7/10** | 1.0295 | **2.4930 (PASS)** | 0.1907 | **3/9** | 2785.22 |
| `combine,shared` (M3) | 6/10 | 1.3159 (0.5095) | **4.1851 (FAIL)** | 0.2036 | 2/9 | 2886.89 |

Per-position dKL vs the combine-only row: 129 **+1.7772**, 130 **+1.2773**, 131 -0.0809,
132 -0.1625, 133 +0.0103, 134 -0.2510, 135 -0.0478, 136 +0.0848, 137 -0.0300. The interior decode
rows 131..137 improve on kl_mean (0.5554 vs 0.6235, -11%) but the boundary pair p129/p130 regresses
by +1.78/+1.28 and dominates the aggregate, so the arm scores 6/10 under the declared gate.

**Decision: the M3 loop-path native shared expert is a measured negative and is NOT adopted.** The
default-off capability, its tests and the captured fixtures stay merged for future composition and
debugging, exactly as M4 does.

**Wall note.** The native shared row is not cheaper in the loop: +102 ms/step (+3.7%, 2886.89 vs
2785.22). Hypothesis (not profiled, recorded as a hypothesis only): the per-call quantise/dequantise
buffers in `native_linear` - the coded F8 weight is not materialised to f32 (it feeds the existing
`matmul_f8_scaled`), but each of the 3 Linears per layer allocates its `QuantisedFp8` code/scale
buffers plus a dequantised activation Vec, and each call allocates two row Vecs, i.e. 3x40 small
allocations per position, plus the layer-0 evidence f32 row. A future optimisation would reuse
`MoeScratch` buffers for codes/scales/dequantised activations and derive the layer-0 comparison
offline; no code change was made and none is planned without a request.

**Protocol observation (recorded, not an excuse).** The arm that most closely reproduces the
released shared-expert arithmetic improves the interior decode rows and regresses exactly the
boundary rows p129/p130, whose oracle values came from the prefill `prompt_logprobs` capture regime.
That is recorded as a protocol observation about the oracle capture regime, not as a justification:
the declared aggregate gate stands and the arm is not adopted.

## Switch and evidence design (for the staging step)

`--native-rounding <none|combine|mhc|shared|comma list>` on the `tp4-chain --forward` arm,
parsed once into a `NativeRounding { combine, mhc, shared }` value in `TokenStageRequest`, printed
in the run log and recorded in the forward evidence (next to the existing per-layer records); the
default is `none` and the default arithmetic path must stay byte-identical. The window row is
attributable only if the active set is in the evidence, so that field is part of the staging edit,
not an afterthought.

## Effort / risk / status

| arm | implementation | host twin | loop wiring | pin edit | effort | risk |
|---|---|---|---|---|---|---|
| M2 combine | `moe_combine_rank_sum` + `moe_combine_native` (landed) | passed, captured m26 activation (bit-equal native; loop chain differs 2813/5120, rel L2 3.66e-3 host) | applied, **measured positive** (m1w30: 7/10, kl_p95 PASS 2.4930; best measured D41 arm) | none | 1.5-2 h | gather wire volume 12x the delta reduce payload (737 KB vs 61 KB wire per rank per layer-position); mixed-arm/sequence guard must be added; 4-rank path cannot be run on the dev host |
| M4 mHC | `bf16_round_row`, `hc_pre_bf16`, `rms_norm_bf16`, `hc_post_bf16` (landed) | passed, captured boundary row (capture self-consistency bit-exact; boundary moves 20479/20480 hc_post and 5120/5120 hc_pre values) | applied, **measured negative** (m1w30: 6/10; `both` worse than combine-only) - not adopted | none | 1-1.5 h | seam digests change by design; replay/QA tagging |
| M3 shared | `shared_expert_rows` / `native_linear` / `add_shared_expert_native` (landed default-off, merged 8c35c5f3) | 5 host tests incl. independent release reference + captured m26 quantiser row (209 engine tests green) | applied default-off, **measured negative** (m1w33 `combine,shared`: 6/10, kl_p95 4.1851 FAIL vs combine 2.4930 PASS; wall +102 ms/step) - not adopted | none | ~2.5 h actual | capability stays merged default-off for future composition/debug; interior rows improve but p129/p130 boundary pair regresses |

Stopping point: **no pin-set edit, no cubin, no pinstep, no node access**. The M2/M4 loop wiring
(merged `67a12f85`) and the M3 loop-path native shared expert (merged `8c35c5f3`) are both
measured; only M2 combine is recommended. Recorded as E on `task/native-shared` (docs only); the
default-off M4 and M3 capabilities and their fixtures stay merged for future composition/debug, and
no further code or window work is planned without a Lead request.

# Task-8: the device MoE router (Phase 3, roadmap 6.3 item 4)

Status: **iteration A delivered and measured.** The engine's host router
(`crates/sparkflash-engine/src/forward/moe.rs`) is **untouched** and is still the path a served
token takes: the device router is an additional path, and the seam removal is a separate MR
(engine-dev) so that it can be bisected. Nothing here claims a host-time saving; that claim needs a
measured seam, which is the later MR. The two device runs in section 5 are the Lead's window runs on
the staged binary, not something this MR ran in CI.

## 1. What the kernel computes

`kernels/moe_router_sm121.cu` takes BF16 activations `[tokens][5120]`, the BF16 gate weight
`[384][5120]` and the F32 noaux_tc bias `[384]`, and writes the released routing outputs for each
token:

* `z = sum_k x[k] * gate[e][k] / gate_temp` with `gate_temp = 1.0`;
* `score[e] = sqrt(softplus(z))`, `softplus(x) = x` above 20 and `ln(1 + exp(x))` below it;
* selection by `score[e] + bias[e]`, top-6, ties keep the lower expert id (the same `>`-not-`>=`
  rule the engine's `Routing::run` uses);
* the weights are the **raw unbiased** scores of the selected experts, normalised by
  `w /= sum(w) + 1e-20` and then multiplied by `route_scale = 1.5`.

Outputs are `ids[tokens][6]` (u32), `weights[tokens][6]` (F32, already normalised and scaled) and
`scores[tokens][384]` (F32, unbiased).

Geometry: one block routes 8 tokens with 256 threads (32 lanes per token, one lane owns one expert
group), walking `k` in 32-value tiles; the full gate tile for a k-slice is staged in shared memory
(384 x 32 bf16), which is why the block owns 8 tokens rather than one. All geometry and the released
constants are `static_assert`ed in the kernel and pinned again from the source in
`crate::router_abi`.

Kept deliberately identical to the engine's arithmetic: the gate GEMM accumulates in **ascending k**
with `__fmul_rn` then `__fadd_rn` and **never an FMA**, which is the engine's serial
`acc += x[col] * w[col]` loop (the engine's own comment: fewer than four million multiplies here).
BF16 widens to F32 exactly on both sides, so no decode rounding differs. The one place the device
cannot be bit-identical is the score function: CUDA's `expf`/`log1pf` are not libm's
`exp`/`ln_1p`. `sqrtf` is correctly rounded, so it is not a source of difference. That split is why
the device pass bar is exact for ids and for the host path's weights against the twin, but
tolerance-based for the device's own scores and weights against the twin (section 4); the harness has
no separate device-vs-host weight comparison, and it does not assert the pre-softplus GEMM separately.

## 2. The CPU twin, and why it is the reference

`crates/sparkflash-kernels/src/router_ref.rs` is a Rust twin of the kernel's arithmetic (same
constants, same softplus, same selection, same fold order for the normalisation sum). Its tests run
on any host and include the two that matter:

* `the_twin_matches_the_engine_on_the_released_configuration` - the twin and
  `sparkflash_engine::forward::moe::route` agree **bit for bit** (ids and weights) on the released
  geometry;
* `the_twin_matches_the_engine_at_tiny_shapes` - the same at `ModelConfig::tiny()`, plus the
  bias-steers-selection-not-weights, tie-keeps-lower-id, `norm_topk_prob = false` and geometry-refusal
  cases.

So the chain the device window needs is: device kernel vs twin (tolerance bars), twin vs engine host
path (already proven exact). That keeps the device comparison independent of re-deriving the engine
semantics in the harness.

## 3. Pins and ABI

* `crates/sparkflash-kernels/src/router_abi.rs` (new, NOT `m4_abi.rs`, which is frozen by the D26
  ratchet): pins the released constants, the kernel's 10-parameter declaration parsed out of the
  source (six pointers, `tokens`, `trace_ring`, `layer_id`, `step_seq`), the kernel's geometry
  `static_assert`s (each evaluated, each required to name a `SPARK_` symbol), and the cubin entry
  from `baseline/kernel_baseline.txt`. Mutation-tested: swapping `ids`/`weights` types fails the
  signature pin.
* `crates/sparkflash-kernels/src/router_launch.rs`: the typed launch (argument words in declaration
  order, `grid = ceil(tokens / 8)` blocks of 256 threads, `tokens == 0` refused). `KernelId::MoeRouter`
  is the driver's symbol `moe_router_sm121`.
* Baseline: the router kernel **and its baseline pin are new in this MR** (on main `4967d68` the
  manifest has seven entries and `kernels/moe_router_sm121.cu` does not exist). It is the **eighth**
  entry in the manifest, in `build.rs`'s `PROBED_KERNELS` table, in both scripts' kernel lists, and in
  `tests/kernel_baseline.rs`'s count/texts (that test also parses the probe table as a body, so a
  rustfmt-wrapped entry still counts).
* Trace: the kernel reuses `TRACE_SCOPE_MOE_MXFP4` (the routed-expert consumer scope); the shipped
  cubin is the trace-disabled build (the launcher passes a null ring), and the trace-enabled source
  is exercised by the build script's combined probe.

## 4. The device window: what to build, run, and expect

Cubin (pinned toolkit): `nvcc -std=c++17 -arch=sm_121a -cubin -I kernels
kernels/moe_router_sm121.cu -o out/moe_router_sm121.cubin`, CUDA 13.0.3 (nvcc V13.0.88),
**37,384 B, sha256
`5992cef91c3e01be46c0691a5d905877b38a2c3002d90d5c31b80a003924769b`, 0 PTX membar lines**
(measured with the same image the baseline check uses). The harness refuses any other cubin **by
hash** before it launches it.

Harness: `./scripts/build-aarch64.sh --example=m9_router`, then on the node

```sh
m9_router --checkpoint <snapshot> --cubin-dir <dir> --tokens 8 --layer 0 --repeat 100
```

The harness prints its **pass bar before any number**, then measures:

| check | bar |
| :--- | :--- |
| ids vs the engine host path and the CPU twin | **exact**, all `tokens x 6` selections |
| host path's weights vs the twin | **exact** (bit-identical; asserted in the harness) |
| device scores vs the twin | max abs delta <= 1e-6, max ULP <= 8 |
| device weights vs the twin | max abs delta <= 1e-6, max ULP <= 16 (so within 16 ULP of the host path too, since host == twin) |
| determinism | two device runs byte-identical |
| cubin | 37,384 B and the sha above |

Why the device bars are not "bit-identical" everywhere: ids are exact, but the harness does not
assert the pre-softplus GEMM separately - the device output starts at the score, so the score bar
(1e-6 and 8 ULP) is the measured bound on the device's arithmetic. The scores pass through CUDA's
`expf`/`log1pf`, which are different implementations from libm's, so their last bits may differ and
the normalisation propagates that into the weights. The bars are declared in the harness source and
printed before the numbers they judge.

Operand provenance: the released `layers.<layer>.ffn.gate.weight` (BF16) and
`layers.<layer>.ffn.gate.bias` (F32), and `embed.weight` rows 0..tokens as the activation rows. The
embed row is a **stand-in**: the real router consumes the post-attention hidden, and the harness does
not have layer 0's FFN-side residual stream. Every claim is about the router arithmetic over released
gate bytes and released activation bytes, not about end-to-end layer text.

Cost note for reading the timing line: each block stages the **whole 384 x 5120 gate** in k-tiles, so
the launch re-reads about 3.9 MB of gate per block (17 blocks at 130 tokens, about 66 MB) - the kernel
is bandwidth-bound on the gate for small batches. The harness reports p50 per launch and per token
with the invocation printed; do not compare it against the host seam's cost until the seam MR
measures both sides on the same window.

## 5. The device runs (Lead's window, 2026-10-18T18:32Z)

Both runs are the Lead's window runs on the staged binary, not something this MR ran in CI. Staged
artifacts: `m9_router` sha256
`8f32174e31da3561a68029c98f0861b194310c1cf2cd1fb06bcb89ee5e78e5cb` (built from commit
`15b86d0`, the reviewed tree) and `moe_router_sm121.cubin`, 37,384 B, sha256
`5992cef91c3e01be46c0691a5d905877b38a2c3002d90d5c31b80a003924769b`. The node's MemAvailable floor
held at about 118 GiB in both runs, and the harness reported 4,028,288 B of device buffers for the
8-token run.

`--tokens 8` (PASS 7 of 7):

* ids: 48 of 48 exact against the engine host path and the twin (the two are identical);
* host path vs twin: weights bit-identical (asserted, as above);
* device scores vs twin: max abs delta 1.1920929e-7, max ULP 1, 2629 of 3072 bit-identical;
* device weights vs twin: max abs delta 5.9604645e-8, max ULP 2, 41 of 48 bit-identical;
* determinism: two runs byte-identical, 12,672 B compared;
* p50 6.8530 ms per launch (0.8566 ms/token) at the printed invocation.

`--tokens 130` (PASS 7 of 7, grid `(17, 1, 1)`):

* ids: 780 of 780 exact;
* device scores vs twin: max ULP 2, 43126 of 49920 bit-identical;
* device weights vs twin: max ULP 4, 707 of 780 bit-identical;
* determinism: 205,920 B byte-identical;
* p50 6.8541 ms per launch (0.0527 ms/token) - the launch is gate-bandwidth-bound, so the per-token
  cost drops with the batch, as the cost note above predicts.

The values are within the bars declared in section 4. These two runs are the reviewed-tree binary
`8f32174e` (built from `15b86d0`); the merged-main re-run follows.

**Merged-main re-run, 2026-10-18T19:14Z-19:16Z.** After !77 merged as `d14306c`, the Lead rebuilt
the harness from that merged main and re-ran both batches in a second window. The binary that ran
was the aarch64 `m9_router` with sha256
`a3ba20e4b3f9d0cc87267bfb90dbd204f9d18ab157a88114146afc228dd1cf85`, 4,970,848 B, built from
`d14306c`; cubin `5992cef9`, 37,384 B; window log `window-b3-20261018T191453Z.log`, `runs3/`. Both
runs PASSED 7 of 7 against the corrected comparison class of section 4:

* `--tokens 8`: host path vs twin ids 48 of 48 equal and weights bit-identical; device vs twin ids
  48 of 48, scores max abs delta 1.1920929e-7 (ULP 1), weights ULP 2; determinism 12,672 B
  byte-identical; p50 6.8530 ms per launch.
* `--tokens 130`: grid `(17, 1, 1)`; ids 780 of 780; device vs twin scores ULP 2, weights ULP 4;
  determinism 205,920 B byte-identical; p50 6.8534 ms per launch.

vLLM was stopped across four nodes for the window and restarted healthy afterwards. These are the
Lead's window measurements, not CI. The !80 review delta changed no kernel code - the cubin is still
`5992cef9` - so the device arithmetic measured above is the delivered one. The harness binary itself
is not byte-identical to a rebuild from the present main, because main also carries the !81 device-M1
engine changes and the delta's source shifts: a rebuild from the !81 main `8b3aef9` gives
`c920c1fdc...`, 4,971,048 B. The Lead re-runs on that final binary if a sha-matched measurement is
needed.

## 6. What this iteration does not claim

* **No host-time saving is claimed.** The seam is still in place; removing it and measuring the
  delta is the follow-up MR.
* The device numbers in section 5 are the Lead's window runs, not an end-to-end layer measurement:
  sections 1-4 are code, pins, host tests and bars, and the runs are the router arithmetic over
  released operands.
* No end-to-end layer text: the embed rows are stand-ins (section 4).
* Iteration B (the K = 5120 routed-expert gate/up FP4 GEMM) is a separate MR, not this one.

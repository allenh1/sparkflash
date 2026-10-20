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
**25,432 B, sha256
`42f6bbdcc6cf26890f2f0b1a24718e65da4ff0d0bedee7bca9a4598018b688cc`, 0 PTX membar lines**
(measured with the same image the baseline check uses). The harness refuses any other cubin **by
hash** before it launches it.

The K3-final warp-per-expert tree (`ffabd235`, 36,344 B) measured 0.1081 ms per 8-token launch, but
its tree reduction moved the scores to max ULP 131 and the weights to ULP 21, outside the serial
bars below. It is not the shipping arithmetic: this pin is the restored ascending-k serial per-output
reduction, one lane per token, so the score is the incumbent serial sequence again.

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
| cubin | 25,432 B and the sha above |

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

Cost note for reading the timing line: one block owns eight consecutive experts and reads each of
their 5,120-value gate rows once per token block, so a 130-token launch has 17 token blocks (about
17 x 3.9 MB of gate reads). The harness reports p50 per launch and per token with the invocation
printed; do not compare it against the host seam's cost until the seam MR measures both sides on the
same window.

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

## 7. The launch geometry is the router's cost: a partitioning plan (not implemented)

Status: **docs-only plan, no kernel change in this MR.** The 6.853 ms in section 5 is a real measured
launch, and at one launch per layer it is 274 ms of routing per decoded token - but the router is not
on the served path yet (the seam MR is separate), so this plan is about the Phase 3/4 kernel, not about
a number a user sees today. This section proposes an order of work and states what must be measured;
it does not promise a speedup.

### 7.1 What the measured launch says about where the time goes

The geometry is fixed by `router_launch.rs`: one block of 256 threads routes
`TOKENS_PER_BLOCK = 8` tokens, so `grid = (ceil(tokens / 8), 1, 1)`; a one-token decode is a
**single block**. Section 5 measured two launches on the Lead's 2026-10-18T18:32Z window:

| tokens | grid | gate bytes read by the launch | p50 per launch |
| ---: | :--- | ---: | ---: |
| 8 | `(1, 1, 1)` | 3.93 MB (one block) | 6.8530 ms |
| 130 | `(17, 1, 1)` | 66.8 MB (17 blocks) | 6.8541 ms |

The 17-block launch reads **17 times** the gate bytes and runs **17 times** the blocks (17 x 8 token
slots) of the 8-token launch, in the same p50. That rules out the reading that the launch is limited by
device-wide gate bandwidth: 66.8 MB in 6.854 ms is **9.8 GB/s**, far from any HBM figure (the same
observation is made about the indexer in
[roadmap-to-serving.md](roadmap-to-serving.md) section 6.2: 4.2 MiB in 4.27 ms is about 1 GB/s and
"equally far from any HBM figure"). The clock of a launch is the **per-block work**, and blocks run in
parallel.

The derivation from here is arithmetic, not a measurement: **6.8530 ms x 40 layers = 274.1 ms per
token** if a decode issues one router launch per layer. No end-to-end decode was timed for this
number.

### 7.2 Why one block is slow: per-thread serial MACs and a 32-way bank conflict

Per block, 256 threads = 8 warps, one token per warp, 32 lanes per token; each lane owns
`EXPERTS_PER_THREAD = 12` experts and walks K = 5,120 in 32-value tiles (`moe_router_sm121.cu:76-92`,
`118-135`). That is **61,440 serial MAC pairs per thread**, 15.73M per block, 31.46M f32 multiplies
and adds. If the SM issued one f32 op per lane-cycle (4 schedulers x 32 lanes = 128 lane-ops per
cycle, a Blackwell-class SM), the block's issue lower bound would be about **246k cycles, about
0.14 ms at a 1.7-1.8 GHz SM clock** (the clock is not recorded in this doc; only the ratio is used).
The measured 6.853 ms is **about 49x** that lower bound, so the gap is a serialisation, not a lack of
issue capacity.

The source points at one cause (a hypothesis until measured): `gate_tile` is
`__nv_bfloat16 [384][32]`, so an expert row is 64 B = 16 words. In one warp every lane has the
same `token_in_block` (`lane / 32`) and reads expert `12 * expert_lane + j` (`expert_lane = lane % 32`,
which inside a warp is `lane`) at K offset `k`; the word
offset is `192 * lane + 16 * j + k / 2`, and **192 mod 32 = 0**, so all 32 lanes of the warp hit
**one shared-memory bank** - a 32-way conflict on every gate load in the inner loop. That predicts
up to a 32x serialisation, the same order as the measured gap.

**This is a source-level hypothesis, not a measurement,** and it fixes the order of the plan: first
the change that tests it with the least code (7.4), then the partition (7.3), then both.

### 7.3 Plan A: `grid.y` expert partitioning plus a second-stage top-6 merge

**Stage 1.** `grid = (ceil(tokens / 8), 12, 1)`: twelve blocks, each owning **32 consecutive
experts** (12 x 32 = 384). With 256 threads and 8 tokens there are 32 lanes per token, so **one lane
owns one expert** instead of twelve. The block computes 32 unbiased scores per token in the same
ascending-K order with the same `__fmul_rn`/`__fadd_rn` pair (no FMA), writes its own 32 columns of
`scores[tokens][384]` (disjoint columns, no atomics) and its own top-6 **candidates** by selection
signal, each `(expert id, score + bias, raw score)` = 12 B, so 12 x 6 = 72 candidates and 864 B per
token. The local candidates use the **same order as the global rule** - highest selection signal,
ties keep the lower expert id, i.e. `(signal desc, id asc)` - so a tie that straddles a partition
boundary still resolves to the lower id after the merge.

**Stage 2.** A second kernel, one thread per token (`grid = ceil(tokens / 256)`), reads the 72
candidates, selects the global top-6 with the same rule as the incumbent scan - highest selection
signal, ties keep the **lower expert id**, which is a `(signal desc, id asc)` sort - then reads the
six raw scores from `scores[]` (written by stage 1; a separate launch is the visibility barrier),
sums them in selection order, adds 1e-20, divides, multiplies by `route_scale`, and writes `ids` and
`weights`. Stage 2 is microseconds: 8 tokens x 78 values, and 130 tokens fit in one block.

**Determinism.** No atomics, no cross-block ordering dependence: every score is one deterministic
f32 sequence (unchanged), and stage 2 is a pure function of the 72 candidates and six scores.

**Arithmetic invariance.** Stage 1 keeps the per-expert accumulation order, the two roundings per
MAC, the softplus branch and the score function; stage 2 keeps the normalisation order (selection
order) and the release constants. The split must therefore be **bit-identical** to the incumbent on
`scores`, `ids` and `weights` for the same inputs - any difference is a defect, not a tolerance.

**Estimate, with its assumption stated and untested.** If per-block time is proportional to
per-thread MAC work, stage 1 is about **6.853 / 12 = 0.571 ms per layer**, i.e. **22.8 ms per token
across 40 layers** (from 274). The 1-block vs 17-block result does **not** support that assumption: it
shows the launch clock is set by the per-block work (17 blocks in the same p50), not that per-block
time scales with per-thread MAC count. The 12x figure is therefore a hypothesis for 7.5 item 7 to
test, not a projection. The bank
conflict also improves, from 32-way to 16-way (one expert per lane gives word offset
`16 * expert + 16 * j + k / 2`, i.e. banks 0 and 16 only), which the estimate does **not** count. Stage 2 and the
extra launch are not in the estimate either: +40 launches per token at a 5-10 us launch (unmeasured
here) is 0.2-0.4 ms per token, which CUDA graphs
([roadmap-to-serving.md](roadmap-to-serving.md) section 6.3 item 3) would absorb; the stage-2 kernel
itself is not a term.

**Cooperative variant (not first).** One kernel with `grid.sync()` between the two stages avoids the
second launch, but the driver wrapper (`crates/sparkflash-kernels/src/cuda_driver.rs`) has no
cooperative-launch entry point today (no cooperative symbol in the module), so it is more code and a
new launch ABI. The second-kernel variant is the smaller change and is graph-capturable; do it first.

### 7.4 Plan B: the layout-only fix that tests the hypothesis

Pad the gate tile's K stride so that expert rows do not alias one bank. `[384][34]` bf16 makes an
expert row 68 B = **17 words**: at one expert per lane the offset is `17 * lane`, and
`gcd(17, 32) = 1`, so the 32 lanes hit **32 distinct banks (conflict-free)**; at the incumbent
twelve experts per lane the offset is `204 * lane = 12 * lane` (mod 32), eight banks, a 4-way
conflict (8x better than today's 32-way). No argument changes, no arithmetic changes, shared memory
goes from 37,888 B to 39,424 B, and the only pins that move are the cubin baseline
(`baseline/kernel_baseline.txt`, `build.rs` probe table, the two scripts and
`tests/kernel_baseline.rs`'s count, as in section 3) and the compiled shared-memory total - the
`gate_tile` size is in `moe_router_sm121.cu`, and the total is the cubin's `.nv.shared` value,
which no pin in the tree records (the section 3 pins are the signature and the constants). This is the cheapest way to test 7.2's hypothesis, and it is worth doing even if Plan A
follows, because the split alone leaves a 16-way conflict on the same load.

Not recommended first: transposing the tile to `[K_TILE][N_ROUTED]`, which moves the conflict to the
staging write and still needs a padded stride.

### 7.5 The test plan (all of it; the speedup is the last item)

Every step runs in the existing `m9_router` harness (section 4) on the same released operands.

1. **Semantics pinned before speed.** The CPU twin (`router_ref.rs`) is unchanged; it is the
   semantic reference for both plans.
2. **Device vs twin, same bars as section 4:** ids exact, host-path weights vs twin exact, device
   scores <= 1e-6 and <= 8 ULP, device weights <= 1e-6 and <= 16 ULP, two device runs byte-identical.
3. **Equivalence to the incumbent kernel:** for tokens 1, 8 and 130, run the incumbent single-block
   kernel and the new path on the same released gate/activation bytes and compare `scores`, `ids`
   and `weights` **byte for byte**. The split is arithmetic-preserving, so this is an equality test,
   not a bar.
4. **Tie rule, in a partition and across partitions:** a synthetic gate/bias with two experts
   tying on `score + bias`, once inside one partition and once in different partitions; both assert
   the **lower** id is selected - stage 1's local candidate order and the merge's secondary key must
   be the same `(signal desc, id asc)` rule.
5. **Top-6 distribution edge cases:** all six selected experts inside one partition; selected six
   spread 6 x 2 across partitions; the last **token block** short at `tokens` not a multiple of 8
   (the `grid.x` block, not a partition - every partition owns a full 32 experts).
6. **Geometry pins:** `router_abi.rs` pins the new grid.y and partition count (12 x 32 = 384) and both
   kernels' declarations (stage 1 keeps the existing ten parameters; stage 2 adds its own); the
   mutation test on the signature argument types stays.
7. **Performance, recorded not promised:** p50 for tokens 1, 8 and 130 with the printed invocation and
   the gate bytes read, for (a) Plan B alone, (b) Plan A alone, (c) both; sweep partitions 2/3/4/6/12
   to test the scaling assumption rather than assuming the 12-way point is best. The pass bar for
   each MR is "arithmetic bit-identical and the p50 recorded"; a speedup is a measurement, not a
   claim in the MR text.
8. **Baseline:** each new or changed cubin gets its own baseline entry and hash before the harness
   will launch it (the harness refuses any cubin whose hash is not pinned).

### 7.6 The honest baseline

* The 6.853 ms is a **correctness harness** over released gate bytes and embed-row stand-ins
  (section 4), not the decode path: `sparkflash serve` still routes on the host. The seam MR must
  measure both sides on the same window before any seam win is claimed, as section 6 already says.
* The 274 ms/token is **derived** (6.853 x 40), not measured end to end.
* The 0.571 ms/layer and 22.8 ms/token figures in 7.3 are **estimates under a named assumption**
  (per-thread MAC work dominates the per-block clock); the measurements in 7.5 item 7 decide whether
  they hold.
* The floor after the split is the **3.93 MB gate read per launch**; the incumbent's aggregate
  9.8 GB/s at 130 tokens is why the launch is nowhere near that floor today. The plan does not put a
  number on how close the split gets, because the conflict hypothesis in 7.2 is not yet measured.

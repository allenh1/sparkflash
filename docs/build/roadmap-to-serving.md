# The road to serving: from one layer on the GB10 to four Sparks in parallel

**Status.** Design document, 2026-10-17, branch `docs/roadmap-to-serving`. It is not a status report:
it says what we are going to build, in what order, why that order, and how each phase is proven. It
continues [device-path-design.md](device-path-design.md) section 11 (the first slice) and
[QUALIFICATION.md](../QUALIFICATION.md) (the acceptance instrument) with the work between them, and
it assumes the reading of the model already recorded in
[../architecture/00-overview.md](../architecture/00-overview.md) and
[../engine/01-forward-order.md](../engine/01-forward-order.md).

**Rules of evidence used throughout.** A number is quoted with the host it was measured on and the
method that produced it; an unmeasured claim is labelled **design** or **hypothesis** in place, not in
a disclaimer at the back. Section 12 collects every measured number in one table, with its host, and
lists what each one does *not* establish. "The node" means `node0` (the head node,
`192.0.2.80`) unless another host is named.

---

## 1. The position we start from

**One layer of forty runs on the device, and two of its three stages are partly on the host.**
Everything else — the other thirty-nine layers, the KV path for a real sequence, the head, sampling,
and all four nodes — does not exist on the device path yet.

That is not a summary anyone has to soften. It is the reason the phases below are ordered the way
they are. What exists is one layer deep and one token wide, and it was built to be a measurement, not
a product: the M7 harness runs **one stage per process**, prints the argv, the device bytes and the
comparison class *before* any number, checks MemAvailable before and after every launch and refuses
to continue under a floor. That discipline is the prototype of the engine's discipline, and it is why
the three stages below can be read at all.

### 1.1 The three stages, and the class of each comparison

All three were measured on `node0` in the M7 device window of 2026-10-17, with vLLM stopped by
the operator (device empty; MemAvailable 118.2 GiB at process start, printed by the harness), by
`crates/sparkflash-kernels/examples/m7_layer0/main.rs` (one module per stage beside it) in review at
`!26`; split from one file by 954d7af with no behaviour change. The harness is
not on main yet; the numbers below are the run's, and the harness's own comparison-class line is
quoted with each row because the class is part of the number.

**Retirement note (2026-10-18):** every number in the table below is a FIRST-WINDOW measurement of
the pre-B0, pre-D8 code, retained as history. B0 retired the stream-mean fold (D22) and D8 moved
the activation quantisers to the reference ceil rule (D23); the current measurements are in
section 12.1, and the class line in row 1 describes a predictor the current code no longer uses.

| Stage | Kernel | Comparison class (printed before the numbers) | Measured on node0, 2026-10-17 | Device bytes |
| :--- | :--- | :--- | :--- | ---: |
| 1. mHC | `mega_mhc_fused_bf16_sm121` | Internal consistency against the in-crate f32 reference on real layer-0 bytes; **not model fidelity** (at the time the predictor was the project's stream-mean fold, retired by B0/D22; current code
consumes the release's `[24, n*d]` projection on both sides). The released class (D22: `hc_attn_base` and `hc_attn_scale` consumed, `C` compared raw, released split-Sinkhorn) is what the current code prints and what the second-window re-run measured. | **20,480/20,480 bit-exact**; scales **160/160**; **4/4 tokens PASS** on the released-arithmetic cubin (second window); kernel time **not re-measured**, p50 0.0227 ms is the pre-released-arithmetic figure | < 1 MiB |
| 2. Attention | `flash_mla_csa2_sm121` | Tolerance 99.5 % exact, abs delta at most 1.6e-2; **sink included on both sides** (task-36 closed: the kernel takes a `[64]` f32 argument, the reference is `mla_ref::attention_with_sink`); candidates harness-supplied; indexer outputs not compared. | **163,838/163,840 = 99.9988 % exact, max abs delta 9.765625e-4, max ULP 1** (second window, measured on the pre-D8 stage-1 chain; the merged-chain number is max abs 2.4414063e-4, recorded in section 12.1); timing **5.73 ms** is the first-window number, re-measure owed | **4.2 MiB** |
| 3. MoE combine | `b12x_mxfp4_moe_sm121` | Tolerance 99.9 % exact, at most 2 bf16 codes; **router on the host seam**; gate/up, SwiGLU, quantiser host; 8/32/64 resident experts. | **20,480/20,480 exact, max ULP 0**; **0.821 / 0.847 / 0.850 ms per token** over four chained launches | 100–802 MB |

The arithmetic that follows is **extrapolation, not a measurement**: 0.0227 + 5.73 + 0.85 is about
**6.6 ms per layer**, times forty about **264 ms per token**, i.e. a ceiling of about **3.8 tok/s**
per node *if every layer were this layer, every token this token, and no host seam existed*. None of
those conditionals holds (section 1.2), and the number is useful only as the denominator for
section 6.

Two secondary results belong here because later phases rely on them:

- **The indexer is the single largest term.** 4.27 ms is 65 % of the per-layer budget, 2.9 times the
  attention proper, and about 5 times one measured MoE-combine stage. Its cost is set by the HSI pool
  geometry (2,048 blocks), not by the 128-candidate window, so it does not shrink on shorter
  prompts: it is the first target of section 6 and the reason a 264 ms/token figure is not a
  "the rest is fine" figure.
- **Kernel builds are reproducible.** The committed baseline has seven entries, all re-pinned under
  the pinned toolkit `V13.0.88`: the D8 update moved `mega_mhc_sm121` to 70,312 B and
  `swiglu_mxfp8_quant_sm121` to 21,776 B (sha256 `4dbef63b...`), and this task-3 branch moves
  `flash_mla_csa2_sm121` to 44,080 B (sha256 `bbb216b7...`) while the other five entries stay
  byte-identical. (History: the `!26` `__syncthreads` fix had moved flash_mla from 32,608 to
  32,624 B before the engine window format changed its source again.) The D8 and task-3 rows below
  record the measured diffs; future kernel changes are task-43 and a Phase 0 exit item.

### 1.2 What the stages do not establish

Every exclusion below is printed by the harness next to the number it limits; they are repeated here
because every downstream phase inherits them.

1. **No stage is model fidelity.** Stage 1 is internal consistency against *our* reference on real
   bytes. Stage 2 is tolerance-based with the sink included on both sides (task-36 closed), a harness-supplied
   candidate list and no comparison of the indexer's own outputs. Stage 3 is tolerance-based over a
   *quantised* activation: the same chain with an f32 activation differs in essentially every output
   (max abs delta 14.19 at 8 experts, 12.91 at 32 and 64), so the exact agreement says the kernel
   computes the reference's arithmetic, not that either reproduces the f32 expert.
2. **No stage is a layer.** Stage 1 is the mHC sweep, stage 2 the attention kernel, stage 3 the
   combine projection. The dense projections that sit between them (`wq_a/wq_b/wkv`, `wo_a/wo_b`), the
   router, gate/up and SwiGLU are absent from the device path. The seventh kernel — the MXFP8
   activation quantiser — has **no device-versus-reference comparison at all**.
3. **No stage is a token.** A token needs all forty layers, the head, and sampling; none exists on
   the device path. The 264 ms figure is arithmetic.
4. **No stage is beside a serving workload.** The device was empty. The same node with vLLM resident
   had 3,115 MB of MemAvailable and refused a second CUDA context outright
   ([device-path-design.md](device-path-design.md) section 9). Every number in section 1.1 is a
   capacity statement about an empty node, not about a serving one.
5. **No number exercises a sequence.** The stage-2 KV rows are quantised by the harness into the
   kernel's format; the engine's own window path rounds KV to FP8 instead. The candidate list is
   supplied, not built. There is no cache, no prefill, no decode step, no window wrap.

### 1.3 The device capacity facts

These constrain every phase, and they are measurements, not estimates:

- The device reports **124,610 MiB** total (`cuDeviceTotalMem_v2`,
  [device-path-design.md](device-path-design.md) section 9).
- With vLLM resident, MemAvailable was **3,115 MB** for the whole node (task-30) and a second CUDA
  context failed with `OUT_OF_MEMORY` on both the primary-context and the explicit-context route
  ([device-path-design.md](device-path-design.md) section 9). The device is shared with the serving
  workload; a device window is an operator decision, not a build step.
- The loader refuses any single tensor above **32 Mi elements** (`REFERENCE_TENSOR_LIMIT`,
  `crates/sparkflash-engine/src/linalg.rs:76`), so `embed.weight` and `head.weight`
  (661,913,600 elements each) can only be read row-wise, and every Engram table (98.3 G elements) can
  never be materialised at all. The upload path is a streaming path by construction
  ([device-path-design.md](device-path-design.md) section 4).

### 1.4 The cluster facts, and which of them are verified

Measured on `node0` on 2026-10-16 with the cross-compiled aarch64 binary (`sparkflash doctor
--local`, read-only; full transcript in [aarch64-pipeline.md](aarch64-pipeline.md) appendix C and
task-30):

| Fact | Value | Verified on |
| :--- | :--- | :--- |
| Ranks | `192.0.2.80, .84, .68, .59`; two addresses per node, one per rail; head is `.80` on rail 1 and `.85` on rail 2 | [crates/sparkflash-comm/tests/hardware_gates.rs](../../crates/sparkflash-comm/tests/hardware_gates.rs) line 18, [DECISIONS.md](../DECISIONS.md) D12 |
| Rail names in the specification | `enp1s0f0np0` / `enP2p1s0f0np0`, both **DOWN**, 40 Gb/s — the wrong function of each dual-function card | head node only |
| Rails actually carrying traffic | the f1 siblings, **ACTIVE at 200 Gb/s** (4X HDR), PCIe `0000:01:00` and `0002:01:00` | head node only |
| MTU | **1500** on all four netdevs; the runbook says 9000 — one of the two is wrong and the question is open | head node only |
| Driver | `580.173.02`, older than the `580.178.04` gate; kernel `6.17.0-1029-nvidia`; CUDA 13.0; `nvidia-smi` exits 2 | head node only |
| Memory | 3 GiB MemAvailable against the 114 GiB gate, because vLLM holds the rest | head node only |

**Unverified, and known to be unverified:** the state of the other three nodes (rail names, MTU,
driver); the RoCE fabric has never carried a byte of this project — the ibverbs backend and the three
cluster gate bodies are real now (typed errors, never `RdmaNotBuilt`), but none has run and no frame
has crossed a wire; the 10 KiB All-Reduce gate has never been measured and its preflight threshold in
[DECISIONS.md](../DECISIONS.md) D17 (22 µs) already fails the 20 µs requirement. Section 7 builds on that fabric anyway, because the plan needs it; it labels
every such statement.

### 1.5 What does not exist at all

- Thirty-nine of forty backbone layers on the device, and the three MTP layers.
- The per-layer dispatch that layer heterogeneity requires (section 4.1).
- A device KV cache for a real sequence, and the compressor/CED/indexer paths that feed it.
- The head on the device, and sampling.
- The four-node system, the collectives it needs, and the acceptance numbers for either.
- A device `InferenceEngine` behind the HTTP surface: the engine seam exists
  ([crates/sparkflash-api/src/engine.rs](../../crates/sparkflash-api/src/engine.rs)), the reference
  engine exists, the device engine does not.

---

## 2. Why the phases are in this order

The order below is a claim and it needs an argument, so here it is. Four propositions carry it, and
one exception cuts across it.

**P1 — An experiment is worth what its attribution is worth.** The M7 harness was built one stage per
run precisely so that a mismatch localises to a boundary. Phase 0 finishes that work: until the
coefficient boundary and the determinism of `hc_stats` are trustworthy, a difference between a
40-layer forward and the reference cannot be assigned to a layer, a kernel or a race. That is why
Phase 0, not Phase 1, is first — and why it is scoped to the two defects that destroy attribution
(task-38, task-39) plus the tensor the kernel and the reference both silently drop (task-40), and
not to the full defect list.

**P2 — Structural errors before numerical errors.** A layer-plan bug (a reuse layer that reads the
wrong source, a ratio applied at the wrong granularity) is found by a **single-token, forty-layer
run** in one window, because it produces a gross, layer-local mismatch. A KV-sequence harness
multiplies the state space by layer mode times position times window wrap and hides exactly those
bugs. Breadth first, depth second: the shape of the forward is checked before the state machine
inside it.

**P3 — The sequence path before performance.** The indexer's cost is a function of the candidate and
pool geometry, and the sparse attention path changes between prefill and decode; the measured
4.27 ms is one geometry in one mode. Tuning it against a harness-supplied candidate list would tune a
proxy. The sequence path is what makes the candidate set real, which is what makes the indexer
rewrite measurable rather than plausible.

**P4 — Performance before tensor parallelism.** The collective budget at the 10 KiB gate's 20 µs over
83 collectives per token is **about 1.7 ms/token** (arithmetic, section 7.2) against 264 ms/token of
measured kernel time. TP is not needed to go faster; it is needed to **fit** (section 7.5). Its cost
is a fourfold multiplication of every tuning decision, and its benefit (memory) only becomes visible
once the single-node memory plan is known. Tuning under sharding before the single-node number exists
would multiply the search space and hide regressions.

**The exception.** The fabric probe is cheap, independent of everything else, and settles the biggest
design risk in this document — whether four nodes on these rails can meet the 20 µs requirement at
all. It is the one Phase 4 experiment that should run **in parallel** with Phase 1 (section 9,
experiments E4/E5), because a failure there changes the architecture, not just the schedule.

**And what is *not* a phase:** the serving surface. The API, scheduler and KV crates already exist
(they serve the reference engine). They must not be reworked before a device engine exists, but their
gates are the acceptance instrument, so Phase 5 is the act of *meeting* an instrument that is already
written — not of writing one.

---

## 3. Phase 0: make the three measurements trustworthy

**What we build.** The correctness floor under the three stages: the mHC coefficient boundary the
kernel publishes, the determinism of the quantisation that follows it, the mHC bias tensor that is
currently dropped on both sides, and the baseline entries that make the seventh kernel and the
`flash_mla` fix reviewable. Nothing here is a feature; everything here is a precondition for
believing a 40-layer run.

### 3.1 task-38 — the mHC coefficient `C` is published unnormalised

**What is wrong.** The reference's Sinkhorn schedule is not a generic "20 row + 20 column
normalisations": it is softmax over the key axis, then `+eps` and divide by `(colsum + eps)`, then
19 alternating passes ([../architecture/02-mhc.md](../architecture/02-mhc.md) section 8, delta D3;
`kernel.py:437-458`). The kernel publishes the raw coefficient logits `C` instead, and the M7 harness
compensates by applying the softmax itself before comparing (the pre-change harness printed
"C-after-softmax within 1e-4"; this MR deletes that line and the
compensation). The compensation is honest in the harness and fatal in an engine: every consumer of
the coefficients would have to reproduce it, and any consumer that did not would still "pass" the
stage that was supposed to catch it.

**What we build.** The kernel applies the reference's normalisation before it writes `C` (and, if the
comparison shows it, the exact epsilon placement of D3 — the test below is what distinguishes a
correct schedule from a plausible one). The harness loses its softmax and compares raw `C`.

**Proven by.** `m7_layer0 --stage 1` compares the kernel's raw `C` against the reference's `C` at
1e-4, with a deliberately non-trivial token (a real embedding row, not zeros); and a test asserts
that the harness no longer transforms the kernel's output — so the check cannot silently return to
the harness-side compensation.

**Status (D22).** The kernel publishes the normalised released `C`, the harness compares it raw (the
softmax is deleted), and `spark_released_split_sinkhorn` plus
`the_released_split_sinkhorn_adds_eps_once_before_the_first_column_pass` pin the epsilon placement;
the stage-1 device run that would re-measure it is owed a window (section 3.6).

### 3.2 task-39 — `hc_stats` is not deterministic

**What is wrong.** The kernel accumulates the sum of squares with `fp32 atomicAdd`, so the addition
order is not fixed. The harness's own finding line says what that costs: `1/rms` differs because of
the atomic order, and on identical input the pre-change stage-4 repeat measured **one differing byte
between run 0 and a later run**. (The separate 6-of-5,120 FP8-code difference is stage 1's
device-versus-reference finding, a different comparison; it is not evidence of nondeterminism.) A nondeterministic quantiser
upstream of every layer's activation makes "byte-identical" untestable and makes the difference
between a real regression and a scheduling accident invisible.

**What we build.** A deterministic reduction of the 5,120-wide sum: one block per row, per-lane
partials, a shuffle tree, then a fixed-order cross-warp combine in shared memory. The cost is
negligible next to the 5.73 ms attention stage (0.0227 ms/launch on the pre-change kernel), and it must
apply to every kernel that accumulates a normalisation statistic, not only this one.

**Proven by.** The measurement is `m7_layer0 --stage 4 --repeat 100`: the same input bytes 100
times, requiring `hc_stats`, the 5,120 FP8 codes and the 160 scale bytes to be byte-identical across
all runs. The host halves pin the fixed-slot structure (`crates/sparkflash-kernels/tests/mhc_determinism.rs`)
and the D21 contract (`crates/sparkflash-kernels/tests/determinism_contract.rs`) without a device. The old fp32 `atomicAdd` reduction failed the
contract by construction (tile arrival order changed `hc_stats`), and on identical input the
pre-change stage-4 run measured one differing byte between run 0 and a later run. One device window
suffices; the harness already exists.

**Status.** The kernel change is in (fixed slot per tile, binary64 tile-order combine,
`kernels/mega_mhc_sm121.cu:197-222`), the harness's stage 4 is the N = 100 exact-bytes measurement,
`crates/sparkflash-kernels/tests/mhc_determinism.rs` pins the kernel structure without a device, and
`crates/sparkflash-kernels/tests/determinism_contract.rs` pins the D21 contract. The device run is owed: on 2026-10-17 the launcher refused with
`cuCtxCreate_v2 -> 2 (OUT_OF_MEMORY)` because the resident model holds the only usable context —
a capacity fact, recorded rather than worked around.

### 3.3 task-40 — `hc_attn_base` is dropped by the kernel and by our reference

**What is wrong.** Every layer's checkpoint carries an `hc_attn_base` of 24 f32 values — an additive
per-coefficient bias — and neither the kernel's `H()` nor our reference's `h_of` has a term for it.
The harness reads the tensor only to print its range and says so; the pre-change harness printed
"not consumed: see the gap line", and this MR deletes both the
line and the gap. The released forward models it; task-40 records the gap. A model-fidelity
claim cannot be made while a per-layer parameter that the released forward consumes is absent from
both sides of a two-sided comparison — the comparison is self-consistent, and wrong.

**What we build.** The reference's `h_of` gains the bias first (the reference is the test oracle and
must be complete before the kernel is judged), then the kernel's `H()` takes the 24 floats as an ABI
argument. The stage-1 comparison starts consuming the tensor it already reads.

**Proven by.** The harness's "not consumed" line disappears; a test drives a non-zero
`hc_attn_base` and asserts the coefficients move by exactly the bias's effect; and the stage-1
comparison passes with the bias consumed. The ABI change is the same shape as every later kernel ABI
change: versioned in the launcher, recorded in the MR.

**Status.** The reference takes the bias first (`bias_into_kernel_rows`, `mega_mhc_ref.rs:606`), the
kernel takes `hc_base` as an ABI argument (`kernels/mega_mhc_sm121.cu:93`, `m4_abi.rs` signature
pin), the harness's "not consumed" line is gone, and
`the_released_scale_multiplies_each_group_before_the_bias_is_added` drives a non-zero bias. This MR
also takes `hc_attn_scale` into the kernel, which closes roadmap-section-10 unknown 1 on the same
path.

### 3.4 task-43 — the baseline did not know the seventh kernel, and the flash_mla fix was unreviewable

**What was wrong (historical).** `!26` changed `flash_mla_csa2_sm121.cu` (one `__syncthreads` fix)
and added `swiglu_mxfp8_quant_sm121.cu`. At that time the baseline in
[kernel_baseline.txt](../../crates/sparkflash-kernels/baseline/kernel_baseline.txt) still recorded the
pre-fix `flash_mla` cubin and had no entry for the seventh kernel, so
[crates/sparkflash-kernels/scripts/check-kernel-baseline.sh](../../crates/sparkflash-kernels/scripts/check-kernel-baseline.sh)
was red and the CI for `!26` could not be read as "the kernels are unchanged where they should be".
The pinned-toolkit result — five of six entries byte-identical, only the fixed kernel moving — was
exactly the evidence that made the seventh kernel reviewable; it had to be recorded, not inferred.

**What was built (task-43 closed by D8).** The baseline was regenerated under the pinned
`nvcc V13.0.88` (the image and the script are both pinned) and now has seven entries: D8 moved
`mega_mhc` to 70,312 B and `swiglu` to 21,776 B (sha256 `4dbef63b...`) while five entries stayed
byte-identical, and this task-3 branch moves `flash_mla` to 44,080 B (sha256 `bbb216b7...`) as the
single further diff. The baseline header names seven kernels.

**Proven by.** `check-kernel-baseline.sh` green in the `cuda-abi` CI job with seven entries, and the
kernel-baseline row in section 12.1 records the measured diff.

### 3.5 The determinism contract we adopt here (adopted as D21)

The project needs one sentence that every later test can cite, and it is now entry
[D21](../DECISIONS.md) in `DECISIONS.md`; `crates/sparkflash-kernels/tests/determinism_contract.rs`
fails if a statement or the harness's citation of it disappears:

- **D-a: kernel determinism.** For fixed input bytes, launch configuration and device, a forward
  kernel's output bytes are identical on every run. The M7 harness's per-stage runs did not test it;
  stage 1's `hc_stats` is the known counter-example Phase 0 removes.
- **D-b: reduction determinism.** The M3 one-shot reduction is bit-identical across ranks *by
  construction*: f32 accumulation in ascending rank order, one rounding to bf16 at the end
  ([crates/sparkflash-comm/src/lib.rs](../../crates/sparkflash-comm/src/lib.rs) module docs). This is
  the property Phase 4's numerical criterion rests on (section 7.3).
- **D-c: no assumed cross-boundary equality.** Device-versus-reference equality is never assumed; it
  is either exact by construction or a named tolerance with a reason, printed before the number. This
  is already the M7 harness's rule and is the only rule that survives four nodes.

### 3.6 Exit criteria for Phase 0

Each of these is a test that can fail, not a checklist item:

1. `m7_layer0 --stage 1` passes with **raw `C`** compared (no harness-side softmax) and
   `hc_attn_base` consumed; the harness prints neither the "not consumed" nor the "C-after-softmax"
   caveat.
2. `m7_layer0 --stage 4 --repeat 100` (same input) passes: `hc_stats`, FP8 codes and scales
   byte-identical; its host-side structural half is `crates/sparkflash-kernels/tests/mhc_determinism.rs`.
3. `check-kernel-baseline.sh` green on `!26`; the baseline names seven kernels and records the
   `flash_mla` fix.
4. A test asserts the three D-a/D-b/D-c statements exist and are cited by the harness's output — a
   "contract is checked, not described" test. It is
   `determinism_contract::the_determinism_contract_is_written_down_and_the_harness_cites_it`
   (`crates/sparkflash-kernels/tests/determinism_contract.rs`).
5. `cargo test -p sparkflash-kernels --test kernel_baseline`, `--test mhc_determinism` and
   `--test determinism_contract` green in CI without a device. The determinism and contract tests
   have no device skip path; `kernel_baseline` skips its nvcc comparison when nvcc is absent, and
   the CI `cuda-abi` job runs `check-kernel-baseline.sh` in the pinned container, so the baseline
   gate is enforced there rather than by the skip.

**Status of this list.** All five are met. Items 3, 4 and 5 in CI (`check-kernel-baseline.sh` green
under V13.0.88, seven entries). The two device halves ran in the second window of 2026-10-17 with the
node's model stopped and cubins rebuilt on the node from these sources: item 1 PASSed 4/4 tokens with
raw `C` and `hc_attn_base` consumed and no caveat lines, and item 2 PASSed `--stage 4 --repeat 100`
(4 tokens x 100 launches, every compared byte identical).

### 3.7 Tasks

`task-38`, `task-39`, `task-40`, `task-43` (all in flight); new: `phase0: determinism test` and
`phase0: raw-C stage-1` (kernel-dev). Cost: one kernel window, one harness edit, no new kernels.

**Why not the sink here?** Because the sink is not a numerics defect; it is a *missing tensor* in the
attention ABI (task-36), and it belongs with the first device attention path that consumes it
(Phase 1, section 4.3). Keeping it out of Phase 0 keeps Phase 0 one window and one failure class.

---

## 4. Phase 1: breadth to one real token

**What we build.** All forty backbone layers in one forward on the device, in one window, for one
token, with the per-layer dispatch that layer heterogeneity requires, the host seams named, and a
per-layer boundary diff against the reference. This is the first time the model — not a kernel —
exists on the device path.

### 4.1 Why layer heterogeneity is the main design problem

The backbone is not a chain. Three config lists make it a DAG, and the reference implements the DAG
with process-global slots ([../architecture/01-csa2-attention.md](../architecture/01-csa2-attention.md)
section 2, [../engine/01-forward-order.md](../engine/01-forward-order.md) section 4):

- `kv_source_layers = [2, 8, 14, 20]` own a compressor and publish the **shared compressed KV**;
  layers 3–7, 9–13, 15–19 and 21–39 read that slot.
- `index_source_layers = [2, 8, 14, 20, 24, 28, 32, 36]` publish `topk_idxs`; layers 24/28/32/36 are
  **Reindex** layers — they keep the source's KV and index K but recompute their own Top-K inside the
  pool.
- `candidate_source_layer = 20` builds the **HSI pool**; every layer above 20 masks its scores with
  that pool.
- Of forty backbone layers: **3 Full, 1 Full + HSI source, 4 Reindex, 30 Reuse**, and two ratio-0
  layers (0 and 1) with no compressor and no indexer at all. The three MTP rows are window-only
  DSpark stages (`LayerMode::Mtp`) and outside this phase.
- Layer 1 and layer 14 additionally carry the **Engram** tables, and layers 37–39 are the **DSpark**
  targets — the base forward computes them as ordinary ratio-1 layers, the draft path does not exist
  yet.

A uniform `for layer in 0..40` loop with one tensor set computes the wrong thing **silently**: a
Reuse layer that projects its own KV, or a ratio-2 layer that reads a ratio-1 slot, produces a
plausible hidden state and a wrong token. This is the single largest design risk between the current
state and a served model, and it is structural: no amount of kernel work finds it.

### 4.2 The layer plan: derive it, assert it, dispatch from it

**What we build.** One `LayerPlan` for the model - one row per backbone layer and MTP stage -
built once from the released config and validated before any weight is read:

    struct LayerSpec {
        mode: LayerMode,                 // SwaOnly, Full, Reindex, Reuse, Mtp (unit variants)
        owns_kv: bool,                   // Full: owns the compressor and compressed KV
        owns_index: bool,                // Full index source or Reindex: publishes topk_idxs
        kv_source: Option<usize>,        // Some(2 | 8 | 14 | 20) for readers
        index_source: Option<usize>,     // Some(2 | 8 | ... | 36); a Full row may reuse too
        candidate_source: Option<usize>, // Some(20); an index source, not necessarily a KV one
        kv_offset: Option<usize>,        // Some(128) on a compressed row; None on a stage
        compress_ratio: usize,           // 0 for layers 0/1, else 2 or 1
        engram: bool,                    // layers 1, 14
        dspark_target: bool,             // layers 37..39 (base forward only)
    }

`Full` means "owns KV", not "owns both": `owns_index` is an independent flag
(`model.py:654-655`), so a Full row can reuse a preceding index source's Top-K while still
publishing its own compressed KV.

The plan is **derived** from the four config fields ([DECISIONS.md](../DECISIONS.md) D18's rule) and
then *asserted* against the released config: forty backbone plus three MTP entries in
`compress_ratios`; exactly the four/four/eight id lists; the Reindex set; the two Engram ids; the three
DSpark targets. The assertions are the point — a plan built from the wrong field produces a wrong
forward with no error, while a plan whose derivation is checked against the released tensors has a
place to fail. The existing `the_plan_names_are_the_released_index_names` test
([crates/sparkflash-engine/tests/tensor_plan.rs](../../crates/sparkflash-engine/tests/tensor_plan.rs))
already does this for the tensor set; Phase 1 extends it to the dispatch set.

The device engine then dispatches on `mode`, not on a layer index: one code path per mode, one tensor
set per mode, and the layer number is data. A test pins the 43-row table (mode, sources, ratio,
Engram, DSpark) so a change to the derivation shows up as a diff, not as a token.

### 4.3 What stays on the host in Phase 1, and why that is labelled

The device path today has nine kernels with baseline entries: the six Milestone-4 ABI kernels, the
SwiGLU quantiser epilogue `swiglu_mxfp8_quant_sm121` (never compared against a reference, task-43),
the task-37 router `moe_router_sm121` (measured in its own correctness harness) and the task-8
routed gate/up kernel `routed_gate_up_sm121` (measured in its own harness). None of the last three is
on the served decode path yet. Phase 1 keeps them on the host and **names each one** in the run output:

| Seam | Why it is still host | Task that moves it |
| :--- | :--- | :--- |
| Router (gate GEMM, sqrt-softplus, noaux_tc bias, top-6, `norm_topk_prob`) | No device kernel reduces K = 5120 against packed FP4 weights; the gate itself is bf16 times quantised A | task-37 |
| gate/up (w1/w3) GEMM | The K = 5120 kernel exists (`routed_gate_up_sm121`) but it is not on the served decode path | task-8, then Phase 3 |
| SwiGLU and the activation quantiser | The seventh kernel exists but is unproven; it must be compared before it is trusted | task-43, then Phase 3 |
| HSI candidate merge (top-512 over the pool) | A pipeline step the attention kernel does not implement (stage-2 label) | Phase 2/3 |
| Head and sampling | `cert_lm_head_sm121` exists but has never run against a reference on the device path | Phase 1 stage 4 / Phase 3 |

A seam that is labelled is a seam whose cost and correctness are known. A seam that is silent is a
number that cannot be attributed — the sink was dropped from both sides of the comparison for
exactly this reason, and task-36 is the fix, not a caveat.

**The sink (task-36) is closed.** The reference half is merged (!27, `0374b69`); the kernel half is
the `[64]` f32 per-head argument on `flash_mla_csa2_sm121`, plus the matching
`mla_ref::attention_with_sink` call and signature pin. The second device window of 2026-10-17 ran
stage 2 with it: 5/5 positions PASS, 163,838/163,840 exact, max ULP 1 (that window recorded max abs
delta 9.765625e-4 on the pre-D8 stage-1 chain). After D8 changed the stage-1 activation-quantiser
chain, the merged-chain numbers were re-measured on 2026-10-18 with a paired diagnostic: the
merged-head binary with the old 15-parameter kernel (cubin `822df7f8...`) and the task-3 binary
(cubin `bbb216b7...`) write identical per-position lines, totals 163,838/163,840 exact, max abs
delta 2.4414063e-4, max ULP 1 (logs `phase1-task3-diag-oldkernel-20261018T060534Z.log` and
`phase1-task3-diag-newkernel-20261018T060534Z.log`); the 9.765625e-4 number is kept as a historical
pre-D8 label, not as a current expectation, and the harness prints the position set and that note
beside the equality line. The harness prints that the sink is consumed on both sides — no "sink
excluded" line. Phase 1's attention path consumes the same argument.

### 4.4 What one token proves, and what it does not

A **one-token prompt** (start position 0) is the cheapest token that exists, and its semantics are
worth stating precisely, because "one token" can hide an empty model:

- Every cache is empty: `compress_len = 0` for every ratio-2 layer, `compress_len = 1` for layer 20
  (ratio 1), and every published `topk_idxs` has shape `[b, s, 0]`. Consumers must tolerate the empty
  set (consequence 3 in [../architecture/01-csa2-attention.md](../architecture/01-csa2-attention.md)
  section 2); a one-token run is the first place that tolerance is exercised.
- Layers 0 and 1 run full attention over one position; the window path is a one-row ring.
- The plan, the weights, the mHC on all forty layers, the host router over all 384 experts, the
  combine over the six routed experts per layer, the head and sampling are all exercised.

It does **not** exercise the sparse path (empty candidates), the compressor pooling, the window wrap,
the CED prefill bypass, the Reindex Top-K, or the shared-KV slice beyond position 0. Those are
Phase 2. To keep the token honest, Phase 1 also runs the same token with a **harness-supplied
candidate list** (the stage-2 mechanism) so the attention kernel runs with 128 real candidates even
though the sequence is one position; that run is labelled "synthetic candidates" in its output.

Expert residency is a separate honesty problem. A one-token forward can route over all 384 experts
— the gate weight is 3.9 MB and the six routed experts per layer stream in as needed — so the
comparison can be full-routing. If a window instead holds only 64 of 384 experts per layer
(section 4.6), the comparison must use the **same bounded router** the device uses: the router
selects over the resident set, and the coverage line reports which top-6 selections it captured
(10/24 at 8 experts, 24/24 at 32 and 64 in stage 3). A bounded run must not be compared against an
unbounded reference; the run prints both numbers.

### 4.5 Exit criteria for Phase 1

1. **Per-layer boundary table.** One row per layer that ran, times {residual streams, quantised
   block input and scales, attention output, FFN output}, in the documented tensor order
   ([phase1-boundary-table.md](phase1-boundary-table.md) section 2; `TENSOR_ORDER`,
   `crates/sparkflash-engine/src/forward/boundary.rs:143`), each with its comparison class and bar;
   the run fails if any row misses its bar. The table is the deliverable: a gross mismatch names its
   layer. Three rules come with the rows, all measured on the host reference:
   - **A first step's table has exactly `n_layers` rows** — 40 for the released configuration. The
     host engine skips the MTP pass at `start_pos == 0`
     (`crates/sparkflash-engine/src/forward/mod.rs:497`), so a first-step capture has backbone rows
     only. The released reference returns no draft there either (`forward_spec` returns `None`,
     model.py:1123-1125, 1274-1282) but still runs each stage's attention to seed its window KV cache
     (`model.py:1044-1052`; `docs/engine/01-forward-order.md` section 3.9), so "40 rows" is the host
     capture's contract, not a claim that the reference does no MTP work on a first step. The engine
     does not seed those stage caches, so a device harness seeded the reference's way holds a
     different MTP history on the next decode step (format document section 1). A decode
     step after a prefill has `n_layers + n_mtp_layers` rows (43), at layer ids `n_layers + stage`
     (`crates/sparkflash-engine/src/forward/mtp.rs:674`,
     `crates/sparkflash-engine/src/forward/mod.rs:508`);
     `the_released_row_count_rules_hold_at_forty_layers_and_three_stages`
     (`crates/sparkflash-engine/tests/boundary.rs:924`) pins the 40/43 counts at the released layer
     counts. A run that captures a decode step must emit 43 rows; criterion 1's "forty rows" is the
     prefill/first-step case.
   - **The quantised block-input pair exists only when `dim % 32 == 0`** — E4M3 codes plus one E8M0
     scale per 32 contiguous elements (kernel.py:108). The observed entry points refuse a
     configuration that violates this before any weight is read
     (`crates/sparkflash-engine/src/forward/mod.rs:276`); the released `dim = 5120` divides. A
     configuration that does not divide has no boundary table, rather than a row with a missing
     tensor.
   - **The host-vs-host bar is bit-exact, as measured.** The reference is deterministic, so two runs
     with the same weights are bit-identical; `BoundaryTable::diff`
     (`crates/sparkflash-engine/src/forward/boundary.rs:592`) compares f32 bits and code bytes, and
     the determinism test `the_observed_step_is_the_plain_step_and_two_runs_capture_identically`
     (`crates/sparkflash-engine/tests/boundary.rs:506`) pins it. A diff of two different steps or
     layer sets is refused (`a_diff_between_different_steps_or_layer_sets_is_refused`,
     `crates/sparkflash-engine/tests/boundary.rs:1040`), including two tables with the same start
     position and chunk token but a different consumed prefix
     (`a_diff_of_the_same_decode_header_over_a_different_prefix_is_refused`,
     `crates/sparkflash-engine/tests/boundary.rs:662`) or a different MTP anchor
     (`a_diff_of_the_same_header_over_a_different_mtp_anchor_is_refused`,
     `crates/sparkflash-engine/tests/boundary.rs:704`). The device-vs-reference bars remain the
     device half's measured obligation, per tensor and not per row
     (`BoundaryTensorDiff::differing`, `crates/sparkflash-engine/src/forward/boundary.rs:845`;
     `max_abs_delta`, `:855`): the device path carries bf16 hidden
     buffers and reduction orders that are not the host's, so its tolerances are measured by its run
     as the single-layer delta compounded over the traversed layers (criterion 3's decision) and are
     not invented here.
2. **Perturbation test.** A test injects a one-code change into layer 17's `wq_b` weights and asserts
   the diff flags layer 17. The general **default attribution rule** is first-differing-row
   attribution: layer L's `residual_out` is layer L+1's `residual_in`, so a change at L moves every
   later row of an end-to-end diff, and the first row whose boundary differs is the layer the diff
   attributes first (`BoundaryDiff::first_differing`,
   `crates/sparkflash-engine/src/forward/boundary.rs:763`). It is a default rule and a heuristic
   about where to look, not a proof: rows after the first difference ran on a diverged stream, and
   localising a particular operation needs the replay below. A literal "flags layer 17 and no other
   layer" therefore holds only when the perturbed layer is the **last captured row**: the shipped test
   `a_one_code_change_in_layer_17s_wq_b_flags_layer_17_and_no_other_layer`
   (`crates/sparkflash-engine/tests/boundary.rs:325`) uses eighteen backbone layers so layer 17 is
   last, and asserts `differing_layers() == [17]` while the other seventeen rows are bit-identical.
   The measured row 17 has **161 differing values in six tensors** (`attention_out` 64,
   `post_attention_hidden` 17, `hc_ffn_comb` 3, `pre_ffn_hidden` 8, `ffn_out` 12,
   `residual_out` 57), and the row's `max_abs_delta` 5.96e-8 is an `hc_ffn_comb` coefficient
   delta, not a residual-stream delta - a one-code change is small and the row records where it
   survives and where it rounds away (format document section 6). These are the numbers measured
   after the comb-orientation correction: an earlier host capture applied the comb transposed and
   reported 155 differing values, calling 5.96e-8 "exactly one f32 ulp at the residual's magnitude";
   both the count and that attribution are superseded (format document sections 3 and 6). The
   propagation test `a_middle_layer_change_propagates_and_the_first_divergence_is_its_layer`
   (`crates/sparkflash-engine/tests/boundary.rs:997`) pins the general case with a layer-5 change:
   the rows before it are identical, the first difference is at layer 5, and every later row differs.
   Testing a **later** layer's own arithmetic, once an earlier row has diverged, needs the
   input-substitution replay of the format document section 7.3 — run the layer from the reference
   row's entry stream and input mix; that tool belongs to the device half. Without a test that can
   fail, the table is decoration: a diff that cannot fail is not a test, and the shipped test fails
   both when the capture records nothing and when the injection does nothing (mutation evidence in
   format document section 7.2).
3. **One token's logits.** Argmax identical to the reference's; top-5 set identical; max abs delta
   within a bar set by the run itself on expert-residency-invariant layers (design decision: the bar
   is the measured single-layer delta compounded over forty layers, not a constant chosen to pass).
   The reference is our CPU forward with the sink and `hc_attn_base` present; the comparison is on the
   final position only. The reference now captures the vector the head multiplied
   (`StepOutput::head_input`: the last position's collapsed residual after the final RMS norm), so the
   device head's input is the reference's own normed vector rather than a recomputation, and
   `boundary_dump --head-input-out FILE --logits-out FILE` writes the reference's head input and
   logits as raw little-endian f32; the device side is the harness's own output, which does not exist yet.
4. **Every seam labelled.** A test greps the run output for the five seam labels of section 4.3 and
   fails if one disappears; the sink line is part of the same test.
5. **Tensor audit.** Every tensor the plan expects is read and every tensor not read on the text path
   is named (`gate.bias_vl`, the vision tower, the aligner and the image delimiters — D18's exclusion
   test already classifies 266 of the 96,085 names); the audit prints a count and a test pins it.
6. **Run determinism.** Two runs of the same token are byte-identical (D-a at model scale).

**Status of this list (host half).** Criteria 1, 4, 5 and 6 now have a runnable host implementation:
`crates/sparkflash-engine/src/report.rs` (`one_token_report`) and
`crates/sparkflash-engine/examples/one_token.rs` print a report whose contract
`crates/sparkflash-engine/tests/one_token.rs` pins - forty rows at the released layer counts (40
backbone + 3 stages), the five seam labels of section 4.3 plus the sink line, the tensor audit
(plan/read/unread counts) and a two-run bit-identical boundary table and logits check. The
criterion-2 attribution tests are already merged (!38). Run it with
`cargo run -p sparkflash-engine --example one_token` for the synthetic mode, or with
`-- --checkpoint <snapshot dir>` on the node for the real table. Criterion 3's host half is also in
place: the step output carries the head input (`StepOutput::head_input`) and `boundary_dump` writes
it with the reference logits (`--head-input-out`, `--logits-out`), which is exactly the pair the
device `cert_lm_head_sm121` harness compares. The harness itself is merged (`m7_layer0 --stage 5`,
!46) and its CPU precheck (`--stage 6`) has run on the node: recomputing all 129,280 logits in the
kernel's order from the f32 head input reproduces the reference exactly (max abs delta 0), and from the bf16 codes
the kernel ABI is fed it keeps argmax 5 and the same top-5 while moving a logit by at most 2.4320781e-2
(measured, `phase1-boundary-table.md` section 11). That is the conversion the matched reference
avoids charging to the head; the measured device-vs-matched delta was 0, and the bar remains the
provisional 2e-2 ceiling.

The device head has now run in a full window: `--stage 5` reports device argmax 5 equal to both
references, all 129,280 entries exact with **max |device - bf16-matched host| = 0** (value equality in
the kernel's order), and 0 certified entries on this token, so the certificate was not exercised and
the harness says so. That closes criterion 3's device half for the argmax and the exact logits; the
top-5 is not claimed because a certified entry stores a bound rather than its value. What remains is
the device column of criterion 1: the device half diffs this reference table against the device path
and measures the per-tensor bars, and the 40-layer token harness that produces those rows. The
groundwork for that diff is now device-measured in the fourth window (2026-10-18): the released
`[24, n*d]` mHC predictor (B0), the B1 probes (B1) and the reference activation ceil rule (D8) run
on the node as one chained layer-0 path - stage 0 PASS, stage 1 PASS at layers 0/1/4, stage 8 PASS
with all four probes bit-exact, stage 2 PASS, stage 3 PASS; the numbers and logs are in
[phase1-boundary-table.md](phase1-boundary-table.md) section 12 and D22/D23 in
[DECISIONS.md](../DECISIONS.md). What remains for criterion 1's device column is B2: the single-layer
row-0 dump-and-diff of all 18 tensors against this reference table, per tensor class, followed by the
40-layer token harness. A synthetic run prints that its source is bytes, not numbers, so its
argmax/top-5 are a shape and determinism record rather than a prediction. The released-checkpoint
reference table is now the harness's --checkpoint mode: ReferenceWeights verifies the plan the way
LoadedWeights does and serves the stored forms the reference decodes, LoadedWeights::read_ranges
pages the two Engram tables row by row (98,308,270,592 and 98,305,579,008 elements, still refused as
whole reads), the loader bounds every other tensor, and the rest of the released plan runs under a
1<<30-element limit. Serving the loader's converted tensors instead would have failed at the first
layer - `wo_a` comes back as BF16 with its scale applied under the plan's F8E4M3 spec - which the
new tiny plan-complete-checkpoint test
(`tests/loader.rs::a_reference_source_runs_one_token_on_a_plan_complete_checkpoint`) pins. What
remains is to run it on the node and diff the device half against its table. The harness did find
one released-geometry bug on the way: the attention scratch buffer was sized from the residual
width, so a 40-layer released-config run panicked on the first layer (64 heads of 512 = 32,768 values
against dim 5,120). The same bug sat in the MTP stage path and would have hit the 43-row decode case once the
MTP pass ran; both buffers are now sized from the head geometry and truncated to the residual row,
pinned by tests/forward.rs::a_geometry_with_wider_heads_than_the_residual_width_runs_and_records_the_narrow_row
and tests/boundary.rs::a_wide_head_decode_records_all_forty_three_rows. The audit also now
partitions the unread set into the four classes criterion 5 names (unselected experts, DSpark
stages, the text-path gate.bias_vl, and an indexer no first step consults) and refuses anything
outside them, so the future real-weight report is not refused for an expected tensor.

The 40-layer device column itself has now run: stage 9 (`m7_layer0 --stage 9`) chains the
released checkpoint's first step through the M7 kernels as 40 rows, writes a device `v1` pair and
diffs it against the host pair with `boundary_diff`. The numbers are
[phase1-boundary-table.md](phase1-boundary-table.md) section 13. The code is rebased onto the
squashed B2 change (`2214fe6`) and reuses the B2 harness: the duplicate safetensors reader is gone,
the stage-7 replay and stage 9 consume the shared snapshot/audit module and its four helpers
(`combine_bf16_routed_with_shared`, `verify_forced_ffn_out`, `forced_control_verdict`,
`verify_forced_operand_provenance`). Evidence of record:
`/home/user/phase1-evidence/phase1-c-stage9-40rows-harness5b-20261018T050416Z.log` (572,420 B,
sha256 `bd0f908e...`, UTC 05:04:16Z-05:09:14Z, wall 298.6 s, exit 0), binary sha256
`c1708abc...`, diff log `phase1-c-stage9-diff-20261018T050941Z.log` (sha256 `b86be968...`),
device pair `stage9-out-r5b/` (manifest 34,767 B `6093c74b...`, blob 13,537,920 B
`fecd813b...`). The device pair is bit-identical to the earlier Stage-C run's and all 1,520 printed
row lines match it: the helper adaptation moved verification and reporting, not measurements. The
two runs are bit-identical (both v1 blobs sha256 `fecd813b...`), the device manifest is
byte-identical to the host pair's (`6093c74b...`), and the diff's control exits 0 while the device
column exits 1 with all 40 rows differing - the expected verdict for the health-check column.
Against the pre-registered bars (frozen Stage-B single-layer deltas, cited from the B2 log of record
`...rereview5-20261018T044122Z.log` - 23,897 B, local sha256 `c2de544c...`, lines 149-166,
binary `2b518a35...`; stream tensors times row+1), row 0 is the Stage-B replay whose bars ARE its
own deltas, so row 0 is reproduction evidence only and only rows 1-39 are independent; no row is
judged against its own measurement, including the last one. Of the 720 tensor-row comparisons, 306
are at or under the bar and 414 over, the first over-bar row is row 1, and the residual_out delta
grows from 1.41 at row 1 to 3.03e3 at row 39. Discrete expert flips occur on 17 of 40 layers and are
reported per layer with their measured pointwise flip and two separate host deltas, never folded
into a bar; layer 36 is the worked zero-effect flip (its two independently computed references are
bit-identical, so the flip cannot move the bytes), and the shared verdict now takes that reference
evidence before allowing ids-differ + bytes-identical instead of aborting it. The audit reads 2,673
of 95,819 planned tensors, classifies all 93,146 unread and leaves 0 unclassified; all 80 K5, 640
kernel-vs-reference and 160 routed-combine checks are exact; 20,338,614,720 B pass through the
harness's reader (2,661 distinct tensors, because the shared reader counts the provenance re-reads
too). The retired earlier log (`phase1-c-stage9-40rows-final-...T023529Z.log`, binary
`f5430527...`), its smoke and diff logs stay on disk as historical evidence and are no longer
cited. **This does not close criterion 1**:
the health-check class measures attribution and gross mismatch (D4 FP4 window KV; f32 host
reference with no GEMM quantiser or bf16 writes). Task-3 has since landed the engine FP8 window
format on the device path and measured its own column (roadmap section 4.4 and the table row above):
that column is a format-fidelity tolerance class, not a closure, and the release-faithful exactness
run still waits for Phase 2. Criterion 1's device half now has a running, deterministic,
self-auditing column; its exactness half is still owed by the later run.

### 4.6 Cost of Phase 1

One dedicated device window (vLLM stopped) and one upload. The weights a token needs are bounded: per
layer the dense attention is 5.07 GB for forty layers, about 127 MB per layer; the mHC set is about
4.2 MB; the experts are 18.8 MB each. With 64 resident experts that is about 53 GB for forty layers —
over the 3 GiB the node leaves free beside vLLM and comfortably inside the empty node's 118 GiB; with
8 resident experts it is about 12 GB. The host reference for one token is one f32 CPU forward (the
head alone is 129,280 times 5,120, about 662 M multiply-accumulates), seconds to minutes, not a
window cost.

### 4.7 Tasks

`task-36` (kernel ABI, api-dev); new tasks: `phase1: layer plan and a pinned 43-row table`
(arch-docs/kernel-dev), the 40-layer token harness task (kernel-dev) - kernel dispatch is not
layer-0-specific: stage 1 passes at layers 1, 4, 14 and 39, stages 2 and 3 at layer 14 (recon logs in
phase1-boundary-table.md section 11); the chained harness remains, and the stages deliberately do not
chain layers.
`phase1: stage 4, head and sampling on device` (kernel-dev) - the certified head harness is merged
(!46) and has run on the released token: argmax matches both references, all 129,280 logits are
within max abs delta 0 of the bf16-matched host, 0 certified; the sampling path remains. `phase1: perturbation and seam-label
tests` (api-dev), `phase1: run the released-checkpoint reference table on the node` (kernel-dev) -
the paged Engram source, the released-geometry scratch fixes, the partitioned audit and the
harness's --checkpoint mode landed; the real-weight run and its device diff are what remain.
`task-37` and `task-41` are *not*
Phase 1; they are Phase 3.

---

## 5. Phase 2: the sequence path

**What we build.** A real KV path: prefill of a prompt longer than the window, decode steps, the
compressed KV of the four source layers, the sliding-window ring, the CED prefill bypass, the
Reindex paths, and a KV cache whose contents are audited against the reference rather than trusted.

### 5.1 What exists, and what is missing

`sparkflash-kv` exists and is older and better specified than the device path: D1's layout (main KV
as FP4 E2M1 with per-16 E4M3 scales, indexer K as MXFP4 with per-32 E8M0 scales), **356 B per entry,
890 B per token, 56,960 B per 64-token page** ([DECISIONS.md](../DECISIONS.md) D1; D16 finding 1), a
paged radix prefix cache, SWA bounded replay, `plan_request`, and the six spec defects recorded in D16
with the code that settles each one. The device path has none of it: the M7 attention kernel reads a
flat structure-of-arrays (`kv_fp4` at `p*256`, `kv_scales` at `p*32`, `selected[]` at most 512 u32 —
[device-path-design.md](device-path-design.md) section 3), and the harness quantises its KV rows
itself. The device-side cache and the host-side radix metadata have never met.

### 5.2 The design: the device layout is the KV store, the crate is its metadata

**What we build.** A device KV store whose arrays are the kernel's ABI, per layer mode:

- four compressed-KV arrays (one per `kv_source` layer), sized for `SPARK_MAX_SEQ_LEN` at that
  layer's ratio: at 1M tokens, 3 ratio-2 sources times 500 k positions times 288 B = 432 MB, plus
  layer 20's ratio-1 1M times 288 B = 288 MB, so about **720 MB**; index K at 68 B per position gives
  about **170 MB**; forty window rings (one per layer) at 128 slots times 528 B each are about
  **2.7 MB**. Total
  about **0.9 GB** at 1M context (design arithmetic from D1's entry sizes and the section 8 layouts
  of [../architecture/01-csa2-attention.md](../architecture/01-csa2-attention.md), not a
  measurement).
- the shared slots the reference implements as process globals become explicit handles: the
  compressed KV of the current source layer, the index K of the current source, the current Top-K
  list, the HSI pool. The plan (section 4.2) is what makes "current" well-defined; a Reuse layer
  takes the handle, it does not search for it.
- the host-side radix cache keeps its role — prefix matching, page tables, replay planning — and the
  device store is the backing memory those pages name. Paging on the device is deliberately deferred:
  at 1M context the flat store is under a gigabyte, so a device paged allocator is a Phase 4/5
  concern (it intersects the per-rank memory plan of section 7.5), not a Phase 2 one.

**Proven by** the audit below, plus one negative test: the store must refuse a write that would
exceed its layer's ratio-derived position bound.

### 5.3 Prefill versus decode, and the CED divergence

The two modes differ in the window write, the window indices, the compressor pooling, the index-K
positions and the reachability predicate
([../engine/01-forward-order.md](../engine/01-forward-order.md) section 3.9). Three consequences are
design decisions, not details:

1. **A prompt shorter than 128 never leaves the window**, so the out-of-window candidate branch — the
   branch the M7 stage-2 run was told to exercise with `--tokens 130` — is untested by short prompts.
   The sequence harness uses prompts **over 128 tokens**, and the acceptance set includes one that
   crosses the boundary.
2. **Our prefill implements the CED bypass (D7): layers 0–20 run over the prompt, layers 21–39 only
   over the last window.** The released reference does not implement CED — it runs all forty layers
   over the whole prompt ([../engine/01-forward-order.md](../engine/01-forward-order.md) section 1;
   the report's 8B-prefill claim is not reproduced by the reference). Our prefill is therefore *not*
   expected to reproduce the reference's logits for positions older than the window; the acceptance
   compares the **last position and the decoded tokens**, and a test pins that exclusion so it cannot
   be widened by accident.
3. **Replay is a correctness path, not a performance path.** D16 findings 2 and 5 show the two
   bounded replays cannot reconstruct a SWA tail on their own; the prefill path that fills all forty
   layers (`cache_swa_tail`) is the one Phase 2 must exercise, and the tail bitmask is part of the
   audit.

### 5.4 The audit

**What we build.** An audit mode that reads the device KV back after a run and compares every row
against the reference's quantise-then-dequantise of the same latent bytes: a per-layer, per-position
manifest of sha256s, a per-token byte count from the live store, and a corruption test that flips one
byte and asserts the audit fails.

Why it is not optional: the KV is the only state that outlives a forward, and the only place where a
quantisation bug can be invisible to a logits comparison (a wrong scale on a position the attention
barely reads changes nothing today and everything at 4,096 tokens). The audit also settles D16's
first defect: the live per-token byte count (890 versus 1,050) is measured from the store, not argued
from the itemisation.

**Proven by:** audit clean on a 130-token prefill plus 8 decode steps; the corruption test failing as
required; the per-token bytes matching the D1 layout exactly; and the last-position logits matching
the reference within the Phase 1 bar.

### 5.5 Exit criteria for Phase 2

1. Prefill of a prompt **over 128 tokens** and **at least 8 decode steps**, every decoded token
   reproduced against the reference (token-for-token, not just the last). The reference is the CPU
   forward run with the same prompt and its own cache.
2. The audit passes on every layer mode (Full, Full + HSI, Reindex, Reuse, SWA-only), with the
   corruption test failing as required.
3. The measured per-token KV bytes equal 356 B times entries for the live geometry and the store's
   total matches the layer-mode arithmetic of section 5.2; D16's 890-versus-1,050 question is closed
   with a measured number.
4. A cross-window prompt: the 130-token run exercises the out-of-window candidate branch (a
   deliberately un-masked score would break the test).
5. The CED exclusion is pinned by a test: prefill logits for positions older than the window are not
   compared, and the test names why (D7).

### 5.6 Tasks

kv-dev: `device KV store with the kernel's SoA ABI`; kernel-dev: `prefill/decode sequence harness`,
`KV audit mode`; new: `D16: measured per-token KV bytes`, `sequence: cross-window prompt fixture`.

---

## 6. Phase 3: performance on one node

**What we build.** The single-node speed-up, in the order that the measurements say, and a tokens/s
number that is comparable to vLLM's on the same prompt and batch.

### 6.1 The measured budget, decomposed

| Term | Per token | Source |
| :--- | ---: | :--- |
| mHC (40 layers times 0.0227 ms) | 0.9 ms | measured stage 1, pre-change kernel |
| Attention proper (40 times 1.46 ms) | 58.4 ms | measured stage 2 |
| **Indexer (40 times 4.27 ms)** | **170.8 ms** | measured stage 2 |
| MoE combine (40 times about 0.85 ms) | 34 ms | measured stage 3 |
| Router, gate/up, SwiGLU, quantiser, head, sampling | **unmeasured** | host seams, section 4.3 |
| Collectives (four nodes) | about 1.7 ms | arithmetic in section 7.2; not needed on one node |
| **Sum of the measured kernels** | **264 ms** | extrapolation |

The table is also the argument: **65 % of the measured cost is one kernel**, and that kernel's cost is
the pool geometry (2,048 blocks), not the candidate count. Nothing else in this phase matters until
the indexer does.

### 6.2 The roofline argument

The measured kernels are nowhere near the device's arithmetic: the indexer does 2,048 blocks times 8
slots times 32 index heads times 128 dims times 2, about **134 MFLOP** per query position, in 4.27 ms
— about **31 GFLOP/s** (a GB10 peak is not in our docs; the argument needs only that it is orders of
magnitude higher, a **hypothesis**-labelled premise that can be checked in one line on the node). Its
device working set is 4.2 MiB in 4.27 ms, about **1 GB/s**, which is equally far from any HBM figure.
The attention proper: 128 candidates times 64 heads times 512 dims times 2, about 8.4 MFLOP in
1.46 ms, about 5.7 GFLOP/s. The combine is the closest to real work: 5,120 times 2,304 times 2, about
23.6 MFLOP per rank step in 0.205 ms, about 115 GFLOP/s.

Three conclusions follow, and they set the work order:

1. **The device is idle, not slow.** At batch 1 these kernels are latency- and occupancy-bound: a
   grid of `(4, 20, 1)` blocks at 128 tokens does not fill a GB10. The fix is parallelism inside the
   kernel (more warps doing independent dots), not a faster GEMM.
2. **The indexer's shape is a thread-assignment problem.** A 128-wide dot is four values per lane in
   a warp; running it as a 256-thread tree spends most of the tree on idle lanes. One warp per
   (block, slot) with a shuffle reduce turns 8 serial block trees into 8 warps of independent work;
   that is the first change and it is measurable in the stage-2 harness alone.
3. **Batch is the real lever once the kernels are right.** At 16 tokens the same launches do 16
   tokens' work; the per-token costs above are batch-1 numbers, and a serving number that ignores
   batch is not comparable to vLLM's.

### 6.3 The work order

1. **Indexer, warp-level dots** (one warp per block-slot; 128-dim shuffle dot), measured in
   `--stage 2` alone. Target **design**: indexer at or below the attention proper (at most 1.5 ms),
   on the basis that the current shape is about 0.03 % of a device peak and the block grid is
   2,048 times 64 slices; the target is a thing to measure against, not a promise.
2. **Indexer, `grid.y` sweep** — the harness notes the indexer's `blockIdx.y` geometry; a sweep is a
   one-window experiment with no new code.
3. **CUDA graphs.** The per-token launch count is over 120 (three stages times forty layers); at
   0.2 ms per launch chain step (stage 3: 0.82 ms for four chained launches) the launch overhead is a
   first-order term for the small kernels. Graph capture per batch shape also exercises QUALIFICATION
   gate 3.8, which currently has no capture code at all.
4. **Device router (task-37)**: BF16 times quantised-A GEMM, sqrt-softplus, noaux_tc bias, top-6,
   `norm_topk_prob` — the seam that costs the most host time and the one that blocks all later device
   work.
5. **The K = 5120 packed-FP4 gate/up GEMM** (and the shared expert): the missing kernel; every dense
   projection on the device path needs a K = 5120 variant.
6. **Attention sink ABI (task-36)** if not already in (it should be, from Phase 1).
7. **Fusion, last**: attention plus candidate mask; mHC plus norm plus quantisation (already fused);
   MoE gate plus gate/up fused dispatch. Fusion before the individual kernels are fast hides the cost.

### 6.4 Exit criteria for Phase 3

1. A **tokens/s** table for one node at batch 1 and batch 16, measured on a fixed prompt with the
   reference engine's own tokenizer and sampling parameters, against **vLLM on the same node, same
   prompt, same batch, same dtype** — same units, same prompt, stated batch. This is the only
   comparison that can be read; a tokens/s number without all four is not a number.
2. The indexer's time is measured after each change, in the same harness, with the same bar, and the
   final indexer is **below the attention proper** (design target) or the gap has a roofline account.
3. Every remaining gap between the measured time and the roofline has a named cause (occupancy,
   launch count, memory traffic, host seam) — the acceptance is not "fast", it is "no unexplained
   gap".
4. The new kernels have device-versus-reference comparisons (the seventh kernel included), so speed
   did not buy a numerics regression.

### 6.5 Tasks

`task-41` (indexer), `task-37` (router), `task-36` (sink, if not closed), new: `K=5120 gate/up
GEMM`, `CUDA graph runner`, `tokens/s harness vs vLLM`.

---

## 7. Phase 4: tensor parallelism across four Sparks

**The honest label.** Every number in this section is arithmetic from our own byte tables and from
the reference's sharding rules; **nothing here has been run on the fabric**. The four-node system has
never carried a byte of this project: the RDMA backend is real and the three cluster gate bodies are
implemented, but no gate has run and no frame has crossed a wire, and the 10 KiB All-Reduce gate has
never been measured ([../QUALIFICATION.md](../QUALIFICATION.md) section 3.1). This phase is design plus
hypothesis, and the design is written so that the first hardware run falsifies it cheaply.

### 7.1 What is sharded, what is replicated, and why

The reference's rules are fixed (`convert.py:56-64`, `129-134`; summarised in
[../architecture/03-moe.md](../architecture/03-moe.md) section 3.1 and
[../architecture/01-csa2-attention.md](../architecture/01-csa2-attention.md) section 10). Our plan
follows them unless a column says otherwise:

| Component | Reference rule | Our plan | Why | Collective |
| :--- | :--- | :--- | :--- | :--- |
| Routed experts (384/layer) | **Whole experts per rank**, 96 at world 4 | **TP=4 over the intermediate width**: every rank holds every expert's 576-value slice, padded to 640 (D30) | The only routed-expert layout the device has run is the kernel's 576 -> 640 slice (M7 K9-K12, four rank launches); whole-expert EP would need a different kernel ABI and is recorded as not implemented (D30) | partial-sum all-reduce, [tokens, 5120] |
| Shared expert | Whole 2,304 per rank, every rank | Same | 1.4 GB total; splitting it would add a collective for 0.4 % of the weights | none |
| `wq_b`, `wo_a` | dim 0 (output) | Same | — | none |
| `wo_b` | dim 1 (input), **fp32 all-reduce** | Same, fp32 payload | The reference reduces fp32 (01-csa2-attention.md section 10); bf16 would be a deliberate divergence | fp32 all-reduce, [tokens, 5120] |
| `wq_a` | **absent from the reference's shard list** | Replicate (design) | The summary in our architecture doc omits it; the rule must be re-read from `convert.py:56-64` before TP opens (open unknown 15) | none |
| `wkv`, compressors, indexers | Replicated | Replicated | Small and per-head; splitting them would make the attention prologue a collective | none |
| Norms, `gate.weight`, `gate.bias` | Replicated | Replicated | 160 MB of gates in total; norms negligible | none |
| mHC (`hc_*_fn/base/scale`) | Replicated | Replicated | Coefficients derive from the full residual; splitting the stream would make every mHC elementwise op a collective — the opposite of why it is fused | none (169 MB total) |
| Engram tables | **Row-sharded**, ceil(rows/4), tail padded (weights 0, scales 1) | Same | The spec's "6 heads per rank" split is a deployment choice the reference does not make (D6, 04-engram.md section 7) | all-reduce [B,L,24,256] bf16 = **12,288 B/token/layer**, 2 layers |
| Engram projections (`q_weight`, `k_weight`) | Replicated | Replicated | 315 MB total | none |
| `embed`, `head` | dim 0 (vocab), **32,320 logits/rank** | Same | Vocab split is the cheap axis | all-gather of logits per position (517 KB at fp32) |
| `attn_sink` | dim 0 | dim 0 | Per-head; 16 per rank | none |
| `weights_proj` | dim 0 | dim 0 | — | none |
| **KV latent** | **Not split** (`n_kv_heads = 1`) | **Replicated, whole** | Head-splitting would turn `wkv` into an all-gather of 1,536-dim slices and is not what the reference does; about 0.9 GB/rank at 1M context | none |
| MTP / DSpark | Experts: **whole experts 32/rank**, top-3 global | **D30's TP=4 intermediate slice + 640 padding, the same kernel ABI, Phase 5** (D6's layout half is corrected by D30; the functional DSpark decision stands) | Out of the base forward | MoE all-reduce per draft step |

**The routed-expert row contradicts the reference on purpose.** `convert.py` shards whole experts (96
per rank at world 4) and D2 recorded that rule; the only routed-expert **device** GEMM in the tree
and its device verification are written to a 576-value intermediate slice per rank padded to 640,
and the M7 chain runs it as four rank launches. D30 adopts the verified layout and prices it:
74.0 -> 82.2 GB/rank of routed experts, backbone and MTP (+11.1 %; the MTP experts are 1.80 GB/rank unpadded and
2.01 GB/rank padded, and the MTP row above now follows D30 rather than the reference's whole-expert
rule), and the 124,610 MiB device (130.66 GB decimal = 121.7 GiB) no longer fits the Engram tables
resident at all (138.3 GB of weights), so section 7.5's NVMe plan becomes required rather than
advisable. The shared expert is not sliced: it stays whole at 2,304 and replicated.

Two rules deserve their own sentence because they decide the fabric's role:

- **The KV latent cannot be head-sharded.** There is one KV head; a head split would require
  `wkv`'s output to be gathered before attention. We replicate the latent and pay 0.9 GB/rank for it
  at 1M context (section 5.2). That is cheaper than one extra collective per layer at 40 layers.
- **The Engram split is by rows, not heads.** Each rank answers **all 24 lookups** and the partial
  vectors are all-reduced (12,288 B per token per Engram layer). It is not tensor parallelism and
  must not be counted as such.

### 7.2 The per-token collective budget

At batch 1 and 1M-context shapes, per token:

| Collective | Payload | Count/token | Bytes |
| :--- | ---: | ---: | ---: |
| `wo_b` attention all-reduce (fp32) | 20,480 B | 40 | 819,200 |
| MoE expert partial-sum all-reduce (bf16) | 10,240 B | 40 | 409,600 |
| Engram lookup all-reduce (bf16) | 12,288 B | 2 | 24,576 |
| Logits all-gather (fp32) | 517,120 B | 1 | 517,120 |
| **Total** | | **83** | **about 1.77 MB** |

The MoE dtype is an assumption (open unknown 16); `wo_b`'s fp32 reduce is stated in
01-csa2-attention.md section 10.

At the gate's 20 µs per collective that is **about 1.66 ms/token** — 0.6 % of the measured
264 ms/token, and about 4 % of a 40 ms/token target. Two consequences:

- **TP is not a performance feature at today's speed; it is a memory feature.** The fabric must not be
  sold as a speedup.
- **The collectives become relevant exactly when Phase 3 succeeds.** At 0.5 ms/layer the same 83
  collectives are about 8 % of the token. The 20 µs gate is therefore a Phase 4 gate, not a Phase 5
  nice-to-have.

**The one-shot path's payload limit.** The RoCEnante one-shot collective takes payloads **up to
2 MiB**; larger payloads go through the patched NCCL ring
([crates/sparkflash-comm/src/lib.rs](../../crates/sparkflash-comm/src/lib.rs) module docs). Decode
(one token) is always one-shot: 10 KiB for the expert reduce is exactly the gate's payload, and
20 KiB for `wo_b` is twice it. Prefill chunks cross the threshold at **204 tokens** for a 5,120-wide
bf16 reduce (2 MiB / 10,240 B) and at **170 tokens** for the Engram 12,288-byte reduce. The design
therefore either chunks prefill to at most 160 tokens per collective or accepts the ring path — and
the ring path's numerical properties are **unmeasured**, which is the numerical-equivalence question
below.

### 7.3 Numerical equivalence under TP: define it, or the tests are theatre

Bit-identity to a single-rank run is **not achievable** and must not be the criterion: the reference's
own single-rank result accumulates in a different order than a rank-ordered reduce. The criterion we
propose (to become a new [DECISIONS.md](../DECISIONS.md) entry when this phase opens) has three parts:

- **R1 — rank-equivalence.** For the same reduced payload, every rank's result is bit-identical to
  every other rank's. The M3 one-shot reduction provides this by construction (f32, ascending rank
  order, one bf16 rounding; [crates/sparkflash-comm/src/lib.rs](../../crates/sparkflash-comm/src/lib.rs)).
  Testable on loopback today, without a device.
- **R2 — run-determinism.** Same input, same world size, same order gives bit-identical results
  across runs. The one-shot path gives this; **the NCCL ring path gives it only if its reduction
  order is fixed**, which we have not established. Until measured, any prefill path that takes the
  ring is *outside* R2, and the run says so.
- **R3 — model-equivalence.** Against the f32 reference at every boundary, the TP result is within the
  *same named tolerance as the single-node result*, not a new one. The tolerance is calibrated by the
  measured single-node-to-reference delta; it is not invented. R3 excludes bit-identity with the
  single-node logits, and it excludes the NaN corner case of the one-shot path (the kernel's
  `__float2bfloat16_rn` maps every NaN to `0x7fff`, which the host cannot reproduce).

A test that cannot fail is not a criterion: R1 and R2 are equality tests on the fabric; R3 is a
tolerance test whose bar is computed from the run's own single-node measurement. If a TP run fails R3
at a boundary while passing R1 and R2, the fault is in the *sharding* (a tensor split where it should
be replicated, or the reverse), which is exactly what the boundary diff is for.

### 7.4 The four nodes: what we know, what we do not

Known (head node only, section 1.4): addresses `.80/.84/.68/.59`, two per node (one per rail;
`.80`/`.85` on the head), the f1 siblings ACTIVE at 200 Gb/s, MTU 1500 measured against a 9000
runbook, driver `580.173.02` against a `580.178.04` gate, `nvidia-smi` exit 2, 3 GiB MemAvailable
beside vLLM.

Not known: the other three nodes' rail state, MTU, driver and CUDA; whether the two rails can carry
the same collective in both directions (D12's dual-path enumerates the ACTIVE functions at runtime,
but the RDMA transport that would use them is real and has not run on the fabric); the real one-shot
latency; the ring
path's numerics; and whether a four-node window can even hold four copies of the model beside the
operating cost (section 7.5: the plan fits with the Engram tables on NVMe and about 43 GB to spare,
and does not fit with them resident).

**How the nodes are addressed in the design.** Discovery is not hardcoded: the rail selection
enumerates ACTIVE functions at 200 Gb/s or more (D12), the rank topology is `[.80, .84, .68, .59]`
with a per-rank second address, and the dual-path routes each collective by payload size (one-shot up
to 2 MiB, ring above). The MTU question is a **blocking prerequisite**: 1500 versus 9000 changes
RoCEv2 throughput by a factor that can decide the 20 µs gate, and the runbook/node disagreement must
be settled (the node is right, or the netdevs need reconfiguring) before any gate number is believed.
The same applies to the driver mismatch: a gate measured under `580.173.02` is not the gate the
qualification runbook specifies.

### 7.5 Memory per rank: the number that decides the architecture

The checkpoint is 510,286,023,000 B ([../architecture/00-overview.md](../architecture/00-overview.md)
section 2.4, closing exactly). Under the plan's sharding with world 4 (design arithmetic; the routed
row is TP=4 with the 576 -> 640 padding, D30):

All figures below are **GB decimal**; the corpus fact tables report the same device as 124,610 MiB
(124,610 x 1,048,576 = 130,663,055,360 B = 130.66 GB = 121.7 GiB).

| Group | Total | Per rank | Rule |
| :--- | ---: | ---: | :--- |
| Routed experts, backbone + MTP | 295.997 GB | 82.2 GB | TP=4 intermediate slice, 576 -> 640 padded (D30; 74.0 GB unpadded) |
| Engram tables | 202.758 GB | 50.7 GB | row shard |
| Dense attention, embed/head, sink, weights_proj | 8.1 GB | about 2.0 GB | dim-0/dim-1 shards |
| Replicated (shared experts, gates, norms, mHC, Engram projections, compressors, MTP extras, image) | 3.43 GB | 3.43 GB | replicated |
| **Resident per rank** | | **about 138.3 GB** | 134.9 sharded plus 3.4 replicated |
| Device total | | **130.66 GB** (124,610 MiB = 121.7 GiB) | `cuDeviceTotalMem_v2` |

**With every Engram table resident, a rank holds 82.2 + 50.7 + 2.0 + 3.43 = 138.33 GB (about
138.3 GB) of weights on a 130.66 GB device: it does not fit, by 138.33 - 130.66 = 7.67 GB (about
7.7 GB).** The EP layout this section carried before D30 left about 0.5 GB to spare; the 8.22 GB of
padding D30 introduces removes even that margin, so the NVMe plan below is required rather than
merely advisable. It is the single most important consequence of the released byte table for
Phase 4.

The plan that makes it serviceable (design):

1. **Engram tables live on the local NVMe through the M2 store, not in device memory.** The store
   already exists for exactly this reason (packer, O_DIRECT, 16-way cache, io_uring prefetch); the
   per-rank row shard is 50.7 GB on disk and the cache budget is 1 GiB. Resident weights drop to
   82.2 + 2.0 + 3.43 = 87.63 GB, **about 87.6 GB**, leaving 130.66 - 87.63 = 43.03 GB, **about
   43 GB**, for KV (about 0.9 GB at 1M context, section 5.2),
   activations (prefill chunks, tens of MB) and the Engram cache. Cost: the I/O budget — 24 lookups
   times 2 layers = 48 rows per token, 5.6–6.5 µs per row on a hit and about 18 µs on a miss, so
   **about 0.3–0.9 ms/token** of Engram I/O, with queue depth at least 48 required
   ([device-path-design.md](device-path-design.md) section 10). That cost is a few percent of a 40 ms
   token; it must be prefetched, not hidden.
2. **The image path (0.97 GB) is not loaded for text serving** — a further 0.97 GB, and one fewer
   family to audit.
3. **The MTP/DSpark layers (7.22 GB of routed experts globally, 1.80 GB/rank; 2.01 GB/rank after
   D30's padding, plus 0.21 GB extras) load only when speculation is on**; the base forward is forty
   layers.
4. **The Engram packer has never been run against the checkpoint** ([../QUALIFICATION.md](../QUALIFICATION.md)
   section 1, check 4: the shards are not packed here). This is a prerequisite, not a Phase 5 detail.

Caveats, stated as hypotheses: the device's *usable* memory is below `cuDeviceTotalMem_v2` (driver
and context overhead, unmeasured); the Engram row shard assumes the reference's `ceil(rows/4)` rule
and the tail padding; and the byte table assumes the released checkpoint, which is the one we have.

### 7.6 How TP is tested without weights (and before the fabric exists)

Three layers, cheapest first:

1. **Loopback, in CI, no hardware.** The comm crate ships a loopback/TCP transport for exactly this
   (the reference reduction tests run on it). A four-rank TP forward over loopback with deterministic
   synthetic weights (a seeded generator keyed by tensor name, so a sharding bug is a reproducible
   tensor) tests: rank-equivalence (R1), the per-layer collective sizes above, the sharding map
   itself (a wrongly sharded tensor changes the result), and R3 against the single-rank reference.
   This is the test that must exist before a four-node window is spent.
2. **Tiny model over loopback.** The engine's small configuration run with four logical ranks and
   real reference weights, end to end, logits against the whole-model reference. This catches
   integration faults (the shared slots, the KV handles, the head gather) without the fabric.
3. **Real weights, four nodes, one token.** The Phase 4 exit: R1/R2/R3 on the fabric, one token,
   every boundary compared, and the collective budget measured.

### 7.7 Exit criteria for Phase 4

1. Loopback TP test in CI: R1 bit-identical across ranks; R3 within the named bar; the sharding map
   pinned by a test (one tensor per rule).
2. Four-node, real weights, one token: R1/R2/R3 pass at every boundary; the logits match the
   single-node reference within R3.
3. **The 10 KiB All-Reduce gate measured on the real fabric** (QUALIFICATION 3.1) and **10,000
   replays bit-identical** (3.2) — these are *our* gates, not someone else's: without them R1/R2 are
   untested on the hardware.
4. All four nodes' preflight facts recorded (rails, MTU, driver, CUDA) and the MTU/rail-name defects
   fixed in the docs and the runbook (task-30).
5. The NCCL ring path either proven to meet R2 or avoided by chunking prefill to at most 160 tokens;
   the choice and its measurement recorded.
6. The per-rank memory plan verified on the node (a MemAvailable census with the real resident set).

### 7.8 Tasks

comm-dev: `fabric: real one-shot probe and 10 KiB gate`, `fabric: ring-path numerical check`,
`fabric: four-node addressing and MTU settlement`; engine/api-dev: `TP-forward: loopback four-rank
test`, `TP-forward: sharding map`, `TP-forward: per-rank memory census`; engram-store: `engram: pack
the real checkpoint and prove the NVMe residency budget`.

---

## 8. Phase 5: the serving surface

**What we build.** A device-backed `InferenceEngine` behind the existing HTTP surface, meeting the
gates that are already written in [../QUALIFICATION.md](../QUALIFICATION.md), and a recorded answer
to "are we ready to serve" that is not a claim.

### 8.1 What exists, and why the surface is not the work

The surface is honest about what it cannot do. The engine capability set has no vision channel,
`GET /v1/models` reports `multimodal: false` and `image_url: false`, and a request carrying an image
part is refused before generation with HTTP 400 `unsupported_capability` and a message naming the
missing vision capability - rather than answered from its text tokens or charged a visual-token count
derived from an assumed 32x32 grid. The vision tower and aligner are 263 checkpoint tensors the
tensor plan does not load (D18), so the channel is absent rather than merely unimplemented
(DECISIONS.md D27). The Phase 5 device engine inherits that boundary: it reports the conservative
capability set and the same typed refusal.

The API crate is a full HTTP surface with an `InferenceEngine` seam and a `MockEngine` behind it
([crates/sparkflash-api/src/engine.rs](../../crates/sparkflash-api/src/engine.rs)); the scheduler
crate has chunked prefill with the bounded HSI indexer, the short-request bypass, the DSpark
confidence cap and the gates module (`crates/sparkflash-scheduler/src/{prefill,bypass,dspark,gates}.rs`);
the KV crate has the radix cache and replay planning; and the CLI serves the reference engine. What
does **not** exist is the device engine those crates would drive. The serving surface is therefore
*not* a rewrite; it is an implementation of an interface that was deliberately written without a GPU.

### 8.2 What changes for the device engine

- **`InferenceEngine` implementation** over the Phase 4 TP forward: request, tokenize, prefill, decode
  loop, sample, stream. The reference engine's control flow is the skeleton; the device path replaces
  its forward and nothing else.
- **Scheduler-device adapter**: the chunked-prefill plan becomes a sequence of device launches; the KV
  page tables name device offsets (Phase 2's store, Phase 4's per-rank replication); the
  bounded-indexer plan must match the device candidate path.
- **Admission control by memory**: the KV pool, activation workspace and Engram cache must fit the
  about 43 GB left after weights (section 7.5), so batch and concurrency limits are computed from the
  measured per-rank census, not configured by hand.
- **CUDA graph capture/replay per batch shape** (gate 3.8) and the **45-step scoped trace** (gate
  3.9) — both have instruments and no producers today.
- **DSpark** (three stages, confidence head passed through a sigmoid per D6, the noise token, the
  confidence cap) — the reference never calls `forward_spec`
  ([../architecture/05-dspark.md](../architecture/05-dspark.md) sections 1–2), so the draft loop is
  ours to prove; it is also the gate with a KL target.
- **Telemetry**: RoCE per-rail counters with the D13 accounting, KV stats, DSpark step stats, and the
  loop-abort detector — the schema exists in the API crate.

### 8.3 The latency budget a token must fit

At batch 1, from the measured kernels: 170.8 ms indexer plus 58.4 attention plus 34 MoE plus 0.9 mHC
= 264 ms, plus host seams, head, sampling and (four-node) 1.7 ms of collectives. A serving latency
target has to be set *after* Phase 3, from the measured tokens/s; the design budget that Phase 3
tests against is **about 40 ms/token (about 25 tok/s) at batch 1** with the indexer below the
attention proper — a **design target**, not a measurement, chosen because it is the smallest number
that makes the current attention-plus-combine costs a minority of the token and leaves the indexer a
plausible rework.

### 8.4 The acceptance test for "ready to serve"

"Ready" means **all nine QUALIFICATION gates have instruments and have passed on the real fabric**,
plus the four gates this roadmap adds:

| Gate | Instrument today | State |
| :--- | :--- | :--- |
| 10 KiB one-shot All-Reduce under 20 µs | `doctor` check 3, `rocenante-probe` | never run |
| Bit-identical collectives, 10,000 replays | `roce_bits_match_the_cpu_reference_over_ten_thousand_replays` | real gate body, never run |
| Dual-rail counters half each | `dual_rail_counters_show_both_interfaces_carrying_half_the_bytes` | real gate body, never run |
| qeval 72/75 | `gates.rs:293` | no producer |
| DSpark KL under 0.007 | `gates.rs:308` | no producer |
| 1M-token needle | `gates.rs:325` | no producer |
| sparkDash concurrency 1..16 | `gates.rs:338` | no producer |
| CUDA graph replay, batch 1..16 | `gates.rs:350` | no producer, no capture code |
| 45-step scoped trace | `gates.rs:362` | planner only |
| **Token-for-token reference agreement** (new) | Phase 2 harness | to build |
| **Determinism of a served request** (new) | D-a/D-b through the engine | to build |
| **Per-token latency at batch 1 and concurrency 16** (new) | Phase 3 harness | to build |
| **Memory headroom under load** (new) | per-rank census | to build |

The table is the answer to "are we close": nine gates that have never run, five without a producer,
one with no capture code — and a device engine that does not exist yet. The distance is legible, and
that is deliberate.

### 8.5 Tasks

`task-18` (E3: engine forward and serve) becomes the Phase 5 parent; new: `serve: device engine`,
`serve: scheduler device adapter`, `serve: latency harness`, `serve: gate producers` (one per gate
without a producer).

---

## 9. What we would measure next, and why

Ordered by information per unit of cost. Each entry names the decision it settles.

1. **E1 — stage-1 determinism, N = 100.** Cost: half a device window, no code beyond the test.
   Settles the determinism contract (D-a) and gives task-39 its evidence. Highest value because
   every later equality claim depends on it.
2. **E2 — stage-2 candidate-count sweep (64/128/256/512) and a no-indexer run.** Cost: one harness
   flag plus one window. Settles whether the 5.73 ms is the candidate merge or the attention, and
   gives the indexer rewrite its target. The harness already supplies the candidate list; a flag and
   a loop are the whole change.
3. **E3 — the indexer's warp-dot variant, measured in isolation.** Cost: one kernel edit plus one
   window. The single most likely 3–5 times win in the model (65 % of the measured budget at about
   0.03 % of a plausible peak). If it does not move, the roofline argument is wrong and Phase 3
   changes.
4. **E4 — four-node read-only preflight.** Cost: an operator window, no model run. Settles the
   unverified cluster table (rails, MTU, driver, CUDA on all four), and it is the cheapest possible
   way to find out that Phase 4's prerequisites are not met.
5. **E5 — the 10 KiB All-Reduce attempt.** Cost: one window with the operator; the instrument and
   backend are real, so this is a run rather than a build task - but no frame has crossed a wire yet.
   Settles whether TP on these rails is viable at all — the biggest single risk in this document.
6. **E6 — the Engram packer against the real checkpoint.** Cost: hours of local I/O, no device.
   Settles the 50.7 GB/rank NVMe plan and the per-rank memory arithmetic of section 7.5. Without it,
   the TP memory plan is a spreadsheet.
7. **E7 — the 43-row layer-plan dry run.** Cost: minutes, local, CI-able. Builds the plan from the
   released config and asserts every tensor name the plan expects exists in the index (and every
   index tensor is classified). Finds plan bugs before any window; this is the cheapest test in the
   document and the one that prevents the Phase 1 window being wasted.
8. **E8 — stage 4, `cert_lm_head_sm121` against a host reference on one token's hidden state.** Cost:
   one window, one harness stage. The head is one of the six kernels and has never run on the device
   path; sampling depends on it.

The order is a claim too: E1, E2 and E7 cost almost nothing and are prerequisites for reading
anything else; E4 and E5 run in parallel because they are the only experiments that can invalidate
the architecture rather than a step.

---

## 10. Open unknowns

Each is a real gap with a test that would close it; none is a caveat to be waved away.

1. **`hc_scale` (the three per-group scales) is not modelled by the kernel.** The M7 harness folds
   `hc_attn_scale` and `hc_ffn_scale` into the predictor it hands the kernel
   (`examples/m7_layer0/main.rs:674-679` and the fold loop at `710-726`; the split left the crate-root
   line numbers unchanged); whether the model applies the same fold at the same point is our
   reading of the reference, not a confirmed device-versus-reference boundary. Test: a stage-1 run
   with a deliberately perturbed scale vector must move the coefficients by exactly the folded
   amount; if it does not, the fold is in the wrong place.
   **Closed (D22).** The fold is gone: the kernel takes `hc_scale[3]` itself and applies it with
   `hc_base` after the projection rstd, the reference is
   `mega_mhc_ref::mega_mhc_layer_with_scale`, and
   `the_released_scale_multiplies_each_group_before_the_bias_is_added` pins the scaled-then-biased
   order against the reference's own arithmetic.
2. **The pre/post activation pairing in mHC is our reading.** The split of the 24 predictor rows into
   `A` (pre, `sigmoid + hc_eps`), `B` (comb, the 16 Sinkhorn rows) and `C` (post, `2 * sigmoid`)
   follows the reference's code, but the
   stage-1 comparison would pass a wrong *pairing* as long as both sides made the same mistake. Test:
   a reference-side unit test against the released kernel's row semantics, not against our port.
   **Closed (D22).** `the_fold_puts_each_released_row_on_its_kernel_row`,
   `the_released_pre_and_post_activations_are_sigmoids_not_softmaxes` and
   `the_released_split_sinkhorn_adds_eps_once_before_the_first_column_pass` (plus
   `the_released_split_sinkhorn_runs_iterations_minus_one_pairs`, which enters the pair loop) pin the
   row mapping, the activations and the released schedule; `mega_mhc_ref.rs:606,683` are the mapping
   functions.
3. **The attention sink (task-36) is consumed on both sides.** **Closed (D22, second window).**
   The kernel takes the `[64]` f32 argument and the reference is `mla_ref::attention_with_sink`; the
   stage-2 run of 2026-10-17 was 5/5 positions within tolerance with the sink consumed, and prints no
   "sink excluded" line.
4. **`hc_attn_base` (task-40)** is consumed by neither side (section 3.3).
   **Closed (D22).** Both sides consume it: the kernel takes it as an ABI argument and
   `bias_into_kernel_rows` puts the 24 values in the kernel's row order; the harness no longer
   prints a "not consumed" line.
5. **The seventh kernel has no device-versus-reference comparison** (task-43): it is the only
   arithmetic in the chain with no numeric check at all.
6. **The mHC `C` normalisation (task-38)** is compared after a harness-side softmax; the epsilon
   placement of D3 is not proven by the stage as it stands.
   **Closed (D22).** `C` is compared raw on both sides, the harness-side softmax is deleted, and
   `spark_released_split_sinkhorn` plus
   `the_released_split_sinkhorn_adds_eps_once_before_the_first_column_pass` pin the released epsilon
   placement (one `+ eps` after the row softmax, one unconditional column pass, then 19 row/column
   pairs, each dividing by `sum + eps`).
7. **`hc_stats` is nondeterministic (task-39)**: the statistic was accumulated with an fp32
   atomicAdd, so tile arrival order changed it between runs (the unrelated 6-of-5,120 code difference
   was stage 1's device-versus-reference finding, not a two-run difference).
   **Closed in code (D21).** The fp32 atomicAdd is gone: each tile writes a slot fixed by
   `blockIdx` and the four partials are summed in tile order in binary64
   (`kernels/mega_mhc_sm121.cu:197-222`), with the structure pinned by
   `mhc_determinism.rs`. The N = 100 device run is owed a window, like unknown 1's stage-1 run.
8. **No bytes have ever moved on the four-node fabric.** The ibverbs backend and the three cluster
   gate bodies are real, but none has run and no frame has crossed a wire, and the 10 KiB gate is
   unmeasured.
9. **The NCCL ring path's reduction order (and therefore R2) is unknown** for prefill payloads above
   2 MiB.
10. **MTU 1500 versus 9000, and the rail-name defect, are unresolved** (task-30); the other three
    nodes are unmeasured.
11. **The Engram packer has never run against the checkpoint**; the 1 GiB cache and the queue-depth-48
    requirement are requirements, not serving measurements.
12. **CED is a divergence from the reference prefill** (D7); our prefill logits for positions older
    than the window are not expected to match, and the exclusion is pinned by a test (section 5.3).
13. **The HSI candidate merge (top-512 over the pool) is not implemented anywhere** — the attention
    kernel takes a candidate list; who builds it, on which device, at what cost, is Phase 2/3 work.
14. **A GB10 arithmetic peak is not in our docs.** The roofline argument of section 6.2 needs only
    "orders of magnitude above 31 GFLOP/s"; the exact figure should be recorded before it is quoted.
15. **`wq_a`'s shard rule is absent from our summary of the reference's mapping** (`convert.py:56-64`):
    the architecture doc lists what is dim-0, dim-1 and replicated, and omits it. It must be re-read
    from the released code before the TP sharding map is pinned; until then the plan replicates it
    (13 MB/layer at most) and the sharding map test records the assumption.
16. **The reference's MoE reduction dtype is not pinned in our docs.** The budget in section 7.2
    assumes a bf16 partial-sum payload; `model.py:895-902` must be read for the dtype (and `wo_b`'s
    fp32 reduce confirmed) before the sharding map is pinned. The map test records the assumption.

---

## 11. Task map

| Phase | Existing tasks | New tasks (proposed) | Exit criterion |
| :--- | :--- | :--- | :--- |
| 0 | task-38, task-39, task-40, task-43 | determinism test; raw-C stage-1 | section 3.6 |
| 1 | task-35 and task-31 (slice close-out) | layer plan and pinned 43-row table; 40-layer token harness; stage-4 head; perturbation and seam-label tests | section 4.5 |
| 2 | none | device KV store; prefill/decode harness; KV audit; D16 measured bytes | section 5.5 |
| 3 | task-41, task-37 | K=5120 gate/up GEMM; CUDA graph runner; tokens/s harness vs vLLM | section 6.4 |
| 4 | task-30 (cluster facts; prerequisite) | one-shot gate on the real fabric; ring-path numerical check; loopback TP test; sharding map; per-rank memory census; Engram packer on the real checkpoint | section 7.7 |
| 5 | task-18 | device engine; scheduler adapter; latency harness; gate producers | section 8.4 |

The roadmap does not close task-35 or task-31: they are the current slice's close-out (the M7 harness
merged and the slice documented), and they feed Phase 0/1 rather than being replaced by them.

---

## 12. Appendix: the measured numbers, their hosts, and the labels

### 12.1 The numbers

| Number | Value | Host | How measured |
| :--- | :--- | :--- | :--- |
| mHC residual streams | 20,480/20,480 bit-exact | node0, 2026-10-18 (fourth window) | `m7_layer0 --stage 1`, real layer-0 bytes, released `[24, n*d]` predictor cubin (B0); PASS 4/4 tokens at layers 0/1/4 |
| mHC E4M3 codes / ue8m0 scales | layer 0: token 0 at **5,119/5,120 codes** (one one-code difference), tokens 1-3 exact; layer 1: **5,120/5,120 codes** at every token; layer 4: token 0 at **5,118/5,120 codes** (two one-code differences), other tokens exact; **160/160 scales** at every token in every case | node0, 2026-10-18 (fourth window) | `m7_layer0 --stage 1`, real layer-0 bytes, released `[24, n*d]` predictor cubin (B0), ceil-rule reference (D8); PASS 4/4 tokens at layers 0/1/4; where a code differs it is the fixed-order `1/rms` reduction residual, inside the bar |
| mHC activation quantiser rule (D8), host recomputation | ceil rule **5,120/5,120 codes + 160/160 scales exact** on the engine's captured row 0; the retired floor rule would give 4,288/5,120 + 134/160 (attention) and 4,096/5,120 + 128/160 (ffn) | node0, 2026-10-18 (fourth window) | `m7_layer0 --stage 0`, engine's captured row-0 `pre_attention_xhat/scales` and `pre_ffn_xhat/scales` recomputed with `act_quant_fp8_row`; log `phase1-d8-stage0-quantiser-20261018T012533Z.log` |
| mHC activation quantiser rule (D8), device stage-1 old-vs-new | **OLD-VS-NEW run pending a device window**: the retained `phase1-d8-stage1-layer0-20261018T012545Z.log` shows the old floor-rule behaviour (4,287/5,120 codes, 134/160 scales on token 0) but its header records the NEW binary and cubin sha256s, so it is not evidence for the old cubin and is not cited as such | node0, 2026-10-18 (mislabelled; regeneration owed) | pre-D8 floor-rule cubin retained at `/home/user/phase1-evidence/b0-out/mega_mhc_fused_bf16_sm121.cubin` (70,152 B, `6487f6ff...`) and old binary `m7_layer0-b0` on the node; rerun `--stage 1 --layer 0` with a labelled `OLD-VS-NEW` wrapper |
| SwiGLU activation quantiser | codes 2,560/2,560 + scales 80/80 (4x640) and 4,608/4,608 + 144/144 (2x2304) bit-exact; pre-D8 probe was 10.16%/0% exact, max code delta 24 | node0, 2026-10-18 (fourth window) | `m7_layer0 --stage 8`, device kernel against `swiglu_ref` on identical bytes; PASS 4/4 probes |
| Chained layer-0 replay (B2) | 18/18 row-0 tensors captured and diffed. Upstream agreement: `pre_attention_hidden` max abs 2.76e-7, `pre_attention_xhat` 5,119/5,120 codes and 160/160 scales. Attention tail: `attention_out` max abs 2.57e-1, `post_attention_hidden` bf16(host)-exact 48.18%, `pre_ffn_hidden` max abs 3.16e-2, `ffn_out` max abs 2.15e-1, `residual_out` bf16(host)-exact 5.96% | node0, 2026-10-18 (sixth window, pass-5 rerun) | `m7_layer0 --stage 7 --layer 0`, 16 chain launches plus 4 forced-routing launches; binary sha256 `2b518a35...`, the cross-build of the pass-5 code commit `1845f2d` (the evidence binary is that commit's own build; the docs head's rebuild reproduces it); every kernel bit-exact against its in-crate reference on the same bytes (K5 32,768/32,768, K9-K12 5,120/5,120 max ULP 0, forced control too); log `phase1-b2-stage7-layer0-rereview5-20261018T044122Z.log` (23,897 B, local copy sha256 `c2de544c1789743e73ccc5d5277dc3165d98296d0f5cc867cc819d9160754ac9`), UTC 04:41:22Z-04:41:26Z, metric/content lines bit-identical to the pass-4 log (only the label/argv/binary-sha/UTC header lines and the device/node MemAvailable lines - header before and after, process start, per-launch and staged-weight - differ); the pass-4 log (`...rereview4-20261018T041707Z.log`, `f99f14bf...`) remains on disk as pre-plumbing evidence and is no longer cited |
| Window-KV format attribution (B2) | The same f32 window-KV latent encoded in the device's FP4 E2M1 format and the host's FP8 E4M3 format predicts `attention_out` with max abs 3.13e-1 between the formats on the same latent and the same tail: that is the format's own contribution. The host-format prediction is 1.05e-1 from the host row against the measured 2.57e-1, but it reuses the device's q and attention-tail operands, so it carries upstream device-side differences and is a diagnostic, not a pure format measurement | node0, 2026-10-18 (fifth window) | same log, "D4 attribution check" lines; the device-format prediction's control reproduces the run's `attention_out` exactly; the same-latent `formats_delta` is the attribution and `predicted_host_delta` is only a diagnostic |
| Router top-6 and its measured effect (B2) | ordered top-6 differs in four slots; expert sets differ by one member (device only 246, host only 236). The corrected forced-routing control re-derives the host row's routing from the host row, builds the forced operands from it and verifies them against the checkpoint before any launch (every rank's w2 slices and every slot's w1/w3-derived activation codes and scales at every rank), requires identical ids with unchanged weights to reproduce identical routed bytes and ties the forced `ffn_out` to the forced routed output, then measures the flip pointwise: max abs |`ffn_out` free - `ffn_out` forced| = 3.750000e-1 at index 5071 (signed -3.750000e-1), 4,853/5,120 points differ; the two host deltas are 2.150760e-1 (free, max at index 1233) and 2.160320e-1 (forced, max at index 5071), and their difference is a change in max error, not a max-abs contribution. The flip is large and the max host-error point is flip-dominated: at index 1233 the flip is 2.500000e-1, 116.24% of the 2.150760e-1 host error there; forcing the host routing does not reduce the max host error | node0, 2026-10-18 (sixth window, pass-5 rerun) | same log, "forced-routing control" lines; the forced path requires identical ids+weights to give bit-identical routed output (else nondeterminism aborts), a flip to give a nonzero difference, and `ffn_out_forced` to be exactly the forced routed bits plus shared output through the free path's combination; both the free-running and forced routed combines are 5,120/5,120 exact against `mxfp4_ref` on the same bytes |
| Chained 40-layer device column (Stage C) - determinism, format and diff | two chain runs bit-identical on all 18 tensors x 40 rows, both v1 blobs sha256 `fecd813b...`; device manifest byte-identical to the host pair's (`6093c74b...`, 34,767 B; device blob 13,537,920 B); the pair and all 1,520 printed row lines are bit-identical to the earlier Stage-C run's, so the B2-helper adaptation moved verification only; `boundary_diff` control (host vs itself) exits 0 with all 40 rows bit-identical, host-vs-device exits 1 with all 40 rows differing | node0, 2026-10-18 (sixth window) | `m7_layer0 --stage 9 --stage9-layers 40 --stage9-runs 2`, released checkpoint token 0 position 0, 763 launches per chain run; binary sha256 `c1708abc...` (C head rebased onto main `2214fe6`); log `phase1-c-stage9-40rows-harness5b-20261018T050416Z.log` (572,420 B, sha256 `bd0f908e...`), diff log `phase1-c-stage9-diff-20261018T050941Z.log` (sha256 `b86be968...`, diff binary `5ffb11f9...`); pair `stage9-out-r5b/`; the retired `...final-...T023529Z.log` / binary `f5430527...` stay on disk as historical and are no longer cited |
| Chained 40-layer device column (Stage C) - bar results | Row 0 (Stage-B replay, bars ARE its own deltas: reproduction only) 18/18 at or under; rows 1-39 the independent ones. Totals 306 of 720 tensor-row comparisons at or under the pre-registered bars, 414 over; first over-bar row 1; residual_out max-abs grows 1.41 (row 1) -> 3.03e3 (row 39), post_attention_hidden -> 2.81e3, ffn_out 1.44e2 at row 20 (column max 7.957832e2 at row 24); discrete expert flips on 17/40 layers [0, 1, 2, 4, 5, 8, 10, 11, 15, 16, 17, 19, 20, 23, 24, 35, 36], the pointwise free-vs-forced flip and the two separate host deltas printed per layer; L36 is the zero-effect flip (references bit-identical, pointwise flip 0.0, host deltas 1.932627e2 both) | node0, 2026-10-18 (sixth window) | log `phase1-c-stage9-40rows-harness5b-20261018T050416Z.log`, row lines and "run 0 summary"/"run 0 routing" lines (all 1,520 row lines bit-identical to the earlier run's); bars are frozen B2 deltas of record (`phase1-b2-stage7-layer0-rereview5-20261018T044122Z.log` lines 149-166, binary `2b518a35...`, 23,897 B, local sha256 `c2de544c...`), stream tensors times row+1; `AT-BAR-PRECISION` class for the frozen log's 7-digit constants; the earlier "free minus forced = flip contribution" reading is withdrawn |
| Chained 40-layer device column (Stage C) - forced-control verdict refinement | ids-differ + routed-bytes-identical is allowed only when the two independently computed routed-combine references are bit-identical; L36 is the measured case (refs identical, pointwise flip 0.0, both host deltas 1.932627e2 at index 2455). A flip whose references differ and whose bytes are identical remains the hard copy abort; references bit-identical with bytes differing aborts as a broken/nondeterministic launch; identical-ids rules unchanged; the operand-provenance verifier is unchanged. B2's row-0 case is unchanged (its references differ, so the copy mutation still aborts) | node0, 2026-10-18 (sixth window) | per-layer forced-control lines of both runs in `phase1-c-stage9-40rows-harness5b-20261018T050416Z.log`; CPU suite `crates/sparkflash-kernels/tests/b2_harness_offline.rs` (14 tests, new allowed/abort cases); the pre-refinement abort log `phase1-c-stage9-40rows-harness5-20261018T045752Z.log` (259,368 B, sha256 `5b8da864...`, exit 1, no pair written) stays on disk as historical evidence |
| Chained 40-layer device column (Stage C) - audit and internal exactness | plan 95,819 / read 2,673 / unread 93,146, all unread classified (unselected-expert 90,660; DSpark-stage 2,401; text-path 40; indexer 32; compressor 11; head-not-claimed 2), 0 unclassified; 80/80 K5 checks 100% exact, 640/640 kernel-vs-reference checks exact, 160/160 routed-combine checks exact, zero warnings; 20,338,614,720 B through the harness's own reader (2,661 distinct tensors; the shared reader counts every read, including the forced-operand provenance re-reads) plus the paged Engram reads; wall 298.6 s, MemAvailable min 122,285,404 kB | node0, 2026-10-18 (sixth window) | log `phase1-c-stage9-40rows-harness5b-20261018T050416Z.log`, audit and per-kernel reference lines; the S7 seam is degenerate at position 0, so compressor/indexer are read differently from the host path (stated in the audit note); the earlier run's 15,826,461,120 B is superseded |
| mHC kernel time | p50 0.0227 ms/launch | node0, 2026-10-17 (first window) | pre-released-arithmetic kernel; **timing re-measure owed** |
| Attention outputs (FP4 latent window format) | 163,838/163,840 = 99.9988 % exact, max abs delta 2.4414063e-4, max ULP 1 - the merged-chain measurement; the earlier 9.765625e-4 was the pre-D8 stage-1 chain and is a historical label only | node0, 2026-10-18 (paired diagnostic after D8) | `--stage 2 --tokens 130 --stage2-kv-format fp4`; merged-head binary `c1708abc...` with the old 15-parameter kernel and cubin `822df7f8...` versus task-3 binary `487ac623...`/cubin `bbb216b7...`: every query-position/attention-out/verdict line identical (pairing logs `phase1-task3-diag-oldkernel-20261018T060534Z.log` 9,122 B sha256 `ef554d46...` and `phase1-task3-diag-newkernel-20261018T060534Z.log` 10,242 B sha256 `061eb529...`; that pairing predates the launch-ABI fix); equality re-run on the final ABI-fixed binary `14e0266f...` in `phase1-task3-abi-stage2-fp4-a/-b` (11,647 B each, sha256 `4613aac5...`/`b455cc57...`, PASS); **sink consumed on both sides** (kernel argument + `attention_with_sink`) |
| Attention outputs (engine FP8 window format, task-3) | 163,834/163,840 = 99.9963 % exact, max abs delta 9.765625e-4, max ULP 1; P0/P1/P127/P128 are 32,768/32,768 exact and P129 is 32,762/32,768; declared class 99.5% / 1.6e-2. Internal consistency in the engine's released window format: both sides consume bf16(E4M3 code x E8M0 scale); the engine's quantiser and `m4_quant::act_quant_fp8_row` agree byte-for-byte on 130/130 rows | node0, 2026-10-18 (task-3 window) | `--stage 2 --tokens 130 --stage2-kv-format engine`; out-of-window ids at or past `window_rows` contribute exactly zero on both sides at every position; ring order pinned oldest -> newest; two device runs per position byte-identical; the ABI-fixed binary `14e0266f...` also prints a host finite-window guard (130 x 512 decoded finite; no 0x7F NaN code or 0x7E x scale-254 overflow pair reaches the kernel) and a non-vacuous out-of-window assertion (3 injected ids verified outside and present); logs `phase1-task3-abi-stage2-engine-a/-b` (14,127 B each, sha256 `3abfdd7d...`/`1577daa2...`); decoded edge cases pinned in `mla_ref::window_decode_covers_every_e4m3_code_and_the_scale_edges`; not a bit-identity claim against the released kernel and the engine reference's f32-dequant seam is named |
| Launch-ABI fix re-check (stage 7 + stage 9) | stage 7 layer-0 replay: the 18-row measurement table and all 18 `single_layer_delta` lines are bit-identical to the merged B2 log (23,897 B, sha256 `c2de544c...`), headers aside and one updated B2-helper sentence in the forced-control separation line; stage 7 PASS 18/18. Stage 9 smoke (2 layers, 1 run) PASS framing to the last row, 0 FAIL; row 1 `residual_out` max-abs 1.411274 equals the merged Stage-C row-1 number (9 over/below-bar rows are attribution, not a gate) | node0, 2026-10-18 (task-3 ABI-fix window) | both runs from the ABI-fixed binary `14e0266f1f554f1372bf6a1f79f8dde9d0aefe2888a345b5ba30217c294757cc` (the build of head `c520c93`, command `./scripts/build-aarch64.sh --example=m7_layer0` run in `/home/user/wt/task3-window-kv`; the aarch64 hash is path-dependent, and the node copy of the evidence binary hashes to the same `14e0266f...`) and flash cubin `bbb216b7...`; logs `phase1-task3-abi-stage7-b2compare` (24,005 B, sha256 `fb87a5e4...`) and `phase1-task3-abi-stage9-smoke` (26,155 B, sha256 `b14a6036...`); all four flash_mla builder calls and six wrapper call sites are covered by the typed builder and the `m4_abi` source-text pin with its 15-parameter mutation; kernel baseline diff remains the single flash_mla line (44,080 B) |
| Attention kernel time | 5.73 ms/launch = 4.27 indexer plus 1.46 attention | node0, 2026-10-17 (first window) | event timing over repeats; the sink ABI changes no timing claim, but re-measure is owed |
| Attention device bytes | 4.2 MiB (`index_key` 4 MiB) | node0, 2026-10-17 | printer from the harness's planned layout |
| mHC determinism | 4 tokens x 100 launches, every compared byte identical | node0, 2026-10-17 (second window) | `--stage 4 --repeat 100`, released-arithmetic cubin; Phase 0 exit criterion 2 |
| MoE combine | 20,480/20,480 exact, max ULP 0 | node0, 2026-10-17 (second window: 8 experts re-run PASS; 32/64 from the first window) | `--stage 3 --experts 8/32/64 --tokens 4` |
| MoE combine time | 0.821 / 0.847 / 0.850 ms per token | node0, 2026-10-17 | same, four rank-chained launches |
| Expert coverage | 10/24, 24/24, 24/24 top-6 selections | node0, 2026-10-17 (first window) | same, coverage line |
| Quantisation gap | 1/20,480 and 2/20,480 outputs exact; max abs delta 14.19 / 12.91; second window 8 experts: max abs delta 13.4375 | node0, 2026-10-17 | same, f32-activation control |
| Device total memory | 124,610 MiB | node0 | `cuDeviceTotalMem_v2` ([device-path-design.md](device-path-design.md) section 9) |
| Free memory beside vLLM | 3,115 MB | node0, 2026-10-16 | `doctor --local` |
| Empty-node MemAvailable | 118.2 GiB at process start | node0, 2026-10-17 | harness floor check |
| Kernel baseline | seven of seven entries re-pinned. The D8 update moved `mega_mhc` to 70,312 B sha256 `e21293e3...` and `swiglu` to 21,776 B sha256 `4dbef63b...`. On this task-3 branch, `flash_mla_csa2_sm121` moves from 36,160 B sha256 `822df7f8...` to 44,080 B sha256 `bbb216b7...` (same pinned toolkit); the other five entries are byte-identical to the D8 baseline. | node0, 2026-10-18 (task-3 branch, `nvcc V13.0.88` sm_121a); D8 numbers from amd64 container `7d2f6a8c2071` | `regen-kernel-baseline.sh` commands re-run on the node for the task-3 branch; the diff is the single flash_mla line; `check-kernel-baseline.sh` is green in the `cuda-abi` CI job |
| Checkpoint bytes | 510,286,023,000 B, closing exactly | released index | [../architecture/00-overview.md](../architecture/00-overview.md) section 2.4 |
| KV entry / token / page | 356 B / 890 B / 56,960 B | design plus `tests/layout.rs` | D1, D16 |
| Per-rank weights (all resident) | 82.2 + 50.7 + 2.0 + 3.43 = 138.3 GB on a 130.66 GB device (does not fit; with the Engram tables on NVMe, 87.6 GB) | arithmetic, released bytes | section 7.5, D30 |
| Per-token collective budget | 83 collectives, about 1.77 MB, about 1.7 ms at 20 µs | arithmetic | section 7.2 |
| HSI indexer (task-7; landed as !35 `826f4f0`) | **measured, window 2026-10-18**: stage 2, 200 launches, p50: indexer proxy at one candidate (`--tokens 1`, engine format) **0.0930 ms**; attention-proper proxy at 128 candidates (A - B) **0.1562 ms**, with the 130-token engine total **0.2492 ms**; FP4 total 0.2471 ms and the merged-chain equality line reproduced exactly (163,838/163,840 exact, 2.4414063e-4, ULP 1); A correctness 163,834/163,840 exact (99.9963%), max abs delta 9.765625e-4, ULP 1, deterministic. Under the A-minus-B proxy the indexer is now about 0.6x the attention proper, so roadmap 6.3 item 1's "at or below" would hold - but that split is a proxy, not a phase-isolated measurement, so the target is **measurable and proxy-met, not established** | node0, 2026-10-18T17:16Z window; binary sha256 `43603e44...`, cubin `5bfee4c7...`, checkpoint `1bae0e10...`; run logs in `/home/user/phase2-evidence/window-2026-10-18/runs/`; method, exact lines and the proxy caveat in [phase3-indexer.md](phase3-indexer.md) | stage-1 printed-class log regenerated 2026-10-18T17:47:23Z: `stage 1: PASS (4 of 4 tokens at every boundary)` with the renamed Mega-mHC cubin; log `/home/user/phase2-evidence/window-2026-10-18/window-stage1-20261018T174723Z.log`; boundary deltas in [phase3-indexer.md](phase3-indexer.md) section 5 |

The stage-1 and stage-2 printed-class logs cited in this section predate the M2 fix of the
harness's printed comparison-class strings (the stage-1 log printed the retired stream-mean fold as
the current predictor). **Both halves have been regenerated in the 2026-10-18 windows.** The
**stage-2** half: runs A and B (engine format) and C (FP4) print the current comparison-class
strings, recorded with the window's numbers in [phase3-indexer.md](phase3-indexer.md) sections 2 and
5; the stage-2 numbers in 12.1's HSI indexer row come from those printed lines, not from a stale
printed class line. The **stage-1** half: run D failed before any GPU work on the Mega-mHC cubin
file name, and the corrected 17:47:23Z retry window printed
`stage 1: PASS (4 of 4 tokens at every boundary)` with the renamed cubin (log
`/home/user/phase2-evidence/window-2026-10-18/window-stage1-20261018T174723Z.log`); its boundary
deltas are quoted in [phase3-indexer.md](phase3-indexer.md) section 5. The stage-1 rows in 12.1 are
the fourth-window values, which that run re-confirms against a current printed-class log.

### 12.2 The labels

- **measured** — run on a named host, with the method and the comparison class printed by the
  harness; the numbers in 12.1.
- **arithmetic** — derived from a measured byte count or a released value; the derivation is written
  in place; the result inherits the uncertainty of each input.
- **design** — a decision this document makes (a layout, an order, a target).
- **hypothesis** — a claim the work will test (a GB10 peak far above the measured throughput, the ring
  path's reduction order, the device's usable memory below its reported total).

### 12.3 Reproduction

- Stage numbers: `m7_layer0 --stage {1,2,3} --checkpoint <released> --cubin-dir <cubins>`; the argv
  and comparison class are printed by the harness itself
  (`crates/sparkflash-kernels/examples/m7_layer0/`, in `!26`; split by 954d7af).
- Kernel reproducibility:
  [crates/sparkflash-kernels/scripts/check-kernel-baseline.sh](../../crates/sparkflash-kernels/scripts/check-kernel-baseline.sh)
  inside the pinned nvcc image.
- Cluster facts: `sparkflash doctor --local` on the node (read-only; exit 1 is the finding, not a
  crash).
- Build and upload path: [scripts/build-aarch64.sh](../../scripts/build-aarch64.sh) and
  [aarch64-pipeline.md](aarch64-pipeline.md).

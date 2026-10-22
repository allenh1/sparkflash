# Task-8 iteration B: the routed-expert gate/up W4A8 GEMM (Phase 3, roadmap 6.3 item 5)

Status: **code, CPU twin, ABI pins, and a device run: PASS in the Lead's window b6.**
One projection per launch is bit-identical to the CPU twin on every bf16 value at both batches
(`--tokens 8` and `--tokens 130`), padding is exactly zero, and two device runs are byte-identical
(section 5). A first window attempt on 2026-10-18T19:51Z panicked in the harness before the device
path; the failure and its fix are recorded in section 5. This note records what the kernel computes,
where its boundary is, how it is pinned, what the window printed, and what is deliberately not
claimed. Nothing here is a host-time, routing or end-to-end MoE result.

## 1. What the kernel computes

`kernels/routed_gate_up_sm121.cu` is one FP8 x packed-FP4 GEMM: **one** projection (w1 or w3) of
**one** routed-expert slice per launch. Two launches give a routed token's gate and up rows.

* `A` = the ffn block input, E4M3 with one ue8m0 scale per 32 values along K
  (`[tokens][5120]` codes + `[tokens][160]` scales) - the same operand `mega_mhc_sm121` emits
  and `b12x_mxfp4_moe_sm121` consumes.
* `B` = the rank's 576-row window of a routed expert's w1 (or w3): packed FP4 E2M1, two values per
  byte, low nibble first, `[576][2560]` bytes, with the per-output-row ue8m0 grid
  `[576][160]` bytes. K = 5120, N = 576 real columns per rank (2304 / 4). This is **D30's**
  routed-expert layout: TP=4 over the intermediate width, every rank holding a 576-row slice of
  every expert. D2's reference rule shards whole experts at the full 2304 width and is **corrected
  by D30** (`docs/DECISIONS.md` D30, merged in !85); this note describes the implemented layout and
  does not make a separate claim about the corpus.
* Output `[tokens * 6][640]` BF16: one row per routed (token, slot), 576 real columns plus 64
  **padding columns the kernel writes as exactly zero** (the combine kernel masks that tail; this
  one produces it).
* Arithmetic per output: 160 blocks of 32; `partial = sum_32 a[i] * w[i]`, then
  `acc += (partial * act_scale) * weight_scale`, one `__float2bfloat16_rn` at the end - the D14
  contract (`kernel.py:552-553`), which applies both scales to the 32-value **partial product**.
  Every ue8m0 scale is a power of two, so multiplying the partial by the scale is exact in F32 and
  the only rounding in the accumulation is the F32 addition itself.
* Guarding: block `(routed_row, tile)` computes columns `tile * 256 .. +256` and the expert id
  comes from `expert_ids[routed_row]`, so the kernel computes nothing for a column >= 576; it
  writes zeros there instead. 256 threads, three tiles, grid `(tokens * 6, 3)`.

Deliberately **not** in this kernel:

* **SwiGLU.** The clamps and `gate * sigmoid(gate) * up` (`model.py:857-885`) stay in the host
  chain / the existing `swiglu_mxfp8_quant_sm121` kernel; this GEMM ends at BF16 rows.
* **The routing weight multiply** (section 4).
* **The routed sum, the shared expert and the shared-add epilogue.** `b12x_mxfp4_moe_sm121` is the
  combine (it takes routed rows and produces the routed sum), and the shared expert is dense FP8
  already served by `b12x_mxfp8_dense_sm121`. This kernel is the piece that was missing between
  them.

## 2. The CPU twin, and why it is the reference

`crates/sparkflash-kernels/src/routed_gate_up_ref.rs` is a Rust twin of the kernel's arithmetic:
the same 160-block fold order, the same scale placement (block partial, then the two scales), the
same low-nibble-first FP4 decode, the same single BF16 rounding, the same padding writes. Its tests
run on any host: a one-hot hand-computed row, low-nibble packing, slots sharing one activation row
and differing only by expert, padding zeros, layout refusals, and a twin-vs-naive-dequantised
comparison.

The exactness argument is short: the checkpoint stores the weights already as FP4, the FP4 -> FP8
cast through float is exact on every E2M1 code, and choosing the same fold order means the twin is
computing the same real number on the same F32 intermediates. The harness also compares the twin
against `b2_harness::gate_up_rows` - the shared host implementation the m7 stage uses for this same
projection - so the two host implementations must agree bit for bit before the device comparison
means anything. That comparison is a cross-check of the arithmetic ordering, not of the decoders:
`b2_harness` shares `mxfp4_ref`'s FP4 loader with the tree's other references. The genuinely
independent decode-and-round path is the twin's naive test, which reads E2M1/E4M3/ue8m0 from the
bit layouts and rounds bf16 from the f32 bits without calling the twin's helpers.

What the twin does **not** model is the routing weight (section 4). The twin's A operand is the
quantised block input; a caller that needs the weighted path must apply the weight before the BF16
cast, outside this kernel.

## 3. Pins and ABI

* `crates/sparkflash-kernels/src/routed_gate_up_abi.rs` (new; `m4_abi.rs` stays frozen under the
  D26 ratchet): pins the kernel geometry (5120, 576, 640, 6, 32, 160 blocks, 256 threads, 3
  tiles), the 10-parameter kernel declaration parsed from the source (six pointers - the two A
  buffers, the two B buffers, expert ids, output - then `tokens`, `trace_ring`, `layer_id`,
  `step_seq`), the kernel's `static_assert`s (evaluated, each naming a `SPARK_` symbol), and the
  cubin entry from the baseline manifest. Mutation-tested: dropping a constant from the source
  fails the signature/constant pin.
* `crates/sparkflash-kernels/src/routed_gate_up_launch.rs`: the typed launch - argument words in
  declaration order, `grid = (tokens * 6, 3)` blocks of 256 threads, `tokens == 0` refused - and
  `KernelId::RoutedGateUp` is the driver's symbol `routed_gate_up_sm121`.
* Baseline: this is the **ninth** entry in `baseline/kernel_baseline.txt`, in `build.rs`'s
  `PROBED_KERNELS` table, in both baseline scripts' kernel lists, and in
  `tests/kernel_baseline.rs`'s count and texts.
* Trace: the kernel records under `TRACE_SCOPE_MOE_MXFP4`, the routed-expert consumer domain the
  combine kernel and the router already use, so the 16-bit scope table
  (`crates/sparkflash-trace/src/scope.rs`, which `kernels/trace_globaltimer_sm121.cuh` must mirror)
  is not extended. The shipped cubin is the trace-disabled build and the launcher passes a null
  ring, so a measurement is unaffected by tracing.
* Cubin pin (built in the pinned CUDA 13.0.3 / V13.0.88 image, trace-disabled): **9,752 B, sha256
  `115200d26c3404b7169c476e350f5575dadc3fec9e8b0ec97113cd297fa6c1af`**. Window b6 loaded this file
  and the harness checked both the size and the sha against the pin (it refuses any other bytes).
  The node did not recompile the kernel, so the pin's build provenance is the pinned image's nvcc,
  which is also what the CI baseline job rebuilds; the bytes are not attested by an on-node nvcc
  run.

## 4. The two launch-level gaps, decided outside the kernel

1. **The routing weight multiply (F6).** The released chain multiplies the *SwiGLU output* by the
   router weight **before** the BF16 cast inside the down projection's quantiser
   (`model.py:849-851`): the weight scales the activation that w2 consumes, not the gate/up rows
   this GEMM produces, so the operand this kernel reads is not weighted. The pinned
   `swiglu_mxfp8_quant_sm121` kernel has no weight operand, so **the weight stays outside**: the
   chain must apply it between SwiGLU and the w2-down activation quantiser, or a weighted
   SwiGLU+quantise variant must be added later. That variant is a named follow-up gap, not part of
   this MR. The harness compares the unweighted projections, and its pass bar says so.
2. **The shared expert and the shared-add epilogue (F8).** The shared expert is a dense FP8
   GEMM at K = 5120, already covered by `b12x_mxfp8_dense_sm121` (the K2 launch of the m7 stage);
   the routed sum is the combine kernel's job. **Neither is in this kernel**, and neither is
   measured by this harness. Integrating them is a chain-order question for the device-forward
   design, not a new gate/up kernel.

Both decisions are restated in the harness's declared comparison class, which is printed before any
number: it compares device vs CPU twin on the unweighted projections only, plus padding and
determinism. It does not compare a weighted activation, SwiGLU, the combine, the shared expert, or
any end-to-end MoE output.

## 5. The device window: what to build, run, and expect

Cubin (pinned toolkit): `nvcc -std=c++17 -arch=sm_121a -cubin -I kernels
kernels/routed_gate_up_sm121.cu -o out/routed_gate_up_sm121.cubin` under CUDA 13.0.3 (V13.0.88);
the harness refuses a cubin whose length or sha256 differs from the pin in
`routed_gate_up_abi.rs`.

Harness: `./scripts/build-aarch64.sh --example=m10_gate_up`, then on the node

```sh
m10_gate_up --checkpoint <snapshot> --cubin-dir <dir> --tokens 8 --layer 0 --experts 8 --repeat 100
```

The harness prints its **pass bar before any number**, then measures:

| check | bar |
| :--- | :--- |
| cubin | 9,752 B and the sha above |
| twin vs `b2_harness::gate_up_rows` (same dequantised activation) | **bit-identical**, all `tokens x 6 x 576` outputs, both projections; an ordering cross-check - the shared path reuses the tree's FP4 loader, so decoders are checked by the twin's own independent test, not here |
| device gate vs twin | **bit-identical**, all `tokens x 6 x 640` bf16 values, 0 ULP |
| device up vs twin | **bit-identical**, all `tokens x 6 x 640` bf16 values, 0 ULP |
| padding columns 576..639 | **exactly zero** in both projections |
| determinism | two device runs byte-identical |
| engine-style F32 control (unquantised block input, same expert slice) | printed as the **quantisation gap** over the **576 real columns only** (the twin's padded tail is not part of the control); max abs delta <= 10% of the control maximum - a sanity bound, not an exactness claim |
| timing | p50 per launch and per token at the requested repeat, printed with the invocation |

Why the device bar can be exact here: every operand already is what the checkpoint stores (FP8
codes, ue8m0 scales, packed FP4 bytes), the twin uses the same fold order, every scale is a power of
two, and the only rounding is one F32 add chain plus one BF16 conversion. The kernel writes the
multiply and the add as separate roundings (`__fmul_rn` then `__fadd_rn`): nvcc's default
`-fmad=true` contracted the plain expression into a single-rounding FMA in the first cut of this
kernel, which would have differed from the twin in the last bits and made the 0-ULP bar
unreachable - the PTX now has separate mul/add at both reduction sites (the remaining FMA
instructions with `x*0.125+1` / `x*0.5+1` immediate constants are exact steps inside the decode
helpers, not the reduction). Unlike the router (which crosses CUDA's `expf`), nothing in this
kernel is a different implementation of a transcendental.
The F32-control line is therefore a separate, bounded number: it shows how far the quantised path is
from an unquantised F32 gate/up over the **same** expert bytes, and it must not be read as the
device error.

Operand provenance: the released `layers.0.ffn.experts.<e>.w1/w3.weight` (stored I8, logical FP4
`[2304, 5120]`, probe reads the rank's 576-row window) and `.w1/w3.scale` (F8_E8M0
`[2304, 160]`), with the release's `embed.weight` rows 0..tokens, BF16-rounded and D8-quantised,
as the activation stand-in (the real input is the post-attention, post-mHC block input). The expert
ids are a deterministic cycle over the loaded experts: the router's selection is `m9_router`'s
claim, not this harness's. Every claim is about the GEMM over released expert bytes and released
activation bytes, not about end-to-end layer text.

Cost note: one launch reads the 576 x 2560 B weight slice once per token-slot row (6 rows per
token), so at 8 tokens a gate launch moves about 70 MB of packed weight - the kernel is
weight-bandwidth-bound on small batches, exactly like the combine. The measured p50 is 1.2734 ms
for the 8-token gate launch (0.1592 ms/token) and 30.1708 ms at 130 tokens (0.2321 ms/token, the
larger batch re-reading more activation rows per slot); those are observations at the printed
batches, not claims against the host seam.

**First window attempt (b4), 2026-10-18T19:51Z.** The Lead ran the harness on the node (binary `dd9921a4`,
from commit `0ff78fe`) and it panicked in the **host** comparison before the device path: the
engine-style control has `tokens x 6 x 576` values while the twin returns the ABI's padded
`tokens x 6 x 640`, and the gap comparison asserted equal lengths (`left 30720 right 27648` at 8
tokens, and **the same failure** at 130: `left 499200 right 449280`). No device numbers were
recorded. The window log `window-b4-20261018T195135Z.log` is on the node only; the two failing runs
are copied to the dev host as `phase2-evidence/window-2026-10-18/runs4/m10-tokens8.log` and
`.../runs4/m10-tokens130.log`. The gap now goes through `control_gap`, which selects the 576 real
columns of each twin row before comparing, and four example tests pin it at 1/8/130 tokens (the
compared lengths, the exact 30,720 / 27,648 / 499,200 / 449,280 numbers, and the helper's real-column
values against a padded twin) so a call-site revert no longer compiles - the guard fails in
`cargo test --workspace --examples` instead of waiting for the next window.

**Device window (b6), 2026-10-18T20:22Z - PASS.** Measured in a Lead window on node0. Log
`window-b6-20261018T202235Z.log`, runs `runs6/m10-tokens8.log` and `runs6/m10-tokens130.log`.
Provenance: binary `m10_gate_up_bb12c547_aarch64`, sha256
`bb12c547aed5f6a1a17c753db60c4330b41da4190ea62564d8e0155f1b9755b8`, 4,557,008 bytes, built from the
Rust/.cu content at `bb09bf3`; that content is byte-identical on the rebased tip `fc97f64`, and a
rebuild on the rebased tree reproduced the same sha256. The cubin was
`cubins/routed_gate_up_sm121.cubin`, 9,752 B / `115200d2...` (checked by the harness against the
pin). vLLM was stopped on all four nodes before the run and restarted afterwards.

At `--tokens 8` (48 routed rows, grid `(48, 3)`), all eight verdicts PASS: the device gate output is
bit-identical to the CPU twin on **30,720/30,720** bf16 values and the up projection likewise
**30,720/30,720**; the padding columns are **exactly zero** (0 nonzero); two device runs are
**byte-identical**; the twin matches the shared harness arithmetic with **0 mismatches over 27,648
real outputs** per projection; and the engine-style F32 control gap is **gate 3.977737e-2, up
3.599775e-2 over a control max of 2.490959e0** (1.6% / 1.4% of the control maximum, inside the 10%
sanity bound - the quantisation gap, not the device error). Timing over 100 launches: gate min
1.2596 / p50 1.2734 / mean 1.2813 / max 1.3862 ms, i.e. **1.2800 ms/launch and 0.1592 ms/token**;
up p50 1.2741 ms, 0.1593 ms/token.

At `--tokens 130` (780 routed rows, grid `(780, 3)`): device gate **499,200/499,200** exact and up
**499,200/499,200** exact, padding **0 nonzero**, two runs **byte-identical**, twin vs shared harness
**0 mismatches over 449,280 real outputs** per projection. Timing over 20 launches: gate p50
30.1708 ms (0.2321 ms/token), up p50 30.2761 ms (0.2329 ms/token); the per-launch cost grows with
the batch because every routed row re-reads its expert's 576 x 2560 B slice, which is the
weight-bandwidth-bound shape the cost note predicts.

What this window does **not** claim: the routing-weight multiply (F6), SwiGLU, the combine, the
shared expert, an end-to-end MoE number, a host-time saving, or a seam change. It is device
correctness of one routed gate/up projection per launch, against released layer-0 expert bytes and
released embed-row activation bytes.

## 6. What this iteration does not claim

* **The device claim is exactly the harness's class**, and it is one measured window (b6): one
  routed gate/up projection per launch, device output bit-identical to the CPU twin on every bf16
  value, padding zero, deterministic, against released layer-0 expert bytes and released embed-row
  activation bytes. It says nothing about a second batch shape beyond the two printed, another
  layer, another rank's slice, or a real post-attention activation.
* **No host-time saving** and no seam change: the host gate/up path is untouched, and a saving claim
  needs both sides measured on one window. The timing lines are launch observations.
* **No routing weight, SwiGLU, combine or shared expert** (section 4): those stay outside this
  kernel by contract, and the harness prints that in its comparison class. F6 (routing weight before
  the w2-down quantiser) and F8 (shared expert + shared-add epilogue) are named follow-up gaps, not
  work this kernel does.
* **No end-to-end MoE or layer text**: the activation is an embed-row stand-in and the expert ids are
  a deterministic cycle, not a router decision.
* The cubin pin is attested by the pinned image's build (the CI baseline job recompiles it with
  V13.0.88 and matches) and by window b6 loading exactly those bytes; the node did not recompile the
  kernel. If the bytes move, the pin moves with them in the same MR.

# Phase 3 m13: the output-projection pair (K6 wo_a grouped, K7 wo_b dense) in isolation

**Status: window b10 PASSED both shapes and the record is below.** This is the m13 deliverable the Lead
asked for: the two output-projection stages of layer 0's attention sublayer, in one branch, at the
released V4.1 dimensions, in the m10/m11/m12 window-harness pattern. Attention itself (K5
flash_mla_csa2, the window/sink/FLSA stage) is the **next** chain stage and is **not** part of this
harness: it is m14 and must not be bundled with this branch.

The harness is [m13_output_projections.rs](../../../crates/sparkflash-kernels/examples/m13_output_projections.rs);
it extends the same discipline as m10 (gate/up), m11 (mHC entry) and m12 (attention projections):
verbatim released bytes, declared provenance and permutation, both cubins verified by size and
sha256 before the first device allocation, CPU twins, determinism, an f32/FP8 control, both
`--tokens 8` and `--tokens 130`, a declared comparison class and an explicit NOT-claimed list.

## What the two stages are

| stage | kernel | pin (baseline manifest) | geometry (layer 0) |
| :--- | :--- | :--- | :--- |
| K6 wo_a | `b12x_mxfp8_wo_a_sm121` | 9,056 B, sha256 `84833781a77d094945d9f8e33168ff75cae4d9bf81b2a24b531b404984160eed`, 0 PTX membar lines | 8 groups x 1,024 output channels over 4,096 reduction values each; one position per launch; grid (8, 4, 1), block (256, 1, 1), 9 argument words |
| K7 wo_b | `b12x_mxfp8_dense_sm121` | 11,824 B, sha256 `03e4c422f1c55141a2abde6a53038e223d0a59ffe29d2ad913b979720f150a4a`, 0 PTX membar lines | N = 5,120, K = 8,192, rows = positions; grid (ceil(rows/64), 80, 1), block (256, 1, 1), 11 argument words |

The grouped kernel is a single-position kernel: the group is a grid dimension, not a GEMM shape,
and there is no rows argument. wo_a therefore launches once per position; wo_b takes rows and
launches once for the whole matrix. The shape table is the measured
`LAYER0_ATTENTION_WEIGHTS` in [m7_layer0_shapes.rs](../../../crates/sparkflash-kernels/src/m7_layer0_shapes.rs):
wo_a `[8192, 4096]` with a `[256, 128]` scale tensor and wo_b `[5120, 8192]` with a `[160, 256]`
scale tensor. The harness asserts those shapes and the scale-grid rule at start-up, so a drifting
table fails closed before anything is uploaded.

## Provenance and the declared transforms

Every weight and scale byte is read verbatim from the released checkpoint
`layers.0.attn.{wo_a,wo_b,wkv}.{weight,scale}` through the snapshot reader, which checks the tensor's
dtype and shape against the measured header before handing back bytes.

| tensor | stored | kernel layout | declared transform |
| :--- | :--- | :--- | :--- |
| `wo_b.weight` | F8_E4M3 `[5120, 8192]`, row-major `[out, in]` | B12x 32x32 micro-tiles | `pack_b12x`: a value-preserving index permutation (one byte per value, no decode, no arithmetic). A unit test proves the bijection on a fixture. |
| `wo_b.scale` | F8_E8M0 `[160, 256]` | the dense GEMM's per-32 grid `[out/32, in/32]` | none: uploaded as stored. The harness asserts the byte count equals `b12x_scale_bytes(5120, 8192)`. |
| `wo_a.weight` | F8_E4M3 `[8192, 4096]` | **eight** per-group B12x buffers of `[1024, 4096]` | declared per-group repack: slice the stored rows by group and apply `pack_b12x` per group. Value-preserving; each group is 1,024 x 4,096 bytes and the eight buffers are concatenated in group order. |
| `wo_a.scale` | F8_E8M0 `[256, 128]` | eight contiguous per-group `[32, 128]` grids | none: the stored grid's rows split 256 = 8 x 32, so the concatenation of the per-group scale buffers is the stored bytes. The harness asserts the grouped and whole-tensor byte counts against the B12x rules, so a wrong group stride fails closed. |
| `wkv.weight`, `wkv.scale` | F8_E4M3 `[512, 5120]`, F8_E8M0 `[16, 160]` | B12x | used **only** to build the declared attention-output stand-in below, never as the stage under test. |

The grouped kernel's arithmetic is the dense kernel's: `mxfp8_ref.rs`'s accumulation contract covers
both, and `mxfp8_ref::grouped_wo_a` is defined as a per-group `dense_gemm` over the same slice -
which is exactly what the harness's grouped twin reproduces. There is no "close enough" bar here.

## Activation operands: declared host-built stand-ins

The checkpoint stores no activation for either stage, so both operands are host-built and declared;
they are identical for the device and the CPU twin.

* **wo_a operand (declared attention-output stand-in).** The released K4 `wkv` latent of each
  position - computed by the CPU twin from the released wkv weight and scales on that position's
  D8-quantised `embed.weight` row - is broadcast across all 64 heads' 512-wide slices: each group's
  4,096-wide slice is the 512-wide latent repeated 8 times. That is the single-candidate attention
  output up to the sink-weighted scalar this harness deliberately does not model. The wkv bytes and
  the D8 quantiser are real; the broadcast is a declared seam, and **no attention, sink, softmax,
  RoPE or window-KV semantics are claimed**. The operand is 32,768 E4M3 codes and 1,024 ue8m0 scales
  per position.
* **wo_b operand (declared K6 -> K7 host seam).** The DEVICE wo_a `lo_out` of each position (bf16)
  widened to f32 and D8-quantised, exactly as M7 stage 7 builds the wo_b operand from the wo_a
  output. The seam itself is not claimed; the operand is built once and shared by the device run and
  the twin. The operand is 8,192 E4M3 codes and 256 ue8m0 scales per position.

## Comparison class (declared before any number)

Every output element of both stages, at both row counts, is compared to the in-crate reference on
the **same** activation bytes, packed B12x weights and scale bytes:

* **bit-identical**: all K6 outputs (positions x 8,192 bf16) against `mxfp8_ref::grouped_wo_a`,
  and all K7 outputs (positions x 5,120 bf16) against `mxfp8_ref::dense_gemm`. This is a legitimate
  exactness claim, not a bar: both kernels' documented scalar accumulation order is the reference's
  contract (K ascending in 32-value blocks, ascending within a block, one f32 accumulation, one
  bf16 rounding), the grouped kernel is explicitly the per-group dense GEMM, and M7 stage 7 measured
  K6 8,192/8,192 and K7 5,120/5,120 exact at this same real geometry.
* **determinism**: every launch is repeated on the same operands and all output bytes must match.
* **f32/FP8 control** (separate from the exactness claim): each stage's D8 activation quantisation
  gap - the codes x ue8m0 scales, decoded, against the f32 operand - bounded at 10% of that
  operand's maximum absolute value. It measures the activation-side FP8 cost only.

A single differing output bit is a FAIL, not a tolerance. There is deliberately no reduction-order
bar: this scalar cubin's accumulation order is the reference's contract. A future tensor-core path
would need a different declared class.

## NOT claimed (unchanged from the harness's declaration)

Attention, the sink, softmax, RoPE, the window-KV stage, the HSI candidate merge, the attention-output
inverse RoPE; the residual add / hc_post and any chain ordering, seam ledger or per-layer schedule;
any token, any text, any end-to-end layer or model output; routing, SwiGLU, the MoE block, the
shared expert, the combine; weight staging, device-budget admission, engine integration or any
host-time saving; and any per-decode-token timing derived from the measured launch times.

## Cubins: build and pin

Both cubins are built with the pinned toolkit (CUDA 13.0.3, nvcc V13.0.88) under
`nvidia/cuda:13.0.3-devel-ubuntu22.04`:

```
nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/b12x_mxfp8_wo_a_sm121.cu   -o out/b12x_mxfp8_wo_a_sm121.cubin
nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/b12x_mxfp8_dense_sm121.cu -o out/b12x_mxfp8_dense_sm121.cubin
```

The harness reads both cubin files, refuses anything that is not the pinned size and sha256 from
`baseline/kernel_baseline.txt` (and refuses if the manifest's own pin moved), and only then loads the
modules and allocates device memory. The table above records the sizes and digests.

## Building and running the harness

The harness is a normal crate example; no extra features are needed.

```
cargo build --example m13_output_projections                              # host check
scripts/build-aarch64.sh --example=m13_output_projections                # node binary
```

Usage: `m13_output_projections --checkpoint <snapshot dir> --cubin-dir <dir with both cubins>
[--tokens 8] [--repeat 100] [--floor-gib 32]`. The checkpoint is a released snapshot directory such
as the one the m12 window used
(`.../models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/<rev>`). `--tokens` is the number of
independent positions (matrix rows), not a batch or sequence dimension; 130 is the fixture prefill
length. `--layer` must be 0 (the default) because the operands are the layer-0 boundary inputs.
The harness checks `MemAvailable` against `--floor-gib` at start and end, so it refuses to run under
the serving stack rather than racing it for memory.

## Window commands (for the Lead window)

The binary is staged on the node at
`~/phase2-evidence/window-2026-10-18/m13_output_projections_6d50c4fc_aarch64` (3,961,480 B, sha256
`6d50c4fc5db4a6d5af1b742e63ff83c701a4c5c4045e08b231ceb478cc13f9b6`) together with both cubins in
`~/phase2-evidence/window-2026-10-18/cubins/`. The exact commands, with vLLM stopped on all four
nodes first and the checkpoint path the m12 window used:

```
~/phase2-evidence/window-2026-10-18/m13_output_projections_6d50c4fc_aarch64 \
  --checkpoint /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins --tokens 8 --repeat 100
```

and the same line with `--tokens 130`.

## Pass bars (the window record must show)

| check | bar |
| :--- | :--- |
| cubin pins | both files match 9,056 B / `84833781...` and 11,824 B / `03e4c422...` before any device allocation |
| K6 wo_a output bits | all positions x 8,192 bf16 bit-identical to `mxfp8_ref::grouped_wo_a` (0 mismatches) at 8 and 130 positions |
| K7 wo_b output bits | all positions x 5,120 bf16 bit-identical to `mxfp8_ref::dense_gemm` (0 mismatches) at 8 and 130 positions |
| determinism | every repeat launch byte-identical for both stages |
| K6 control | worst D8 gap <= 10% of the operand maximum |
| K7 control | max D8 gap <= 10% of the operand maximum |
| exit code | 0 at both row counts |
| summary | "10 verdicts, 0 failed" and "m13_output_projections: PASS" |

## Timings: what was measured and what is not

The harness prints per-launch event timing (min / p50 / mean / max over `--repeat` launches, plus the
batched ms/launch) for each stage at the row count it ran with. These are whole-stage measurements on
one node with vLLM stopped. **No per-decode-token cost is claimed or derived from them.** wo_a is a
single-position kernel, so its reported time is one position's launch, while wo_b's is one
whole-matrix launch of N rows; the two are not per-token figures, and any comparison against the m12
numbers must use the same row counts (8 vs 8, 130 vs 130). If a per-token figure is ever wanted it
would have to be labelled a hypothesis, state which row count it assumes, and would still assume one
row per token; it is not measured here.

## Window record: b10 (PASS)

Ran in a Lead window on 2026-10-18: window b10 start 23:32:08Z (host node0), vLLM stopped on all
four nodes from 23:32:08Z, GPU compute apps 0 at 23:33:20Z, and vLLM restarted after the runs at
23:33:40Z (the Lead verified the served endpoint separately). The evidence is node-only:
`/home/user/phase2-evidence/window-2026-10-18/window-b10-20261018T233208Z.log`,
`runs10/m13-tokens8.log` (run 23:33:20Z-23:33:26Z, exit 0) and `runs10/m13-tokens130.log`
(23:33:26Z-23:33:40Z, exit 0) under the same directory.

Provenance: binary `m13_output_projections_6d50c4fc_aarch64`, 3,961,480 B, sha256
`6d50c4fc5db4a6d5af1b742e63ff83c701a4c5c4045e08b231ceb478cc13f9b6`; cubins
`b12x_mxfp8_wo_a_sm121` 9,056 B sha256 `84833781a77d094945d9f8e33168ff75cae4d9bf81b2a24b531b404984160eed`
and `b12x_mxfp8_dense_sm121` 11,824 B sha256
`03e4c422f1c55141a2abde6a53038e223d0a59ffe29d2ad913b979720f150a4a`, both verified by the harness by
size and sha256 before any device allocation; MemAvailable 118.3 GiB at start and 117.6 GiB
(123,276,056 kB) at end for the 8-position run, 118.3 GiB to 117.5 GiB (123,214,508 kB) for the
130-position run, against the 32-GiB floor.

Result: **PASS** at both position counts - 8 of 8 verdicts (two cubin pins, plus output bits,
determinism and control for each stage), 0 failed; determinism was byte-identical for both stages at
both shapes, and every output max abs delta was 0 (bit-identical, not within a bar).

Measured at 8 positions:

| stage | output bits exact | control gap (bar 10%) | per-launch ms min / p50 / mean / max (batched) |
| :--- | :--- | :--- | :--- |
| K6 wo_a | 65,536 / 65,536 (100%) | 2.0833% (1.25e-1 of 6.0) | 0.6425 / 0.6438 / 0.6512 / 0.7995 (0.6511) |
| K7 wo_b | 40,960 / 40,960 (100%) | 3.7736% (1.25e-1 of 3.3125) | 21.0654 / 21.2368 / 21.2337 / 21.3856 (21.2462) |

Measured at 130 positions:

| stage | output bits exact | control gap (bar 10%) | per-launch ms min / p50 / mean / max (batched) |
| :--- | :--- | :--- | :--- |
| K6 wo_a | 1,064,960 / 1,064,960 (100%) | 2.0833% (1.25e-1 of 6.0) | 0.6521 / 0.6532 / 0.6609 / 0.8005 (0.6603) |
| K7 wo_b | 665,600 / 665,600 (100%) | 3.7736% (1.25e-1 of 3.3125) | 38.7021 / 38.7264 / 39.0305 / 41.0090 (38.7795) |

The control rows are the two D8 activation quantiser gaps (worst over positions for K6, over the whole
K7 operand matrix for K7), each against its own operand maximum. K6's per-launch time is one
position's launch (the grouped kernel is single-position, so the 130-position run performs 130 of
them); K7's is one whole-matrix launch of `rows = positions`.

Performance note, unchanged and honest: these are whole-stage measurements on one node with vLLM
stopped at the stated position counts. **No per-decode-token cost is claimed or derived from them**,
and the two stages are not directly comparable per token because K6 is per position while K7 is per
matrix; any comparison against the m12 numbers uses the same position counts (8 vs 8, 130 vs 130),
and any derived per-token figure would be a labelled hypothesis that states which count it assumes.

The NOT-claimed list above is unchanged: no attention/sink/softmax/RoPE/window-KV (K5 flash_mla is
the next stage, m14), no residual add/hc_post, no chain ordering or seam ledger, no token/text/
end-to-end output, no routing/SwiGLU/MoE/shared-expert/combine, no staging/device-budget/engine
claim, no host-time saving, and no per-token derivation of the timings.

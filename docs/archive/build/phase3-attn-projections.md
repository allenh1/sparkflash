# Phase 3 m12: the attention projections in isolation

The second device stage of the forward chain, run on its own: layer 0's attention projections - the
dense MXFP8 GEMMs K2 wq_a, K3 wq_b and K4 wkv of `kernels/b12x_mxfp8_dense_sm121.cu` - on released
checkpoint bytes at the real released dimensions, with the CPU twin (`mxfp8_ref::dense_gemm` on the
same bytes) and the typed launch builder (`mxfp8_dense_launch::Mxfp8DenseParams`) as the only way the
arguments are formed. This is the m10/m11 discipline applied to the stage that follows the mHC entry.

## What this harness proves, and what it does not

It proves the **attention projections alone**: the released `wq_a`, `wq_b` and `wkv` weights and
their ue8m0 scale grids reach the kernel in the layouts it consumes, every one of their output
bits is the in-crate reference's bits on the same operands, the launches are deterministic, and the
activation quantiser's own gap is bounded. Both row counts (8 and 130) run every projection at its
real N/K.

It does NOT prove: attention itself (flash_mla, the sink, softmax, RoPE, the main-KV packing, output
inverse RoPE), q/kv norm as a measured step, the residual add and the hc_post merge that follow the
attention block, K5-K16, any chain ordering, per-layer schedule or seam ledger, any token/text/
end-to-end output, routing/SwiGLU/MoE/shared expert/combine, weight staging, device-budget admission,
engine integration, or any host-time saving.

## Provenance and permutation, declared

The weights are read verbatim from the released checkpoint. The shapes are not assumed: they are the
headers measured in `sparkflash_kernels::m7_layer0_shapes` (the table records them from the snapshot's
safetensors headers, and its own unit tests re-derive every scale grid from the 32x32 rule).

| projection | released tensors | weight dtype/shape | scale dtype/shape |
| :--- | :--- | :--- | :--- |
| wq_a (K2) | `layers.0.attn.wq_a.weight`, `.scale` | F8_E4M3 [1280, 5120] | F8_E8M0 [40, 160] |
| wq_b (K3) | `layers.0.attn.wq_b.weight`, `.scale` | F8_E4M3 [32768, 1280] | F8_E8M0 [1024, 40] |
| wkv (K4) | `layers.0.attn.wkv.weight`, `.scale` | F8_E4M3 [512, 5120] | F8_E8M0 [16, 160] |

Two byte layouts matter, and both are stated rather than silently reshaped:

* **Weights.** The checkpoint stores row-major [out, in] E4M3; the kernel walks B12x 32x32
  micro-tiles. The host applies `pack_b12x`, a declared **value-preserving index permutation** - no
  arithmetic, no decode, one byte per value - and the harness's unit test proves every packed byte is
  the checkpoint byte at the B12x index and that the map is a bijection.
* **Scales.** The kernel's scale index is `(n/32) * k_blocks + (k/32)`, which is exactly the
  checkpoint's stored [out/32, in/32] row-major grid, so the `.scale` tensor is uploaded **as
  stored**, with no transform. The harness asserts the stored tensor's byte count equals the GEMM's
  scale-grid byte count for each projection, so the claim fails closed if it is ever wrong.

The **activation operands** are host-built and declared, because the checkpoint stores no activation
tensor for this stage:

* wq_a and wkv: the released `embed.weight` row of position t, bf16 widened to f32, quantised by the
  engine's own D8 rule (`m4_quant::act_quant_fp8_row`). This is the declared stand-in for the
  layer-0 hidden state that K1 (the mHC entry) produces; K1 is not launched here.
* wq_b: the **device** wq_a result of position t, bf16 widened to f32, passed through the released
  `layers.0.attn.q_norm.weight` RMS normalisation (`NORM_EPS = 1e-20`) and the D8 quantiser - the
  chain's K2 -> K3 host seam exactly as M7 stage 7 performs it. The operand is built once and is
  identical for the device and the twin, so the seam is declared and **not** claimed as part of the
  stage's arithmetic.

## Run it

Cubin (pinned toolkit, CUDA 13.0.3 / nvcc V13.0.88):

```sh
nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/b12x_mxfp8_dense_sm121.cu \
    -o out/b12x_mxfp8_dense_sm121.cubin
```

The harness refuses any other compiled form before the first device allocation: 11,824 B and sha256
`03e4c422f1c55141a2abde6a53038e223d0a59ffe29d2ad913b979720f150a4a` (the `baseline/kernel_baseline.txt`
pin, read through `baseline::pin`; 0 PTX membar lines, the scalar dense path carries no fence).

Binary (cross-compiled on the dev box, copied to the node):

```sh
./scripts/build-aarch64.sh --example=m12_attn_projections
```

Window launch commands (N is the matrix row count; the kernel has no other batch dimension and every
launch is a whole [N, inner] matrix; 130 is the fixture prefill length):

```sh
m12_attn_projections --checkpoint <snapshot> --cubin-dir <dir> --tokens 8 --repeat 100
m12_attn_projections --checkpoint <snapshot> --cubin-dir <dir> --tokens 130 --repeat 100
```

## Declared pass bar

| check | bar |
| :--- | :--- |
| cubin | 11,824 B, sha256 `03e4c422...`, 0 PTX membar lines; verified before allocation |
| every output bit, all three projections | **bit-identical** to `mxfp8_ref::dense_gemm` on the same activation bytes, packed weights and scales (the kernel's documented accumulation order is the reference's; M7 stage 7 measured K2 1,280/1,280, K3 32,768/32,768 and K4 512/512 exact at this real layer-0 geometry) |
| determinism | the launch repeats byte-identically on the same operands |
| f32/FP8 control | the D8 activation quantisation gap (decoded codes x scales against the f32 operand) <= 10% of that operand's maximum absolute value; printed per projection, not part of the exactness claim |

There is deliberately no reduction-order bar: this cubin is the scalar path whose accumulation order
is the reference's contract, and the M7 stage-7 chain already measured the real geometry exact. A
future tensor-core fast path would need a different declared class, because its accumulation order is
the hardware's.

## Window record: b8 (PASS)

Ran in a Lead window on 2026-10-18: window b8 start 22:32:59Z, with vLLM stopped on all four nodes
before the runs and restarted after 22:34:33Z (the Lead verified the served endpoint separately).
The evidence is node-only: `/home/user/phase2-evidence/window-2026-10-18/window-b8-20261018T223259Z.log`,
`runs8/m12-tokens8.log` (run 22:34:11Z-22:34:18Z, exit 0) and `runs8/m12-tokens130.log`
(22:34:18Z-22:34:33Z, exit 0) under the same directory.

Provenance: binary `m12_attn_projections_b1df5306_aarch64`, 3,888,216 B, sha256
`b1df5306c755cd4ba6e1102b69fe9d720442255694323c201b7f509fdf09fd7c`; cubin
`b12x_mxfp8_dense_sm121`, 11,824 B, sha256
`03e4c422f1c55141a2abde6a53038e223d0a59ffe29d2ad913b979720f150a4a`, verified by the harness before
any device allocation; MemAvailable 118.2 GiB at start and 117.6 GiB (123,362,072 kB) at end against
the 32-GiB floor.

Result: **PASS** at both row counts - 10 of 10 verdicts (three projections x output bits,
determinism and control, plus the cubin pin), 0 failed; determinism was byte-identical for every
projection at both shapes. Every output max abs delta was 0: the bits are the reference's bits, not
a within-bar match.

Measured at 8 rows:

| projection | output bits exact | control gap (bar 10%) | per-launch ms min / p50 / mean / max (batched) |
| :--- | :--- | :--- | :--- |
| wq_a (1280 x 5120) | 10,240 / 10,240 (100%) | 3.2922% (3.125e-2 of 9.4921875e-1) | 11.8396 / 11.8409 / 11.8414 / 11.8660 (11.8412) |
| wq_b (32768 x 1280) | 262,144 / 262,144 (100%) | 2.9775% (2.3996639e-1 of 8.05943) | 6.3708 / 6.4083 / 6.4182 / 6.6449 (6.4422) |
| wkv (512 x 5120) | 4,096 / 4,096 (100%) | 3.2922% (same released embed operands as wq_a) | 12.0985 / 12.1023 / 12.1059 / 12.2176 (12.1023) |

Measured at 130 rows:

| projection | output bits exact | control gap (bar 10%) | per-launch ms min / p50 / mean / max (batched) |
| :--- | :--- | :--- | :--- |
| wq_a (130 x 1280 x 5120) | 166,400 / 166,400 (100%) | 3.2922% (3.125e-2 of 9.4921875e-1) | 13.1321 / 13.1941 / 13.2050 / 13.5018 (13.2611) |
| wq_b (130 x 32768 x 1280) | 4,259,840 / 4,259,840 (100%) | 5.5564% (4.920082e-1 of 8.854728) | 28.9524 / 29.6577 / 29.6871 / 30.5491 (29.7124) |
| wkv (130 x 512 x 5120) | 66,560 / 66,560 (100%) | 3.2922% (same released embed operands as wq_a) | 12.3744 / 12.3768 / 12.3783 / 12.4770 (12.3769) |

Performance note, honest: these are the measured per-launch times of whole-matrix launches at 8 and
130 rows on this node with vLLM stopped. **No per-decode-token cost is claimed or derived from
them.** The rows are independent positions, not a decode stream, and the two row counts differ by
16x while wq_b's time differs by about 4.6x because the 130-row launch is a larger whole matrix, not
a token step. Any derived figure (for example a "ms per token") is a **hypothesis**, would have to
say which row count it assumes (8 or 130), and would still assume one row per token; it is not
measured here.

The NOT-claimed list above is unchanged: no attention/softmax, no KV window, no chain, no
token/text/end-to-end output, no staging/device-budget claim, and no host-time saving (including no
per-token derivation of the timings).

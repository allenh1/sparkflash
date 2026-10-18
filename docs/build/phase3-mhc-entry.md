# Phase 3 m11: the mHC entry stage in isolation

The first device stage of the forward chain, run on its own: layer 0's fused Mega-mHC entry
(`kernels/mega_mhc_sm121.cu`) on released checkpoint bytes, with the CPU twin
(`mega_mhc_ref::mega_mhc_layer_with_scale`) on the same bytes and the typed launch builder
(`mega_mhc_launch::MegaMhcParams`) as the only way the arguments are formed. This is the m10
pattern applied to the chain's first stage: one kernel, one declared comparison class, real bytes.

## What this harness proves, and what it does not

It proves the **mHC entry stage alone**: the released `hc_attn_fn` / `hc_attn_base` /
`hc_attn_scale` and `attn_norm.weight` tensors reach the kernel through the typed builder, the
kernel's 4 x 5,120 residual-stream update, its FP8 block-input quantiser, its 24 mixed coefficients
and its 1/rms are the reference's values (bit-identical where the arithmetic allows it, within the
declared bars where the reduction order differs), two device runs are byte-identical, and the FP8
quantisation gap of the block input is bounded.

It does NOT prove: attention (the next stage); any chain ordering, per-layer schedule or seam
ledger; any token, any text or any end-to-end layer/model output; routing, SwiGLU, the MoE block,
the shared expert or the combine; weight staging, device-budget admission or engine integration; any
host-time saving. The host seams *before* this stage are named and not measured: the embedding row
(the X_-1 seed), the layer-0 boundary coefficients (the checkpoint stores no layer -1 coefficients,
so B = identity, C = 0, A = [1, 0, 0, 0], Y_-1 = 0), and the staging of the mHC tensors themselves.

## Run it

Cubin (pinned toolkit, CUDA 13.0.3 / nvcc V13.0.88):

```sh
nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/mega_mhc_sm121.cu \
    -o out/mega_mhc_sm121.cubin
```

The harness refuses any other compiled form before the first allocation: 70,312 B and sha256
`e21293e3f6288dbe413a2b998efce6c43c8a56dd993eca5cd1bfc6138cdca618` (the tree's
`baseline/kernel_baseline.txt` pin, read through `sparkflash_kernels::baseline::pin`; 4 PTX membar
lines, the kernel's grid rendezvous).

Binary (cross-compiled on the dev box, copied to the node):

```sh
./scripts/build-aarch64.sh --example=m11_mhc_entry
```

Window launch commands (one position per launch; there is no batch dimension, so `--tokens 8` and
`--tokens 130` run the same stage over 8 and 130 independent released embed rows, the fixture
prefill length being 130):

```sh
m11_mhc_entry --checkpoint <snapshot> --cubin-dir <dir> --tokens 8 --repeat 100
m11_mhc_entry --checkpoint <snapshot> --cubin-dir <dir> --tokens 130 --repeat 100
```

The harness prints its comparison class and pass bar before any number, then per position:

```
--- position N (embed row N, four copies) ---
  hc_coeffs B (the mixer): 16 f32 values, ...
  hc_coeffs A (the input mixes), hc_coeffs C (the block-output weights), ...
  xhat_fp8 (E4M3 codes), xhat_scales (ue8m0), residual streams X_l, 1/rms(X_hat), ...
  block input f32 capture (xhat_normed), ...
  f32 control (FP8 quantisation gap): ...
summary: N positions x 2 verdicts, 0 failed
m11_mhc_entry: PASS ...
```

## Declared pass bar

| check | bar |
| :--- | :--- |
| cubin | 70,312 B, sha256 `e21293e3...`, 4 PTX membar lines; verified before allocation |
| residual-stream write X_l | **bit-identical** after bf16 rounding, 4 x 5,120 values |
| ue8m0 scales | **bit-exact**, 160 bytes |
| f32 block-input capture (xhat_normed) | **bit-exact** against `block_input_normed` from the same inputs and the device's own 1/rms |
| E4M3 codes | >= 99.5% exact and every code within 1 (the kernel's fixed-order tile reduction vs the reference's sequential accumulation) |
| B/A/C coefficients | max abs delta <= 1e-4 |
| 1/rms(X_hat) | max abs delta <= 3e-4 |
| determinism | two device runs of every position byte-identical across all six outputs |
| f32 control (FP8 quantisation gap) | max abs gap <= 10% of the capture's max abs value; printed separately, not part of the exactness claim |

The bars are the comparison class m7 stage 1 declares for this kernel, unchanged. They are not a
loosened exactness claim: the kernel reduces the two sums of squares in a fixed tile order and
rounds once, the reference accumulates sequentially, and 1/rms moves values across E4M3 boundaries;
the harness says so in the per-position finding when a code differs.

## Window record: b7 (PASS)

Ran in a Lead window on 2026-10-18, start 22:01:38Z, with vLLM stopped on all four nodes before the
runs and restarted after 22:02:52Z (the Lead verified the restore separately). The evidence is
node-only: `/home/user/phase2-evidence/window-2026-10-18/window-b7-20261018T220138Z.log`,
`runs7/m11-tokens8.log` (22:02:51Z) and `runs7/m11-tokens130.log` (22:02:52Z) under the same
directory. Both runs exited 0 and each took under a second.

Provenance: binary `m11_mhc_entry_8da8b0b2_aarch64`, 4,008,760 B, sha256
`8da8b0b292e507821b9f91afcb780a055f11e3340250f5033660c2646fc88be6`; cubin `mega_mhc_sm121`,
70,312 B, sha256 `e21293e3f6288dbe413a2b998efce6c43c8a56dd993eca5cd1bfc6138cdca618`, verified by the
harness before any device allocation; MemAvailable 118.0 GiB at start and 117.8 GiB at end against
the 32-GiB floor.

Result: **PASS** at both shapes - 8 of 8 positions at every boundary, and 130 of 130 positions with
0 failed of 130 x 2 verdicts.

Measured at 130 positions (per position unless stated):

| boundary | measured |
| :--- | :--- |
| residual streams X_l | 20,480 of 20,480 bf16 exact (100%), max abs delta 0 |
| xhat_scales (ue8m0) | 160 of 160 exact (100%) |
| xhat_fp8 (E4M3 codes) | 5,120 of 5,120 exact (100%) in every position |
| block-input f32 capture (xhat_normed) | 5,120 of 5,120 exact (100%), max abs delta 0 |
| 1/rms(X_hat) | max abs delta <= 2.746582e-4 (bar 3e-4; e.g. position 128 2.2888184e-4, position 129 1.66893e-5) |
| hc_coeffs B/A/C | within the declared 1e-4 bar (position 129: B 1.1920929e-7, A 2.682209e-6, C 6.7055225e-7) |
| determinism | two device runs byte-identical in every position |
| f32 control (FP8 quantisation gap) | 3.7352517e-3 against capture max 7.922271e-2 = 4.7149% (bar 10%); position 129: 3.212206e-3 / 9.540956e-2 = 3.3668% |

Timing (100 launches, one position each): tokens 8 min 0.0330, p50 0.0342, mean 0.0343, max 0.0387
ms per launch (batched 0.0341 ms); tokens 130 min 0.0334, p50 0.0340, mean 0.0341, max 0.0350 ms
(batched 0.0338 ms). These are timing measurements of one isolated stage, not a host-time saving
claim.

What the PASS covers is unchanged from the section above: the mHC entry stage in isolation. It does
not claim attention, any chain ordering/schedule or seam ledger, any token/text/end-to-end output,
routing/SwiGLU/MoE/shared expert/combine, weight staging, device budget, engine integration, or any
host-time saving. The bit-identical boundaries and the declared bars are the class stated above; the
numbers above are the measured values, not a relaxation of it.

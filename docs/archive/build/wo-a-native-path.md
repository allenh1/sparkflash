# The native wo_a path (task-6 / F1 / M1)

The grouped output projection wo_a is the one attention projection where the released model does
not run an FP8 activation: it keeps the BF16 attention output and dequantises the wo_a weight to
BF16 at conversion, so the GEMM is a BF16 einsum. The chain ran an E4M3 activation quantiser and FP8
weight codes there, which made wo_a the coarsest arithmetic deviation in the chain
(docs/build/native-arithmetic-audit.md, F1). This change adds the native path as a chain option and
records the deviation's magnitude.

## The native rule

- model.py:766-772: the body output is a BF16 tensor and the inverse rotation is applied into it.
- model.py:783-787: the weight is viewed as [n_local_groups][o_lora_rank][-1] and multiplied with a
  grouped einsum. BF16 operands, f32 accumulation, BF16 output, no activation quantiser.
- convert.py dequantises the checkpoint's E4M3 codes with their ue8m0 scales into BF16. That
  dequantisation is exact: an E4M3 significand has four bits and the scale is a power of two, which
  BF16 (eight significand bits) represents exactly.

## The two implementation paths, and the measured decision

Measured from the released checkpoint's safetensors headers (no GPU): layers.N.attn.wo_a.weight is
F8_E4M3 [8192, 4096] = 33,554,432 B and .scale is F8_E8M0 [256, 128] = 32,768 B per layer. The
BF16 copy is 67,108,864 B (exactly 64 MiB). Over the 40 backbone layers that is 1,343,488,000 B in
FP8 versus 2,684,354,560 B in BF16, a delta of 1,340,866,560 B.

- Path A, BF16 weight copy at load (what convert.py does). Arithmetic exactly native, but the chain
  re-stages weights per layer per position, so it doubles wo_a's H2D traffic (+1.34 GB per position,
  about +23.7 ms at the measured 56.6 GB/s aggregate staging rate) and needs a new BF16 grouped
  kernel; keeping the BF16 set resident instead would add 1.34 GB of device memory and break the
  no-resident-matrix-set design.
- Path B, FP8 codes stay in HBM and are dequantised inside the GEMM, with a BF16 activation. The
  products are the same f32 values as Path A because the dequantisation is exact; only the
  accumulation association differs. H2D does not grow, host memory does not grow, per-element ALU
  drops (one FP8 decode instead of two), and the kernel, pure-Rust twin, typed launch, pins and
  host tests already existed on batch/2026-10-19-25.

Path B is what this branch wires (Lead decision 2026-10-19). Path A is not implemented.

## What the chain runs

- A chain-level option --wo-a-mode native|mxfp8, default native. The run output records a
  "wo_a path: mode=... " line, so a window log shows which arithmetic ran and the two paths differ
  only where native differs.
- Native mode: the captured inverse-RoPE'd row is rounded to BF16 exactly as the reference's
  in-place unit round leaves it (F5 for this seam), uploaded as 65,536 B of BF16 (the current path
  uploads 32,768 B of E4M3 plus 1,024 B of scales), and launched with
  b12x_bf16a_mxfp8w_wo_a_sm121. The checkpoint's FP8 weight codes and scales are still read,
  cached and staged exactly as before. The activation quantiser call is gone.
- The reused device buffer set grows by one 65,536 B BF16 activation buffer.

## Per-layer comparison, and the offline re-run

When the existing --moe-layer0-diagnose flag is on, or when --dump-wo-a-operand DIR is given, each
layer runs the comparison once, at its first position, and the run log gains one line per layer:

    wo_a native comparison: mode=... layer=N device_vs_native_mismatches=.../8192 device_max_abs=...
      mxfp8_codes_equal=.../8192 mxfp8_max_abs=... output_max_abs=... operand_checksum=...

device_vs_native is the native kernel against the BF16 reference computed from the same BF16
codes, which must be zero mismatches. mxfp8_codes_equal and mxfp8_max_abs are the current MXFP8 path
against that same reference on the same captured operand: that is the F1 magnitude. The dump writes
wo_a_layer_NN_operand.f32 (the captured operand, before the BF16 round) and wo_a_layer_NN_device.bf16.

The same comparison re-runs without a device with

    cargo run -p sparkflash-kernels --example wo_a_native_compare -- --config CONFIG --layer N
      --operand OPERAND.f32 --weight WO_A.weight --scales WO_A.scale --rope none

(or --rope inverse with --position P for an engine capture taken before the inverse rotation).
The tool is CPU-only and prints the same quantities.

Measured offline on three captured operands at position 129 (task14 attention-case captures, layers
21/30/39), with the released layer weights and the released config:

| layer | exact fraction | max abs delta | output max abs | relative max | mean abs delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 295/8192 = 0.036 | 5.468750e-2 | 1.6796875 | 3.256e-2 | 1.0713e-2 |
| 30 | 315/8192 = 0.038 | 5.273437e-2 | 1.6406250 | 3.214e-2 | 6.7953e-3 |
| 39 | 360/8192 = 0.044 | 6.250000e-2 | 1.7812500 | 3.509e-2 | 1.0806e-2 |

The capture is the pre-inverse-rotation attention output; the tool applies the layer's inverse
rotation with the engine's own table and rotation routines, and the rotation touches only the last
64 of each head's 512 values (the last 2 of 16 blocks per head), so the magnitude above is
representative of the operand wo_a is called with.

## Cubin

b12x_bf16a_mxfp8w_wo_a_sm121 is pinned in crates/sparkflash-kernels/baseline/kernel_baseline.txt at
8,976 B, sha256 d11a8610ff557617116c437b0bc7a0367f66a8069ccfc97373433ebd99ffd249, 0 PTX membar lines.
It recompiles on the node with nvcc 13.0.88 (CUDA 13.0.3):

    /usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_121a -cubin -I kernels \
        kernels/b12x_bf16a_mxfp8w_wo_a_sm121.cu -o DIR/b12x_bf16a_mxfp8w_wo_a_sm121.cubin

and the output digest equals the pin, so the device chain's cubin verification accepts it. The
window's cubin directory must contain it before a native-mode run.

## What this does not claim

No GPU run happened with this change: the device-versus-reference numbers print on the next window
(N9, task-12), including where the kernel and the twin disagree. The offline numbers above are the
arithmetic deviation of the previous path from native on captured operands, not device measurements.

## Audit wording correction (task-29)

docs/build/native-kernel-arithmetic-audit.md Appendix A calls the batch-25 BF16-activation kernel
pair "not native arithmetic". That is correct for wq_a, wq_b, wkv and wo_b, whose native activations
are FP8, but wrong for wo_a: native wo_a's activation is BF16. Wiring the wo_a variant therefore
implements the native rule for exactly one projection, and the dense variant stays unwired. This
removes a quantisation rather than raising precision beyond native, so it does not reopen D39.

## Staging assumptions

The path keeps the chain's per-layer, per-position weight staging and its single-rank shape
(n_groups = o_groups = 8 here; the checkpoint stores the same [8192, 4096] per layer on every rank).
A serving path that keeps weights resident, or one that changes the per-rank weight layout, is
tp-dev's scope and would revisit the staging part of this change; the arithmetic and the mode
selection would not change.

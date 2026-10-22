# Kernels and native arithmetic

The device path runs the checkpoint's **native arithmetic**, not a software approximation of it. This
page lists the pinned kernel surface, the arithmetic contract, and what has been measured. The
single source of the pin is [`crates/sparkflash-kernels/baseline/kernel_baseline.txt`](../../crates/sparkflash-kernels/baseline/kernel_baseline.txt)
(27 kernels, nvcc V13.0.88, `sm_121a`).

## What "native" means

| Item | Rule |
| :-- | :-- |
| Activation quantiser | Smallest power of two at or above `amax/448`, after a 1e-4 amax floor; E4M3 codes (D23). |
| Weight grid | Floor-log2 per-32 `ue8m0` scale (different helper from the activation rule; D23). |
| Routed experts | W4A8: FP8 E4M3 activations, FP4 E2M1 weights on `[out, in/32]`, f32 accumulate (D14, D39). |
| Dense / attention projections | MXFP8 E4M3 activations with 32-element `ue8m0` scales; f32 accumulate. |
| `wo_a` | Native activation is **BF16**, not FP8 (the D39 exception; there is also a BF16a/MXFP8w kernel pair). |
| mHC | Released arithmetic: projection rstd, `mix*scale+base`, sigmoid pre/post, released split-Sinkhorn, raw C, `X_hat` block norm, `[24, n*d]` predictor (D22). |
| Rounding | BF16 round-to-nearest-even at the declared boundaries; f32 accumulate inside a dot product. |
| Reductions | Canonical rank order on the host path. Device tree/split reductions are a declared D54 tolerance arm. |

A kernel is "native" when its rule and dtype match the released forward. The tensor-core accumulation
order is not reproducible bit-for-bit by software; that is why acceptance is numerical (D54), not
byte-identical, and why a per-kernel exactness class is stated rather than an absolute claim.

## Pinned kernel inventory

| Kernel | Purpose | Used by |
| :-- | :-- | :-- |
| `mega_mhc_sm121` | mHC entry/post/merge (fused) | attention/FFN mHC stages |
| `layer_seams_sm121` | mHC entry/apply, FFN seams, FFN epilogue (multiple symbols) | device layer seam stages |
| `b12x_mxfp8_dense_sm121` | MXFP8 dense GEMM | wq_a, wq_b, wkv, wo_b, shared-expert w1/w3/w2 |
| `b12x_mxfp8_dense_m1_sm121` | columns-only M=1 dense projection | decode dense |
| `b12x_mxfp8_gemv_m1_sm121` | bandwidth-first M=1 GEMV | decode projections |
| `b12x_hostw_gemv_m1_sm121` | M=1 GEMV over checkpoint f32/bf16 host weights | `reference_seam` (compressor/indexer) |
| `b12x_mxfp8_wo_a_sm121` | grouped MXFP8 `wo_a` | output projection |
| `b12x_bf16a_mxfp8w_dense_sm121` | BF16-activation / MXFP8-weight dense GEMM | precision path (D37) |
| `b12x_bf16a_mxfp8w_wo_a_sm121` | BF16-activation `wo_a` | `attention_output` |
| `b12x_mxfp4_moe_sm121` | routed down + combine (FP4) | routed MoE |
| `b12x_mxfp4_moe_stored_sm121` | stored-layout routed combine (288-byte code / 18-byte scale rows) | routed MoE (task-6) |
| `routed_gate_up_sm121` | routed-expert gate/up W4A8 GEMM | routed MoE |
| `moe_router_sm121` | router scoring + top-k | `moe_router` |
| `swiglu_mxfp8_quant_sm121` | SwiGLU + activation quantise (unweighted) | shared / routed epilogue |
| `swiglu_mxfp8_quant_weighted_sm121` | weighted SwiGLU + quantise | routed combine (D33) |
| `flash_mla_csa2_sm121` | format-2 (CSA union) attention body | attention body |
| `flash_mla_csa2_seq_sm121` | position-causal CSA2 sequence entry | deferred sequence (D36) |
| `flash_mla_csa2_ring_sm121` | 134-slot ring-capacity CSA2 entry | DSpark/MTP ring (D49) |
| `cert_lm_head_sm121` | certified LM head / screen | serving tail |
| `lm_head_reduce_sm121` | four-rank head block reduce + argmax | serving tail (D35) |
| `rmsnorm_sm121` | standalone final RMSNorm | serving tail (D34) |
| `engram_embed_sm121` | Engram embed row + activation quantise | Engram layers 1, 14 |
| `engram_wkv_sm121` | Engram W8A8 wkv | Engram |
| `engram_gate_write_sm121` | Engram gate + write | Engram |
| `kv_path_sm121` | `kv_norm_rope_bf16`, `window_quant_fp8_bf16`, `kv_append_bf16` | `kv_path` |
| `ref_select_sm121` | top-k selection over compressed candidates | `ref_select` |
| `compressed_append_sm121` | compressed-store append | `reference_seam`/`stage_attention` |

The six Milestone-4 kernels, the M7 SwiGLU epilogue, the router, gate/up, RMSNorm, head reduce, CSA2
sequence/ring entries, BF16a projections, three Engram kernels and the stored-layout MoE variant are
all in the baseline. A new kernel is not "landed" until it has a cubin pin, a `KERNEL_CONTRACTS` row,
a launcher and a CPU twin test.

## ABI and pins

- The frozen attention entry is 19 words; Path A's 20th argument is a separate pin. The mHC entry is
  20 words (was 21); the seam kernels are 8 words; `comm_all_reduce_bf16` is `(stream, in, out,
  count, handle)`.
- The `m4_abi` family is pinned by `ci/check-m4-abi-size.sh` and the m4 split verifier; the family's
  aggregate line count may not drift (D26).
- The baseline check (`scripts/check-kernel-baseline.sh`) verifies cubin bytes, sha256,
  `ptx_membar_lines` and the toolkit version. `scripts/check-kernel-baseline.sh` must pass locally
  with `SPARKFLASH_NVCC` before a window stages a binary.
- A new pin is a deliberate re-baseline, not a side effect of a speed slice.

## Harness status

Each stage has an isolated harness (m10-m20) that compares the kernel against its CPU twin at fixed
token counts and geometry. The status that matters:

| Area | Status |
| :-- | :-- |
| Attention projections (m12) | PASS at both token counts, bit-exact against the twin. |
| Output projections (m13) | PASS against `mxfp8_ref::grouped_wo_a`. |
| CSA2 format-2 body (m14) | PASS on its stage bar. |
| mHC entry/post (m11, m15) | PASS at layers 0/1/4 on the engine twin. |
| LM head (m16) and head reduce (m19) | PASS 17/17; block arrays, tie-break, refusal, sentinel. |
| Routed MoE (m17a) | PASS 35 checks at tokens 8 and 130; weighted epilogue wired. |
| Shared expert (m17b) | PASS at both token counts. |
| Final RMSNorm (m18) | 0 mismatched BF16 bytes at 8 and 130 tokens; wired (b41). |
| CSA2 sequence entry (m20) | PASS 50/50 at positions 0, 1, 2, 129..137. |
| Full-chain token parity (criterion 1) | **OPEN** — the chain is stage-faithful; the native path does not yet match the f32 reference token-for-token (D37/D39). |
| Device-layer stages | Twin-pinned where testable, but **not numerically validated on a node** until a whole-layer row passes the D54 gate. |

"PASS" in a harness means the kernel matches its declared CPU twin on that harness's geometry. It is
not a token-parity claim.

## Exactness and tolerance

- **Strict mode** (pure layout/pointer/geometry change): byte identity via `identity-gate.sh`.
- **Product mode** (any slice that reorders a sum): D54 numerical gate; see [`../RULES.md`](../RULES.md).
- **Admissible:** split reductions, tree sums, device kernels whose parallel dimension divides a sum.
- **Not admissible:** a different quantiser/dtype/set of terms, non-finite values, a bar loosened
  without a measured noise floor, or an edited frozen reference.
- Declared deviations already accepted: device silu uses `expf`/true divide (f32-reference arm);
  device latent is bf16 for `kv_path` (2 % bound); NCCL's ring order for the per-layer reduces.

## Native-audit findings

The native-arithmetic audit found real deviations before they became gate failures: `wo_a` was MXFP8
where native is BF16 (F1), the attention body did not round softmax probabilities to bf16 (F2), the
indexer query/score dtype (F3), Engram host f32 (F4), missing bf16 round points at host seams (F5),
the shared SwiGLU (F6), and the routed combine's bf16 chain (F7). The settlements are recorded in
D30-D34 and D39. F8 (accumulation association) is the class D54 now admits; it is not a defect.

## Archived sources

`phase3-indexer.md`, `phase3-router.md`, `phase3-gate-up.md`, `phase3-mhc-entry.md`,
`phase3-attn-projections.md`, `phase3-csa2-sequence.md`, `phase3-flash-mla.md`,
`phase3-lm-head-reduce.md`, `phase3-lm-head.md`, `phase3-mhc-post.md`, `phase3-moe-routed.md`,
`phase3-moe-shared.md`, `phase3-output-projections.md`, `phase3-rmsnorm-tail.md`,
`phase3-attn-precision.md`, `native-arithmetic-audit.md`, `native-kernel-arithmetic-audit.md`,
`native-rounding-arms.md`, `indexer-native-dtype.md`, `wo-a-native-path.md`,
`moe-stored-layout-variant-plan.md`, `paged-read-attention-kernel-design.md`,
`task7-f2-kernel-pin.md`, `task14-union-attention-case.md`, `kernel-occupancy-inventory.md`,
`launch-count-reductions-note.md`, `logits-dump.md`, `gate-artifact.md`.

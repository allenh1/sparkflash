# Native-kernel arithmetic audit against the released inference code

**Status: docs-only audit, no code changes, no new device run (2026-10-19).** Branch
`batch/2026-10-19-25` (batch 25 plus the kernel-count doc corrections), commit is the one that adds
this file. The direction that produced it: the deliverable is the checkpoint's **native arithmetic**,
not a higher-precision replacement, and acceptance is numeric with tolerance (not token equality,
which generation nondeterminism makes unusable); the behavioural oracle is vLLM, "when in doubt do
what vLLM does".

The BF16-activation pair from batch 25 is **not** the native arithmetic and is not the fix: native
activations are FP8 E4M3 with per-32 ue8m0 power-of-two scales (below). That batch stays measured-
but-unwired on this branch; its GPU comparison (m21) is not needed for this audit.

## 0. Sources, and what "native" means here

"Native" is the released inference implementation in `/home/user/dsv41-research/inference/`:

| file | role |
| :--- | :--- |
| `model.py` | module graph, quantiser call sites, MoE/router/mHC/attention arithmetic |
| `kernel.py` | the tilelang kernels that define the actual block arithmetic |
| `convert.py` | checkpoint conversion; the one weight that is dequantised at conversion time (wo_a) |
| `generate.py` | torch default dtype, so every module boundary is BF16 (`generate.py:118`) |

The globals are set once in `model.py:26-30`: `default_dtype = torch.float8_e4m3fn` (Linear weight
storage), `fp8_block_size = 32`, `fp4_block_size = 32`, `scale_fmt = "ue8m0"`,
`scale_dtype = torch.float8_e8m0fnu`.

"Ours" is the device kernel in `kernels/` and its Rust twin in `crates/sparkflash-kernels/src/*_ref.rs`
(the twin is the definition of what the kernel computes per the repo convention). The engine's
activation quantiser is `crates/sparkflash-engine/src/quant.rs`.

**Evidence caveat that applies to every line below.** Every measured window in this repo compares a
device kernel against its own Rust twin, never against the released Python code. No vLLM run and no
negative log-likelihood/logit comparison against native has been made. So "measured" in this document
means "device equals our declared twin on released bytes", which pins reproducibility, not native
parity. No arithmetic below has been changed by this audit.

Verdict vocabulary: **match** = the native rule is the same rule and our implementation follows it;
**mismatch** = a different quantisation or rounding structure (a defect to report before touching
code); **unverified** = same declared rule, but the reduction/order/implementation is not measured
against native.

## 1. Cross-cutting native rules

| # | rule | native reference |
| :-- | :--- | :--- |
| N1 | Activation FP8 E4M3, one scale per 32 values: amax floored at 1e-4; scale = 2^ceil(log2(amax/448)) stored as E8M0; value clamped to +/-448 then E4M3 round-to-nearest-even | `kernel.py:41-95` (floor `:76`, ceil scale `:78`, clamp/cast `:89`, E8M0 store `:91`), wrapper `kernel.py:98-124`, call sites `model.py:187,197,707,1042,1062`, globals `model.py:27,29,30` |
| N2 | FP8 weight scales: checkpoint grid `[ceil(out/32), ceil(in/32)]` E8M0, one per 32x32 block; no re-quantisation at inference | `model.py:225-233` (shape), `kernel.py:232,263` (B index `(n//32, k)`) |
| N3 | FP4 expert weight scales: `[out, in/32]` E8M0, one per output row per 32 K values | `model.py:219-224`, `kernel.py:582` (`b_s.shape == (N, K // 32)`) |
| N4 | GEMM: tensor-core dot of 32 FP8 values accumulated in FP32, then `C_local_accum += C_local * scale_a * scale_b` in FP32; one block-32 at a time; output BF16 | `kernel.py:250-274` (scales), `kernel.py:208-260` (geometry), `kernel.py:299` (output dtype) |
| N5 | FP4 GEMM: activations are FP8 (A8, D14), weights FP4 cast to FP8 through FP32; same per-32 scale application and FP32 accumulation; output BF16 | `kernel.py:534-553` (K loop: cast `:538-540`, scales `:542-548`, accum `:552-553`), output `kernel.py:583` |
| N6 | FP8 window KV: post-RoPE `act_quant(kv, 32, ue8m0, ..., inplace=True)` -> quantise and dequantise back to BF16 storage; block 32, E8M0 ceil scale | `model.py:700-707` (and DSpark `model.py:1042,1062`) |
| N7 | Compressed KV latent: FP4 E2M1 with **block 16 and E4M3 scales** (not E8M0), in-place dequantised | `model.py:759-760`; kernel `kernel.py:128-181` with `scale_dtype=float8_e4m3fn`, scale `E4M3(amax/6)` |
| N8 | Indexer q/k: FP4 with **block 32 and E8M0 scales**, in-place dequantised | `model.py:546,552` |
| N9 | All accumulation is FP32; module boundaries are BF16; the residual stream is BF16 and every `hc_pre`/`hc_post`/expert output is rounded to BF16 before it is consumed | `generate.py:118`, `model.py:851,893-904,957-966` |

### 1.1 Ours against the cross-cutting rules

- N1: `crates/sparkflash-engine/src/quant.rs:400-437` (`e8m0_encode_ceil`), `:626-682`
  (`quantize_activation_fp8`: floor 1e-4, 2^ceil(log2(amax/448)), clamp +/-448, RNE E4M3).
  `crates/sparkflash-kernels/src/m4_quant.rs:264-400` is the device-side twin of the same rule.
  **Match.** One documented difference: our total function ceils a subnormal product from its
  highest set bit where the reference's `fast_log2_ceil` (`kernel.py:22-27`) would mis-ceil it;
  the 1e-4 floor makes that unreachable (quant.rs:413-416), so it is not a checkpoint-visible
  divergence.
- N2/N3: our packing is a pure byte permutation of the checkpoint codes and scales
  (`mxfp8_ref.rs:40-75`: `b12x_weight_index`/`b12x_scale_index` and `b12x_row_scale_index`), no
  decode and no arithmetic; the scale bytes are the checkpoint's own E8M0 grid. **Match.**
- N4: our GEMM kernels accumulate in FP32 in ascending K order and round once to BF16 at the end
  (dense kernel `kernels/b12x_mxfp8_dense_sm121.cu:95-110`). The dense kernel multiplies each
  product by both scales (both exact powers of two) instead of scaling the block partial; because
  power-of-two scaling commutes with FP32 rounding (no over/underflow here), that is the same f32
  number as N4's block-partial form, up to the summation order. **Match to the rule; exact order
  unverified** (N4 sums inside the 32-block in the tensor core's order, ours is a scalar ascending
  loop). This is a tolerance-class difference, not a quantisation difference.
- N5: `kernels/b12x_mxfp4_moe_sm121.cu:180-197` and `kernels/routed_gate_up_sm121.cu` apply both
  scales to each 32-value block partial in FP32, exactly N5's shape. **Match to the rule; order
  unverified.**
- N6/N7/N8: `crates/sparkflash-engine/src/forward/attention.rs:64-66,175-229` (`round_trip_rows`)
  implements exactly the three native in-place rules, including the block-16 E4M3 compressed-KV
  scale; the attention kernel's window format 1/2 reads E8M0 powers of two per 32 channels
  (`kernels/flash_mla_csa2_seq_sm121.cu:91-108`, `m4_common_sm121.cuh:83,103`) and the compressed
  FP4 latent per 16 with E4M3 scales. **Match** for the format-2 (CSA union) entry, which is the
  released attention path (see 2.9 for the retired format-0 window).
- N9: see the mismatches in section 3: not every BF16 boundary is modelled.

## 2. Per-kernel audit

For each kernel: the native rule, what our kernel/twin does, verdict, and the measured evidence
(always device-versus-twin). Line references are to this checkout.

### 2.1 `b12x_mxfp8_dense_sm121` (wq_a, wq_b, wkv, wo_b, shared expert w1/w3/w2)

Native: each of these is `Linear` with FP8 weights and goes through `linear()` ->
`act_quant(x, 32, ue8m0)` then `fp8_gemm(block_size=32)` (`model.py:181-205`);
`wq_a/wq_b` at `model.py:770-771`, `wkv` at `model.py:705` (DSpark at :1040), shared-expert/ffn GEMMs at `model.py:836-838,843-844,851`
via `Expert`, `wo_b` at `model.py:788`. Weight grid is 32x32 E8M0 (N2), GEMM is N4, output BF16.

Ours: `kernels/b12x_mxfp8_dense_sm121.cu:58-122`; twin `mxfp8_ref.rs:137`
(`dense_gemm`). The kernel takes pre-quantised E4M3 activations plus E8M0 block scales, B12x
32x32-packed FP8 weights and scales, accumulates FP32 in ascending K, one BF16 round.

Verdict: **match to the native rule; exact accumulation order unverified** (tensor-core MMA order vs
our scalar order). The operands are the same rule (N1/N2), the output rounding is the same.

Measured evidence: m12 window b8 PASS, both row counts, every output bit vs the twin (wq_a
10,240/10,240, wq_b 262,144/262,144, wkv 4,096/4,096 at 8 rows; 166,400 / 4,259,840 / 66,560 at
130 rows), two launches byte-identical (`docs/build/phase3-attn-projections.md`); m17b for the
shared-expert widths (m17b window b28, w1/w3 18,432 and 299,520 bf16 values exact); and the M7
stage-7 layer-0 replay, where K2/K3/K4/K6/K7 are each bit-exact against their in-crate references
(K2 1,280/1,280, K3 32,768/32,768, K4 512/512, K6 8,192/8,192, K7 5,120/5,120).

### 2.2 `b12x_mxfp8_wo_a_sm121` (K6 wo_a) - **MISMATCH**

Native: `wo_a` is the one weight `convert.py` **dequantises to BF16** at conversion time
(`convert.py:154-173`: `weight.float() * scale.float()`, then `.bfloat16()`), and the model
computes it as a BF16 grouped einsum over those BF16 weights with the BF16 attention output
(`model.py:783-788`). There is no activation quantisation and no FP8 weight operand on this
projection: the comment at `model.py:784` says so explicitly ("convert.py dequantizes it to bf16; an
fp8 grouped GEMM would halve the memory").

Ours: `kernels/b12x_mxfp8_wo_a_sm121.cu:53-108` takes **FP8 E4M3 activations with per-32 E8M0
scales** and **FP8 E4M3 weights with the checkpoint's 32x32 E8M0 scales**, accumulates FP32 and
rounds once to BF16; twin `mxfp8_ref.rs:194` (`grouped_wo_a`). The kernel's own file header
(`kernels/b12x_mxfp8_dense_sm121.cu:4-6`) assumes every dense projection uses the MXFP8 form; that is
not true for wo_a in the released model.

Impact: relative to native, this path adds an FP8 E4M3 block-32 quantisation of the attention output
(4-bit mantissa) that native never performs, and uses the unrounded FP8 weight value (code x E8M0
scale in FP32) instead of the BF16-rounded dequantised weight. Both differences land before wo_b, so
they propagate. This is the largest single arithmetic deviation found in the audit.

Measured evidence: m13 window b10 PASS at both token counts against `mxfp8_ref::grouped_wo_a` - i.e.
it pins the device against our own MXFP8 twin, not against native (`docs/build/phase3-output-projections.md`).

### 2.3 `b12x_mxfp4_moe_sm121` (routed down + combine) - **MISMATCH**

Native: each routed expert's `w2` is an FP4 GEMM whose output dtype is the default BF16
(`kernel.py:583` `c = ... dtype=torch.get_default_dtype()`, `generate.py:118`), so every expert's
contribution is **rounded to BF16 before it is added**. `MoE.forward` keeps `y` in FP32 and adds
each expert's BF16 result in ascending expert-id order (`model.py:893-900`), adds the shared expert's
BF16 result (`:903`), and rounds once to BF16 on return (`:904`). The residual is *not* part of that
sum: it is added later by `hc_post` in FP32 and rounded then (`model.py:962-966`).

Ours: `kernels/b12x_mxfp4_moe_sm121.cu` computes the down projection per routed slot and then, in
one FP32 chain, accumulates the six slots' block-scaled partials **plus the residual**, with a single
BF16 rounding at the end (`:180-197`, comment at `:195-197`); twin `mxfp4_ref.rs:224`
(`moe_combine`). There is no per-expert BF16 rounding, the slot order is top-k order rather than
ascending expert id, and the residual is inside the same rounding.

Impact: three structural differences (per-expert bf16 rounding, accumulation order, residual
placement). They are small individually but they are the native arithmetic, and they change the
last bits of the MoE output before `hc_post`.

Measured evidence: M7 stage 3 (level 0, so the layer-0 replay's combine class): combine
20,480/20,480 BF16 exact, max ULP 0 against the twin. The measurement cannot catch M2 because the
twin encodes the same single-rounding chain.

### 2.4 `swiglu_mxfp8_quant_sm121` (unweighted epilogue, shared expert) - **MISMATCH**

Native: `Expert.forward` computes `gate`/`up` in FP32, applies the asymmetric clamps, forms
`F.silu(gate) * up` in FP32, and then `x.to(dtype)` **rounds the activation to BF16 before w2's
quantiser reads it** (`model.py:843-851`); the next `linear()` call then applies N1 to those BF16
values (act_quant reads a BF16 tensor).

Ours: `kernels/swiglu_mxfp8_quant_sm121.cu:131-140` keeps the activation in FP32 and feeds it
straight into `spark_activation_quantize_block` (no intermediate BF16 round); the twin
`swiglu_ref.rs:124-140` (`quantise_row`) does the same. The weighted variant *does* round first
(2.5), which is why this is a per-entry-point defect rather than a file-wide one.

Impact: the quantiser sees a different amax (and different values) than native on every shared-expert
row; one extra BF16 rounding is missing before the FP8 quantisation.

Measured evidence: m17b window b28 PASS at both token counts (stage 2 E4M3 codes 18,432 and 299,520
exact, scales 576 and 9,360 exact) - against `swiglu_ref::quantise_row`, which encodes the same
missing round, so the measurement cannot catch it.

### 2.5 `swiglu_mxfp8_quant_weighted_sm121` (routed epilogue)

Native: `x = weights * (F.silu(clamped_gate) * clamped_up)` in FP32, then `x.to(dtype)` BF16, then
w2's `act_quant` (`model.py:843-851`, and the quantiser ordering note at `mxfp4_ref.rs:182-205`).

Ours: `kernels/swiglu_mxfp8_quant_weighted_sm121.cu`; twin `swiglu_ref.rs:168-181`
(`quantise_row_weighted` = `weighted_activation` bf16 round then `act_quant_fp8_row`). The routing
weight is inside the activation before the quantiser, exactly as native.

Verdict: **match to the rule**; the silu/exp implementation and warp amax reduction order are
unverified against native (tolerance class).

Measured evidence: m17a window b33 PASS 35 checks at tokens 8 and 130 (D33), device-vs-twin.

### 2.6 `moe_router_sm121`

Native: `scores = linear(x.float(), gate.weight.float()) / gate_temp` (FP32 F.linear, weight BF16
widened), `softplus.sqrt()` for sqrtsoftplus, bias added only for selection, weights from the raw
scores, normalise over the selected top-k with `+1e-20`, multiply `route_scale` (1.5), top-k via
`torch.topk` (`model.py:809-827`); constants `model.py:66-69` and `config.json` (sqrtsoftplus,
gate_temp 1.0, norm_topk_prob true, route_scale 1.5).

Ours: `kernels/moe_router_sm121.cu:94-104,106-239`: FP32 score per expert accumulated in ascending
K with explicit no-FMA (one rounding per multiply and add, comments `:160-161`), softplus with the
reference threshold 20, sqrt, bias only in the selection key, top-6 scan, normalise with 1e-20,
route scale 1.5; twin `router_ref.rs`.

Verdict: **match to the scoring rule; unverified against native** in two places: (a) the native score
matmul is cuBLAS SGEMM, whose reduction order is not our ascending K order; (b) ties in
`torch.topk` are resolved by the framework's selection order, ours by lower expert id. Either can
change the selected set near a tie, which is a bigger effect than a last-bit logit difference.

Measured evidence: m9 twice, 7/7 checks each, device-vs-twin (`docs/build/serving-readiness.md:66`).
The router is still the chain's host seam (serving-readiness item 7: chain wiring only), so no
chain run exercises the device router.

### 2.7 `rmsnorm_sm121` (final norm)

Native: `RMSNorm.forward`: `x.float()`, `var = x.square().mean(-1, keepdim=True)`,
`rsqrt(var + eps)`, `(weight * x).to(dtype)` (`model.py:281-293`); released `norm_eps = 1e-20`
(`config.json`). Native's mean is torch's CUDA reduction order and `torch.rsqrt` is not our
`1/sqrt`: neither is specified bit-for-bit by the Python.

Ours: `kernels/rmsnorm_sm121.cu:12-48`: row-serial ascending sum of squares in FP32, `/cols`,
`1/sqrt(mean+eps)` with correctly-rounded intrinsics, `(x*inv)*w`, one BF16 RN-even rounding; twin
is the engine's own `linalg::rms_norm` (`rmsnorm_ref.rs:24-64`).

Verdict: **match to the engine reference; unverified against native** on the reduction order,
torch's mean implementation and rsqrt approximation. The engine reference is not native.

Measured evidence: m18 window b40, bit-exact 0 mismatched BF16 bytes at 8 and 130 tokens, odd width
513 exact, two launches identical (D34); wired as the c4c tail in b41, `bf16_mismatches=0` of
10,240 bytes vs the engine path.

### 2.8 `lm_head_reduce_sm121` (block reduction + greedy argmax)

Native: `ParallelHead.forward` casts the hidden to FP32 and does `F.linear(x.float(), weight)` with
an FP32 weight (the checkpoint's BF16 weights widened exactly), producing FP32 logits; greedy
decoding is `torch.argmax` over logits (lowest index on ties), sampling divides by an exponential
without touching the values (`model.py:1008-1017,1288-1292`). The cert kernel's reduction is not in
native; native computes all logits and takes the argmax directly.

Ours: `kernels/lm_head_reduce_sm121.cu` + twin `lm_head_reduce_ref.rs:1-60`: no arithmetic is
performed on a stored logit, block maxima fold in ascending global block order, ties go to the lowest
global entry - the same rule as `torch.argmax`'s first occurrence; status bits report a certified
winner instead of asserting. The logits themselves come from the certified screen kernel
(`cert_lm_head_sm121`), whose FP32 accumulation over four rank shards is a different order from
native's single cuBLAS FP32 GEMM.

Verdict: **reduction semantics match** (ties/argmax); the upstream logits are **unverified** against
native FP32 cuBLAS.

Measured evidence: m19 window b44, PASS 17/17 (real-token control, concentrated certificate,
synthetic tie/offsets/bound/refusal; two-launch determinism). The device reduce is bit-for-bit the
host assembly (token 5 / 13.237006187); the host assembly itself differs from the phase-1 f32
reference logits by ~6.3e-3 (13.230665207), a reminder that the logits are not native cuBLAS.

### 2.9 `flash_mla_csa2_seq_sm121` (CSA union attention)

Native: `sparse_attn` over the concatenated KV: the sliding-window rows are the post-RoPE KV after
N6 (FP8 E4M3 block 32 E8M0, quantise+dequantise to BF16), the compressed rows are the FP4 latent
after N7 (block 16, E4M3 scales), q is BF16 with scale `1/sqrt(d) = 1/sqrt(512)`, and the softmax is
online FP32 with running max initialised to -1e30, sink added as `exp(attn_sink - max)` in the
denominator, output BF16 (`model.py:765-789`, `kernel.py:311-389`). The indexer that chooses the
compressed tops is `model.py:527-580` (q/k FP4 block 32 E8M0, N8).

Ours: `kernels/flash_mla_csa2_seq_sm121.cu` (format 2 = CSA union): window rows read the engine's
FP8 E4M3 + per-32 E8M0 format (`:91-108`) and dequantise to BF16; compressed candidates read the
FP4 latent with per-16 E4M3 scales; score scale is the f32 literal `0.04419417382415922f`
(`m4_common_sm121.cuh:676`, 512^-0.5); sink enters the denominator after the maximum; output BF16.
Twin `mla_ref.rs`.

Verdict: **match to the native KV formats and softmax formula; unverified** on (a) the reduction
order - ours resolves max/sum over the selected set in a fixed three-pass order, native is tilelang's
block-online order with `exp` per element; (b) the compressed/HSI candidate selection, which is host
code fed into the kernel in both implementations but has not been compared to native's
indexer+top-512 merge; (c) the format-0 (FP4 window latent) entry is **not** native - it stores the
window in FP4 block 16 instead of the native FP8 window - and must not be used as the released path
(format 2, the deferred sequence entry, is the one m20 measured).

Measured evidence: m20 window b50 PASS 50/50 at positions 0,1,2,129..137 including position-0
byte identity (D36); the format-2 body's stage bars in the b55-b62 chain - alignment, pre-attention
hidden and the body pass at their own bars (D37) - all device-vs-twin.

### 2.10 `mega_mhc_sm121` - **MISMATCH (plus unverified statistics)**

Native: `hc_mixes`: flatten to `n*d` FP32, `rsqrt(mean(x^2)+norm_eps)`, `F.linear(x, hc_fn)` in
FP32, then `hc_split_sinkhorn` FP32 (sigmoid pre + eps, 2*sigmoid post, softmax + eps then 19
alternating row/column normalisations, `kernel.py:426-458`). `hc_pre`: `sum(pre_mix * x.float(),
dim=2)` then `.to(x.dtype)` = **BF16 rounding** (`model.py:957-960`); `attn_norm` then consumes
that BF16 tensor and returns BF16. `hc_post`: `post*x + sum(comb*residual)` FP32 then BF16
(`model.py:962-966`). So native has a BF16 rounding at every hc_pre/hc_post boundary and at every
sublayer output.

Ours: `kernels/mega_mhc_sm121.cu:20-54,184-188,296-341`: computes `X_hat = A X_l` from registers in
FP32, keeps it FP32 for the block RMSNorm and for the activation quantisation (`:296-341` prints and
quantises the FP32 normed value), writes `X_l` back as BF16 (`:181`) and returns the quantised
block input. The step-1 update is FMA-contracted by construction
(`crates/sparkflash-engine/src/forward/layer_zero.rs:607-615`: "the C Y term is one plain multiply
and each B term is a fused multiply-add ... a mul-then-add twin is not the device's program").

Mismatch: (a) native's residual stream is the BF16 write-back, so `hc_pre` reads BF16 values and
rounds its output to BF16, and the norm output is BF16 before the quantiser; ours computes `X_hat`
from the unrounded FP32 update and keeps FP32 through the norm and quantiser - the same missing-
BF16-round family as 2.4, one level larger;
(b) native `hc_post` is separate rounded multiplications/additions, ours is FMA-contracted, which
differs in the last bit. Unverified: native's mean/rsqrt are torch CUDA reduction/rsqrt, ours is the
declared statistic chain (binary64 combine of tile partials, `mega_mhc_sm121.cu` around the stats
block) - reproducible, not native.

Measured evidence: M7 stage 1 PASS at layers 0/1/4 against the engine twin (4/4 tokens each; layer 0
token 0 5,119/5,120 codes, one one-code difference; layer 1 exact; layer 4 token 0 5,118/5,120), and
the stage-7 layer-0 replay's K1 is exact on the bytes it reads. The residual-stream X_l is
20,480/20,480 BF16 exact in the stage-1 runs. See `docs/build/phase1-boundary-table.md:627`
and `docs/build/phase3-indexer.md:71`.

## 3. Mismatch list (defects; reported before any code change)

| id | kernel/ref | native | ours | why it matters |
| :-- | :--- | :--- | :--- | :--- |
| M1 | `b12x_mxfp8_wo_a` / `mxfp8_ref::grouped_wo_a` | BF16 grouped einsum over BF16-dequantised wo_a weights, BF16 attention operand, no activation quant (model.py:783-788, convert.py:154-173) | FP8 weights + E8M0 32x32 scales, FP8 E4M3 block-32 quantised activation, FP32 accumulate (kernel :53-108) | adds an activation quantisation native does not have, and uses the unrounded weight; affects every attention output before wo_b |
| M2 | `b12x_mxfp4_moe` / `mxfp4_ref::moe_combine` | per-expert w2 output rounded to BF16, added in FP32 in ascending expert-id order; shared added BF16; final BF16; residual added later by hc_post (model.py:893-904,962-966) | six slots' block-scaled partials + residual accumulated in one FP32 chain, one BF16 round, slot order (kernel :180-197) | missing per-expert BF16 rounding, wrong accumulation order, residual inside the same rounding |
| M3 | `swiglu_mxfp8_quant_sm121` (unweighted) / `swiglu_ref::quantise_row` | activation rounded to BF16 by `x.to(dtype)` before w2's act_quant (model.py:848-851) | FP32 activation quantised directly (kernel :131-140, ref :124-140) | different amax/values into the FP8 quantiser on every shared-expert row |
| M4 | `mega_mhc_sm121` / `mega_mhc_ref` | the residual stream is BF16, hc_pre reads its BF16 values, sums in FP32 and rounds the result to BF16; the norm consumes that BF16 tensor and returns BF16; hc_post is separate rounded mul/add; torch mean/rsqrt (model.py:948-966) | X_hat is computed from the unrounded FP32 update and kept FP32 through the norm and its quantiser; FMA-contracted update; declared binary64 statistics (kernel :181-188,296-341; layer_zero.rs:607-615) | missing the BF16 residual/stream roundings and the BF16 norm output before quantisation, plus last-bit FMA differences vs native |

The BF16-activation pair on this branch (`b12x_bf16a_mxfp8w_*`) is deliberately **not** on this
list-of-defects because it is not trying to be native; it is a precision experiment (Appendix A).
`swiglu_mxfp8_quant_weighted` is not on the list: it already has the BF16 round native requires.

## 4. Unverified (rule matches, not measured against native)

| id | area | why unverified | evidence we do have |
| :-- | :--- | :--- | :--- |
| U1 | FP8/FP4 GEMM reduction order (2.1, 2.3 down-projection, 2.5 gate/up, 2.10) | native uses tensor-core MMA inside each 32-block; ours scalar ascending. Same rule, different summation order, so last bits can differ | device-vs-twin exact: m10, m12, m13, m17a/b, M7 |
| U2 | routed gate/up partition (2.5) | native computes the full 2304-wide w1/w3; we slice the intermediate width at 576 (+64 padding) per rank (D2 corrected by D30). Per-output arithmetic is identical; the partition is a layout choice | m10 window b6 PASS, 8/8 verdicts |
| U3 | router (2.6) | native score matmul is cuBLAS FP32 and native top-k ties are torch's; ours is ascending scalar FP32 and lower-id tie-break | m9 7/7 twice, host path bit-identical to twin |
| U4 | final RMSNorm (2.7) | native mean/rsqrt are torch CUDA implementations; ours is the engine's sequential rule | m18 b40/b41 bit-exact vs engine |
| U5 | LM head logits (2.8) | our certified screen's FP32 accumulation over four rank shards vs native one cuBLAS FP32 GEMM; the reduction/argmax itself is the same tie rule | m19 b44 17/17; the device reduce is bit-for-bit the host assembly (token 5 / 13.237006187), and the host assembly differs from the phase-1 f32 reference logits by ~6.3e-3 (13.230665207) |
| U6 | CSA union attention softmax order and candidate selection (2.9) | three-pass fixed-order max/sum vs tilelang online softmax with `exp`; candidate/indexer set construction has not been compared to native | m20 b50 50/50; position-0 identity |
| U7 | every "measured" number in this repo | every window compares device vs our twin; no vLLM/native numeric comparison exists yet. Only a tolerance-class, vLLM-referenced comparison can supply that | the windows cited per kernel above |

## 5. The Lead's explicit checks, one line each

| check | native | ours | verdict |
| :--- | :--- | :--- | :--- |
| activation block 32, E8M0 ceil | `kernel.py:41-95,76-91`, `model.py:27,29,30` | `quant.rs:400-437,626-682`; `m4_quant.rs:264-400` | **match** (subnormal fix documented, unreachable) |
| weight 32x32 E8M0 | `model.py:225-233`, `kernel.py:232,263` | `mxfp8_ref.rs:40-63`; B12x packing is a permutation | **match** |
| FP4 expert weights, 32-element K scales | `model.py:219-224`, `kernel.py:582` | `mxfp4_ref.rs` row-scale index (`mxfp8_ref.rs:69-82`); kernel ABI rows `:68-83` | **match** |
| FP8 KV block 32 | `model.py:700-707` (and 1042,1062) | engine `attention.rs:195-229`; kernel `:91-108` | **match** (format-2 path; format-0 is retired) |
| FP4 q/k block 16 | actual release: **compressed KV** FP4 block 16 with E4M3 scales (`model.py:759-760`); indexer q/k are FP4 block 32 E8M0 (`model.py:546,552`) | compressed KV block-16 E4M3 in engine `attention.rs:64-66,207-229` and kernel scale channels 16 (`m4_common:83`); indexer operands dequantised BF16 at block 32 by the caller | **match**; note the common shorthand "fp4 q/k block 16" is not the release's split |
| f32 accumulation | `kernel.py:208-274,545-591` (FP32 accum and scaled adds) | all kernels accumulate FP32, one BF16 at the boundary | **match to the rule**; order unverified (U1) |

## 6. Appendix A - the batch-25 BF16-activation pair is not native

`kernels/b12x_bf16a_mxfp8w_dense_sm121.cu` and `..._wo_a_sm121.cu` (`bf16a_mxfp8w_ref.rs`,
launch modules, KernelIds `Bf16aMxfp8wDense`/`Bf16aMxfp8wWoA`) were added as a precision-first
experiment: BF16 activations, FP8 weights still, FP32 accumulate. Native activations are FP8 E4M3
block 32 E8M0 (N1), so this is explicitly *not* the checkpoint's arithmetic. It stays unwired on
this branch and is not an acceptance candidate for native parity. (Lead direction: not merged, no
GPU run needed.)

## 7. Appendix B - evidence summary

| kernel | best measured window | bar | compared against |
| :--- | :--- | :--- | :--- |
| b12x_mxfp8_dense | m12 b8 / M7 stage 7 | every output bit | our twin |
| b12x_mxfp8_wo_a | m13 b10 | every output bit | our MXFP8 twin |
| b12x_mxfp4_moe | M7 stage 3 | BF16 exact, max ULP 0 | our twin |
| swiglu unweighted | m17b b28 | codes/scales exact | `swiglu_ref::quantise_row` |
| swiglu weighted | m17a b33 | 35 checks exact | `swiglu_ref::quantise_row_weighted` |
| moe_router | m9 (twice) | 7/7 checks | `router_ref` |
| rmsnorm | m18 b40, c4c b41 | BF16 bytes exact | engine `linalg::rms_norm` |
| lm_head_reduce | m19 b44 | 17/17 checks | host assembly |
| flash_mla_csa2_seq | m20 b50 | 50/50 checks, position-0 identity | `mla_ref` |
| mega_mhc | M7 stage 1 | 4/4 tokens; a few one-code diffs | `mega_mhc_ref` |
| routed_gate_up | m10 b6 | 8/8 verdicts | `routed_gate_up_ref` |

## 8. What this audit does not do

- It does not change code, pins, manifests or the chain. All four mismatches (section 3) are
  reported before any fix, per the Lead's instruction.
- It does not run a GPU or vLLM. The only way to close U7 is a numeric comparison (hidden state,
  logits, per-stage max-abs / relative error with a declared tolerance) against the released
  implementation or vLLM; token equality is not an acceptance test (sampling is stochastic:
  `model.py:1288-1292`).
- It does not re-audit the engine's host seams (q/kv norms, indexer top-k merge, rope) except where
  they change the operands of the kernels above; those remain named seams.

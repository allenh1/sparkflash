# 03 — MoE: gate, experts, quantisation, and the hash-layer contrast

References: /home/user/dsv41-research/inference/model.py (Gate 792-827, Expert 830-851, MoE 854-904),
convert.py, kernel.py, config.json, tech_report.txt.

---

## 1. Shape and count inventory

| Quantity | Value | Source |
| :--- | :--- | :--- |
| routed experts per backbone layer | 384 | config.json n_routed_experts |
| activated per token | 6 | config.json num_experts_per_tok |
| shared experts | 1 (asserted == 1) | model.py:886, config.json n_shared_experts |
| expert intermediate width | 2304 | config.json moe_intermediate_size |
| MoE layers | all 40 backbone blocks (+ 3 MTP) | model.py:929; tech_report.txt p.22 line 1002 |
| MTP routed experts | 128, top-3 | config.json dspark_n_routed_experts / dspark_num_experts_per_tok |
| gate weight | [384, 5120] fp8 + scale | layers.N.ffn.gate.weight, 40 tensors |
| gate.bias | [384] fp32 | from e_score_correction_bias (convert.py:115) |
| gate.bias_vl | [384] fp32 | image-span routing bias (model.py:807) |
| routed expert tensors | 40 x 384 x 3 weights + 40 x 384 x 3 scales = 92,160 | index.json |

---

## 2. Gate: exact scoring and routing

model.py:809-827, verbatim semantics:

```
scores = linear(x.float(), weight.float()) / gate_temp          # gate_temp = 1.0 (default; not in config)
if score_func == "softmax":     scores = scores.softmax(-1)
elif score_func == "sigmoid":   scores = scores.sigmoid()
else:                           scores = F.softplus(scores).sqrt()      # sqrtsoftplus, this model
bias = gate.bias
if image_mask is not None and gate.bias_vl is not None:
    bias = where(image_mask.unsqueeze(-1), gate.bias_vl, bias)  # per-token modality switch
indices = (scores + bias).topk(6, dim=-1)[1]                    # SELECTION uses biased scores
weights = scores.gather(1, indices)                             # WEIGHTING uses raw scores
if norm_topk_prob and topk > 1:
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
weights = weights * route_scale                                 # 1.5
```

Points that matter:

- `score_func` = "sqrtsoftplus": sqrt(softplus(z)) elementwise. There is no division by sqrt(dim)
  and no learned temperature in this checkpoint (gate_temp defaults to 1.0 and is absent from
  config.json).
- The whole gate runs in fp32: x and the weight are cast to float32 before the linear (model.py:811).
- noaux_tc is realised exactly as the report describes (tech_report.txt p.8 lines 325-334): the
  correction bias steers selection, the raw scores weight the outputs, and image tokens use
  gate.bias_vl instead of gate.bias. Two independent bias vectors are updated during training with
  bias update speed 0.001 for each modality (tech_report.txt p.22 line 1028).
- norm_topk_prob divides by sum + 1e-20, with a comment that this is deliberately not norm_eps
  (model.py:825). Then the single global routed_scaling_factor 1.5 is applied once, after
  normalisation, not per expert.
- The top-k is per **token**, not per token group: MoE.forward flattens to [n, dim] and calls the gate
  on all n tokens at once (model.py:890-892). There is no DeepSeek-V3-style node-limited routing in
  this checkpoint.

---

## 3. Experts and the MoE dispatch

```
Expert.forward(x, weights=None):
    gate = w1(x).float(); up = w3(x).float()
    if swiglu_limit > 0:
        up   = clamp(up,   -swiglu_limit, +swiglu_limit)     # both sides
        gate = clamp(gate,                +swiglu_limit)     # upper side only
    x = F.silu(gate) * up
    if weights is not None: x = weights * x
    return w2(x.to(dtype))                                                    model.py:841-851

MoE.forward(x, image_mask):
    weights, indices = gate(x, image_mask)
    y = zeros_like(x, float32)
    counts = bincount(indices.flatten(), minlength=384)
    for i in range(experts_start_idx, experts_end_idx):       # only this rank's experts
        if counts[i] == 0: continue
        idx, top = where(indices == i)
        y[idx] += expert(x[idx], weights[idx, top, None])
    if world_size > 1: all_reduce(y)
    y += shared_experts(x)                                    # unweighted, unconditional
    return y.type_as(x)                                                       model.py:889-904
```

Notes for an engine:

- w1 is the **gate** branch (silu) and w3 is the **up** branch (model.py:843-844). Clamping is
  asymmetric: up is clamped on both sides, gate only from above (model.py:845-847), matching "SwiGLU
  activation function with clamping at a threshold of 10" (tech_report.txt p.22 lines 1003-1004).
- swiglu_limit = 10.0 in config.json. MoE uses it for both routed and shared experts
  (model.py:879, 887).
- The shared expert output is added with weight 1.0 and is not routed, gated or normalised
  (model.py:903).
- Accumulation is fp32 (y is created float32) and only the return value is cast back to the input
  dtype.
- The per-expert loop is the reference's model-parallel dispatch: each rank owns a contiguous block
  of whole experts and the partial sums are all-reduced.

### 3.1 Expert parallelism, not tensor parallelism

convert.py shards **whole experts** across ranks: rank i keeps experts
[i*384/mp, (i+1)*384/mp) and drops the rest (convert.py:129-134); model.py mirrors this with
n_local_experts = n_routed_experts // world_size and experts_start_idx = rank * n_local_experts
(model.py:867-870). At world_size 4 that is **96 whole experts per rank, each 2304 wide**, not 384
experts of width 576. The expert width is never split.

The dense matrices use column/row parallelism per convert.py's mapping table (convert.py:56-64):
wq_b and wo_a are sharded on dim 0 (output), wo_b on dim 1 (input), embed/head/attn_sink/weights_proj
on dim 0; wkv, gate.weight, all norms, hc_*, engram.wkv and the compressors are **replicated**.

---

## 4. Quantisation scheme

| Tensor class | Weight (checkpoint) | Weight scale grid | Activation (compute) | Activation scale grid | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dense Linears (attn, gate, engram.wkv, indexer, ...) | fp8 e4m3, 1 byte/param | [ceil(out/32), ceil(in/32)] ue8m0, one scale per 32x32 block | fp8 e4m3, dynamic (**W8A8**) | [tokens, in/32] ue8m0 | model.py:225-233, 27, 197; kernel.py:293-298 |
| Shared expert (ffn.shared_experts) | fp8 e4m3 (constructed without a dtype, so it takes ModelArgs.dtype) | [ceil(out/32), ceil(in/32)] ue8m0 | fp8 e4m3, dynamic (**W8A8**) | [tokens, in/32] ue8m0 | model.py:26, 887, 196-205 |
| Routed experts (w1/w2/w3) | fp4 e2m1 packed 2-per-byte, stored [out, in/2] int8 | **[out, in/32]** ue8m0, one scale per output row per 32 along K: no blocking along out | **fp8 e4m3, dynamic (W4A8)** | [tokens, in/32] ue8m0 | model.py:219-224, 186-195; kernel.py:486-499, 511-515, 581-582 |
| MTP experts (mtp.N.ffn.experts) | same as routed experts | same | same (**W4A8**) | same | model.py:872-885 |
| wo_a | converted to bf16 at load time | dequantised, no scale retained | bf16 | -- | convert.py:156-173 |
| Embedding / head / norms / hc_* | bf16 (head kept fp32 at runtime) | -- | bf16 | -- | model.py:1005-1006 |

**No activation tensor exists in the checkpoint**: the safetensors constrain the weights only. The
activation *scheme* is nevertheless declared in it, and it is FP8: `quantization_config` states
`quant_method: "fp8"` with `activation_scheme: "dynamic"`, `weight_block_size: [32, 32]`,
`scale_fmt: "ue8m0"` and `expert_dtype: "fp4"` (config.json:12-21). `expert_dtype: "fp4"` names the
routed **weight** format; the declared activation scheme is dynamic FP8, and the reference forward
implements exactly that (below). The dense and MoE kernels are separate in this tree because the
weight formats differ (8-bit block-scaled vs 4-bit row-scaled), **not** because the activations
differ: both take the same FP8 activation.

Kernel-level rules:

- Both GEMMs share **one** activation quantiser. model.py:187 (fp4 weights) and model.py:197 (fp8
  weights) make the identical call `act_quant(x, fp8_block_size=32, scale_fmt="ue8m0",
  scale_dtype=e8m0fnu)`, which returns E4M3 values with per-32 ue8m0 scales; amax is floored at 1e-4
  and, because scale_fmt is set, the scale is rounded to a power of two (kernel.py:74-80).
- **The routed-expert GEMM is W4A8, not W4A4**: fp4 e2m1 weights against fp8 e4m3 activations. The
  kernel that implements it is documented as "FP8 act x FP4 weight GEMM" and typed
  `A: T.Tensor[(M, K), FP8]`, `B: T.Tensor[(N, K), FP4]` (kernel.py:486-499, 511-515). The dispatch
  site states the rule: "Quantized weights need a quantized activation, and both fp4 and fp8 weights
  take an fp8 one -- for fp4 the kernel handles the mixed precision" (model.py:181-183). The FP4
  *activation* quantiser exists but is never a GEMM input: its only call sites are the indexer K, the
  indexer Q and the compressed-KV cache (model.py:546, 552, 760), i.e. KV storage only. The MMA itself
  is fp8 x fp8: the kernel casts each FP4 weight sub-block to FP8 through float before the dot product
  (kernel.py:496-498), so an E2M1 activation would change the arithmetic, not just the operand type.
- **The two weight scale grids differ:**
  - fp8 weight (dense + shared expert): `[ceil(out/32), ceil(in/32)]`, a true 2-D block grid.
    `fp8_gemm` asserts `b_s.shape == ((N + block_size - 1) // block_size, K // block_size)`
    (kernel.py:293-298) and the module allocates the same `((out+31)//32, (in+31)//32)` shape
    (model.py:225-233).
  - fp4 weight (routed and MTP experts): `[out, in/32]`, one ue8m0 scale per output row per 32
    values along K, with **no** blocking along the output dimension. `fp4_gemm` asserts
    `b_s.shape == (N, K // 32)` (kernel.py:581-582) and the module allocates exactly
    `(out_features, in_features // fp4_block_size)` (model.py:219-224).
  - activations, both paths: `[tokens, in/32]` (`a_s.numel() == M * (K // 32)`, kernel.py:294 and
    kernel.py:581).
- The spec's only statement about the MoE activation format is the trace-scope table's "576-wide
  grouped **W4A4** GEMM" (spec line 1130); see delta D7 in section 6.
- convert.py can optionally lift the whole expert set from fp4 to fp8 losslessly
  (`--expert-dtype fp8`, convert.py:174-181): the helper re-scales each 32x32 fp8 block by a shared
  power-of-two offset and derives `MAX_OFFSET_BITS = 6` from 6.0 * 2^6 = 384 < 448, the e4m3 maximum
  (convert.py:36-38, 47-53). That trades the W4A8 expert path for a W8A8 one; it is a
  storage/throughput option, not a bit-exact one.

---

## 5. Contrast: DeepSeek-V4-Flash hash layers (num_hash_layers, gate.tid2eid)

The task asks for the V4-Flash hash-layer design as a contrast. What the sources in this checkout
support:

- **This checkpoint has no hash layers.** A scan of all 96,085 tensor names finds zero tensors
  matching tid2eid / tie2eid / hash / ape, and config.json (HF and runtime) contains no
  num_hash_layers key. All 40 backbone layers carry a learned ffn.gate.weight of shape [384, 5120].
- The loader nevertheless anticipates those tensors: convert.py:116-121 lists "tie2eid", "tid2eid",
  "ape" and "image_" among the keys that are stored without a trailing ".weight", i.e. it was written
  to handle sibling checkpoints that do have a token-id-to-expert-id table. On this checkpoint the
  list is inert.
- Gate.forward has no hash path: it always computes scores from a learned projection
  (model.py:809-827). There is no code path that reads expert ids directly from token ids.
- For completeness, the report contrasts V4.1-Flash against DeepSeek-V4-Flash in KV terms only
  (890 bytes/token, roughly 1/4 of V4-Flash's global KV; tech_report.txt p.1 lines 17-23), not in
  routing terms.

The exact semantics of V4-Flash's num_hash_layers and gate.tid2eid are **not derivable from the
sources in this checkout** (no V4-Flash config or implementation is present). Treat any statement
about them as an assumption. What is certain is the delta that matters for SparkFlash-RS: this model
needs 40 learned gates, not a baked-in id table, and each gate needs **two** correction-bias vectors
(text and image), not one.

---

## 6. Spec deltas

**D1 — The spec's TP=4/EP=1 claim is inverted relative to the reference.** The spec says
"Running TP=4 with EP=1 shards every expert's intermediate dimension (2304 / 4 = 576 per rank) rather
than distributing whole experts". The released reference does the opposite: whole experts are
partitioned, 96 per rank at world_size 4, each still 2304 wide, and the per-rank partial sums are
all-reduced (model.py:867-870, 895-902; convert.py:129-134). Following the spec would produce a
different memory plan (no per-rank 576-wide GEMM tiles) and a different communication pattern.

**D2 — The spec's sharding table is wrong for wkv, gate.weight and norms.** The spec implies every
parameter is partitioned by 4. convert.py's mapping (convert.py:56-64) shards only wq_b, wo_a, wo_b,
embed, head, attn_sink and weights_proj; wkv, gate.weight, all RMSNorm weights, hc_*, compressor and
engram.wkv are replicated on every rank.

**D3 — The spec's shared-expert description ("576 padded to 640/rank") assumes an intermediate-dim
split.** The reference keeps the shared expert whole (2304) per rank and runs it on every rank
(model.py:886-887, 903), so there is nothing to pad.

**D4 — The spec never states the asymmetric swiglu clamp.** Up is clamped on both sides, gate only
from above (model.py:845-847). A symmetric clamp changes the FFN output for any token whose gate
pre-activation exceeds -10.

**D5 — The spec's gate description omits the modality-specific bias switch and the exact eps.** The
bias is selected per token by image_mask between gate.bias and gate.bias_vl (model.py:818-820), and
the top-k normalisation uses 1e-20 with an explicit "not norm_eps" comment (model.py:825).

**D6 — The spec's "MXFP4 (32x32 ue8m0)" for experts is wrong on two counts.** Expert weights use a
per-output-row scale with granularity 32 along K only (kernel.py:582), not 32x32 blocks; and the
*activation* is fp8 e4m3 with per-32 ue8m0 scales, not fp4 (model.py:186-195).

**D7 — The spec calls the MoE kernel "W4A4"; the checkpoint and the reference are W4A8.** The spec's
only wording on the MoE activation format is the `MOE_MXFP4` telemetry row: "Route planner, 576-wide
grouped W4A4 GEMM, deterministic sum" (spec line 1130); milestone 4 names the kernel without any
activation claim (spec line 1338). Three sources contradict W4A4: (1) the checkpoint declares
`quant_method: "fp8"` with `activation_scheme: "dynamic"` and `expert_dtype: "fp4"`, i.e. dynamic FP8
activations with FP4 expert weights (config.json:12-21); (2) the reference GEMM is documented and
typed as "FP8 act x FP4 weight", `A: T.Tensor[(M, K), FP8]` (kernel.py:486-499, 511-515), fed by
`act_quant(x, 32, "ue8m0", ...)` in the fp4 branch of the dispatcher (model.py:187); (3) the spec's
own section 2.4 has the fused mHC kernel emit the block input as dynamic FP8 `32x32 ue8m0` (spec
line 86), which is the tensor the MoE consumes, and nothing in the spec re-quantises it to E2M1. W4A4
would need an extra FP8 to E2M1 step that no source defines, and it would quantise a tensor that the
preceding kernel already produced in FP8. Recorded as D14 in `docs/DECISIONS.md`.

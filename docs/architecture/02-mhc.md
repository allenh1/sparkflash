<!--
Copyright (c) 2026 Hunter L. Allen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 02 — Hyper-Connections (mHC): exact math, tensor inventory, and what Mega-mHC must compute

References: <checkpoint>/inference/model.py, kernel.py, tech_report.txt.
Config: hc_mult = 4, hc_sinkhorn_iters = 20, hc_eps = 1e-06 (config.json text_config).

---

## 1. State and parameter inventory

The residual stream is carried as n = hc_mult = 4 parallel copies. Where a normal transformer has a
[b, s, 5120] stream, this model has h = [b, s, 4, 5120] (model.py:1258).

Each block owns **two independent coefficient predictors** — one for the attention sublayer and one
for the FFN sublayer (model.py:940-946):

| Tensor | Shape | dtype | Note |
| :--- | :--- | :--- | :--- |
| hc_attn_fn | [mix_hc, hc_dim] = [24, 20480] | fp32 | mix_hc = (2 + n) * n = 24, hc_dim = n * d = 20480 |
| hc_attn_base | [24] | fp32 | per-coefficient bias |
| hc_attn_scale | [3] | fp32 | one scalar per coefficient group: pre, post, comb |
| hc_ffn_fn / hc_ffn_base / hc_ffn_scale | same | fp32 | the FFN twin |

All six are created under "with set_dtype(torch.float32)" (model.py:940-946), and the checkpoint
follows (layers.N.hc_attn_fn, hc_attn_base, hc_attn_scale and their ffn twins: 40 x 6 = 240 tensors,
plus 3 x 6 for the MTP stages). The spec's single (A_l, B_l, C_l) = H(X_l) is therefore an
oversimplification: there are two predictors per block, 480 in total.

The 24 output rows of one predictor split as (kernel.py:426-431):

| Row index | Symbol | Activation |
| :--- | :--- | :--- |
| j in [0, n) | A (pre) | sigmoid(mix_j * scale[0] + base_j) + eps |
| j + n in [n, 2n) | B (post) | 2 * sigmoid(mix_j * scale[1] + base_j) |
| j*n + k + 2n in [2n, 2n + n^2) | C (comb) | raw: mix * scale[2] + base, then Sinkhorn |

With n = 4 that is 4 pre rows, 4 post rows and 16 comb rows, and 2n + n^2 = 24 = mix_hc.

---

## 2. Coefficient prediction: hc_mixes (model.py:948-955)

```
x_flat = x.flatten(2).float()                                     # [b, s, n*d] = [b, s, 20480]
rsqrt  = rsqrt(mean(x_flat^2, dim=-1, keepdim=True) + norm_eps)   # one statistic per token
mixes  = F.linear(x_flat, hc_fn) * rsqrt                          # [b, s, 24]
(A, B, C) = hc_split_sinkhorn(mixes, hc_scale, hc_base, n, iters, eps)
```

Three points a reimplementation must get right:

1. The RMS statistic is taken over the **flattened n*d stream of one token**, not per copy
   (model.py:952-953).
2. eps is ModelArgs.norm_eps = 1e-20, not hc_eps = 1e-06 (model.py:937, 953).
3. The projection is fp32 throughout, so hc_fn must be kept in fp32 (or emulated with a
   higher-precision accumulator) to reproduce it.

---

## 3. hc_split_sinkhorn: the exact Sinkhorn schedule (kernel.py:406-474)

```
pre[j]      = sigmoid(mix[j]        * scale[0] + base[j])       + eps        (kernel.py:427)
post[j]     = 2 * sigmoid(mix[j + n] * scale[1] + base[j + n])               (kernel.py:429)
comb[j][k]  = mix[j*n + k + 2n] * scale[2] + base[j*n + k + 2n]             (kernel.py:431)

# 1. row softmax, then shift into row-stochastic form:
comb = exp(comb - rowmax) / rowsum      (a real softmax, no eps)             (kernel.py:437-443)
comb = comb + eps
# 2. column normalisation with eps in the denominator:
comb = comb / (colsum + eps)                                                (kernel.py:445-448)
# 3. (sinkhorn_iters - 1) = 19 more alternating passes:
repeat 19 times:
    comb = comb / (rowsum + eps)                                            (kernel.py:450-454)
    comb = comb / (colsum + eps)                                            (kernel.py:455-458)
```

Accounting: **20 row normalisations** (the softmax counts as the first, plus 19 more) and **20 column
normalisations**, matching hc_sinkhorn_iters = 20. eps is added once, before step 2 (kernel.py:443),
and appears in every later denominator; the very first row normalisation has no eps in its
denominator. Row sums are over dim -1 (the destination copy k); column sums are over dim 0 (the
source copy j), so comb ends up **column-stochastic**: sum_j comb[j][k] = 1 for every destination k.
All of this is exact fp32 and per token, only n x n = 16 floats of state per token, so it fits in
registers.

---

## 4. Applied to the stream: hc_pre and hc_post

```
hc_pre(x, A)  = sum_c A[c] * x[c]                                       model.py:957-960
hc_post(y, X, B, C)[k] = B[k] * y + sum_j C[j][k] * X[j]                model.py:962-966
```

hc_pre collapses [b, s, n, d] to [b, s, d] (the actual sublayer input).
hc_post expands the sublayer output y = [b, s, d] back to [b, s, n, d]: the elementwise product
C.unsqueeze(-1) * X.unsqueeze(-2) has shape [b, s, n, n, d] and is summed over dim 2, i.e. over the
**source** copy j (model.py:965). With C column-stochastic, each output copy is a convex combination
of the old copies plus its own post-scaled sublayer output -- the U-connection / Hyper-Connections
residual mixing of the report's Eq. (2) (tech_report.txt p.12 line 561).

---

## 5. The exact block dataflow, and where the shift lives

```
residual = X_l
(A_attn, B_attn, C_attn) = hc_mixes(X_l, hc_attn_fn, ...)        # from the PRE-attention stream
x        = hc_pre(X_l, pre_mix)                                  # pre_mix comes from the CALLER
x        = attn(attn_norm(x), ...)
X'       = hc_post(x, residual, B_attn, C_attn)
residual = X'
(A_ffn, B_ffn, C_ffn)    = hc_mixes(X', hc_ffn_fn, ...)          # from the POST-attention stream
x        = hc_pre(X', A_attn)                                    # uses the coefficient just computed
x        = ffn(ffn_norm(x), ...)
X_next   = hc_post(x, residual, B_ffn, C_ffn)
return X_next, A_ffn                                             # A_ffn feeds the NEXT block
                                                                  model.py:981-994
```

And in Transformer.forward:

```
pre_mix = make_identity_pre_mix(h, hc_mult)     # one-hot on copy 0: [1,0,0,0]   model.py:1159-1163
for layer in layers: h, pre_mix = layer(h, start_pos, pre_mix, image_mask)      model.py:1267
h = layer.hc_pre(h, pre_mix)                    # final collapse uses the LAST block's A_ffn
logits = head(norm(h))                                                          model.py:1268-1269
```

So the concrete shift rule is:

- **Attention of block l** uses A_ffn computed by block l-1 (from block l-1's post-attention stream);
  block 0 uses the one-hot identity.
- **FFN of block l** uses A_attn computed by block l itself (from block l's input stream).
- B and C are always the ones computed by the sublayer that consumes them.
- The final logits use the last block's A_ffn as the collapse vector, not a mean over copies.

This is sharper than "coefficients shifted by one block" as the report phrases Eq. (6): within a
block the FFN's A comes from the same block's pre-attention predictor, while attention takes the
previous block's FFN predictor. Both satisfy the Single-Pass requirement (neither A depends on the
sublayer's own input), but a kernel implementing the report's one-predictor-per-block abstraction
will produce different numbers.

The DSpark path re-uses the same machinery: DSparkBlock.forward_embed expands the draft embeddings to
hc_mult copies (model.py:1134), forward_head calls hc_pre(x, pre_mix) before the head norm
(model.py:1144-1145), and forward_spec re-initialises a one-hot pre_mix (model.py:1277).

---

## 6. What a fused Mega-mHC kernel must compute

Given X_in [n, d] for the current sublayer, A_prev [n], the locally computed B/C, and the fp32
hc_fn/hc_base/hc_scale, one fused pass must produce X_out [n, d], A_here [n], and the quantised
block input:

1. **Read X_in once, in hidden-dimension tiles** (the report's tiling; tech_report.txt p.13
   lines 596-602).
2. **Residual update**: X_next = B . X_in + C . Y for the previous sublayer's output Y, or consume
   an already-updated X_in if the caller materialised it (the reference always does, model.py:981).
3. **Input mixing**: X_hat = sum over c of A_prev[c] * X_in[c]. This is a contraction over n only,
   so it is tile-local and can start immediately.
4. **Coefficient prediction accumulation**: per tile, accumulate the sum of squares of X_in (for the
   reciprocal RMS) and hc_fn @ X_in_tile (a [24, d_tile] by [d_tile] tile GEMM); after the sweep
   multiply by rsqrt(mean(x^2) + 1e-20).
5. **Sinkhorn** on 16 comb values per token: softmax over k, add eps, divide by (colsum + eps), then
   19 alternating row and column normalisations. All of it fits in registers or shared memory.
6. **Input pre-norm**: RMSNorm(X_hat) with the sublayer's norm weight (attn_norm or ffn_norm).
7. **Dynamic FP8 quantisation of the normalised block input**: act_quant with block_size = 32 and
   scale_fmt = "ue8m0", i.e. the SMALLEST power of two at or above amax/448 (the exponent is rounded
   up), with amax floored at 1e-4 (kernel.py:36-37, 74-80). This is the reference's activation rule,
   not the checkpoint's weight-grid rule `floor(log2(amax)) - 8`: the device kernel implements it
   in `kernels/m4_activation_quant_sm121.cuh`'s `spark_act_quant_scale_code` and the two rules are
   separate, named helpers (D23/D8). The mHC reference and the harness use
   `m4_quant::act_quant_fp8_row` for the same reason.
8. **Write X_out once**, plus the quantised X_hat and its per-32 scales.

Data the kernel needs that the report omits: **A_prev must arrive as an input** (it is the previous
sublayer's coefficient, already on the critical path), and a FlashMLA-style attention kernel sits
between the two mHC passes, so "single pass" is per sublayer, not per block. The published lower
bound of (2n + 2)d = 10d stream elements per sublayer (tech_report.txt p.12-13 lines 564-602; the
spec quotes the same number) is the floor for read(X_in) + write(X_out) plus the n extra copies, and
it is reachable only because input mixing and coefficient prediction share one traversal.

---

## 7. ParallelHead

ParallelHead is the vocabulary projection used by both the backbone head and the DSpark Markov head
(model.py:997-1017):

- The weight is stored fp32 even though the checkpoint holds it bf16, so the logits come out in fp32
  (model.py:1005-1006; comment: kept as fp32 here so the logits come out in fp32 directly).
- Shape [vocab_size // world_size, dim]; vocab_size = 129280 divides by 4, giving 32320 rows per
  rank.
- forward(x, full_logits=False) uses only x[:, -1] by default, i.e. generation computes one row;
  full_logits=True is used by the DSpark head to get all 5 draft positions at once
  (model.py:1010-1012, 1145).
- With world_size > 1 the per-rank logits are all-gathered and concatenated to form the full
  vocabulary (model.py:1013-1016). This is an all-gather, not an all-reduce, so an engine that wants
  a distributed argmax must implement it itself rather than materialising 129280 fp32 values per
  position.

---

## 8. Spec deltas

**D1 — One fused Mega-mHC kernel per block is not the granularity the reference supports.** The spec
lists four steps in one sweep, including writing only X_l and the quantised block input. In the
reference each block performs **two** predictor evaluations at two different points in the stream
(pre-attention at model.py:982 and post-attention at model.py:989), with the attention kernel running
between them. A single kernel per block would have to contain attention. The correct fusion
granularity is one mHC sweep per sublayer.

**D2 — The spec has one coefficient predictor H(X_l) per block; the checkpoint has two.**
hc_attn_fn and hc_ffn_fn are separate [24, 20480] fp32 tensors in all 40 layers plus the MTP stages,
and the FFN's input mixing consumes the same block's attn_pre while attention consumes the previous
block's ffn_pre (model.py:981-994). A kernel written against the spec's single (A_l, B_l, C_l) will
apply the wrong pre-mix to half the sublayers.

**D3 — The spec's Sinkhorn description hides the exact schedule.** The reference does: softmax over
k, **add eps**, divide by (colsum + eps), then 19 alternating row and column passes
(kernel.py:437-458). The eps placement is not symmetric: the first row normalisation is the softmax
itself and carries no eps in its denominator, and eps is added to comb once before the first column
pass. A generic "20 alternating normalisations" implementation will differ in the last ulps and can
produce a comb that is not exactly column-stochastic.

**D4 — The spec does not state which eps the coefficient RMS uses.** It is ModelArgs.norm_eps =
1e-20 (model.py:953), evaluated over the flattened n*d = 20480 stream of one token, not hc_eps =
1e-06. hc_eps is only used inside Sinkhorn (model.py:937, 955).

**D5 — A_l for the first block is not the previous layer's coefficient.** The reference seeds it with
an explicit one-hot [1, 0, 0, 0] mix (model.py:1159-1163, 1260). The spec's Eq. (6) with A_{l-1}
leaves the l = 0 case undefined.

**D6 — The final collapse is not an average.** After the last block,
h = layer.hc_pre(h, pre_mix) uses the last block's A_ffn as a weighted sum over the 4 copies
(model.py:1268), whereas the DSpark target-layer tap uses h.mean(dim=2) (model.py:1266). Both appear
in the reference; the spec mentions neither.

**D7 — The spec's (2n+2)d = 10d traffic claim is the report's activation-traffic figure, but the spec
attributes it to a kernel that also writes the quantised block input.** The report counts activation
reads and writes only, not the 24 x 20480 fp32 hc_fn reads (tech_report.txt p.12-13 lines 564-602). A
kernel that additionally materialises X_hat in FP8 writes more than (2n+2)d.

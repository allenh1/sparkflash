# Inkling 2 Small's drafter (`mtp.safetensors`): the forward a port has to reproduce

Status: **the reference the device drafter follows** (`SPARKFLASH_INKLING_DRAFT=dspark`,
`crates/sparkflash-model-inkling2/src/device/draft.rs` and `kernels/inkling_draft_sm121.cu`;
measurements in [the TP4 note, "DSpark drafting"](inkling2-small-tp4.md)). Written 2026-11-14
from the public sources in §14. The equations are restated, not copied; the one Python excerpt (§3, the
tap capture) quotes SGLang, Apache License 2.0, Copyright 2023-2024 SGLang Team. The engine's own
drafters (DeepSeek's DSpark stage, GLM's DFlash2) differ in shape, and §1-§12 are what this
checkpoint needs.

## 0. Source status (read this first)

- **No Thinking Machines (TM) paper or tech report describes the drafter.** The TM pages say
  nothing about it:
  - the HF model card (https://huggingface.co/thinkingmachines/Inkling-2-Small);
  - the model card page (https://thinkingmachines.ai/model-card/inkling-2-small/);
  - the announcement (https://thinkingmachines.ai/news/inkling-2-small/).
  The announcement's only related sentence is about RL infrastructure: prefill, draft and
  verify run on CUDA graphs. The alphaXiv listing of "Introducing Inkling 2 Small" is the same
  announcement, and it is silent too.
- **What TM-adjacent public text does say** (vLLM pages, not TM papers):
  - the vLLM blog (https://vllm.ai/blog/2026-11-12-inkling2-small);
  - the vLLM recipe (https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small, and
    `vllm/repo/recipes/models/thinkingmachines/Inkling-2-Small.yaml:48,120-121`).

  The claims:
  - the drafter is DSpark with a 5-layer parallel block drafter;
  - its head is "the RNN variant from the DSpark paper";
  - it drafts 7 tokens per step, with `draft_sample_method: probabilistic`;
  - acceptance length is about 3.3 to 3.5;
  - greedy drafting at T=1 accepts about 15 to 25% less.
- **Primary source for exact arithmetic:** the SGLang branch `inkling-2-small-support`, at
  commit `778fb834e745bffd9a303ff388435fb44b644746` (2026-11-14). It is the only public
  implementation found. The vLLM sparse clone (`a16bfee11b92`) has the recipe but no Inkling-2
  drafter code.
- **Method lineage.** The checkpoint combines three published pieces and adds two TM-specific
  ones (they appear in code only):
  - **DFlash** (Chen, Liang, Liu): the block-diffusion parallel drafter.
  - **DFlare** (Zhang et al.): per-layer adaptive layer fusion, plus heterogeneous context KV
    projections.
  - **DSpark** (DeepSeek): a semi-autoregressive RNN Markov head.
  - **TM-specific:**
    - `markov_head_type = "rnn_hidden"`, which differs from DSpark's `rnn`;
    - a 6th fusion row that feeds the head.

Path abbreviations used below:

- `S/` = `python/sglang/` of SGLang at `778fb834e745bffd9a303ff388435fb44b644746` (branch
  `inkling-2-small-support`, https://github.com/sgl-project/sglang);
- `DS/` = the `sections/` of DSpark's arXiv source (arXiv:2607.05147v1);
- `DF/` = `acl_latex.tex` of DFlare's arXiv source (arXiv:2606.02091v2);
- `HF/` = the release's `config.json` and `hf_quant_config.json`
  (https://huggingface.co/thinkingmachines/Inkling-2-Small).

Line numbers are those files' at those versions.

## 1. Constants (release config)

`HF/config.json` → `mtp_config`. SGLang derives more values in
`S/srt/configs/inkling2.py:115-205`.

| symbol | value | source |
|---|---|---|
| d (hidden) | 4608 | mtp_config.hidden_size (must equal the target's, inkling2.py:151-152) |
| D (draft layers) | 5 | mtp_config.num_hidden_layers |
| H / H_kv / d_h | 48 / 8 / 128 (GQA 6:1) | mtp_config |
| d_ff | 18432, SiLU-gated | mtp_config.intermediate_size; hidden_act="silu" (inkling2.py:185) |
| eps | 1e-6 (every RMSNorm) | mtp_config.rms_norm_eps |
| RoPE θ | 1e6, rotary over all 128 dims | mtp_config.rope_theta; `S/srt/models/dflash.py:323-350` |
| local (sliding) layers | {0, 1, 3, 4}; layer 2 is full | mtp_config.local_layer_ids; layer_types at inkling2.py:191-194; `window_anchor_at_start` on local layers (inkling2_mtp.py:146) |
| sliding_window_size | 1024 (HF semantics), SGLang window_left = 1023 | `S/srt/speculative/dflash_utils.py:439-440` |
| block B | 8 = anchor + 7 masks; γ = 7 drafted | mtp_config.block_size; `S/srt/models/inkling2_mtp.py:122-124` |
| r (markov_rank) | 256 | mtp_config.markov_rank |
| head type | `rnn_hidden` (the only one allowed) | inkling2.py:153-154 |
| target taps L_j | [5, 11, 17, 23, 29, 35, 41, 48]; T = 8 | mtp_config.target_layer_ids |
| V_pad / V | 201024 / 200058 | text_config.vocab_size / unpadded_vocab_size; `S/srt/configs/inkling.py:128-140` |
| final softcap | 30.0 | text_config.final_logit_softcapping |
| mup multiplier | **absent** | not in `HF/config.json` (checked text_config and the top level) |
| is_causal | False (bidirectional) | inkling2.py:186 |
| sample_from_anchor | False | inkling2.py:188; inkling2_mtp.py:124 |
| mask_token_id | 0 (a placeholder, see §5) | inkling2.py:198 |

## 2. Checkpoint inventory (84 BF16 tensors)

The group list comes from `HF/hf_quant_config.json`, `mtp.parameter_groups[0]`: 84 tensors,
with compute and output dtype bfloat16. Shapes come from the SGLang module constructors. The
safetensors header itself was not read, but every load goes through a shape-checking
`weight_loader`, and BF16 is required (`inkling2_mtp.py:374-377`).

Per layer i ∈ 0..4 (15 tensors, 75 in total):

```
layers.i.input_layernorm.weight            [4608]
layers.i.post_attention_layernorm.weight   [4608]
layers.i.dflare_norm.weight                [4608]       ctx-fusion RMSNorm (DFlare)
layers.i.self_attn.q_proj.weight           [6144, 4608] no bias (q_bias/o_bias false)
layers.i.self_attn.k_proj.weight           [1024, 4608] block rows (W_K^d)
layers.i.self_attn.v_proj.weight           [1024, 4608] block rows (W_V^d)
layers.i.self_attn.o_proj.weight           [4608, 6144]
layers.i.self_attn.q_norm.weight           [128]
layers.i.self_attn.k_norm.weight           [128]
layers.i.self_attn.k_proj_ctx.weight       [1024, 4608] context rows (W_K^t)
layers.i.self_attn.v_proj_ctx.weight       [1024, 4608] context rows (W_V^t)
layers.i.self_attn.k_norm_ctx.weight       [128]
layers.i.mlp.gate_proj.weight              [18432, 4608]
layers.i.mlp.up_proj.weight                [18432, 4608]
layers.i.mlp.down_proj.weight              [4608, 18432]
```

Globals (9 tensors):

```
fuse_weight                       [6, 8]      rows 0-4 = draft layers, row 5 = Markov head
mask_embed                        [4608]
norm.weight                       [4608]      final draft norm
markov_head.markov_w1.weight      [201024, 256]   (V_pad rows; inkling2_mtp.py:131-133, 37)
markov_head.rnn_proj.weight       [768, 9728]     9728 = 2*256 + 2*4608 (inkling2_mtp.py:38)
markov_head.rnn_proj.bias         [768]           768 = 3*256 = gate|cand|out
markov_head.hidden_out.weight     [4608, 256]     (inkling2_mtp.py:39)
markov_head.hidden_out.bias       [4608]
markov_head.dflare_norm.weight    [4608]          RMSNorm for f_rnn (inkling2_mtp.py:40)
```

The checkpoint has **no** lm_head, no embedding, no `fc`/`hidden_norm` (the DFlash shared
projection), no `markov_w2`, and no confidence head. The model leaves them unset
(`inkling2_mtp.py:128-130`). From these shapes the drafter has about 1.71 B parameters (about
3.4 GB in BF16).

Shared target modules:

- **embedding:** `model.llm.embed.weight` with `model.llm.embed_norm.weight`, both BF16;
- **lm_head:** `model.llm.unembed.weight`, BF16 with `fwd_output_dtype: float32`
  (`HF/hf_quant_config.json`, `model` group).

SGLang requires the lm_head to be unquantized (`inkling2_mtp.py:168-169`).

## 3. Target taps: what the eight features are (and what "48" means)

Source: `S/srt/models/inkling2.py:270-278`, with the attention-residual state in
`S/srt/models/inkling_common/inkling2_attn_res.py:116-168`.

```python
state = Inkling2AttnResState(attn_res, hidden)        # partial = embedding, blocks = []
for i, layer in enumerate(layers):
    if i in layers_to_capture:
        captures.append(state.partial.to(lm_head.weight.dtype))   # BEFORE layer i
    layer(state, ...)
hidden = norm(state.compute_output())
captures.append(hidden.to(lm_head.weight.dtype))                  # id 48
```

**Block attention residuals.** The Inkling-2 trunk uses block attention residuals with 6-layer
blocks (`n_layers_per_attn_res_block: 6` in `HF/config.json`).

- At a block boundary (`l % 6 == 0`): `blocks.append(partial)`, then
  `partial = attn_l + mlp_l`.
- Otherwise: `partial += attn_l + mlp_l` (inkling2_attn_res.py:147-156).

So, at token position t (all taps are BF16, `[d]`):

- **c^(j) for j = 0..6** (L_j = 5, 11, …, 41): c^(j) = Σ_{l = L_j−5}^{L_j−1} (attn_out_l + mlp_out_l).
  - This is the in-block partial sum of the first five layers of block ⌊L_j/6⌋.
  - It is **not** the residual stream and **not** a post-layer hidden state.
  - The embedding is its own block 0, so it is not in tap 5.
- **c^(7), "target layer id 48"**: a sentinel equal to `num_hidden_layers`. It means the
  post-final-norm hidden, `norm(attn_res_output)`, which is exactly the lm_head input.
  - The config and the model both require the list to end with 48 (inkling2.py:165-175).
  - The model's error text names it "the post-final-norm hidden state" (inkling2.py:380-392).
  - The DFlash-style `target_layer_ids` handed to the generic code drop it: `features[:-1]`
    (inkling2.py:199-203).

**Packing.** Captures are packed as `[N, 8·d]` in tap order and viewed as `[N, 8, d]` by the
drafter (`inkling2_mtp.py:298-300`, `236-240`).

**Capture API.** SGLang sets the capture list with `set_dspark_layers_to_capture` (pre-layer
partials, which is the Inkling-2 path). A path that captures a post-layer residual hidden at
these ids would feed the drafter the wrong features.

## 4. Fusion weights (DFlare adaptive layer fusion, plus a TM extra row)

- **Paper:** α^(i) = softmax(W_fuse[i, :]) over the T target layers, then
  f_t^(i) = RMSNorm(Σ_j α_j^(i) h_t^(j)). Here W_fuse ∈ R^{D×T}, and α can be precomputed (`DF`
  §"Adaptive Layer Fusion", lines 199-212).
- **Inkling-2** stores `fuse_weight` as `[D+1, T] = [6, 8]` (`inkling2_mtp.py:136-139`).
  - Rows 0-4 feed draft layers 0-4.
  - Row 5 (`row = -1`) feeds the Markov head's f_rnn (`inkling2_mtp.py:241`).
  - The 6th row is TM-specific; DFlare has only D rows.
- **Softmax and rounding.** The softmax runs over the last axis (the 8 taps) in fp32, and the
  result is **rounded to BF16**: `A = bf16(softmax(fp32(fuse_weight), -1))`, `[6, 8]`
  (`inkling2_mtp.py:411-417`).

`mix_rmsnorm(c, a, w)` is the Triton kernel at `S/kernels/ops/inkling2/mtp.py:7-57`:

```
m       = Σ_{j=0..7} fp32(a_j) · fp32(c^(j))          fp32 accumulate, tap order
inv_rms = rsqrt( Σ m² / d + eps )
out     = bf16( m · inv_rms · fp32(w) )               [d]
```

## 5. Draft block inputs and positions

Let P be the committed sequence length, so the anchor sits at position P. The anchor is the
token the target emitted last (the bonus token), and its KV is not in the target cache yet.

- **Input embeddings** (`forward_embed`, `inkling2_mtp.py:209-218`), `[8, d]` BF16:
  - `e_0 = embed_norm(E_tgt[anchor])`. This is the target's own input embedding
    (`get_draft_input_embeddings` → `embed_norm(embed_tokens(ids))`; inkling2.py:245-251,
    398-399; inkling2_mtp_worker.py:57-58).
  - `e_k = mask_embed` for k = 1..7. This is the raw learned vector; **no** embed_norm is
    applied to it.
  - The block's input ids are `[anchor, 0, 0, …]` (mask_token_id 0), but rows 1..7 are
    overwritten, so id 0 never matters.
- **Positions:** p_k = P + k for k = 0..7 (`positions_2d = prefix_lens + arange(stride)`;
  `S/kernels/ops/speculative/dspark/dspark_verify_window.py:501`;
  `S/srt/speculative/dspark_components/dspark_draft.py:618-624`).
- **Prediction target:** row k (k ≥ 1) predicts the token *at* position P+k, as in DFlash mask
  filling. Row 0 (the anchor) is dropped (`select_draft_hidden_without_anchor`,
  dspark_draft.py:95-113).
- **Difference from the DSpark paper:** DSpark uses the anchor as the first prediction
  (`DS/arch.tex:34-37`). Inkling-2 does not (`sample_from_anchor = False`).

## 6. Context K/V injection (per draft layer, per committed target position)

Source: `write_target_hidden_kv`, `inkling2_mtp.py:284-345`. DFlare's heterogeneous KV
(`DF` lines 216-227) gives the context features their own projections, W_K^t and W_V^t.

For each draft layer i and each committed position t, with c_t the 8 taps at t:

```
F_t^i  = mix_rmsnorm(c_t, A[i], layers.i.dflare_norm.weight)          [d] bf16
[k|v]  = F_t^i · [k_proj_ctx ; v_proj_ctx]^T                          [2048] bf16  (no bias)
k      = RMSNorm_per_head(k.view(8,128); k_norm_ctx, eps 1e-6)        V gets no norm
k      = RoPE(k, position = t)      NeoX rotate-half, θ = 1e6, 128 dims  (apply_k_rope, dflash.py:534-538)
store k, v into layer i's draft KV cache at the slot for position t
```

**When it is written:**

- **Prefill:** all prompt positions 0..P−1 (`S/srt/speculative/dspark_components/dspark_kv_inject.py:65-125`).
- **After every verify:** positions P..P+commit_len−1, from the target hidden of the verify rows
  0..commit_len−1 (`inject_ragged`, dspark_kv_inject.py:374-430).
  - The block's own K/V, which the draft wrote at those same slots during drafting, gets
    overwritten.
- **SWA layers** keep only the window, in a separate SWA pool (`inkling2_mtp.py:301-303`).

## 7. Draft transformer (5 layers, bidirectional)

Sources: `DFlashDecoderLayer` (`S/srt/models/dflash.py:927-1023`), `DFlashAttention`
(dflash.py:271-405, forward 446-494), `DFlashMLP` (dflash.py:724-780), model forward
(dflash.py:1222-1278). Every layer is pre-norm, and SGLang fuses add+RMSNorm, so the residual
stream is stored in BF16.

```
h ← E                                                      [8, d]   (h = e_0..e_7)
for i in 0..4:
    x       = RMSNorm(h; input_layernorm)
    q, k, v = x·W_q^T [8,48,128],  x·W_k^T [8,8,128],  x·W_v^T [8,8,128]
    q       = RMSNorm_per_head(q; q_norm);  k = RMSNorm_per_head(k; k_norm)
    q, k    = RoPE(q, p_k), RoPE(k, p_k)                   NeoX, θ = 1e6
    Keys/Vals = [ctx K/V for t ∈ Ω_i ; block k/v for p_0..p_7]
        Ω_i = [max(0, P−1024), P−1]   for local layers {0,1,3,4}
        Ω_i = [0, P−1]                for layer 2 (full)
    o       = softmax(q·K^T / sqrt(128)) · V               no mask; every row sees every key
    h       = h + o·W_o^T
    x       = RMSNorm(h; post_attention_layernorm)
    h       = h + W_down( SiLU(x·W_gate^T) ⊙ (x·W_up^T) )  no MLP softcaps
h = RMSNorm(h; norm.weight)
h_k = h[k], k = 1..7                                       [7, d] bf16
```

- **Bidirectional attention.** `is_causal = False` maps to `AttentionType.ENCODER_ONLY` for both
  the full and the sliding layers (dflash.py:229-263).
- **The sliding window is block-anchored, not per query** (`window_anchor_at_start = True` on
  local layers; inkling2_mtp.py:146).
  - The FA4 backend computes `start = cache_seqlens − query_lens − (window_left + 1)` with
    `cache_seqlens = P+8`, `query_len = 8`, `window_left = 1023`. That gives start = P − 1024,
    and the window is then `(−1, −1)`.
  - So all 8 queries see the same key set: the last 1024 context positions plus the whole
    block (`S/srt/layers/attention/flashattention_backend.py:1828-1875`).
  - It is supported only on FA4 (inkling2_mtp_worker.py:63-70). The draft window (1024) must be
    ≤ the target window, which is 1024 under DSPARK (inkling2_mtp_worker.py:71-76;
    inkling2.py:394-396).
- **Attention scale** is `head_dim^-0.5` (dflash.py:353). There are no sinks, and `v_scale` is
  unset for this config (dflash.py:366-391).
- **Tap order is preserved** in both the context and the block paths. DFlash's shared `fc`
  projection is disabled (`project_target_hidden` raises; inkling2_mtp.py:279-282).

## 8. Base logits

```
U_k = bf16( h_k · W_lm^T )     [V_pad] → cropped to lm_head.org_vocab_size      (k = 1..7)
```

- Sources: `compute_base_logits`, inkling2_mtp.py:200-207; `project_through_lm_head`,
  `S/srt/models/dspark.py:124-142`, which is `torch.matmul(hidden.to(weight.dtype), weight.T)`
  and therefore gives BF16 output; crop at dspark.py:117-121.
- If the target declared `logits_mup_width_multiplier = m`, the hidden would be divided by m
  first (inkling2_mtp.py:202-203; the worker reads it from the target config at
  `S/srt/speculative/dspark_components/dspark_worker_v2.py:260-264`). **Inkling-2-Small
  declares none, so there is no mup scaling.**
- There is an optional fp8 or mxfp8 knob for this GEMM (`speculative_dspark_lm_head_dtype`,
  dspark.py:131-139). It is off by default.

## 9. The `rnn_hidden` Markov head (sequential, 7 steps)

### 9.1 f_rnn (per request, staged before the block)

```
f = mix_rmsnorm(c_{P−1}, A[5], markov_head.dflare_norm.weight)      [d] bf16
```

- c_{P−1} holds the 8 taps at the **last committed position**, the target row whose logits
  produced the anchor.
  - After prefill, it is the last prompt row (`_on_prefill_target_hidden`,
    inkling2_mtp_worker.py:80-89).
  - After a verify, it is verify row commit_len−1 (`store_verify_commit`,
    inkling2_mtp.py:250-259, called from inkling2_mtp_worker.py:41-47).
- It is kept in a per-request buffer (inkling2_mtp.py:220-248), gathered per batch in
  `forward` (inkling2_mtp.py:266-268), and handed to `init_state` (inkling2_mtp.py:76-85).

### 9.2 Fold (computed once at load)

The fold is computed in fp64 and stored in the lm_head dtype, BF16 (inkling2_mtp.py:46-67):

```
W_f = bf16( W_lm · W_ho )       [V_pad, 256]
b_f = bf16( W_lm · b_ho )       [V_pad]
```

Both are also divided by m if a mup multiplier existed; it does not here. W_ho and b_ho are
`markov_head.hidden_out.{weight,bias}`.

### 9.3 Step loop

Sources: `draft_markov_block`, dspark_draft.py:263-308; `_draft_step`, 208-260; `step_bias`,
inkling2_mtp.py:87-97. The kernels are `S/kernels/ops/inkling2/mtp.py:60-160`, and the sampler
is `S/kernels/ops/speculative/dspark/markov_sample.py:384-489`.

```
s_0 = 0 ∈ R^256 (bf16);  x_0 = anchor;  f from 9.1
for k = 1..7:
    z_k = [ s_{k−1} (256) ; W1[x_{k−1}] (256) ; h_k (4608) ; f (4608) ]   ∈ R^9728 bf16   (order: mtp.py:105-139)
    p   = bf16( W_rnn · z_k + b_rnn )                                       ∈ R^768
    g   = bf16( sigmoid(p[0:256]) )                    gate
    ĉ   = bf16( tanh(p[256:512]) )                     candidate
    s_k = bf16( bf16(g ⊙ s_{k−1}) + bf16( bf16(1 − g) ⊙ ĉ ) )
    ℓ_k = bf16( tanh(p[512:768]) )                     output latent (taken from z_k, not s_k)
    B_k = bf16( b_f + W_f · ℓ_k )                      [V]   (torch.addmm, bf16)
    v   = bf16( fp32(U_k) + fp32(B_k) )
    v   = bf16( 30 · tanh(v / 30) )                    target final softcap (inkling2_mtp.py:134)
    v[j ≥ 200058] = −inf                               (inkling2_mtp.py:135)
    x_k = argmax_j v_j                                 greedy request (top_k ≤ 1)
        = argmax_j ( v_j / T − log E_j ),  E_j ~ Exp(1)  otherwise (Gumbel-max ≡ sample softmax(v/T))
          ties → lowest index; Philox noise keyed by (seed, row, step, per-block counter)
    keep v as the draft logits of step k (q_k = softmax(v / T) for verification)
```

- **Gate math** comes from `_rnn_gates` (mtp.py:65-81). It rounds every op to BF16, with the
  comment "Each op rounds to BF16 like the eager torch ops it replaces". It is compiled with
  `enable_fp_fusion=False`.
- **Temperature** is clamped to at least 1e-5, and the step temperature is 0 on greedy rows
  (dspark_draft.py:147-177). The noise counter advances once per block (dspark_draft.py:307).
- **Draft mode.** SGLang drafts probabilistically at the request temperature unless
  `SGLANG_DSPARK_USE_GREEDY_DRAFT` is set (dspark_worker_v2.py:374). vLLM's recipe also asks
  for `probabilistic`.
- **No fused greedy kernel.** The head's `greedy_w2_weight` is the base class's `None`
  (dspark.py:181-184), so Inkling-2 always takes the `step_bias` path.

### 9.4 What `rnn_hidden` means

This is an interpretation that follows from the code. Because B_k = W_lm(W_ho ℓ_k + b_ho):

  U_k + B_k = W_lm · ( h_k + δ_k ),   with δ_k = W_ho · tanh(W_o z_k) + b_ho ∈ R^d

The recurrent head predicts a **hidden-space correction** δ_k, which is added before the shared
frozen lm_head and then softcapped like the target's logits. It differs from the DSpark paper's
RNN head (`DS/arch.tex:83-93`) and from DeepSpec's `RNNHead`
(DeepSpec `005e03b`, `deepspec/modeling/dspark/markov_head.py:125-173`) in three ways:

1. B_k = W_2^T tanh(W_o z_k) with its own `markov_w2` [r × V] is replaced by `hidden_out`
   (r → d), followed by the lm_head. That is the "hidden" in the name.
2. z_k carries the extra fused target feature f (+4608), so 9728 = 2r + 2d instead of 2r + d.
3. The output softcap (30) and the padding mask come after the bias.

The gate equations, the single joint projection split as gate|cand|out, and s_0 = 0 match the
paper.

## 10. Verify, accept, commit, rollback

- **Verify input.** The target runs 8 tokens `[anchor, x_1..x_7]` at positions P..P+7
  (`verify_num_draft_tokens = γ+1 = 8`; dspark_worker_v2.py:270-275;
  `S/srt/speculative/dspark_components/dspark_config.py:106-154`). The SGLang cookbook says it
  "verifies 8 tokens per request per step" (`docs/cookbook/autoregressive/ThinkingMachines/Inkling-2-Small.mdx:79`).
- **Target logits** are softcapped at 30, with columns from 200058 up set to −inf
  (`Inkling2LogitsProcessor`, inkling2.py:149-196). The unembed's forward output is fp32.
- **Greedy rows** accept the longest prefix where x_{j+1} = argmax(target row j). The bonus is
  the target argmax at the first mismatch row, or at row 7 (`AcceptGreedy`,
  `S/kernels/ops/speculative/dspark/dspark_accept.py:761-801`; dspark_verify.py:1422).
- **Sampled rows** use classic chain rejection sampling (blockwise_rejection.py:1-18;
  dspark_accept.py:108-229):
  - p_j = softmax(target_j / T);
  - q = softmax(v / T) of the kept step logits (`_draft_probs`, dspark_verify.py:1330-1348);
    a greedy draft makes q one-hot;
  - accept x iff u · q(x) < p(x);
  - on a reject, resample from norm(max(p − q, 0));
  - if all 7 are accepted, the bonus comes from p_7.
- **Commit.** commit_len = n_accepted + 1 ∈ [1, 8]. The next anchor is the bonus token, at
  P' = P + commit_len.
- **Draft-state refresh** after each verify:
  - context K/V at P..P'−1, from target taps rows 0..commit_len−1 (§6);
  - f_rnn from row commit_len−1, i.e. position P'−1 (§9.1).

  The RNN state restarts at 0 every block, and the draft block slots ≥ P' get rewritten by the
  next draft.
- **Target rollback** (traced only partly):
  - KV and seq_len are trimmed to P'.
  - SWA-pool commit locations are translated by `S/kernels/ops/speculative/dspark/commit_swa.py`.
  - Short-conv windows (kernel 4) are committed at the last correct step
    (`update_mamba_state_after_mtp_verify` → `commit_conv_state_after_mtp_verify`,
    `S/srt/layers/attention/linear/inkling_sconv_backend.py:683-695`). The docstring says "conv
    windows only, with no SSM state".
  - Attention residuals carry no cross-token state.
- **Not traced:** the target's `d_rel`/`rel_extent` relative-attention mechanism (`HF/config.json`
  text_config).

## 11. Confidence signal, tokens per step

- **No confidence head in the checkpoint**: of the 84 tensors, none is a confidence head, and
  `confidence_head = None` (inkling2_mtp.py:130).
- **SGLang's optional estimator.** SGLang can build its own online acceptance estimator from the
  step logits to cut verify lengths. It does so only when there is no confidence head and the
  ragged verify is not STATIC (dspark_worker_v2.py:331, 558-574; dspark_draft.py:535-541). This
  is serving-side policy, not part of the model.
- **Per step:** 7 tokens drafted, 8 verified, 1 to 8 committed. The vLLM blog reports acceptance
  length 3.3 to 3.5 (32K to 256K context, TP8) and about 630 tok/s single-user at TP4, with
  probabilistic drafting.

## 12. Dtype, mup and softcap quirks (summary)

- Everything is BF16: weights, taps, α, f_rnn, s, ℓ, the fold, base logits and step logits
  (`step_logits_dtype = promote(bf16, bf16)`, inkling2_mtp.py:99-100). The fp32 islands are the
  mix_rmsnorm accumulation, the gate math before each rounding, and the sampler's comparison
  arithmetic.
- The step logits are rounded to BF16 twice: once after U + B, and again after the softcap
  (markov_sample.py torch reference).
- There is no mup scaling for this checkpoint (§8).
- The softcap of 30 and the padding mask apply to draft step logits exactly as to target logits.
  `test/registered/unit/models/test_inkling2.py:414-445` checks that padding stays masked after
  the softcap.
- The draft MLP has **no** gate/up softcaps. The target MLP has `mlp_gate_softcap 4.0` and
  `mlp_up_softcap 25.0` (configs/inkling2.py:22-23).
- embed_norm applies to the anchor row only; mask_embed is used raw.

## 13. Open uncertainties

1. **The training reference is unpublished.** Every equation above is SGLang's inference code,
   checked against the DFlash, DFlare and DSpark papers where they overlap. `rnn_hidden`, the
   6th fusion row, f_rnn and the fold are documented nowhere outside code.
2. **Window semantics.**
   - SGLang gives all 8 block queries one shared window: the last 1024 context tokens plus the
     whole block. A per-query sliding window over the same keys would differ.
   - The explicit `window_anchor_at_start` flag and the "≤ target window" check suggest this
     matches training, but no TM text confirms it.
3. **RoPE style.**
   - NeoX rotate-half is SGLang's default for DFlash drafts, and mtp_config has no
     `rope_is_neox_style` key.
   - Context K uses each context token's own absolute position, as in DFlash. That is code, not a
     TM statement.
4. **Logit precision.**
   - SGLang computes U in BF16, and folds the bias into BF16 W_f/b_f, rather than computing
     W_lm(h + δ) in one GEMM or in fp32. The target unembed declares an fp32 forward output.
   - Training may have used a different rounding. For bit-exact agreement with SGLang, follow
     the fold and the rounding order in §9.3.
5. **α rounded to BF16** before mixing (inkling2_mtp.py:411). Training likely kept fp32 (the
   DFlare paper says the softmax can be precomputed). The difference is small but not zero.
6. **f_rnn position** (taps at P−1, the row that produced the anchor) is clear in code, but its
   training-time definition is unstated.
7. **Draft distribution with top-k/top-p** in verification (the `build_speculative_verify_target_probs`
   path) was not traced. Only the plain-temperature path was read.
8. **Target rollback** of the relative-attention (`d_rel`, `rel_extent`) state and the exact SWA
   ring handling were not traced.
9. **mup.** The released config has no `logits_mup_width_multiplier`. Any local code that applies
   one to Inkling-2 is likely carried over from another model or version. SGLang would divide
   both U and the fold if a multiplier were declared.
10. **Shapes** come from module constructors plus shape-checked loading. The `mtp.safetensors`
    header itself was not downloaded.
11. **vLLM's implementation** (the `inkling-2-small` image) is not in the public tree that was
    cloned, so there is no second implementation to cross-check against.

## 14. Attribution

| Work | Authors / owner | Where | License | Copyright line (exact) |
|---|---|---|---|---|
| SGLang, branch `inkling-2-small-support` @ `778fb834e745bffd9a303ff388435fb44b644746` (primary implementation) | SGLang Team | https://github.com/sgl-project/sglang | Apache License 2.0 (`LICENSE`); no NOTICE file | `Copyright 2023-2024 SGLang Team` (LICENSE:189). `srt/models/dflash.py:1`: "# Adapted from the DFlash reference implementation (HF)" |
| DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation, arXiv:2607.05147 (v1, 2026-07-06) | Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, Yi Qian, Jiaqi Zhu, Shirong Ma, Xiaokang Zhang, Jiasheng Ye, Qinyu Chen, Chengqi Deng, Jiping Yu, Damai Dai, Zhengyan Zhang, Yixuan Wei, Yixuan Tan, Wenkai Yang, Runxin Xu, Yu Wu, Zhean Xu, Xuanyu Wang, Muyang Chen, Rui Tian, Xiao Bi, Zhewen Hao, Shaoyuan Chen, Huanqi Cao, Wentao Zhang, Anyi Xu, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang (Peking University; DeepSeek-AI) | https://arxiv.org/abs/2607.05147 | CC BY 4.0 | None printed. The class defines "© <year> DeepSeek. All rights reserved", but its footer use is commented out (`deepseek.cls:138,175`). |
| DeepSpec @ `005e03b81cec38b7da6399833d609ee89a2587f2` (DSpark training code) | The DeepSpec Authors | https://github.com/deepseek-ai/DeepSpec | MIT; its NOTICE credits SpecForge (Apache-2.0) | `Copyright (c) 2026 The DeepSpec Authors` |
| DFlare: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding, arXiv:2606.02091 (v2, 2026-06-02) | Jiebin Zhang, Zhenghan Yu, Song Liu, Eugene J. Yu, Zheng Li, Dawei Zhu, Jiangshan Duo, Weimin Xiong, Yifan Song, Guanghua Yu, Jianchen Zhu, Sujian Li | https://arxiv.org/abs/2606.02091 | CC BY 4.0 | None in the paper |
| AngelSlim (DFlare code) | Tencent | https://github.com/Tencent/AngelSlim | Apache License 2.0 | `Copyright (C) 2025 Tencent.  All rights reserved.` (LICENSE:3; LICENSE:75 also carries a third-party `Copyright (c) 2023 MIT HAN Lab`) |
| DFlash: Block Diffusion for Flash Speculative Decoding, arXiv:2602.06036 (v2, 2026-05-28; ICML 2026) | Jian Chen, Yesheng Liang, Zhijian Liu | https://arxiv.org/abs/2602.06036 | CC BY 4.0 | None in the paper |
| z-lab/dflash (DFlash reference code) | Z Lab | https://github.com/z-lab/dflash | MIT | `Copyright (c) 2026 Z Lab` |
| Inkling-2-Small checkpoint and config (`mtp.safetensors`, `config.json`, `hf_quant_config.json`) | Thinking Machines Lab, Inc. | https://huggingface.co/thinkingmachines/Inkling-2-Small | apache-2.0 (README front matter, lines 2-3) | `Thinking Machines Lab © 2026` (model-card page footer; no copyright line in the HF README) |
| vLLM recipe / blog (secondary: layer count, tokens per step, acceptance) | vLLM project | https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small; https://vllm.ai/blog/2026-11-12-inkling2-small | Apache License 2.0 (repo); the LICENSE has only the template copyright text | none |

A Rust/CUDA port of the arithmetic in §3-§10 derives from the SGLang code and the three papers.
It needs:

- header attribution naming SGLang (Apache-2.0, "Copyright 2023-2024 SGLang Team");
- NOTICE entries for SGLang, plus DFlash (MIT, Z Lab) via SGLang's own adaptation note;
- paper citations for DFlash, DFlare and DSpark.

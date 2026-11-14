# Inkling 2 Small on four Sparks (TP4)

Status: **text backbone runs end to end at TP1 through the f32 host reference, and TP4 rank
files convert, verify and load; the TP4 device arm, the drafter and vision are the remaining
milestones.**


This note answers two questions about `thinkingmachines/Inkling-2-Small`:

1. can the released checkpoint run on the four-Spark tensor-parallel configuration this tree
   already serves (four GB10 nodes, TP=4, 128 GB unified memory each);
2. what a model arm for it would have to add to SparkFlash.

The short answer is **yes on memory, and now yes on parsing, planning and host execution**. The
NVFP4 checkpoint is 390.06 GB of weights and fits four Sparks with about 11 GiB a rank still
free at a 1M-token context. The tree now has an `inkling2` arm: the shared parser reads the
released schedule and geometry, the tensor plan reproduces the 390,056,448,320-byte index, the
f32 host reference runs the relative attention, short convolutions, block residual, softcapped
MoE and mixed NVFP4/MXFP8 storage, and the five device kernel sources compile with nvcc for
sm_121a. The remaining device work is a TP4 device arm over those kernels and the rank-file
converter; the host arm is the runnable path today.

## The checkpoint

| property | value |
| :-- | :-- |
| HF repo | `thinkingmachines/Inkling-2-Small` (Apache 2.0) |
| architecture | `Inkling2ForConditionalGeneration` / `model_type: inkling2` |
| parameters | 594,050,672,720 total, 17B active |
| layers | 48; 40 `hybrid_sliding` (window 1,024), 8 `hybrid` full-attention (5, 11, 17, 23, 29, 35, 41, 47) |
| attention | GQA 48 query / 8 KV heads, head 128, **no RoPE**, learned relative bias, attention sink, sigmoid output gate, per-head L2 clipping |
| residual | block attention residual (8 blocks of 6 layers), not a plain residual stream |
| convolutions | 4 causal depthwise short convolutions a layer (K, V, attention output, MLP output), kernel 4; K/V are dynamic on the full layers |
| MoE | 384 routed experts, top-8, no shared experts, intermediate 2,304, sigmoid router, `route_scale` 8, softcapped SwiGLU |
| vocabulary | 201,024 padded; 200,058 real; separate `embed` and `unembed`, untied |
| context | 1,048,576 |
| vision | 23-layer ViT, width 1,024, 16 heads, MLP 4,096, patch 20, merge 2, projector 4,096 -> 4,608 |
| drafter | DSpark, 5 layers, block 8, 7 drafted tokens a step, RNN Markov head (rank 256) |
| checkpoint formats | NVFP4 release (main), separate MXFP8 release |

### Stored bytes

The NVFP4 snapshot's index reports **390,056,448,320 bytes** of backbone tensors; the separate
`mtp.safetensors` adds **3,423,429,800 bytes**; the HF repo's total stored size is
393,509,649,692 bytes. The dtype split is exact:

| storage | bytes | where |
| :-- | --: | :-- |
| `U8` packed E2M1 | 247.67 GB | routed experts, layers 0-39 |
| `F8_E4M3` | 128.42 GB | routed experts, layers 40-47 |
| `BF16` | 13.96 GB | attention, embeddings, norms, vision, router |
| `F32` | 2.26 MB | attention-residual queries, router bias, expert amax |

A layer is 7.08 GB in the NVFP4 range and 12.81 GB in the MXFP8 range:

| component | size |
| :-- | --: |
| experts, one NVFP4 layer (384 experts) | 6.88 GB |
| experts, one MXFP8 layer (384 experts) | 12.61 GB |
| attention, one layer | 0.196 GB |
| embed + unembed | 3.705 GB |
| vision tower | 0.655 GB |
| MTP | 3.423 GB |

### Quantisation

`hf_quant_config.json` is authoritative; `config.json`'s single
`quant_method: "modelopt", quant_algo: "NVFP4"` hides the mixed-precision layout.

* **Layers 0-39, routed experts:** NVFP4 E2M1, block 16, E4M3 scales, per-expert
  `amax`/`outer_scale_denominator` rather than Qwen's per-tensor `weight_scale_2`.
  Activations are per-token, block 16, with the **four-over-six** scale-selection rule
  (`fouroversix_mae`).
* **Layers 40-47, routed experts:** MXFP8 E4M3, 32-element blocks, `ue8m0` scales
  (`mxfp8_nearest`).
* **Everything else:** BF16 (attention, router, embeddings, vision, MTP).

The model was trained in NVFP4 and the model card recommends it. The MXFP8 checkpoint needs
about 800 GB of aggregate memory and **does not fit four Sparks**.

## Memory on TP4

The engine's per-node budget is `121.7 GiB - 8 GiB = 113.7 GiB`
(`NODE_MEMORY_BUDGET_BYTES`, `MEMORY_FLOOR_BYTES` in `crates/sparkflash-model/src/arm.rs`).
Four nodes therefore have 486.8 GiB (522.7 GB) in total and **454.8 GiB (488.3 GB) after the
guard band**.

With heads, experts, vocabulary and the attention projections sharded over four ranks, and the
vision tower replicated, one rank holds **91.4 GiB** of backbone weights (98.1 GB). The DSpark
checkpoint is small enough to replicate: **3.2 GiB** more, for 94.6 GiB. The KV arithmetic is
8 global layers x 2 KV heads/rank x 128 x K+V x BF16 = 8 KiB a token, plus 40 MiB of fixed
sliding-window state:

| context | KV a rank | weights + KV | headroom of 113.7 GiB |
| --: | --: | --: | --: |
| 32K | 0.29 GiB | 94.9 GiB | 18.8 GiB |
| 256K | 2.04 GiB | 96.6 GiB | 17.1 GiB |
| 1M | 8.04 GiB | 102.6 GiB | 11.1 GiB |

This is a real fit, but not a comfortable one. Four Sparks' 512 GB nominal clears the model
card's "at least 500 GB of aggregated VRAM" guidance, while the engine's 488.3 GB usable budget
is slightly under it; the checkpoint itself plus 1M KV is about 428 GB, so the fit does not
depend on the missing 12 GB. TP=1 (128 GB) and TP=2 (256 GB) cannot hold 390 GB of weights;
**TP=4 is the minimum and only Spark width**.

Two corrections matter before trusting the 1M row:

* the shared `Architecture::kv_plan` currently counts every GQA layer as full-context, so it
  would charge 48 KiB a token a rank (48 GiB at 1M) and cap the browser's estimate near 0.4M
  tokens. An Inkling `KvPlan` must put the 40 sliding layers in the fixed term (window x
  bytes) and only the 8 full layers in `per_token`. The table above is that plan.
* 11 GiB of headroom has to hold prefill scratch, the 4 convolution states a layer, the block
  residual buffers, and allocator fragmentation. A 1M prefill with a 4,096-token attention
  chunk can spend several GiB on scores alone; use a smaller attention chunk and validate 1M
  before advertising it. 256K is the comfortable default.

Decode is bandwidth-bound. One token reads 7.84 GB of active expert weights, 9.43 GB of
attention weights and 1.85 GB of unembed across the group: **19.1 GB a token, 4.78 GB a rank**.
At the GB10's 273 GB/s that is a 57 tok/s roofline before network and kernel inefficiency; the
DSpark drafter's ~3.5 acceptance length could raise the effective rate substantially, but only
after the drafter exists. Treat 40-60 tok/s without speculation as the planning number, not a
promise.

## What the engine has, and what it lacks

The current `target/debug/sparkflash` refuses the checkpoint at the first layer:

```text
sparkflash serve: --checkpoint ...: config.json is not a usable architecture:
unsupported configuration: layer 0 has no MoE intermediate width
```

That is the first of several parser gaps. The released config uses the Transformers aliases
`intermediate_size` -> `moe_intermediate_size`, `route_scale` ->
`routed_scaling_factor`, and `conv_kernel_size` -> `sconv_kernel_size`; it names
`local_layer_ids` (not `layer_types`); its `mtp_config` is nested; its vision keys are
`native_patch_size`, `adapter_merge_size`, `projection_hidden_dim`, `text_hidden_size`,
`mlp_ratio` and `posemb_grid_size`; and its per-layer precision lives in
`hf_quant_config.json`, not `config.json`.

Even with those aliases, the shared parser would still be wrong:

| released fact | current parse |
| :-- | :-- |
| 40 sliding + 8 full layers | no `layer_types`; every layer gets `sliding_window = 1024` |
| no RoPE | `rotated_dims = head_dim` (full RoPE) |
| relative bias `d_rel=16`, `rel_extent=1024` | no field |
| attention sink, output gate, q/k/rel L2 clip | only `output_gate` exists, no sink/clip |
| dynamic K/V short conv on full layers | no field or state |
| block attention residual, 1 or 3 queries | no field; `Residual` only Plain or HyperConnection |
| softcapped SwiGLU (`tanh(g/4)*sigmoid(g)*tanh(u/25)`) | `Activation::Silu` |
| sigmoid router, `route_scale` 8, gate bias | defaults to softmax, scale 1, no gate bias |
| `final_logit_softcapping` 30, logits sliced to 200,058 | not modelled |
| nested `mtp_config` (5 layers, block 8, RNN head) | no `Part::Mtp`; only flat `dspark_*` keys |
| mixed NVFP4/MXFP8 experts | one global NVFP4 format |

The parts vocabulary is also short. `Part::GqaAttention` exists and accepts 48/8 (group 6,
head 128), but its plan and kernel have no relative bias, sink, L2 clip or convolution state.
`Part::Vision` exists but the shared tower is 14-patch and the catalogue claims only
`VISION_PATCH_SIZES = [14]`. `Part::Mtp` exists but the engine's DSpark logic is
DeepSeek's 3-layer, block-5 drafter, not a 5-layer RNN-headed model. There is no
`ShortConv`, no `BlockAttentionResidual` and no relative-attention part.

## What needs to be added

### 1. Configuration, registry and profile

* `crates/sparkflash-model/src/architecture.rs`
  - derive `layer_types` from `local_layer_ids` (or accept `hybrid`/`hybrid_sliding`),
    with a window only on the sliding layers;
  - read the Transformers aliases above, `swa_head_dim`/`swa_num_*_heads`,
    `rel_extent`, `d_rel`, `qk_l2_clip_threshold`, `rel_l2_clip_threshold`,
    `log_scaling_n_floor`, `log_scaling_alpha`, `mlp_gate_softcap`, `mlp_up_softcap`,
    `final_logit_softcapping`, `unpadded_vocab_size`, `use_gate_bias`, and the
    `scoring_func` default of `sigmoid`;
  - add typed geometry for relative bias, short convolution (static and dynamic), block
    attention residual, and the DSpark stage;
  - parse the vision aliases and `mlp_ratio`;
  - read `hf_quant_config.json` (from a directory or alongside a Hub snapshot) and store a
    per-layer expert format, not one global format.
* `crates/sparkflash-model/src/descriptor.rs` and `crates/sparkflash-web/src/config.rs`:
  a served id, weights id, descriptor (1,048,576 context, 201,024 vocab, 200,058 real),
  and a deployment entry with `world_size = 4`, `worlds = [4]`.
* A new `crates/sparkflash-model-inkling2` crate with its typed config, profile TOML,
  descriptor, host reference and device arm. The profile names
  `model = "thinkingmachines/Inkling-2-Small"` and its weights id; every switch is a typed
  profile field with an `ENV_OVERRIDES` alias, per `docs/adding-a-model.md`.

### 2. Parts and plans

* New `Part` variants (model side and catalogue side): `ShortConv`,
  `BlockAttentionResidual`, and a relative-bias attention part. Keep `Part::GqaAttention`
  for Qwen/GLM/Dory; do not silently change its arithmetic.
* New `AttentionPlan`/geometry fields for `d_rel`, `rel_extent`, sink, L2 clip and the
  log-scaling tau, or a dedicated relative-attention plan. The existing
  `PartGeometry` has `window` but no relative-bias fields.
* A short-convolution plan with a decode state: four states a layer, kernel 4, static taps
  for sliding K/V and attention/MLP outputs, static+dynamic taps for full-layer K/V. The
  output is `(conv(x) + x) / sqrt(kernel + 1)`, and `1 / sqrt(2*kernel + 1)` for the
  dynamic form.
* A block-residual plan: 8 block summaries, an unweighted RMSNorm over the summary stack,
  scaled dot-product attention with 1 learned query (sliding) or 3 (full), and the block
  boundary rule (layer `i % 6 == 0` appends its input and returns the sublayer sum without
  the ordinary additive residual).
* An MoE plan for 384 experts/top-8/2,304, with a per-layer expert format and the
  softcapped-SwiGLU epilogue. At TP=4 the rank-local intermediate is 2,304/4 = 576, which is
  the width the b12x body already uses for DeepSeek, but the arithmetic is NVFP4/MXFP8, not
  MXFP4.

### 3. Kernels

No existing image covers these; a direct-launch arm in the style of Dory's is the pragmatic
first form, because the block residual spans layers and the four convolutions are not a
single `LayerPlan` step.

| kernel family | arithmetic it must add | reusable base |
| :-- | :-- | :-- |
| `inkling_attn` | no-RoPE GQA, relative-bias gather (`r @ proj` over 1,024 distances), sink logit, sigmoid output gate, q/k/rel L2 clip, tau log-scaling on full layers, 1,024 window | q4e attention body for MMA/GQA layout only |
| `inkling_conv` | causal depthwise conv 4, static and dynamic taps, residual + `1/sqrt(k+1)` scale, decode state | none; GDN conv is recurrent, not depthwise |
| `inkling_block_res` | normalized 8-way block mix, 1/3 queries, block append | none |
| `inkling_moe` | 384 x top-8, NVFP4 four-over-six, MXFP8 ue8m0, softcapped SwiGLU | `moe_plan`, grouped MMA and combine |
| `inkling_vision` | patch 20, 23-layer ViT, LayerNorm, GELU, layer scale, axial 2-D RoPE, 2x2 merger | `opt_vision` is patch 14 and no axial RoPE |
| `inkling_dspark` | 5-layer drafter, block 8, 7 tokens, RNN Markov head | scheduler's confidence-capped verify; no model |
| LM head | final `tanh(logits/30)*30`, slice to 200,058 | existing head reduce |

Every source belongs in `kernels/` and is exposed through `sparkflash_kernels::images`; no
`.cu` in the model crate (`one_stack.rs` enforces it). Add catalogue entries with the real
symbol masks (NVFP4, MXFP8, BF16; prefill/decode) and shape constraints, and keep DeepSeek
canonical for shared parts.

### 4. Weights, conversion and TP

* `sparkflash_weights::nvfp4` must decode the ModelOpt **per-expert** `amax` /
  `outer_scale_denominator` form, not only Qwen's per-tensor `weight_scale_2`.
* A new MXFP8 expert loader: E4M3 codes, `ue8m0` scale a 32-element block, and a per-expert
  scale where the release has one.
* A family `ArchitectureTensorPlan` (or a `SlotPack` arm) for the 48 layers, the 422 vision
  tensors, the separate `mtp.safetensors`, and the checkpoint's names
  (`wq_du`, `wk_dv`, `wv_dv`, `wr_du`, `gate_attn`, `attn_sink_proj`, `wo_ud`,
  `rel_logits_proj.proj`, `k_sconv`/`v_sconv`, `attn_sconv`, `mlp_sconv`, `w13_weight`,
  `w2_weight`, `w13_weight_scale`, `w13_weight_amax`, ...).
* TP4: heads 48/4 = 12, KV 8/4 = 2, experts 384/4 = 96, vocabulary 201,024/4 = 50,256
  (padded to 50,304 at a 128 alignment; use 64 or the padded width rather than reusing
  Qwen's `vocab_multiple(128)` rule). Routed experts are split by **expert**, not by expert
  *and* intermediate width: a rank holds 96 whole experts of 2,304 intermediate columns.
  Splitting both axes gives a rank 96 experts of 576 columns, a sixteenth of the expert
  weights, and four such files hold a quarter of the model. Expert parallelism also matches
  the router — a rank computes its own experts whole and the collective exchanges tokens —
  and keeps the intermediate width the shape the kernels are built for.
* Measured rank payload: 97.49 GB per rank at TP4 (389.95 GB of files, 390.06 GB of
  distinct checkpoint bytes). Four ranks cover the checkpoint exactly once — every sharded
  tensor tiled with no gap and no overlap, every replicated tensor identical on all ranks —
  and each rank clears the 113.7 GB budget. `inkling2::convert`'s tiling test is the check.
* `sparkflash convert` writes Inkling's ranks through the shared rank-file container:
  `sparkflash-model-inkling2::convert` walks `InklingTensorPlan` in plan order with
  `SlotPlan`/`SlotFileWriter`, chunking payloads over 512 MiB and recording the source
  ranges an inner-axis split needs, and `verify_rank` compares every slot against the
  checkpoint. `--inspect` reads a file back. The Hub resolver needs an entry for the repo
  and its sibling `mtp.safetensors`.

### 5. Serving, tokenizer and KV

* `PackArm` implementation and a `Pack::Inkling` variant in
  `crates/sparkflash-cli/src/serve/pack.rs`; descriptor, fixture writer, configuration,
  tokenizer, open and response layer.
* `InklingCodec`: the tokenizer is a 26 MB tiktoken-style `tokenizer.json` with custom
  markers (`<|message_user|>`, `<|content_text|>`, `<|content_thinking|>`,
  `<|content_invoke_tool_json|>`, `<|end_message|>`, ...), not GLM's `[gMASK]` or
  ChatML. The chat template carries reasoning effort (0.0-0.99), tool declarations and
  canonicalized tool-call JSON; the current codec detection refuses it.
* Image input: `image_token_id = 200054`, `<|unused_200054|>` in the template, 20-pixel
  patches, 2x2 merge, class token, 32x32 learned position grid interpolated per image, axial
  2-D RoPE, 40-4,096-pixel input guidance.
* `KvPlan`: 8 full layers in `per_token`, 40 sliding layers as a fixed 1,024-token
  allocation, 4 convolution states a layer, and the DSpark stage's cache. The shared
  `Architecture::kv_plan` needs a per-layer window term (or the arm's own plan), otherwise
  `fit.rs` reports only ~0.4M tokens on a 1M model.
* `--context` default 262,144, allowed up to 1,048,576 after validation; TP4 group serves
  one request at a time until fused batching is extended.

### 6. Drafter and vision, separately

* **DSpark:** the scheduler already has confidence-capped verification and a rank-256
  Markov-head shape, but for DeepSeek's 3-layer/block-5 stage. Inkling's 5-layer draft
  model, `mtp_config.block_size = 8`, `markov_head_type = rnn_hidden`, `dflare_norm`,
  `k_proj_ctx`/`v_proj_ctx` and `markov_head.rnn_proj`/`markov_w1` need a device
  drafter and a `DraftDevice`. It can be deferred; the backbone serves without it.
* **Vision:** the 23-layer ViT and merger can be deferred for a text-only bring-up. It is
  only 0.655 GB, but the patch-20 tower and the image-token path are a separate kernel and
  preprocessing project. The dynamic host ignores vision; a device arm must not claim it
  until it runs.

### 7. What runs today

* **Serving (TP1, f32 host reference).** One request against the released 367 GB checkpoint
  answers end to end. The first measured request — one prompt token plus one generated token —
  took 165.3 s; after the routed experts were spread over the machine's cores and the MXFP8
  rows decoded inline instead of through a one-byte slice per value, the same request took
  **10.6 s** and produced the same token. The eight MXFP8 layers were 15.8 s each before the
  change and are 0.22 s after. `SPARKFLASH_INKLING_TRACE=1` prints the per-stage split.
* **Routed experts and the router, verified on hardware.** `kernels/inkling_moe_sm121.cu` adds
  the two expert weight formats and the sigmoid router, and all three pass parity against
  `reference` on a GB10. The formats differ from DeepSeek's in what a scale multiplies:
  ModelOpt NVFP4 with a per-expert `scale_2 = amax / (6 * outer_scale_denominator)`, and MXFP8
  with a `ue8m0` power of two a 32 columns and no per-expert factor. Because both factors are
  linear, **an expert's scale rides on its own output**, `scale_2 * (W x)`, so one body serves
  every expert and no kernel needs a per-expert amax.

  The host forward now streams those products too (`expert_matvec`): codes and scales for one
  expert are read and multiplied straight into the result, so a routed expert costs a few MB of
  I/O instead of a four-gigabyte dequantised buffer, and the host and the parity harness call
  the same function.

* **Kernels, verified on hardware.** `crates/sparkflash-model-inkling2/tests/kernels.rs` runs
  each sm_121a kernel against `inkling2::reference` — the same functions the f32 host forward
  calls, so a kernel cannot pass by agreeing with a second hand-written oracle. All seven cases
  pass on a GB10 at one rank's geometry (12 query heads, 2 key/value heads of 128, 2,304
  intermediate columns, a 1,024-token window, 8 relative dimensions).

  The harness found five real defects, all fixed: attention read the learned `q_l2_clip`
  and `k_l2_clip` scales as one gain per head instead of per element, ignored
  `rel_l2_clip.scale` entirely, and clipped the raw query and key instead of the scaled ones;
  the relative bank was indexed as `[heads, distance, d_rel]` where the release stores
  `[d_rel, rel_extent]`; the prefill convolution read its taps in the wrong order (the
  decode path was right, so prefill and decode disagreed); and the softcapped SwiGLU dropped
  the cap that multiplies back after the tanh.

* **Vision (host reference).** `inkling2::vision` loads the released `model.visual` tensors and
  encodes a canvas to image tokens: patch-20 embedding, the class token, the learned 32x32
  position grid bilinearly resampled to the patch grid, 23 pre-norm blocks with layer scale,
  axial 2-D RoPE (a head's first half turns with the patch row and its second half with its
  column), the 2x2 merger and the two-layer adapter. `plan_grid` enforces the released
  40..4,096-pixel range and the patch-times-merge cell. Tests pin the merged token count, that
  the class token reaches the output, that a patch one row down produces different tokens (the
  row angle is live), determinism and the 4,096-token cap.

  The release ships no modelling code, so the choices a config cannot express — GELU's tanh
  approximation, pre-norm order, the merger's patch order, rope base 10,000 — are stated in the
  module. Nothing serves images yet: `ModelSession` has no image input, so the tower is an
  oracle and a loader until the request path carries images.

* **Converted weights.** `sparkflash convert --checkpoint DIR --tp-world-size 4` writes four
  rank files; `--inspect FILE --verify-weights` checks every slot digest;
  `SPARKFLASH_INKLING_CONVERTED_DIR` makes `serve` read a staged rank file instead of the
  sharded checkpoint, and the banner names whichever source is in use. The host reference
  serves identical logits from either (tests/convert.rs).

  Measured on the four-node cluster (DGX Spark, sm_121a), one 367 GB checkpoint:

  | rank | file bytes | weights | header hash | file digest | `--inspect --verify-weights` |
  | ---: | ---: | ---: | :-- | :-- | :-- |
  | 0 | 97,488,232,448 | 1,420 | `0x1f53298aa68134af` | `0x30fd86d74b3a4f84` | ok on sg1-jackson |
  | 1 | 97,488,232,448 | 1,420 | `0x04cc72543b1ad22a` | `0xa0f3013070d457d5` | ok on sg1-teilc |
  | 2 | 97,488,232,448 | 1,420 | `0x4d85e9f6f93b16ea` | `0xbe02d64ab0dec264` | ok on sg1-carter |
  | 3 | 97,488,232,448 | 1,420 | `0x8b743f3f1ddcc102` | `0x2de0634eee1e41ca` | ok on sg1-oneill |

  Each rank took 872-952 s to write and verify against the checkpoint on the converting node
  (399 GB of files, 381 GB of payload). Every file was then copied to a different node and
  re-verified there from its own header, digest and payload alone — the node that will read
  that rank is the node that checked it.

### 8. Open risk: the dynamic short convolution's tap order

The released checkpoint stores the full layers' convolution as `static_w` of shape
`[kernel, channels]` plus a per-token projection, and ships no modelling code. Two readings are
possible: row `k` is conv1d index `k` (PyTorch correlation, so `k` covers distance
`kernel - 1 - k`), or row `k` is already distance `k`. The host originally used the second for
the dynamic form while the static depthwise form (`[channels, 1, kernel]`) used the first; both
now use the PyTorch reading, which is the only one that makes the two convolutions of the same
layer agree.

That choice is observable: the first token for the raw completion "Hi" changed from `719` to
`Ġ` (a space, top-1 at -2.12 nats, next `Ġwherein` at -2.67) when the dynamic order was made
consistent. Nothing in the release settles which reading the exporter used, so this stays the
first thing to check against a reference implementation, and the kernels follow the host either
way — `inkling_short_conv_dynamic` and its decode form are held to the same oracle.

## Bring-up order

1. **Text, TP4, 256K, no drafter or vision.** Parse the config, add the parts and the host
   reference forward, load/converge the weights, run the 48-layer loop, and pass layer-by-layer
   parity for relative attention, short convolution, block residual, MoE and the softcapped
   head. This is the smallest milestone that can answer a prompt.
2. **1M and KV.** Add the sliding-window KV plan and the arm's own `KvPlan`; validate
   memory at 1M with a large prefill before advertising it. The difference between the shared
   plan and the real one is 40 GiB at 1M.
3. **DSpark.** Add the 5-layer drafter and RNN head; measure acceptance and per-rank reduce.
4. **Vision.** Add the patch-20 ViT, axial RoPE, merger and image preprocessing; then
   declare `Capabilities::ALL` rather than text-only.

Run the gates in `docs/adding-a-model.md`: `cargo test -p sparkflash-ops`,
`cargo test -p sparkflash-model`, `cargo check -p sparkflash-model-inkling2 --all-targets`,
the layer-by-layer `sparkflash_ops::parity` harness against the host reference, and the
released-config `engine_plan` test at TP=1 and TP=4. Capture the new profile's arithmetic in
`numerics/registry.toml` before claiming parity.

## Risks to pin early

* **Four-over-six** is not Qwen's ModelOpt NVFP4 path. The per-token amax, block-16 scale
  choice between 4 and 6, per-expert `amax_denominator`, and the exact MAE tie-break all
  need a reference test, not an approximation.
* **Relative bias** is a per-head 16-profile mix over 1,024 backward distances, zero outside
  `rel_extent`; the full layers still attend globally, with a zero bias past 1,024. A kernel
  that only handles the sliding window will be numerically wrong on the 8 full layers.
* **Block residual** changes the residual topology: 8 block summaries, 1 query on sliding
  layers and 3 on full layers, and no additive residual at a block boundary. This is the
  easiest mechanism to implement plausibly and get wrong.
* **Router bias.** The checkpoint carries `mlp.gate.bias` and `use_gate_bias: true`; the
  current parser only models `selection_bias` for `topk_method = noaux_tc` and otherwise
  defaults to softmax. Pin the exact sigmoid/top-k/bias/re-normalize order against the
  reference and vLLM before writing the kernel.
* **1M memory.** 11 GiB of headroom is enough for the weights and KV, not necessarily for a
  naive 1M prefill. Chunk attention and measure.

# Inkling 2 Small on four Sparks (TP4)

Status: **the device arm serves HTTP as a four-rank group on four Sparks: text and images,
chunked tensor-core prefill to a million positions, and concurrent requests through fused lanes
(§15), which draft with DSpark and keep their conversations. The numerics entry for the TP4
configuration is not registered.**


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
* `--expert-split intermediate` writes the other expert layout: every rank holds all 384
  experts at a quarter of their intermediate width, not 96 whole experts. It changes bits and is
  the layout served since 2026-11-16 (D122); §15, "Experts split by intermediate columns", has
  the measurements.

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
  drafter and a `DraftDevice`. It can be deferred; the backbone serves without it. The forward
  it has to reproduce (the taps are each block's in-progress residual partial and the final
  norm's output, not post-layer hidden states; the `rnn_hidden` head corrects the hidden before
  the shared lm_head) is [inkling2-small-drafter.md](inkling2-small-drafter.md); no Thinking
  Machines paper describes it, and SGLang's implementation is the source.
* **Vision:** done (§15, "Images"): the patch-20 tower runs on rank 0's device and the image
  rows reach every rank through the group's sum.

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
  | 0 | 97,488,232,448 | 1,420 | `0x1f53298aa68134af` | `0x30fd86d74b3a4f84` | ok on node1 |
  | 1 | 97,488,232,448 | 1,420 | `0x04cc72543b1ad22a` | `0xa0f3013070d457d5` | ok on node2 |
  | 2 | 97,488,232,448 | 1,420 | `0x4d85e9f6f93b16ea` | `0xbe02d64ab0dec264` | ok on node3 |
  | 3 | 97,488,232,448 | 1,420 | `0x8b743f3f1ddcc102` | `0x2de0634eee1e41ca` | ok on node0 |

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

### 9. The device schedule

`inkling2::device_plan` is the layer between the host reference and an executor: a typed,
CPU-inspectable list of every launch one pass issues, at one rank, with the image and entry each
launch loads. It is derived from the released configuration, a `TpPlan` and the served tensor
plan — never from a family name — so the rank-local widths and the weight slots cannot drift from
what the converter wrote.

Three properties are tests rather than claims:

* **Every served weight is read exactly once.** The union of the schedule's weight references is
  exactly the served tensor plan, with no tensor read twice, at TP1 and at TP4 on ranks 0 and 3.
  A weight the converter wrote and the schedule never names is paid for and unused; a weight
  named twice double-counts a residual.
* **Every named kernel is carried.** A step that names a `Part` must resolve to a catalogue entry
  on the same image and symbol (`every_named_kernel_is_carried_by_its_image`, both passes); a step
  with no `Part` is arm-private glue and must name a symbol the Inkling image exports.
* **The state is accounted.** `BufferPlan` sizes the residual stream, the block-residual inputs,
  the query/gate/relative/sink projections, the expert band, the logit slice, the four
  convolution histories a layer, the block-summary stack and the KV cache at the admitted
  context.

What the schedule says, in numbers:

| fact | value |
| :-- | :-- |
| launches for one decode step | 1,367 |
| a sliding layer | 28 launches |
| a full layer | 30 (three input norms, two dynamic K/V convolutions) |
| layer 0 | 27 (no attention block-residual bank: it consumes the embedding) |
| a block-boundary layer | unchanged (a summary push replaces the residual add) |
| TP4 per rank | 12 query heads, 2 KV heads, 96 whole experts, 50,256 vocabulary rows |
| state at 4,096 tokens | 77.9 MiB |
| state at 1,048,576 tokens | 8 GiB KV, 5.6 MiB convolution histories, 147 KiB summaries |

The KV figure is the per-layer window allocation: the 8 full layers keep the whole context and
the 40 sliding layers keep 1,024 rows each, which is 8 GiB instead of the 25 GiB a full-context
cache for all 48 layers would need.

Two design points the schedule makes explicit because the executor would otherwise get them
wrong silently. **Routed experts are expert-parallel**: the router is replicated, so every rank
selects the same global top-8 and computes only its own experts, and one all-reduce over the
residual width completes the mix — the schedule records that as `Collective::ExpertDelta`, which
is why the band is followed by the reduce and then a plain combine and not a local weighting.
**The attention output projection needs its own reduce**: `wo_ud` shards its *contraction* axis,
the heads, so a rank's dense result is a partial sum of every element of the layer's attention
output. The schedule records that as `Collective::AttentionDelta`, placed between the projection
and the short convolution that follows it, so the convolution mixes a complete row and its
four-tap history is the single-rank history instead of a quarter of it. A layer is therefore two
residual-width exchanges, attention and experts, which is the same pair the V4.1 TP4 chain posts.
**The static taps keep their stored layout**: the sliding layers store `[channels][kernel]` and
the full layers `[kernel, channels]`, the two decode entries read those two layouts, and the plan
records which is which per layer — a transposed tap table is the same size, in bounds, and wrong.

#### The two seam kernels

The schedule needed two launches no shared image has, so they are additions to
`kernels/inkling_epilogue_sm121.cu` rather than fits into an unrelated part:

* `inkling_kv_store` narrows the f32 key/value projections into the BF16 cache rows the attention
  body reads. The released model's cache is BF16, and the projection that feeds it is f32, so the
  narrowing is its own launch instead of a silent argument to the attention kernel.
* `inkling_residual_add` is `out = residual + delta`, in place. The block-attention topology has
  two residual forms and the boundary layers take the other one, so this is called on every layer
  but a boundary.

Both are exact operations — a round-to-nearest-wide conversion and an f32 addition — so
`tests/kernels.rs` compares them bit for bit instead of within the pack's relative bound. All
twelve parity cases pass on a GB10 (12/12: the ten previous ones plus the two new).

#### Two catalogue gaps the schedule found

* `b12x_hostw_gemv_m1_bf16w_f32o_sm121` — the host-weight M=1 GEMV with BF16 weights and f32
  accumulators — was exported by the DeepSeek image and **not catalogued**. It is the launch every
  one of this arm's dense projections needs: the query, gate and relative projections are clipped
  in f32 before use, so the BF16-store variant is a different vector. It is now a `DenseGemm`
  symbol; the catalogue's omission would have surfaced as a resolver refusal, not a wrong number.
* The released vocabulary is 201,024 rows, which is `64 * 3,141`: a multiple of 64 and not of
  128. The shared `(Embedding, Qwen4Exp)` and `(LmHead, Qwen4Exp)` shape rules require 128, so an
  `Architecture`-driven resolve refuses this checkpoint's embedding and head even though
  `q4e_embed` is a plain row gather with no alignment requirement. The device arm loads by symbol
  and is unaffected; the resolver path needs either a 64 rule with that argument recorded, or an
  Inkling entry. This is the next catalogue item, and it is why the schedule names the pair
  explicitly rather than pretending the entry resolves.

#### What is still missing

The schedule is not an executor. What remains, in order: the executor that walks it (kernel
loads by image and symbol, the three collectives, the position/state ring for the four
convolutions, the expert loop over the local selected experts); folding the router weight into the
MXFP8 expert body, which today has no scale argument (NVFP4 already takes it as `scale_2`); the
DSpark drafter; and the image path. The dynamic-convolution tap order of section 8 is still the
one thing a reference implementation could overturn, and the schedule follows the host either way.


### 10. The device arm, on hardware

\`inkling2::device\` walks the schedule and launches it. It owns no ordering: it matches each
[Op] with its [Target] and dispatches to the entry the plan named, so the plan is the single
description of a pass and the walk is a dispatcher. It opens one rank of a world of one and
refuses a larger world rather than run without the three collectives.

**The arm is the host reference, to the cache's rounding.** \`tests/device.rs\` runs both backends
on the tiny fixture, from the same weights and the same sampler, and compares the logits:

| pass | host best | device best | largest difference | row scale |
| :-- | :-- | :-- | :-- | :-- |
| prefill of 3 tokens | 35 (0.4437) | 35 (0.4434) | 0.00108 | 0.4514 |
| one decode step | 19 (0.6114) | 19 (0.6133) | 0.00193 | 0.6114 |

That is 0.2-0.3% of the row, which is what narrowing the key/value cache, the attention output
and the logits to BF16 costs; the bound in the test is 1%. The per-layer trace agrees to four
decimal places (layer 0: 0.00224/0.03323 device against 0.00228/0.03321 host).

**Two defects in the f32 reference, both found by the device's first two comparisons.** The host
was the oracle, so a difference was as likely to be its fault as the kernel's, and twice it was:

* **The block-summary stack accumulated across tokens.** \`forward_token\` pushed a summary a
  block boundary a token and never cleared the stack, so the second token read the first token's
  streams as its own summaries and the stack grew 8 rows a token. The device's push wrote past
  its arena and the process died; the host had been silently answering with an ever-longer
  summary stack. It is now cleared at the start of a token, in both walks.
* **A sliding layer ignored its own block-residual bank.** \`layer()\` computed the attention
  input from the bank, then normalized the *stream* instead and dropped the result. The released
  checkpoint stores a bank for every layer and the shapes encode one query on a sliding layer and
  three on a full one, so the bank is read; the sliding branch now normalizes the row it just
  computed. This changes the host's answers for every sliding layer, which is why the first
  token moved again.

**The schedule itself had one bug the arm found immediately: the head ran first.** The plan kept
one \`globals\` list, and a walk that ran it before the layers projected the *embedding* through
the head. The plan now carries a prologue (embed, gather, norm) and an epilogue (the head's block
residual, its norm, the vocabulary projection, the softcap, the slice gather), and the walk runs
prologue, layers, epilogue.

**Three more seam kernels, for the same reason as the first two:**

* \`inkling_widen_f32\` — the attention body writes BF16 (the released cache and output
  precision) and the projection that follows reads f32. Exact, so it is compared bit for bit.
* \`inkling_expert_mix\` — the mix is \`sum over slots of scale * partial\`. One body serves all
  three expert formats because they differ only in where the per-expert factor comes from: an
  NVFP4 amax over six times its denominator, nothing for MXFP8, and the fixture's dense BF16. The
  arm hands the mix the router's share times that factor.
* the attention body gained a \`cache_rows\` argument: an absolute slot addresses
  \`slot % cache_rows\`, so a sliding layer's cache is its window and is read as a ring. Without
  it every layer would need a context-length cache: 41 GiB at a million tokens instead of 8.

Both new parity cases pass on the GB10 (14/14: the twelve previous, the mix and the ring), the
ring case proving a ring cache reads the same rows as a linear one, and the cubin baseline is
regenerated (the attention image grows by 4,336 bytes, the epilogue by 9,640).

**TP4, and the three defects the four-rank test found.** `inkling2::collective` carries the
exchange (`all_reduce_f32`, `all_gather_f32`, the embedding row as a zero-padded sum) over the
shared `sparkflash_comm::tcp` framing, and `tests/device.rs` runs four converted rank files on
four threads of one GB10 over TCP loopback against the single-rank host reference. Getting that
test from "the logits are NaN" to 0.26% of the row took three fixes, all of them invisible at TP1:

* **The band ran every selected expert on every rank.** The router's indices are global and the
  band looped over all top-k slots, so a rank read `w13`/`w2` at a global expert offset inside
  its own 96-expert slice — past the end for three of the four slots — and mixed the result. The
  band now skips the slots outside `InklingDevicePlan::expert_range()` and addresses its weights
  and its per-expert amax table with the local index. The zeroed share is what makes the following
  all-reduce the whole exchange.
* **The key/value projections were sharded on the dimension axis, not the head axis.** `TpAxis::KvHeads`
  splits a *head count* (the shared rule: two heads over four ranks are one head each, read twice),
  but the plan was handing it `kv_heads * head_dim` rows. With eight key/value heads over four
  ranks both readings agree, which is why only the fixture (two heads, four ranks) refused to
  match: ranks 0 and 1 held `wk_dv` rows 0-3 and 4-7 — half a head each — while the kernel
  computed one head. The plan now splits the head axis and scales to rows, for the projections
  and for the four key/value short-convolution tables (`weight`, `static_w`, `weight_proj`),
  whose per-head blocks are 128 channels or one kernel width.
* **The dynamic convolution's tap projection asked for the world's heads.** `inkling_short_conv_dynamic`
  was launched with `num_key_value_heads * kernel` rows of `weight_proj` output while the rank's
  `weight_proj` holds one head a rank.

### 11. The four-node run, and what is still wrong

`sparkflash inkling-tp4` runs one rank of a four-host cluster. On the four DGX Sparks the four
converted ranks load, connect over TCP, and answer in lockstep: 97.48 GB of weights a rank in
96-123 s, a 5-token prefill in 0.6-3.4 s at TP4, and 111-147 ms a decode token, with all four
ranks reporting the same tokens and the same top logit. That is the arm running at the released
geometry, and it is the first thing the cluster has ever done with this checkpoint on the device.

**Its answers were not yet the model's.** The first run produced a repeating token. Diffing the
released transformers implementation (`modeling_inkling2.py`) against `forward.rs` and
`reference.rs` found three arithmetic defects, all now fixed: the dynamic short convolution
dropped its input term (`(conv + x) / sqrt(2k + 1)`, not `conv / sqrt(...)`), its tap tables
count the distance back rather than conv1d order, and a sliding layer normalized the raw stream
instead of its block-attention residual input - the last one silently dropped the released
softmax mix on 39 of 48 layers.

A fourth defect came from the checkpoint rather than the code. The NVFP4 per-expert factor is
`amax / outer_scale_denominator`, and the stored denominator is already six times the E4M3
maximum (the file carries 2,688 = 6 * 448). Dividing by the six again scaled every routed
expert's weight to a sixth. The evidence is arithmetic: the largest stored block scale is 448, the
largest code is 6, so `6 * 448 * amax / 2688` reproduces the stored `amax` exactly, and the
decoded matrix's rms (0.014726) matches the layer's BF16 projections (0.014731) instead of being
six times smaller.

**Still open.** With those fixed the model is coherent but not right: asked
"What is the capital of France? Answer with one word." it echoes the instruction, and the
top-8 after "The quick brown fox jumps over the lazy" is `" "`, `" The"`, `" the"`,
`" brown"` rather than " dog". The symptom is a model whose sublayer deltas are too small beside
the embedding stream, so it behaves like a copy-the-last-token model; the next step is a
layer-by-layer comparison against `modeling_inkling2.py` on the same checkpoint, not more
inspection of our own two implementations, which agree with each other by construction.

**What the arm still did not do at this point:** a chunked prefill (it loops the decode walk,
which is correct and slow), the DSpark drafter and the image path. The serving path for the device
arm landed afterwards; section 14 is that work and the tokens it served.

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

## 12. Independent arithmetic check: the forward is right, the completions are the model

> **Superseded by §15.** Both legs of this check read `w13` as a gate half followed by an up
> half; the checkpoint interleaves them row by row. The agreement below is real - two
> implementations of the same misreading - and so is the conclusion's error: the looping
> completions were the defect, not the model.

The defects in §10 were found by reading the release's `modeling_inkling2.py` against ours,
which cannot rule out a defect both share in *spirit* — a misread sign, a transposed weight, an
off-by-one in the tap order. So an independent numpy implementation was written from the HF
source alone (bf16/f32/NVFP4/MXFP8 read straight from the 24 shards, no torch, no SparkFlash
code) and the two residual streams were compared at every layer.

**Result: all 48 layers agree.** With one five-token prompt, 1 decode step, no cache:

| layer | numpy `x0, x1` | host forward `x0, x1` | `Δrms` |
|---|---|---|---|
| 0 | 0.176000, −0.022951 | 0.17600, −0.02295 | 0.0000 % |
| 5 | −0.33718, 1.1805 | −0.33720, 1.1805 | 0.0000 % |
| 24 | 0.11153, 0.40054 | 0.11153, 0.40055 | 0.0000 % |
| 40 | 0.95748, 0.28299 | 0.95724, 0.28299 | 0.0000 % |
| 47 | −0.64874, −0.62371 | −0.64883, −0.62371 | 0.0000 % |

Across all 48 layers the largest stream discrepancy is `0.270 %` (layer 29), the largest
`Δrms` `0.0023 %` (layer 42), and the median `Δrms` `0.0003 %`. The ends match as well:
`final_input` `0.696008, −0.16553` against `0.69601, −0.16560`, `final_norm`
`4.92836, −1.68709` against `4.92838, −1.68777`, and `raw_logits[0]` `20.3493` against
`20.34900`. The numpy reference's top-8 at the last position is
`220 (28.485)`, `151451 (27.925)`, `149876 (27.375)`, `149884 (26.902)`, `149878 (26.781)`,
`149935 (26.663)`, `11 (26.509)`, `150386 (26.476)` — token 220 first, which is exactly what
the host forward and the four-rank device arm both pick.

So the host forward, the kernels and the TP4 device arm are computing the released model. What
is left is **the model's own behaviour on raw text**: it is a chat-trained preview and greedy
continuation of `"The capital of France is"` really is `" of"` (`" Paris"` is not first). That is
not a defect in this arm, but it does mean "the completion looks right" is not a usable end-to-end
oracle — the oracle is the two-way trace, and it now has an independent leg.

Two habits this check produced:

* A trace line that is printed *before* the operation it appears to measure (here `embed_row`
  prints the raw row, then normalises) reads as a bug and is not one. Print the value you mean.
* A reference that reports a 10–40 % divergence at layer 1 and an exact match at layer 0 is
  usually wrong in the *reference*: layer 0 matched because its attention output was compared
  before the block-residual path ran. Compare the last stage first, then bisect backwards.

The numpy implementation lives at `~/inkling-bin/hf_reference.py` on node0, with
the clean trace at `pyref_trace.log`; it runs in about 500 s for one 48-layer forward.

## 13. The converted artefacts, pinned

The four rank files the arm ships and loads, as they stand after the head-axis split
(97,484,103,680 bytes each, 1,420 weights per rank):

| rank | bytes | sha256 |
|---|---|---|
| 0 | 97,484,103,680 | `d62c2d1b716ea89831b106caaf5ed254798df0f96b53c813b2f111467e6578c4` |
| 1 | 97,484,103,680 | `6ed4c11f022cf3396ffae78fb0180176e679e3e8313e10ee10a4bb909cd1f828` |
| 2 | 97,484,103,680 | `7c7a7c971f4dfcd4bd727b527c62c071b2ba398619906b1e12b94d9021b09b28` |
| 3 | 97,484,103,680 | `21b950f03853ebf2bf4efe8432980a4702595d73b4778561b880c838c055211d` |

Hash a file before a run and compare, not after: a rank file that predates a sharding fix still
loads, because the loader checks each slot's byte width against the plan and only the tensors
whose axis rule changed were resized — the rest of the file is simply the wrong arithmetic, which
no load-time check can see. Two of this arm's four worst hours came from that shape of mistake.

## 14. The four-rank server

The three pieces the previous revision of this section named are in, and the arm answers HTTP.

1. **Rank identity.** `sparkflash serve --rank/--world/--peer/--control-port/--converted` — the
   batch command's own flags, moved behind `ServeOptions` (`GroupOptions` in the arm's `group`
   module, parsed by `group_from_cli` in `serve.rs`). They
   parse unconditionally and refuse with a feature message when the binary has no `tp4-serve`.
   `--peer` carries each rank's collective address (`host:port`, rank order), `--control-port`
   defaults to 29700 at `peers[0]`, and `--converted` falls back to
   `SPARKFLASH_INKLING_CONVERTED_DIR`. The DeepSeek host group keeps `--tp4-*`: the two groups
   name their transports differently (bare addresses plus a base port, against per-rank
   addresses), and unifying them is not this change. A group is refused for any other pack, for
   `--synthetic`, `--concurrency` above one, `--continuous-batching`, and alongside
   `--tp4-rank` or `--recurrence-rank`.
2. **The execute loop.** Rank 0 owns HTTP, the tokenizer and the control channel; every rank
   opens its own converted file and device state. Before each step rank 0 broadcasts the frame
   the workers mirror — `Request{prompt_ids}` before its own prefill, `Token{id}` before its own
   decode — so no rank waits on a collective the others were not told to join. A session dropped
   before its next prefill broadcasts `End`; a completed request keeps its state on both sides,
   which is what the host driver's prefix reuse expects, and the next `Request` resumes or
   resets that state identically on every rank. A prompt longer than the admitted context is
   refused before it is broadcast. The model collective keeps its 900 s frame deadline; the
   control channel is the shared `tp4_control` Hello/Ready/Request/Token/End protocol.
3. **The codec.** Unchanged: `InklingCodec` was already registered, and the served path renders
   the chat template, encodes and detokenises exactly as the TP1 host reference does.

The rank machinery lives in `sparkflash-model-inkling2::group` - the backend, the session that
broadcasts before each step, and the worker mirror. The control frames and the hub/worker moved to
`sparkflash-comm::control`, where the DeepSeek host group reads them too. Both moves were
forced by the CLI's source-line ratchet (`sparkflash-ops/tests/one_stack.rs`, 34,000 lines):
the CLI was already at 34,245 before this work, and the server would have taken it to 35,274. It is
33,964 now, which is under the gate and not a licence to grow.

A group that names no `--context` admits 256K, not the released 1M window: every rank allocates
its KV at the admitted context, and 1M is 8 GiB a rank. `--context` still reaches 1,048,576, and
the launcher (`scripts/inkling-serve-group.sh`) now names it, which is what the window below was
served at.

### The first served tokens

Four DGX Sparks, the converted ranks pinned in section 13, greedy, context 4,096:

* **Batch reference** (`inkling-tp4 --text "The capital of France is"`): prompt
  `[371, 151451, 149879, 151194, 149884]`, generated
  `[149879, 149876, 151451, 149879, 149876, 151451, 149879, 149876, 151451]`, text
  `" of the capital of the capital of the capital"`; all four ranks agree, best
  `(149879, 27.875)`.
* **Served** (`POST /v1/completions`, the same prompt, `max_tokens` 9, temperature 0): the same
  prompt ids, the same fed token ids on rank 0 and on the workers (their
  `SPARKFLASH_INKLING_TRACE=1` lines are identical), `finish_reason: length`,
  `completion_tokens: 9`, and the same text.
* **Prefix reuse across requests.** `"The capital of France is"` for one token returns `" of"`;
  then `"The capital of France is of Paris"` reports `prompt_tokens 7`, `cached_tokens 6` and
  forwards one position on every rank; a fresh `"Hello"` reports `cached_tokens 0`.
* **Chat and streaming.** `/v1/chat/completions` answers the 27-token template (16 tokens,
  `finish length`), and `stream: true` emits role and content deltas and a terminal
  `finish_reason: length` chunk. The served surface reports `context_length 4096`,
  `world 4`, `fabric_mode "tcp tp4 device"` and numerics `unregistered`, which is the honest
  state until the configuration's arithmetic is captured in `numerics/registry.toml`.

The served TTFT for the 16-token chat prompt was 2.26 s (7.1 tok/s prefill) and decode 5.0 tok/s;
the batch walk's decode is 112-147 ms a token. The difference is the per-step control round trip
and a prefill that forwards one position a step - which is what the next two sections close.

### The chunked prefill

A prompt's positions are independent in every dense projection, in the routed band and in the
convolutions' taps, so they can be forwarded together. Three things are per *position* rather than
per launch, and each is why the chunked pass is a schedule of its own instead of a wider loop:

* **The convolution's seam.** A row's filter reads the three rows before it; a chunk's first rows
  read the ones the previous launch ended on. Each of the four convolution inputs therefore
  carries three history rows in front of its first row, filled from the decode layout by
  `inkling_conv_history_copy` and written back the same way. The two layouts hold the same
  numbers, so the copy rounds nothing.
* **The block-summary stack.** The stack is a token's residual chain, not a history, so every row
  of a chunk builds its own: the stack is indexed by row, and the boundary layer of every row
  pushes into it in one strided launch (`inkling_summary_push`).
* **The per-position scalars.** A full layer's log scaling is a function of the absolute
  position, so a chunk's rows differ. The table is computed on the host by the plan's own rule
  (`TauRule::at`) and passed beside the launch.

`InklingDevicePlan::build(..., Pass::Prefill, rows, context)` is the same step order as the
decode schedule with five entries swapped for row forms: `inkling_attn_decode` →
`inkling_attn_prefill` (one block a (row, head)), `inkling_short_conv_decode` →
`inkling_short_conv` (a `lead` of stored history rows), `inkling_short_conv_dynamic_decode` →
`inkling_short_conv_dynamic`, `q4e_rmsnorm` → `q4e_rmsnorm_rows`, and `inkling_kv_store`
writes the chunk's positions and lets the sliding layers' ring wrap each of them.
`SPARKFLASH_INKLING_PREFILL` selects the walk: `tensor` (the default since D117, §15), `chunk`
(the walk this section describes, with the decode step's bits), `rows` (the token-at-a-time one)
or `gemm` (the reordered tiled projection, below).

**The gate is the two-way trace, not the completion.** With `SPARKFLASH_INKLING_TRACE=1` the
chunk prints each row's post-layer stream (`[inkling] chunk rank R layer L row N x0=… x1=… rms=…`)
and the row walk prints one line a token. On the recorded five-token prompt the 240 lines - five
rows by forty-eight layers - are identical to the last printed bit, and the chunk generates the
walk's exact ten ids `[149879, 149876, 151451, 149879, 149876, 151451, 149879, 149876, 151451,
149879]` with the same best logit `(149879, 27.875)`.

Four defects stood between the first chunk and that trace, and each is a test now:

1. **The band's plan carries two different indices.** The gate/up projection reads the *token's
   row* of the stream and writes the pair's *band slot*; the down projection reads the slot it was
   activated in. One number for both read the wrong activation for every row but the first, which
   silenced the rank's own experts and still produced fluent text.
2. **A boundary takes the convolution's output.** At a block layer the sublayer output *is* the
   stream, and the decode walk's convolution writes in place, so the chunk's copy has to come from
   the convolved buffer and not from the projection.
3. **f32 addition is not associative, and the taps were accumulated newest-first** where the
   decode walk accumulates oldest-first. The terms were the same and the bits were not: the traces
   parted at layer four and the greedy token flipped. Both convolution entries now mirror the
   decode entry's order.
4. **A tensor-core projection cannot keep the walk's bits.** The tiled `q4e_gemm_bf16` sums a
   row's inner dimension in its own order, and any true tensor-core product would round the
   activation to TF32's ten bits or BF16's eight before multiplying. The residual stream carries
   that rounding: with every projection on the tiled kernel the chunk still picks the walk's own
   token `149879` and then answers `[149879, 151194, 149884, ...]` where the walk answers
   `[149879, 149876, 151451, ...]`; leaving only the dynamic convolution's tap projection on it
   moved the prefill's own token as well, which is how that projection was found. The chunk
   therefore takes
   `q4e_gemv_bf16`'s **batched** walk (up to eight rows a launch), which the kernel documents and
   `crates/sparkflash-model-inkling2/tests/prefill.rs` asserts to be bit-identical to its
   one-row launches. `gemm` is kept for the comparison below and is not a served path.

Measured on the four Sparks, greedy, batch arm (`scripts/inkling-batch.sh`):

| prompt | rows | chunk | ratio |
| --- | --- | --- | --- |
| the recorded five ids | 627-715 ms (128 ms a position) | 264-292 ms | 2.4x |
| 256 ids | 37,995 ms = 6.7 tok/s | 7,455 ms = 34.3 tok/s | 5.1x |

The chunk's throughput is bounded below by the routed band's weight traffic rather than by its
launches: eight of ninety-six local experts is 7.6 GiB a rank a position, and a chunk of five rows
selects nearly the same union of experts as five positions do one at a time. What the chunk buys
is the launches - the schedule's 1,367 go from one position to one chunk - and the weight reads
the band shares across the rows that picked the same expert.

### A million tokens

`--context 1048576` is the released window, and the launcher now names it. The arithmetic: eight
full layers, two KV heads of 128 values, f16 each, k and v - 8 KiB a token a rank, so 1M positions
is 8.04 GiB a rank. The four rank files are 97,483,910,480 bytes each (91.4 GiB of weights), so a
rank's whole state is 99.4 GiB of the engine's 113.7 GiB budget. The sliding layers' caches are
their 1,024-token windows and do not grow with the context.

Served at that window (`CONTEXT=1048576 scripts/inkling-serve-group.sh launch`): rank 0 opened
97,483,911,488 weight bytes in 97 s at the time (the load is now 15 s; see "Loading a rank" below),
the card reports `context_length 1048576` and
`max_output_tokens 32768`, and `scripts/inkling-serve-probe.py` measures

| prompt | TTFT | prefill | decode |
| --- | --- | --- | --- |
| the recorded five ids | 0.99 s | 5.1 tok/s (overhead-bound) | 7.1 tok/s |
| a 20-token chat template | 1.50 s | 13.0 tok/s | 6.9 tok/s |
| 258 tokens | 8.50 s | 30.3 tok/s | 5.4 tok/s |

The five-token prompt's time to first token is not prefill: the batch arm forwards the same five
ids in 264 ms, and the rest belongs to the session's first request, which is where `open_session`
allocates and clears the 8.04 GiB of cache and the chunk buffer. The 258-token prompt is the
honest number: 30 tok/s against the row walk's 6.7, which is the batch arm's ratio arriving at the
socket.

The session's other new cost is the chunk buffer: 128 rows of every scratch buffer, 99,572,416
bytes measured, of which 21,233,664 are the rows' summary stacks. (Since §15 the buffer is sized
to the prompt's first chunk - the next power of two, at least 128 rows - and grows on demand up
to the widest chunk, so a short prompt still pays the 128-row figure.) It is allocated on the first
prefill and released with the session (`Drop for InklingDeviceSession`, which also gives back the
cache, the four convolution histories and the summary stack). A served request ends by dropping
its session, so that release is what keeps a long window servable across requests rather than
once.

### Two defects the server found

* **A CUDA context is per-thread.** The first request failed with
  `cuMemAlloc(524288) failed: 201` (`CUDA_ERROR_INVALID_CONTEXT`) from `open_session`. The
  backend retained the primary context on the thread that opened it, and `sparkflash-serve`'s
  host driver opens a sequence on its own worker thread; the batch walk never saw it because it
  allocates on the opening thread. `InklingDeviceBackend::open_session` now calls
  `Cuda::make_current` before `State::allocate`.
* **The served surface claimed no cluster.** The same first response reported
  `"fabric_mode": "none (host backend)"`, false for a four-host device group. `ModelBackend`
  gained `fabric_mode` (default `None`, so no other arm changes), the host driver uses it for the
  cluster snapshot and each response's transport block, and the Inkling group declares
  `"tcp tp4 device"`.

### What the server still does not do

One request at a time; no DSpark, so no speculation; no image path; the numerics entry is
unregistered. A worker that fails ends its process, and rank 0 then stalls on the next collective
until the transport's 900 s deadline rather than receiving a typed error - the protocol has no
worker-to-hub failure frame yet. The launcher used for these runs is
`scripts/inkling-serve-group.sh`; the four-node window's addresses are its defaults, and
`scripts/inkling-serve-probe.py` is the client that produced the tables above (`/v1/models` for
the card, then one streamed request, timing the first chunk against the rest).

(§15 closes the speed, the looping, one request at a time and the image path.)

The two-way trace is still the bit gate, but this section's last claim was wrong: the greedy
`" of the capital of the capital"` was the `w13` misreading that §12's reference shared, not the
model. A completion that loops on a plain prompt is a symptom worth chasing even when two
implementations agree.

## 15. The looping was `w13`; the prompt runs on tensor cores

### The defect

Every completion degenerated into a loop (`" of the capital of the capital of the capital"`), and
the independent numpy reference of §12 looped the same way, which read as a model trait. It was
the weights' row order. The checkpoint's `w13` is `[2 x moe_intermediate][hidden]` with the two
projections **interleaved row by row** - row `2c` is gate column `c`, row `2c + 1` is up column
`c` - and the released loader de-interleaves it (`Interleave(dim=1)`) before it chunks the gate
and up halves. Both of our legs read it as a gate half followed by an up half, so every routed
expert paired each gate with the wrong up. Nothing failed: the dense path, the router, the shared
expert and the residual chain were all right, and the experts' contribution was plausible noise,
which is why the text stayed fluent and only the long-range behaviour broke.

The fix is in the reading, not the conversion: the host forward's softcapped SwiGLU, the decode
band, the chunk band and the grouped MMA's epilogue all take the interleaved order, and the
converted rank files are unchanged. The numpy reference with the interleave (`pyref_interleave`)
and the device arm now agree again, this time on the model's own answer: both put token `152717`
first for the recorded five ids (28.92 in the reference, 28.875 on four ranks), and the device
walks generate the same 25 ids under `chunk` and `tensor`.

### The tensor walk (D117)

`SPARKFLASH_INKLING_PREFILL` unset is now `tensor`. It is the chunk walk of §14 with the two
heavy operations moved onto the tensor cores, reusing the workspace's kernels rather than adding
an Inkling form:

* **Dense projections** take the shared wide tile, `glm_prefill_bf16_wide` - the body GLM's and
  Dory's prompt projections take - whenever the projection's width and inner width are whole
  32-column tiles (`wide_fits`); the rest stay on the batched GEMV.
* **NVFP4 experts (layers 0-39)** take Qwen's grouped W4A16 MMA as Qwen runs it: `q4p_split_bf16`,
  the shared `moe_plan`, `q4p_moe_gate_up_capped` (the same kernel with Inkling's softcapped
  SwiGLU as its epilogue, `inkling_softcap_sm121.cuh`, shared with the decode band) and
  `q4p_moe_down`, writing the `[slot][hidden]` partials the pair walk writes. The router's choice
  never leaves the device: `inkling_band_shares` writes the shares and the plan's local ids.
* **MXFP8 experts (layers 40-47)** first stayed on the pair walk, since no grouped MXFP8 MMA
  existed in the workspace to reuse. They now take the same grouped MMA over E4M3 bytes (D120,
  "The MXFP8 layers on the grouped MMA" below); `SPARKFLASH_INKLING_MXFP8=pairs` keeps the pair
  walk.

Both MMA forms carry each f32 activation as a BF16 head and remainder (16 significant bits) and
accumulate in f32 in the tile's order, so this is a declared precision change, not a
reassociation. D117 records the decision and its measurement: on twelve teacher-forced cuts of a
6,000-token text, tensor against exact agrees on the top-1 token at 11 of 12 cuts, with KL at most
6.1e-3 nats and a mean absolute logit difference of 0.07-0.50 a cut. `chunk` stays as the
opt-out with the decode step's bits.

What the tensor walk keeps is chunk-width invariance, bit for bit: an output's sum depends on K
alone and never on which rows share its tile, so the same cuts at 4,096- and 8,192-row chunks are
identical in every logit. What it gives up is that a prompt row is the bits a decode step would
give it, which is why a one-position tail stays a chunk under `tensor` (a prompt's last row would
otherwise depend on where the chunks fell), and why a continued conversation does not resume on
the bits of a cold prefill until decode runs on the same forms.

### Chunk widths

`ChunkWidths` cuts a prompt into 4,096-row chunks and widens to 8,192 while at least 16,384
positions remain; `SPARKFLASH_INKLING_PREFILL_CHUNK=n` fixes the width and turns the boost off.
The session's chunk buffer is sized to the first chunk (the next power of two, at least 128 rows)
and grows on demand, so a short prompt does not pay for the widest chunk; at 8,192 rows it is about
6.4 GB a rank (778 KB a row), which still fits beside 1M of cache (§14's 99.4 GiB).

Four ranks, greedy, batch arm, second pass where two were run:

| prompt | walk | chunk | prefill | prompt tok/s |
| --- | --- | --- | --- | --- |
| 4,096 ids | chunk (exact) | 128 / 4,096 | 24.2 / 25.2 s | 163-169 |
| 4,096 ids | dense on tiles only | 4,096 | 18.4 s | 222 |
| 4,096 ids | tensor | 4,096 | 9.25 s | 443 |
| 6,000-token text | chunk (exact) | 4,096 | 33.5 s | 179 |
| 6,000-token text | tensor | 4,096 / 8,192 | 13.7 / 13.3 s | 438 / 451 |
| 16,384 ids | tensor | 4,096 | 46.2 s | 355 |
| 16,384 ids | tensor | 8,192 | 45.7 s | 358 |
| 16,384 ids | tensor | default (8,192 then 4,096) | 45.8 s | 358 |

The tensor walk is 2.5-2.7x the exact one. The 8,192 boost is worth about 1 % at 16K: past a few
thousand rows the MMA forms already read each expert's weights once a chunk, so a wider chunk
saves little, and the 16K prompt's extra time over 4K is attention over the growing context, not
weights. The boost is kept because it is free in bits and buys a little at long prompts.

That was before the MXFP8 layers took the grouped MMA (D120) and the row split's gathers went
under the compute. Since then a chunk's fixed costs (each expert's weights read once a chunk, a
collective's latency, a band's two halves) are a larger share of a long prompt, and the width
matters. The 93K prompt on four ranks under the row split (batch arm, 131,072-position context):

| chunk | prefill | prompt tok/s | lead's lowest MemAvailable |
| --- | --- | --- | --- |
| 4,096 fixed | 90.95 s | 1,024 | |
| default (8,192 while 16,384 remain) | 81.1 s | 1,148 | |
| 16,384 fixed | 76.8 s | 1,213 | 9.1 GB |

The served launch fixes 4,096 rows at the 1M window to keep its memory margin, so its 93K prompt
takes about 12 % longer than the default width would. The bits do not depend on the width (D117),
so a wider pin at 1M is a question of memory alone: it is the maintainer's call, and its margin
under a held 1M session has to be measured first.

Decode, unchanged in arithmetic, is 39-40 ms a step after a five-token prompt, 45 ms after 4K and
55-58 ms after 16K, against 112-147 ms when §14 was written. Three changes landed in between: the
collectives moved from the TCP frames onto the device through NCCL (`RunReduce`, the communicator
GLM's arm opens), the decode attention became the split walk in `inkling_attn_sm121.cu` (key
scales, parallel scores, then the softmax recurrence alone, with the serial body's bits), and the
sliding layers' caches became rings of their window plus the widest chunk.

### The doorbell, the grouped GEMV and the two-shot sum

All three reuse DeepSeek's code and none changes a bit: each run below generates the same ids and
the teacher-forced logits of the 6,000-token text are identical.

* **The doorbell.** With the `rdma` feature, the device leg's rounds of at most 256 KiB go over
  DeepSeek's direct RDMA write exchange (`sparkflash_kernels::write_exchange`, the GB10
  doorbell) on the null stream, and wider rounds stay on NCCL. Decode is 37-38 ms a step against
  NCCL's 39-40. The first run's five-token prefill took 2,097 ms: the ranks finished loading
  minutes apart and the first exchange round absorbed the skew. A host-leg meeting at the end of
  `open_rank` now takes it, and the same prefill is 109 ms.
* **The grouped GEMV.** The narrow projections the tiles do not take ran as 8-row launches on one
  or two blocks: 8.8 s of GPU time in a 4,096-row prompt. The groups now go on the grid's y axis
  and take 0.3 s. The chunk walk's 6,000-token text went from 33.5 s to 28.8 s, the tensor walk's
  to 12.16 s, and a 4,096-id prompt under `tensor` from 9.25 s to 8.2 s.
* **The two-shot sum.** A wide sum is DeepSeek's pattern: scatter, sum this rank's slice in rank
  order, then gather the summed slices. It moves half the bytes. The 4,096-id prompt is 7.85 s,
  522 prompt tok/s.

### Concurrency: fused lanes

`LANES=N` (`--concurrency N --fuse-batch`) serves N requests at once through the workspace's
`FusedEngine`, the engine Qwen and GLM use (Dory's recurrence group has its own scheduler).
Requests are admitted within a 300 µs window. A prefill
runs alone, and the decodes due together run as one multi-row step that reads each weight once for
every lane. Rank 0 broadcasts each lane's prompt (`LaneRequest`) and each fused step's members in
its own order (`LaneStep`), so every rank steps the same rows in the same order. A lane decodes the
bits it would decode alone: every request at widths 2 and 4 returns its width-1 text.

| width | aggregate | per request | time to first token |
| --- | --- | --- | --- |
| 1 | 23.2 tok/s | 23.9-24.5 tok/s | 0.16-0.20 s |
| 2 | 29.7 tok/s | 15.2-15.8 tok/s | 0.17-0.35 s |
| 4 | 39.0 tok/s | 9.9-10.6 tok/s | 0.18-0.64 s |

(Four requests, 64 greedy tokens each; width 1 runs them one after another;
`scripts/inkling-lanes-probe.py --widths 1,2,4`; plain decoding, before the three changes below.)

#### Lanes draft, keep their conversations, and back their caches as they grow

Serving now defaults to four lanes (`LANES=4` in `scripts/inkling-serve-group.sh`; `LANES=1` is
the serial engine). Three changes make that the right default at a 1M context:

* **Lanes draft together.** With a drafter loaded (DSpark), the lanes that need a round in the
  same step draft their chains and verify them in one walk (joint rounds,
  `sparkflash_runtime::rounds`; `SPARKFLASH_JOINT_ROUNDS=off` runs each lane's round on its own).
  The runtime shares the verify rows across the lanes by DeepSeek's row share, each row to the
  draft likeliest accepted while it pays. Each lane accepts from its own rows, so its tokens are
  its alone-run's bits. A lane's accepted run goes back to its handle whole (multi-token runs,
  [engine/03-runtime.md](../engine/03-runtime.md)), so the lanes need their next round at the same
  step.
* **Lanes keep their conversations.** The host's prefix pool (`HeldPool` in
  `sparkflash-serve/src/host.rs`) is shared by every worker. A finished request parks its session
  in its slot (`ModelSession::park`, `Work::Park`), and the next request claims the slot whose
  kept tokens are its longest prefix, else the least recently used one. A kept lane continues: on a
  group, rank 0's `LaneRequest` carries `keep`, so the workers skip the renew and every rank
  extends the same cache. The sampler travels with the request, so a kept lane samples on every
  rank as rank 0 does.
* **Caches back as they grow.** `sparkflash_kernels::lane_memory` is DeepSeek's lane backing made
  shared (`deepseek-forward`'s `tp4_forward/lanes.rs` now runs on it): each lane reserves virtual
  memory for the whole context, and 2 MiB chunks from one pool are mapped (and zeroed) as its
  rows grow, in 2,048-row steps plus 1,024 rows past the position (tiles read past it and the
  drafter's block writes past it). Inkling's lane memory is the full layers' keys and values,
  their key-scale tables and DSpark's whole-context rings; the sliding layers and scratch stay
  per session. The pool is one full-context lane by default (10.1 GiB at 1M with DSpark) and the
  same on every rank, so every rank makes the same decisions. `SPARKFLASH_LANE_MEMORY_MB`
  (`runtime.lane_memory_mb`) sets it. When a lane needs chunks the pool lacks, the engine releases
  the least recently parked slot that holds a conversation (`LaneRelease` on a group) until it
  fits; a prompt or step that still cannot fit is refused on rank 0 before anything is broadcast.

Served at 1M positions with DSpark (adaptive depth, at most 7), four 160-token greedy answers:

| | one lane (lda) | four lanes |
| --- | --- | --- |
| each request alone (first after launch; not an A/B) | 28.7-37.0 tok/s | 35.9-50.6 tok/s |
| the four at once | | 10.6-10.9 tok/s each, 42.4 tok/s aggregate |
| second turns at once | | each continues its own lane (cached 27-30 of its prompt) |
| MemAvailable, lead, lowest | 7 GiB | 14 GiB |

Every answer, alone and together, is the one-lane build's text (the same digests). These four at
once barely beat one request's speed in aggregate, because each member then drafted on its own.
Joint rounds raise it to 54.5-56.3 tok/s greedy (51.9 sampled) against 43.2 for that build on the
same gate, and 42.9 with them off; every answer is still the same text. What concurrency also buys
is that four clients are served without queueing behind each other and that each keeps its own
conversation. Memory falls with it, because an idle lane backs nothing: the one-lane build held its
1M cache from the start.

Reclaim is checked with a pool that holds one short conversation but not two
(`SPARKFLASH_LANE_MEMORY_MB=120`): after `a1` and `b1`, `a2` runs cold (`b1` took `a`'s chunks) and
answers its one-lane text, and `a3` then continues `a2` (cached 47 of 56). Batch decoding is
unchanged: plain decoding's digest is `ec85a691b74bbc6b`, DSpark's ids equal plain's (greedy and
sampled), and the 93K text's ids equal the previous build's. The gate is
`scripts/inkling-lanes-gate.py PORT full|tiny|alone` (four topics alone and together, concurrent second turns, the
small-pool reclaim).

### Images (D118)

Rank 0 holds the vision tower on its device (653,318,144 bytes, BF16) and encodes every image of a
request before the prompt runs. The ViT is DeepSeek's geometry (1024 wide, 16 heads of 64), so its
GEMMs (with their bias, GELU and residual epilogues) and attention are DeepSeek's `opt_vision_sm121`
kernels, which `kernels/inkling_vision_sm121.cu` includes. That file adds Inkling's four: the
patch-20 patchify (each patch padded from 1,200 to 1,216 values), the LayerNorm, the interleaved
axial RoPE and the layer-scaled residual. The class row, the half-pixel resampling of the 32x32
position grid and the RoPE tables are made on the host by the reference's own functions and copied
in. The release folds the temporal axis into the stored patch projection (`[1024, 1200]`), so a
config's `temporal_patch_size` of 2 is a still image's shape and not a refusal.

The image rows reach every rank through the group's own sum: rank 0 writes its rows into a zeroed
f32 buffer, every other rank contributes zeros, and each rank copies the result over the
placeholder rows after the embedding norm, inside the chunk walk. A request's images travel in the
broadcast frame (`Request`, `LaneRequest`) as spans and keys, never pixels. `SpanCheck` refuses,
on rank 0 before the broadcast and identically on every worker, any span a rank could not prefill:
a request with images when no tower is loaded, a span whose stride, grid or delimiters are not the
codec's, one past the prompt's end, overlapping spans, and a stray placeholder outside every span.
`SPARKFLASH_VISION=off` (or `0`, `false`) serves text only and unset or `on` (`1`,
`true`) loads the tower; any other value is refused. The switch was
`SPARKFLASH_INKLING_VISION`, which is now refused with `SPARKFLASH_VISION` in the message. The
deployed site file that still sets the old name changes in the deploy that carries this rename,
and that deploy waits for the owner's go-ahead.

**Bits.** The device tower is held to the f32 host reference
(`tests/vision.rs::the_device_tower_matches_the_host_reference`: the released width, heads, patch,
merge and folded temporal patch, two blocks deep): the lowest token cosine is 0.999917 and the
largest difference 8.5e-3 of a largest magnitude 1.28 (6.6e-3 relative), against the test's floor
of 0.999 and 2e-2.

**Served** (four ranks, `LANES=4`, `CONTEXT=131072`, synthetic PNGs, greedy, the default reasoning
effort, so each answer follows a short reasoning turn):

| image | prompt | first token | answer |
| --- | --- | --- | --- |
| 400x300, left red, right blue | 122 | 0.92-0.97 s | "The left half is red and the right half is blue." |
| 256x256 green | 78 | 0.52-0.78 s | "Green" |
| 320x320 black disc on white | 94 | 0.54-0.55 s | "A solid black circle on a white background." |
| 320x320 quadrants | 117 | 0.61 s | "Top-left: red; top-right: green; bottom-left: blue; bottom-right: yellow." |
| 320x240, red bottom 40 rows | 88 | 0.57 s | "a horizontal band along the bottom, covering roughly the bottom fifth" |
| 240x320, blue right 40 columns | 88 | 0.56 s | "a vertical band along the right edge, covering roughly one-fifth" |
| 360x240, four 60-row bands | 93 | 0.58 s | "4 horizontal bands. The top band is yellow, and the bottom band is black." |
| 360x240, six 40-row bands | 86 | 0.55 s | "Five horizontal bands ... starting and ending with yellow" (wrong: six, ending black) |
| green, then halves (two images) | 169 | 1.02-1.18 s | "First image: green. Second image: red and blue." |

The six-band miss is the model's count, not a lost row: the bottom-strip, right-strip and
four-band images show that the last cell row and column reach it. Decode runs at 24.6-25.9 tok/s
behind an image, the text rate. A second turn over the same image ("Which of those two colors is
warmer?") answers "Red", and the first request sent again returns its first text. Run together,
two and four image requests each return the text they return alone (width 4: 13.8 s wall,
28.6 tok/s aggregate). `scripts/inkling-image-probe.py` produced the table (`--geometry` for the
layout rows); it draws its PNGs in pure Python, so it needs no image library.

`reasoning_effort` cannot turn the reasoning off yet: the shared API parser accepts DeepSeek's
integer 1-100 and the named levels, so Inkling's `"none"` (and its fractional 0-0.99 scale) is
refused before the prompt builder sees it.

### Loading a rank

The rank file was read through its `WeightSource`: one buffered `read_exact` a tensor, on one
thread, into a fresh host buffer, then one `cuMemAlloc` and copy each. That ran near 1 GB/s (97 s
for rank 0) and left about 19 GiB of the file in the page cache beside the weights it had been
copied into, with 3 GiB free on rank 0 at the 1M window. That launch's image probe saw one 18.6 s
first token among 1.5-2.8 s ones; the direct-read launch did not repeat it (1.4-3.8 s), though one
probe is not proof the cache was the cause.

The served ranks now load the way the GLM and DeepSeek ranks do
(`InklingDeviceBackend::open_rank_file`): the whole data section into one allocation with the
shared `upload_file_parallel` (16 MiB `O_DIRECT` reads, eight in flight), and each tensor a
pointer into it. A tensor past one 512 MiB chunk is several slots; the writer places each on the
next 256-byte slot boundary and the chunk is a multiple of that, so a tensor's slots abut and it is
one span. `RankFileSource::tensor_span` checks that rather than assume it, and every span is
checked against the plan before the allocation. A checkpoint source (the tests' fixtures, the host
reference) still uploads a tensor at a time through `open_rank`.

| at the 1M window | before | parallel direct read |
| --- | --- | --- |
| rank 0 open (91.4 GiB of weights) | 97.9 s | 15.3 s |
| ranks 1-3 open | not recorded | 14.6 s |
| launch to `/v1/models` | 110 s | 22 s |
| rank 0 page cache after load | 19 GiB | 3 GiB |

The disk reads 11.8 GB/s on one direct stream, so the read alone is about 8 s of the 15; the
rest (not split out here) is the host-to-device copy after each 1 GiB window, which the shared
reader does not overlap with the next read, and the open's own setup. The four-rank device test
opens through the new path and still answers like the host reference.

### Holding the conversation across a re-spelled reply and a retry

An agentic client on the 1M window stopped at an 84K-token turn: every request ran a cold
prefill of the whole conversation (7-10 minutes on the exact attention walk), the client gave up
first, and its retry started cold again. Two things broke the held sequence:

* **The reply came back in other ids.** The model had written `ĠRibb` + `on`; the client sent
  the reply back as text and the tokenizer spelled it `Ġ` + `Ribbon`. The bytes agree, the ids do
  not, and the host driver's reuse needed the prompt to extend the held ids exactly, so the
  79,325 held tokens were dropped. The host driver now matches the held tokens by their bytes
  (`HostOptions::reuse_by_text`): from the first id the two disagree on, the prompt must spell
  the held ids' vocabulary spellings byte for byte, ending on one of its own id boundaries, and
  the codec's decode of the result must equal its decode of the prompt. The sequence then
  continues from its own ids and forwards the rest. The model reads the ids it chose for that
  span, not the ones a cold prefill would read. A request with images never takes this path.
  `SPARKFLASH_REUSE_BY_TEXT=0` turns it off (any value other than on/off is refused at start-up);
  `scripts/inkling-env.sh` forwards it to every rank.
* **A request whose client left threw its work away.** A finished prefill whose first send
  failed dropped the session (the group's `End`), so the retry of the same prompt began cold. A
  request that ends because its client stopped reading now keeps the sequence and the step after
  its last token; a retry whose prompt is exactly that sequence, under an equal sampler, answers
  from the step without a prefill or a new seed. A backend failure still drops the session, and
  a queued request whose client is already gone is skipped before it touches the held one.

On the four-rank group: a completion forced to write `ĠRibb` + `on`, then a prompt that sends
` Ribbon` back, reported 10 of 13 prompt tokens cached and forwarded 3. A 6,331-token chat
abandoned 4 s into its prefill, then retried, reported 6,331 cached and the same message as a
cold run of that prompt (10.7 s including the abandoned prefill's remainder, against 14.5 s
cold).

### Fast attention: prompts on the tensor cores, decode split by keys (D119)

The 84K turn above ran at 115 prompt tokens a second, and its decode at about 5 tokens a
second. Past a few thousand positions both are attention: a prompt row's attention grows with
every position before it, and a decode row's exact walk is one serial recurrence a head, so a
full layer at 93K positions is 2,900 dependent 32-key tiles on 12 blocks. With
`SPARKFLASH_INKLING_ATTENTION` unset, both now take a fast walk.

**Prompt chunks** run on the tensor cores, reusing the body Dory's prompt attention runs rather
than adding an Inkling loop:

* **The body** is `shared_attn_mma_body` in `kernels/shared_attn_mma_sm121.cuh`, which was
  `dory_attn_mma`'s (Dory's cubin is byte-identical after the move), on the `mma.sync`,
  `ldmatrix` and `cp.async` helpers of `opt_mma_common_sm121.cuh` that DeepSeek's, GLM's and
  Qwen's tensor-core kernels share. A block is one kv head and 16 query positions, a warp a query
  head of its group, so one shared key and value tile feeds all six heads; the scores and the
  value sum are `m16n8k16` BF16 products accumulated in f32 over 32-key tiles, with q and the
  softmax weights carried as a BF16 head and remainder (16 significant bits), and the softmax is
  f32 in key order.
* **Inkling's arithmetic is the policy** (`InklingAttnPolicy` in `inkling_attn_sm121.cu`), and its
  inputs are the exact walk's own: the query side is `inkling_attn_query`'s result
  (`inkling_attn_prepare` writes it once a chunk), each key's clip gain and value scale are
  `inkling_attn_key_scales`' over the slots the chunk sees, the sink opens the softmax, and the
  output is the sum over the denominator times the gate's sigmoid. The relative bias of a key
  within `rel_extent` of the block's rows is one more 16 x 16 x 48 MMA a warp a tile, the row's
  relative state split as q is.

A one-position tail stays a chunk under `tensor`, so it takes the tensor-core walk too.

**A decode row** splits its keys ("flash-decoding", the partials-then-merge order of Dory's
`dory_attn_split` and `dory_attn_combine`). The key scales and the scores are the exact walk's
launches; then `inkling_attn_part` takes a block for each kv head and run of 256 keys, writing a
partial per query head of the group (the run's largest score, its sum of `exp(score -
largest)` and the value sums weighted by those terms and the keys' value scales, a value row
read once a group), and `inkling_attn_merge` merges a head's runs in run order with the sink,
divides and applies the gate. What changes is the order of the denominator's and the value
sums' additions, and that each run's terms are taken against its own maximum. The order is fixed
by the row's key count alone. A lane row (several sessions in one step) keeps the exact mix.

The cache is the same bytes under every walk, the sliding layers' rings included.

`SPARKFLASH_INKLING_ATTENTION` is `fast`, `split` (the exact walk), `serial` (its reference) or
`check`. Unset follows the prompt walk: `fast` under `tensor`, and `split` under `chunk`, `rows`
and `gemm`, so `chunk` keeps the decode step's bits. Unset on a shape the tensor-core walk is not
built for (a head width other than 128, a relative state past 16, more than eight query heads a
kv head, or past the shared memory) stands down to `split` with one printed line. `fast` or
`check` named there is refused, as is any other value. `check` serves the exact walk's output,
runs the fast walks beside it on every prompt chunk and decode row, and prints each layer's
largest and mean difference (`[inkling] attention check rank R layer L positions a..b: ...`):
the per-layer bound on a real prompt, every layer from exact inputs. The batch arm reads a long
prompt from a file (`--text-file`, the same path on every rank), since an 80K-token text is past
what one argument may hold.

**Checks.** `tests/prefill.rs` holds both walks to the exact ones:

* The tensor-core walk runs against `inkling_attn_prefill` on full and sliding layers, a ring, a
  long prefix, sinks, the tau rows, `d_rel` 0 and 16, and a short `rel_extent`. Every output is
  within one BF16 step of the exact one (or within 1e-5 of the largest), and 99.6-99.7 % carry
  its exact bits.
* The split-key walk runs against `inkling_attn_mix` for one row: a full layer over three passes
  and over one pass of 157 runs, one whole run, the first position, and a ring past its wrap over
  two passes. It is held to the same bound, and 99.5-100 % carry the exact bits.

On the four ranks:

| 93,159-token prompt, then 24 greedy steps | exact (`split`) | fast (unset) |
|---|---|---|
| prompt | 115 tokens/s (13.5 min) | 772 tokens/s (120.6 s) |
| decode step after it | 187-189 ms (5.3 tokens/s) | 47-49 ms (20.4-21.3 tokens/s) |
| the 25 greedy ids | — | the same 25 as exact decode on the same prompt |

The decode row's attention was 148 of its 188 ms. The split walk reads each full layer's keys
twice and its values twice (key scales, then scores, then the value sums), about 0.7 ms a full
layer at 93K positions, so the step is now within about 8 ms of its 40 ms floor at a short
context.

**Distance.** Teacher-forced on twelve cuts of the 6,000-token text (the D117 method), the
tensor-core prompt walk against the exact one, both under `tensor`: top-1 agrees on 11 of 12,
top-5 overlap 4-5, mean abs logit difference 0.07-0.31 a row (max 2.11), **KL at most 1.2e-2
nats**. `check` on the same text gives the per-layer bound, every layer from exact inputs:

* Every prompt chunk's layer, on all four ranks, is within one BF16 step of its value at its
  largest difference (3.1e-2 at 5.66 the worst), with a mean at most 5.5e-6.
* Every decode row's layer, over eight steps on rank 0, is likewise within one BF16 step (7.8e-3
  at 1.03 the worst), with a mean at most 7.0e-6.

### Full layers keep each key's scales

The split walk read each full layer's keys twice and its values twice. The first read of both
was `inkling_attn_key_scales` (each key's clip gain and its value's RMS scale), which depends
only on the key's own row. A full layer now writes those two numbers once, when the key is
stored, into per-layer tables at the key's absolute slot (`KeyTables`, `[kv_heads][context]`
f32 a table, 2 x 8 MiB a full layer at 1M). The split walk and the tensor-core walk read the
tables instead of recomputing them, so a decode step reads a full layer's keys and values once
each, and a prompt chunk no longer rescales the whole prefix before its walk. A sliding layer
keeps computing its window's scales per pass, since its ring overwrites slots.

This is the same arithmetic: the same kernel runs over the same row. After the 93,159-token
prompt, the 25 greedy ids match the deployed fast walk's.

| on the four ranks | before | key tables |
|---|---|---|
| decode step after the 93K prompt (batch arm) | 47-49 ms | 43-46 ms |
| 93K prompt (batch arm) | 120.6 s | 121.2 s |
| served decode, 2,223-token prompt | 26.1 tokens/s | 26.0 tokens/s |
| served decode, 78,540-token prompt | — | 22.5 tokens/s |
| served decode, 99,848-token prompt | 19.0 tokens/s | 21.4 tokens/s |

### DSpark drafting

`SPARKFLASH_INKLING_DRAFT=dspark` drafts with the release's own drafter (`mtp.safetensors` beside
the checkpoint's `config.json`, on every rank), whose forward
[the drafter note](inkling2-small-drafter.md) restates. `SPARKFLASH_INKLING_DRAFT_DEPTH` is the
most drafts a round verifies (default 7, at most the block's 7). How many it does verify is the
shared depth policy's choice from the drafter's probabilities, unless
`SPARKFLASH_INKLING_DRAFT_POLICY=fixed`. The arm has three parts:

* **Taps.** Every prompt chunk, decode step and verify row writes the target's eight features
  into a tap buffer: the residual entering layers 5, 11, 17, 23, 29, 35 and 41, and the
  final-normed row, each rounded to BF16.
* **Injection.** As rows commit, each of the drafter's five layers mixes the eight taps with its
  fusion row, normalises them, and stores their context keys and values. The sliding layers keep
  the last 1,024 positions; the full layer keeps the whole context.
* **The draft.** It is one forward of the five layers over the anchor and seven mask rows,
  attending to the context and to the block. The `rnn_hidden` Markov head then picks each draft
  in turn. Its output projection is folded into the vocabulary rows once, at load
  (`bf16(W_lm W_ho)`), so a step costs a 256-wide GEMV, not a 4,608-wide one.

The round is the verify walk prompt-lookup drafting uses, so every token is the token plain
decoding computes, whatever the sampler. A drafter costs a rank about 0.97 GB of weights, plus
its full layer's context keys and values (2 GB at 1M positions) and the tap buffer (604 MB at an
8,192-row chunk). Lanes refuse any drafter: a pending round's state is one session's, and a fused
step would walk it.

On the four ranks, greedy, the ids are plain decoding's at every depth:

| prompt | plain | depth 3 | depth 7 |
|---|---|---|---|
| repeating 5K prompt, 96 tokens | 39 ms a token | 22.5 ms (4.0 tokens in a 90 ms round) | 16.0 ms (8.0 in 128 ms) |
| 93,159-token text, 24 tokens | 44.5 ms | 41.4 ms (3.0 in 124 ms) | 45.8 ms (4.0 in 183 ms) |

At 93K, verifying an 8-row round costs four plain steps. Each row routes its own eight experts,
so eight rows read about 59 distinct experts against one row's 8, and every row walks the full
layers' keys. The drafter therefore stays off by default, and depth 3 is the choice when it is
on.

### Decode launches: fused projections, the residual's rows side by side

After the 93K prompt a decode step ran about 1,765 kernels on a rank. A single-row kernel that
finishes in a few microseconds leaves the device idle for about as long between launches. Two
changes take launches and serial work out of the step without moving a bit:

* **Same-input projections in one launch.** A decode row's consecutive projections that read the
  same input at the same width run as one `q4e_gemv4_bf16` launch of up to four matrices: a
  sliding layer's q, k, v and relative-state projections, then its gate and sink. That kernel
  computes each output row with the same `q4e_row_dot_bf16` and `q4e_warp_sum` as the
  one-matrix GEMV, so every output is the same number. A sliding layer goes from six projection
  launches to two, a full layer to four.
* **The block residual's rows side by side.** `inkling_block_residual` runs one block a query
  and walks the summary stack's rows one after another, two block reductions a row: about 21 µs
  a call, 96 calls a step. A call of at most eight (row, query) pairs, which is every decode row,
  now runs `inkling_block_residual_stats` (one block a stack row: its RMS inverse and score) and
  then `inkling_block_residual_mix` (the softmax, then the mix across the hidden width). Both use
  the one-block form's row body, softmax and mix in the same orders, so the output is
  bit-identical. A chunk's pairs already fill the device and keep the one-block form. A stack
  deeper than the kernels' 17 rows is refused; the one-block kernel skipped it.

Each was checked against the deployed build on the four ranks: teacher-forced logits identical
on all twelve D117 cuts, and the greedy ids of both prompts below equal.

| on the four ranks | key tables (before) | fused projections | and the residual |
|---|---|---|---|
| decode step, repeating 5K prompt (batch arm) | 39.0 ms | 38.5 ms | 37.4 ms |
| decode step after the 93K prompt (batch arm) | 44.3 ms | 43.8 ms | 42.7 ms |
| served decode, 2,223-token prompt | 26.0 tokens/s | — | 26.2 tokens/s |
| served decode, 99,848-token prompt | 21.4 tokens/s | — | 22.2 tokens/s |

### Programmatic dependent launch for the Inkling kernels

The `inkling_` kernels launch with programmatic stream serialization, as the `opt_`, `q4e_`,
`q4p_` and `dory_` kernels already did. Each one opens with `OPT_PDL_ENTRY`
(`griddepcontrol.launch_dependents`, then `griddepcontrol.wait`). A kernel's launch and block
setup then overlap the tail of the grid before it, but the kernel reads and writes nothing until
that grid has completed. `tests/pdl_entry.rs` holds every prefixed kernel to the wait.
`SPARKFLASH_PDL=0` turns it off for every prefix.

It was checked against the previous build on the four ranks, as above. Teacher-forced logits were
identical on all twelve cuts, and the greedy ids of both prompts were equal.

| on the four ranks | fused projections and residual | and PDL |
|---|---|---|
| decode step, repeating 5K prompt (batch arm, same session) | 37.45 ms | 35.57 ms |
| decode step after the 93K prompt (batch arm) | 42.7 ms | 40.7 ms |
| served decode, 2,223-token prompt | 26.2 tokens/s | 28.6 tokens/s |
| served decode, 99,848-token prompt | 22.2 tokens/s | 23.1 tokens/s |
| served decode, 127,603-token prompt | 20.7 tokens/s | 22.5 tokens/s |
| served time to first token, 99,848-token prompt | 156 s | 134 s |

### Strided score warps and held part scores

`inkling_attn_scores` gave every warp one run of four keys. A 93K-key full layer launched 5,824
blocks a KV head, and every warp prepared its eight queries for just those four keys. A warp now
prepares its queries once and walks the pass in runs of four, `gridDim.x * WARPS * RUN` keys
apart, with the next run's key loads in flight while it scores the current one. The host caps the
grid at 96 blocks a row (`SCORE_GRID`): one wave of four blocks an SM at two KV heads. Every
score is the same expression on the same operands whatever the grid; only which warp computes a
key changes.

`inkling_attn_part` read each of a thread's scores twice, once for the run's maximum and again
for the terms against it. A thread now holds its keys' scores (two a head) in registers across
both phases, which are the same operations on the same values in the same order.

| one full layer at the rank's shape (min of five) | scores before | after | part before | after |
|---|---|---|---|---|
| 1K keys | 8.9 us | 8.2 us | 18.4 us | 16.4 us |
| 5K keys | 22.5 us | 14.3 us | 18.4 us | 16.4 us |
| 93K keys | 337.6 us | 236.5 us | 296.8 us | 267.4 us |
| 262K keys | 951.9 us | 686.0 us | 805.7 us | 732.3 us |

Every output word of both micro-benches equalled the previous kernel's. On the four ranks the
teacher-forced logits were identical on all twelve cuts, and the greedy ids of both prompts were
equal.

| on the four ranks | PDL | strided scores and held part scores |
|---|---|---|
| decode step, repeating 5K prompt (batch arm, same session) | 35.61 ms | 35.14 ms |
| decode step after the 93K prompt (batch arm, same session) | 40.68 ms | 39.68 ms |
| served decode, 2,223-token prompt | 28.6 tokens/s | 28.75 tokens/s |
| served decode, 99,848-token prompt | 23.1 tokens/s | 23.8 tokens/s |
| served decode, 127,603-token prompt | 22.5 tokens/s | 22.4 tokens/s |
| served time to first token, 99,848-token prompt | 134 s | 135 s |

The served rows are single runs; the 127,603-token one moved by less than its noise.

### The small decode kernels: weights before the wait, rows held, run factors shared

Under PDL a kernel is launched while the grid before it is still running, and everything up to
`griddepcontrol.wait` overlaps that grid. A decode step's one-block kernels were latency bound
on loads they issued only after the wait: a cold weight read costs several microseconds, and a
row read twice costs its latency twice. Four kernels now load their weights before the wait
(`OPT_PDL_LAUNCH`, the weight loads, then `OPT_PDL_WAIT`), hold what they read in registers, or
share work across the block. Each one runs the same operations on the same operands in the same
order, so no output word moves:

* **`q4e_rmsnorm`** (shared with GLM, Qwen and the DFlash drafter) loads its gains before the
  wait and holds its row between the sum of squares and the scaled write, up to 32 values a
  thread; a wider group keeps the two-loop walk.
* **`inkling_logit_softcap`** ran one block over the gathered row's 200,058 entries. It now takes
  a grid of column blocks a row, each striding `gridDim.x` blocks apart.
* **`inkling_attn_merge`** computed every run's `exp(largest_r - largest)` once in every dim's
  thread. The block now takes the runs' maxima strided over its threads (`fmaxf` returns one of
  its operands, so the largest is the same in any order) and each run's factor once, into shared
  memory. Each dim's thread then folds the denominator and its value sum in run order with those
  factors, the `fmaf` chain it ran before. A row past 4,096 runs (over 1M keys) keeps the
  one-thread walk.
* **`inkling_router_topk`** rescanned shared memory for every one of its eight picks, checking
  each expert against the picks so far, and thread 0 took the block's best behind a second
  barrier. A thread now holds its experts' biased scores in registers and marks its chosen ones
  in a mask. Every thread takes the block's best from the warps' bests itself, one barrier a
  pick, with the warps' bests alternating between two buffers. The bias loads before the wait.
  The comparisons are the same, so are the picks: the first index holding the greatest biased
  score, with NaN and -inf never chosen.

| micro-bench, behind a 20 µs kernel under PDL unless noted | before | after |
|---|---|---|
| `q4e_rmsnorm`, one 4,608-wide row | 12.45 µs | 2.2 µs |
| `inkling_router_topk`, 384 experts, top 8 | 11.9 µs | 5.7 µs |
| `inkling_attn_merge`, 93K keys, 12 heads (alone, min of seven) | 47.4 µs | 13.6 µs |
| `inkling_attn_merge`, 1M keys | 763.1 µs | 190.4 µs |
| `inkling_attn_merge`, a sliding row | 3.2 µs | 4.2 µs |

Every output word of each micro-bench equalled the shipped kernel's. The router's inputs
included heavy ties, NaN and infinite logits, and rows with fewer live experts than picks. On the
four ranks the teacher-forced logits were identical on all twelve cuts, and the greedy ids of
both prompts were equal.

| on the four ranks | strided scores and held part scores | small kernels |
|---|---|---|
| decode step, repeating 5K prompt (batch arm, same session) | 35.23 ms | 33.81 ms |
| decode step after the 93K prompt (batch arm, same session) | 39.50 ms | 37.91 ms |
| served decode, 2,223-token prompt | 28.75 tokens/s | 29.92 tokens/s |
| served decode, 99,848-token prompt | 23.8 tokens/s | 25.8 tokens/s |
| served decode, 127,603-token prompt | 22.4 tokens/s | 24.9 tokens/s |
| served time to first token, 99,848-token prompt | 135 s | 130 s |

Two things measured nothing to gain. The lm_head GEMV, at 1.93 ms a step, reads 462 MB a rank
at the DRAM floor. The 93K trace's MXFP8 pair walks, 6.6 ms a step at first sight, were the
prompt's last chunk falling after the last tensor-core attention launch the split marks decode
from.

The served rows are single runs.

### Experts split by intermediate columns (served since 2026-11-16, D122)

Expert parallelism gives each rank 96 whole experts, so a decode row's eight experts land on the
ranks unevenly: one rank often holds three or four of them while another holds none. A profile
after the 93K prompt put the slowest rank's expert phase at 20.4 ms a step against a mean of
13.5 ms. The other ranks wait out the difference in the exchange, which is where rank 0's
largest share of a step went (8.4 ms).

`sparkflash convert --expert-split intermediate` instead gives every rank all 384 experts at a
quarter of their intermediate width, 576 of 2,304 columns: the gate and up rows and the down
projection's contraction columns. The all-reduce after the experts already finishes the down
projection's contraction, so no collective is added. 576 is a whole number of NVFP4 (16) and
MXFP8 (32) scale blocks, of the decode band's 16-column runs and of the prompt MMA's 64-wide
stages; a world that would cut a scale block is refused. The rank file records the split and the
loader takes it from there, so a server needs no flag. DeepSeek's routed experts take the same
slicing at TP4 (D2, D30). The files are the same size as expert parallelism's (97,484,185,600
bytes a rank).

The split changes where the down projection's sums are rounded: each rank's BF16 exchange row is
a partial sum over all eight experts rather than the whole of a few. Its bits therefore differ
from expert parallelism's, and it stays opt-in at conversion until the maintainer decides.

| on the four ranks | expert parallel | intermediate split |
|---|---|---|
| decode step, repeating 5K prompt (batch arm) | 39.0 ms | 33.2 ms (same 96 greedy ids) |
| decode step after the 93K prompt, with the fused projections | 43.8 ms | 38.6 ms |
| 93K prompt (batch arm) | 121.2 s | 121.1 s |

After the 93K prompt the greedy ids agree for 15 tokens and part at the 16th.
Teacher-forced on the D117 text against expert parallelism: top-1 agrees on 12 of 12 cuts, top-5
overlap 4-5, mean abs logit difference 0.08-0.37 a row (max 2.37), KL at most 1.19e-2 nats (mean
4.1e-3). That is the size of D119's distance, and 2,253,663 of the 2,400,696 logits differ.

Those figures are the token walk's. Under the tensor walk (D117, D120) the split's MXFP8 layers
first took the pair walk: the MXFP8 tile read a stage's two ue8m0 codes as four bytes at the row's
codes, and a 576-column down row holds 18 codes, so every other row's codes start two bytes into a
word. The tile now copies the aligned word that holds the stage's pair and takes the half the row
starts in, so it walks any row of whole 64-column stages, and the released split's rows read the
same bytes as before: the teacher-forced logits and the 93K digest are unchanged. Against the row
split (the files served since 11:05), the same build and launch:

| on the four ranks, tensor walk | row split | intermediate split |
|---|---|---|
| repeating 5K prompt | 6.49 s | 5.88 s |
| its decode step (88 steps after the first 8) | 26.03 ms | 25.1 ms (same 96 greedy ids) |
| 93K prompt | 83.4 s | 81.0 s |

Teacher-forced on twelve cuts of the 6,000-token text:

| | top-1 | top-5 overlap | mean abs logit difference a cut | max | KL max (mean), nats |
|---|---|---|---|---|---|
| intermediate split against exact | 12 of 12 | 4-5 | 0.059-0.354 | 2.06 | 1.53e-2 (5.57e-3) |
| intermediate split against the row split | 12 of 12 | 4-5 | 0.084-0.367 | 2.37 | 1.19e-2 (3.68e-3) |
| row split against exact (D117, for scale) | 12 of 12 | 4-5 | 0.057-0.339 | 2.03 | 1.55e-2 (4.58e-3) |

So the split sits as far from exact as the row split does. After the 93K prompt its greedy ids
agree with the row split's for 15 tokens and part at the 16th.

The maintainer took the split for serving on 2026-11-16 (D122). The gate before the swap ran the
served binary on both file sets in one window, with DSpark at adaptive depth 7 on the split:

| on the four ranks, served build | row split | intermediate split |
|---|---|---|
| repeating 5K prompt | 7.24-7.90 s | 5.28-5.39 s |
| its decode step, plain (two runs) | 26.00, 26.55 ms | 25.25, 26.33 ms |
| its decode step, DSpark d7 | | 13.19 ms (81 of 82 drafts accepted, ids equal to plain) |
| 93K prompt | | 67.5 s |
| its decode step, plain / DSpark d7 | | 29.26 / 24.96 ms (ids and logits digest equal) |

The split's digest repeats run to run (`ec85a691b74bbc6b` on the repeating prompt) and DSpark
emits plain decoding's ids and logits on it, so speculation stays lossless. Each host keeps the
row files as `converted-rows`; moving them back to `converted` restores the row split's bits.
The converter's default is unchanged (expert parallelism); the split is still named with
`--expert-split intermediate`.

### A verify's rows read the keys together; the residual's statistics hold their row

Two more changes move no bit:

* **A verify's rows walk the same keys at once.** A verify scores and sums all its rows over the
  same full-layer keys. The score grid gave each row its own 96 blocks, though, and the part grid
  put the rows on its slowest axis. Each row therefore read the keys and values from DRAM after
  the row before it had left L2. The 96 score blocks are now shared by the rows, and the part grid
  takes the rows on y and the runs on z. A verify's rows are resident together, and all but the
  first find the keys in L2. Which block or warp takes a key never changes its bits, and a decode
  step (one row) launches the grids it did.
* **`inkling_block_residual_stats` holds its row.** It read its query only after the previous
  grid ended and read its row twice, once for the sum of squares and once for the dot. A thread
  now loads its share of the query (a weight) before the PDL wait and holds its share of the row
  in registers for both passes. The loads are predicated rather than cut short by a break, so all
  of a pass's loads are in flight at once. The additions are `inkling_residual_row`'s in the same
  order. A row wider than 32 values a thread keeps the two-pass walk.

On the four ranks, in one session, the teacher-forced logits were identical on all twelve cuts,
and every run's greedy ids equalled plain decoding's:

| on the four ranks, same session | before | after |
|---|---|---|
| plain decode step, repeating 5K prompt | 33.85 ms | 33.77 ms |
| depth-3 verify, repeating 5K prompt | 85.3 ms (20.66 ms a token) | 84.7 ms (20.52 ms a token) |
| depth-3 verify after the 93K prompt | 110.3 ms (33.57 ms a token) | 105.0 ms (31.96 ms a token) |

DSpark against plain decoding on this build:

| prompt | plain | depth 3 | depth 7 |
|---|---|---|---|
| repeating 5K prompt, 96 tokens | 33.77 ms a token | 20.52 ms (4.1 tokens in an 84.7 ms round) | 15.41 ms (7.9 in 122.0 ms) |
| 93,159-token text, 24 tokens | 37.74 ms | 31.96 ms (3.3 in 105.0 ms) | 33.30 ms (4.6 in 153.2 ms) |

Depth 3 now beats plain decoding after the 93K prompt as well, by 15%. The drafter stays opt-in.
It costs a rank about 3.6 GB at 1M positions (lanes now take it; §"Lanes draft"), and its gain rests on how many
drafts the text accepts.

### The dense projections in a lossless packed form

A decode step reads every dense BF16 projection once, and at one row those GEMVs run at the
DRAM floor, so their bytes are their time. A rank reads 401 of them in a pass: the attention
projections, the router, the dynamic short convolution's tap projection and the head. That is
2,989 MB, the head alone 462 MB. A BF16 word spends 8 bits on its exponent, but the weights one
lane reads together sit within a few binades of each other.

The packed form (`kernels/bf16_packed_sm121.cuh`) follows `q4e_row_dot_bf16`'s walk: lane `l`
reads 16-byte chunks `l`, `l + 32`, ... in steps of four chunks, then perhaps two, then one. For
each step and lane it keeps three planes:

* each weight's sign and mantissa as a byte;
* its exponent as a 4-bit offset under the largest of the lane's 32 weights at that step;
* that largest exponent, once.

That is 12.25 bits a weight. A weight 15 or more binades under its step's largest (zeros, mostly)
is an escape: its offset reads 15, and its whole word sits in the matrix's escape list, by row
and column. A row's columns must be a multiple of 256, so every lane walks the same steps; every
dense projection here qualifies.

`q4e_gemv_packed` and `q4e_gemv4_packed` rebuild each lane's words in registers, patch in its
escapes, and hand them to `q4e_dot_chunk` in the grouping `q4e_gemv_bf16` and `q4e_gemv4_bf16`
use. A packed projection is therefore the BF16 projection's bits, from 77% of the bytes.

The arm packs at load, in place, a group of rows at a time through one scratch buffer the size
of the widest matrix but the head. Before a group overwrites its BF16 rows, `q4e_unpack_bf16`
rebuilds the group and counts the words that differ from those rows. A single differing word
refuses the rank. A prompt's tensor-core tiles read BF16, so `wide` and `gemm` unpack a
projection into the scratch first; at an 8,192-row chunk that costs nothing measurable.
`SPARKFLASH_DENSE_WEIGHTS=checkpoint` keeps the checkpoint's rows (`packed`, the default, packs), and any other value is refused (the switch is shared with every arm that packs, `sparkflash_ops::packed`; its old names `SPARKFLASH_INKLING_DENSE` and `SPARKFLASH_BF16_FORM` are refused with a pointer to it).

| standalone, one rank's slice (BF16 -> packed, every output word equal) | batch 1 | batch 4 |
|---|---|---|
| query, 1,536 x 4,608 | 63.4 -> 51.5 us (1.23x) | 1.23x |
| attention out, 4,608 x 1,536 | 61.9 -> 48.9 us (1.26x) | 1.25x |
| head, 50,176 x 4,608 | 323.4 -> 258.9 us (1.25x) | 1.23x |

On the four ranks, against the build before, the teacher-forced logits were identical on all
twelve cuts. The greedy ids were equal on both prompts, plain and with DSpark depth 3, and with
the bf16 opt-out:

| on the four ranks | BF16 rows | packed |
|---|---|---|
| decode step, repeating 5K prompt (batch arm, same session) | 33.82 ms | 31.64 ms |
| decode step after the 93K prompt (batch arm, same session) | 37.74 ms | 35.39 ms |
| depth-3 verify after the 93K prompt | 105.0 ms | 102.7 ms |
| 93K prompt (batch arm) | 121.5 s | 120.3 s |
| served decode, 2,223-token prompt | 29.92 tokens/s | 31.74 tokens/s |
| served decode, 99,848-token prompt | 25.8 tokens/s | 26.89 tokens/s |
| served decode, 127,603-token prompt | 24.9 tokens/s | 26.12 tokens/s |
| served time to first token, 99,848-token prompt | 130 s | 133 s |

The served rows are single runs; the BF16 column's are the small decode kernels' build. The
verify gains less than a plain step because each of its rows routes its own experts, so the
dense projections are a smaller share of its bytes.

### Fewer launches in the split-key walk: scored runs and the sliding window's cluster

The split-key walk scores a pass's keys into DRAM, reads them back for each run's partial, and
merges the runs. Two kernels now fold those launches together:

* **A long pass's scores and partials in one block.** `inkling_attn_scored_part` scores one run
  and takes its partial in the same block, with the scores held in shared memory. The scores and
  part kernels share their helpers with it (`inkling_group_queries`, `inkling_key_gain`,
  `inkling_score_load`, `inkling_group_score`, `inkling_part_run`), so every number is still
  theirs. The host takes it for a pass of at least 128 runs and a group of at most six heads:
  on the 93K-key full layer the kernel goes from 525.6 to 470.1 us (262K keys: 1,459 to
  1,310 us), but at 16K keys it is 7% slower.
* **A sliding layer's whole walk in one cluster.** A sliding layer's row sees at most 1,024
  keys: four runs of 256. `inkling_attn_window` runs a (KV head, row) as a cluster of four
  blocks, one run a block. Each block scores and takes its run's partial, and the merge reads
  the four partials through distributed shared memory. That one launch replaces the passes and
  the merge. One row's layer goes from 38.3 to 32.4 us with the cache flushed, and a four-row
  verify's from 50.7 to 31.7 us. Under Check the passes still run, for the exact mix.

Both are bit-identical. On the four ranks, the decode logits digests of the 93K prompt, the
repeating prompt and DSpark depth 3 all equal the two-launch build's:

| on the four ranks (batch arm, ms a token after the first) | before | scored | window |
|---|---|---|---|
| decode step after the 93K prompt | 35.52 | 35.35-35.57 | 34.78-34.91 (35.43 between) |
| decode step, repeating 5K prompt | | 31.71 | 31.24 |
| DSpark depth 3 after the 93K prompt | | 31.57 | 31.39 |
| 93K prompt | 120.3 s | | 121.8 s |

### Experts split by residual rows: expert parallelism's bits, a quarter of every pick a rank

The intermediate split above fixes expert parallelism's imbalance, but it moves where the down
projection's sums round. `sparkflash convert --expert-split rows` splits the same work without
moving a bit:

* **gate/up** is the intermediate split's: every rank holds a quarter of every expert's
  interleaved w13 rows (576 of the 2,304 intermediate columns, gate and up together), so it
  computes 576 activation columns of each of the eight picks.
* **The activation is gathered.** The ranks gather those runs in place on the device leg, and
  `inkling_join_pieces` lays them side by side as whole 2,304-wide rows. It is a copy.
* **down** is split on its output rows: a rank holds 1,152 of w2's 4,608 hidden rows with all
  2,304 contraction columns, and its block scales on the same rows. Every output element is the
  expert-parallel walk's dot product, in the same order.
* **The mix rebuilds expert parallelism's fold.** `inkling_expert_mix` takes the picks' expert
  indices. An expert-parallel rank owns experts `96 r` to `96 r + 95`, so for each of those four
  owners in turn the kernel sums the owner's picks in slot order. It then folds the four mixes
  from 0.0 in rank order, which is the order `exchange_sum` adds the ranks' rows. The ranks then
  gather their 1,152-row pieces, where expert parallelism ran an all-reduce.

The loader takes the split from the rank file. It refuses a split without the device leg, a BF16
expert layer, or pieces that are not whole 16-byte words. The files are the same size as expert
parallelism's (97,484,185,600 bytes a rank). The checkpoint's shards are only on the lead, so
the lead converts every rank (about 10 minutes each) and ships ranks 1-3 to their hosts.

On the four ranks the decode logits digests were equal to the deployed build's: the 93K prompt
(`724cc71b52c00a7e`, twice), the repeating prompt (`0fe1437d677108a4`) and DSpark depth 3 after
the 93K prompt. The greedy ids were equal too. In one session, A/B/A:

| on the four ranks (batch arm, ms a token after the first) | expert parallel | row split |
|---|---|---|
| decode step after the 93K prompt | 34.91 | 30.00, 31.26 |
| decode step, repeating 5K prompt | 31.32 | 26.13 |
| DSpark depth 3 after the 93K prompt | 31.39 (window build) | 31.83 (7 rounds of 104.6 ms) |
| 93K prompt | 120.7 s | 132.0 s, 132.7 s |
| repeating 5K prompt | 8.7 s | 9.6 s |

The prompt pays for the gathers. At an 8,192-row chunk each rank sends 566 MB a layer: the two
BF16 planes of its activation runs (151 MB to each of three peers) and its mix rows (38 MB to
each). Expert parallelism's two-shot all-reduce sends 226 MB.

A plain decode step gains 10-14% after the 93K prompt and 17% on the repeating prompt, because
every rank now reads the same quarter of each pick's expert bytes. A depth-3 verify routes 32
picks, which already spread fairly evenly over expert parallelism's ranks. DSpark therefore
gains nothing here, and after the 93K prompt plain decoding is as fast or slightly faster. The
split stays opt-in at conversion: the maintainer weighs the prompt's 10% against the decode
step's 10-17%.

### What a verify costs: thirty experts, and the launches around them

A depth-3 round verifies four rows (the anchor and three drafts) in one walk. After the 93K
prompt, traced on rank 0, that walk took 61.0 ms of device time against a plain step's 32.3 ms.
About 22.6 ms of the difference is expert weights. The four rows' 32 top-8 picks name about 30
distinct experts, and a plain step's one row names 8. Every distinct expert's bytes are read
once, whichever rows picked it, so this part is the routing's and stays. The rest was launches
and walks that did not need to cost what they did:

* **The band is steered on the device** (`run_steered_band`). It used to read the router's
  choice back to the host on every layer. From five rows up, `moe_plan` sorts the slots by
  expert, and the item walk reads an expert once for all of its slots. At four rows the 32 picks
  are nearly all distinct experts, so the plan saves nothing and its launches cost more.
* **The drafter's span attention runs as walks and a merge.** Each block walks one eighth of the
  keys for four queries of a key head, and loads and scores eight keys ahead of its fold. The fold
  is the only chain from key to key. The merge combines a query's eight walks.
* **Fewer launches for a few rows.** The projections that read the same input share a launch, as
  in the decode step: a sliding layer's six attention projections take two launches. The full
  layers' tau table is uploaded once per walk, not once per layer, because each upload waited on
  the stream. Hidden-width norms hold their rows.
* **The scored partial scores a key with a quad of four threads.** It used to take a warp a key.
  Thread `u` holds dims `u + 4m + 32 step`, so the warp tree's first three levels add a thread's
  own partials and its last two add across the quad. Every sum adds the same operands as the warp
  walk did. The value sums now use the whole block and skip the head test when every head is
  live.
* **The drafter's projections are packed** in the target's lossless form, except the block's key
  and value, which measured slower packed at eight rows. A verify leaves the convolution
  histories as they were, and the round's settle restores all of them from the kept tails in one
  launch.

None of these changes moves a bit. On the four ranks, after each change, the decode logits
digests and the greedy ids were unchanged, and so was every round's accepted count. The 93K
digest stayed `724cc71b52c00a7e` and the repeating prompt's `0fe1437d677108a4`. Rounds in ms:

| on the four ranks (batch arm) | 93K, depth 3 | repeating 5K, depth 3 | repeating 5K, depth 7 |
|---|---|---|---|
| host-steered band | 103.0 | 84.0 | 128.0 |
| steered band on the device | 90.1 | 68.1 | 108.3 |
| expert items from five rows | 92.0 | 68.2 | 101.5 |
| span walks and merge, tau table once | 80.6 | 67.0 | 99.1 |
| few-row fused projections, held norms | 78.6 | 64.7 | 97.1 |
| quad scores, whole-block value sums | 72.1 | 65.6 | 97.8 |
| packed drafter, one-launch restore | 70.4 | 63.3 | 95.0 |
| a token, now (plain) | 21.43 ms (29.57) | 15.32 ms (26.09) | 12.00 ms (26.09) |

The "expert items" row takes its four-row rounds from the build without the plan and its
eight-row round from the build with it, which is what five rows as the threshold runs. Every
number includes about 0.8 ms a token that the batch arm spends digesting each token's logits.
Serving does not pay it.

The drafter's full layer is what grows. Its span attention costs about 2.9 ms a round at 93K
and 14.5 ms at 500K. Each walk is bound by its chain of folds, not by bandwidth. Giving a warp
several queries measured slower, and an L2 prefetch of the next keys was 8% faster on the full
layer and slower on the sliding ones. The drafter's LM head (1.63 ms) is at its DRAM floor.
DSpark stays off by default. At a million positions it costs a rank about 3.6 GB.

### The MXFP8 layers on the grouped MMA (D120)

A trace of the 93K prompt under the row split put the MXFP8 layers' pair walk at 31 % of the
prompt: 41.9 s of 132 s. An MXFP8 layer cost about 915 ms of expert time an 8,192-row chunk, an
NVFP4 layer on the grouped MMA 34 ms. The pair walk is one launch an expert over the rows that
chose it, after the router's picks are read back to the host.

The MXFP8 layers now take the NVFP4 layers' grouped MMA (`q4p_moe_body` in
`kernels/qwen4_prefill_sm121.cu`) in an `MX8` weight form, `q4p_moe_gate_up_capped_mx8` and
`q4p_moe_down_mx8`. A stage holds 128 weight rows of 64 E4M3 bytes and the rows' ue8m0 codes,
two stages instead of three so the shared memory stays where it was. Each weight is formed as the
pair walk forms it, the E4M3 value times `2^(code - 127)` in f32, and then rounded to BF16. That
rounding is exact except under codes 1 and 2, where the smallest E4M3 step times the power of two
falls under BF16's subnormal step. The open reads every MXFP8 scale code (`admit_mxfp8_scales`)
and refuses a checkpoint that holds either, naming `SPARKFLASH_INKLING_MXFP8=pairs`;
Inkling-2-Small's codes are all admitted. The table's outer scales are ones, since MXFP8 stores
no per-tensor scale.

So the MXFP8 layers carry D117's declared precision: the activation as a BF16 head and
remainder, f32 sums in the tile's order, against weights that are exactly the pair walk's.
`SPARKFLASH_INKLING_MXFP8=pairs` keeps the pair walk, and under it every bit is the deployed
build's: the teacher-forced logits are identical and the repeating prompt's decode digest is
`0fe1437d677108a4`. `mma` named under any walk but `tensor` is refused, as is any other value.

Four ranks, row split, batch arm, in one session:

| on the four ranks | pair walk (deployed) | grouped MMA |
|---|---|---|
| 6,000-token text, twelve teacher-forced cuts | 13.0 s | 9.46 s |
| repeating 5K prompt | 9.71 s | 6.93 s |
| 93K prompt | 131.5 s, 131.8 s | 96.6 s |
| prompt tokens a second at 93K | 708 | 964 |
| decode step after the 93K prompt | 29.68, 29.45 ms | 29.50 ms |
| decode step, repeating 5K prompt | 26.13 ms | 26.04 ms |

The distance, teacher-forced on twelve cuts of the 6,000-token text (the D117 method): against
the deployed build, top-1 agrees on 12 of 12 and KL is at most 2.3e-3 nats. Against the exact walk
(`PREFILL=chunk`) the MMA is as far as the deployed build is: top-1 12 of 12 for both, KL at most
1.55e-2 nats (mean 4.58e-3) against 1.52e-2 (mean 4.55e-3), and a mean absolute logit difference
of 0.057-0.339 a cut against 0.056-0.339. The 93K prompt's 24 greedy ids and the repeating
prompt's 96 are the deployed build's.

### The row split's band in two halves, its gathers under the compute

Under the row split a layer's band makes two all-gathers on the wire: every rank's run of the
SwiGLU rows before the down pass (about 36 ms at 8,192 rows on NCCL) and every rank's run of the
mix rows after it (about 12 ms). Both sat on the run's stream, so the GPU waited through them.

A chunk of 512 rows or more now runs its band as two halves of its rows (`run_band_halves` in
`device/chunk.rs`). The run's stream computes gate_up for half 0, then for half 1, then down and
mix for each half once its activation is in, and then each half's join once its mix rows are.
The four gathers go on a side stream in the same order on every rank: half 0's activation under
half 1's gate_up, half 1's under half 0's down and mix, and half 0's mix rows under half 1's down.
The side stream waits for the run's stream before each gather, and the run's stream waits for the
last gather before the next layer's exchange, so the group still issues its collectives one at a
time and in one order.

Each half is planned and run as a chunk of its own rows would be. A grouped-MMA output depends
only on its own slot's row and its K order (the chunk-width invariance the tensor walk keeps),
and the mix and the join work row by row, so the halves give every bit the whole band gives.
Only where the runs wait changes: the activation runs are half-major in `partials`, and the mix
rows wait in `activation`, which the down pass over the runs leaves unread.

Two smaller changes come with it, also bit for bit. The down pass reads the gathered runs where
they land (`q4p_moe_down_runs` and its `_mx8` form), so the join that copied the whole activation
first is gone. The row split's `partials` holds the larger of the gathered runs and the mix rows,
not a hidden-wide row a slot, which is half the bytes.

`SPARKFLASH_INKLING_OVERLAP=off` runs the band whole on the run's stream. Unset or `on` runs the
halves wherever they can run: the row split, more than one rank, the device leg and the tensor
walk. `on` named anywhere else is refused, as is any other value.

Four ranks, row split, batch arm, one session, each prompt twice:

| on the four ranks | D120 build | runs read in place | and the halves |
|---|---|---|---|
| 93K prompt | 93.4 s, 92.6 s | 91.0 s, 91.3 s | 81.1 s, 81.2 s |
| prompt tokens a second at 93K | 1,001 | 1,022 | 1,148 |
| repeating 5K prompt | | 6.78 s | 6.39 s |
| 6,000-token text, twelve teacher-forced cuts | 9.46 s | 9.34 s | 8.54 s |

All three give the same bits. The teacher-forced logits are identical to the D120 build's in
every value, with the halves and without them. The repeating prompt's decode digest is
`90f777a3c48a6bc4` and the 93K prompt's is `c6e0ef10fd49a8e0` in every column.

That left the layer's other collective on the run's stream: the attention delta's all-reduce, a
scatter, a sum and a gather of every row's residual-wide partial, about 11.5 s exposed at 93K. It
now runs in halves too. Under the overlap and the tensor-core attention, a chunk of 512 rows or
more launches the attention's MMA twice, for the rows before a 16-row block's edge near the
middle and for the rows from it. After each launch the run's stream widens that half's output
and projects it, and the side stream sums the half's delta in `exchange_sum`'s two shots
(`attention_half` in `device/chunk.rs`). So half 0's sum crosses the wire under half 1's
attention, and the run's stream waits for the second sum before the convolution.

A block of the MMA reads only its own 16 rows' query side, sink and gate, and takes its keys, its
tiles and its bias distances from its first row's position, so at a block's edge every block is
the one launch's block. The widen is per element, the projection per row, and each element's sum
is the same fold in the same order whichever rows share the launch, so the bits do not move.
Half 1's sum takes the slots past half 0's. `SPARKFLASH_INKLING_OVERLAP=off` runs it whole with
the band, and the trace keeps the whole form, since it reads the projection back before its sum.

| on the four ranks, interleaved | without (pf6) | the attention delta in halves |
|---|---|---|
| 93K prompt | 79.14 s, 79.21 s | 77.28 s, 77.31 s |
| prompt tokens a second at 93K | 1,176 | 1,205 |
| repeating 5K prompt | | 6.15 s |
| 6,000-token text, twelve teacher-forced cuts | | 8.19 s |

The teacher-forced logits are identical in every value, and the 93K and repeating prompts keep
their digests. In a trace the exposed NCCL time at 93K falls from 19.7 s to 13.8 s, all of it now
on the side stream where the other half's compute does not cover it. The compute beside a
collective runs a little slower, so the prompt gains 1.9 s of the 5.9 s.

### The block residuals read their summary stack once

Every layer opens its attention and its MLP with a block residual: a softmax over the summary
stack and the current stream, scored by the layer's queries (three on a full layer's attention,
one elsewhere). The one-block form read the stack twice for each query, once to score it and once
to mix it, and the walk's 97 calls a chunk took 7.01 s over the 13 chunks of the 93K prompt.

A chunk walk now chains them (`inkling_block_residual_chain_*`). Each call's mix pass also scores
the next call's queries against the summary rows it reads, and leaves the scores for that call in
the residual's statistics rows. The next call then reads only the rows the stack gained since,
and the stream. The query counts are compile-time entries, a call's and the next call's
(`q1_n0` through `q3_n3`), so a thread keeps the next call's dots in registers: 40 to 116 a
thread, against 175 for a first form with the counts at run time. Every score and mix is the
one-block form's reduction in its order, so the bits are the same: the teacher-forced logits are
identical in every value, and the 93K and repeating prompts keep their digests.

Its kernels take 5.00 s at 93K against 7.01 s. End to end, that is inside the run-to-run noise
of one interleaved pair (79.5 and 82.7 s chained, 81.0 and 81.8 s not).
`SPARKFLASH_INKLING_RESIDUAL_CHAIN=off` runs the one-block form. A model whose stack outgrows the
chained form's eight summary rows stands down to it when the switch is unset and is refused under
`on`, and any other value is refused.

### Where the 93K prompt's time goes now

The 93K prompt took 132 s (706 prompt tokens a second) before D120. At default bits it now takes
77.3 s (1,205), and 72.2 s with the opt-in head-only activation (measured before the attention
halves). Rank 0's trace of the default build, a 77.1 s prefill span with 0.4 s of it idle:

| stage | own time | share |
|---|---|---|
| NCCL send/recv (13.8 s of it exposed) | 16.0 s | 21 % |
| NVFP4 expert gate_up and down (layers 0-39) | 24.0 s | 31 % |
| attention MMA | 11.3 s | 15 % |
| dense projections (`glm_prefill_bf16_wide`) | 5.9 s | 8 % |
| MXFP8 expert gate_up and down (layers 40-47) | 5.4 s | 7 % |
| block residuals | 5.0 s | 6 % |
| residual add, row norms, short convolutions | 4.9 s | 6 % |

These levers were looked at and not built:

- The collectives on the direct-RDMA exchange over both rails. The exchange already carries every
  sum of at most 256 KB, but a prompt chunk's rounds stay on NCCL over one rail until the lead's
  second rail works. Every NCCL call above is on the side stream, and the exposed 13.8 s is the
  most this could recover.
- More query rows a key tile in the full layers' attention. The MMA block holds 86 KB of the
  101 KB a block may take, at 212 registers a thread, so a wider query tile does not fit without
  a new tile layout, and the kernel is bound by compute and imbalance, not by reading keys.
- One launch for the projections that read one input (the query, gate, relative state and, on a
  sliding layer, key and value). The narrow ones underfill the device: a 256-column projection
  is 64 tiles on 48 SMs, 0.91 ms at 8,192 rows. Grouped, they would save about 0.9 ms a layer at
  that width, about 0.5 s at 93K.
- Folding the residual add into the convolution before it, or into the block residual after it.
  Each saves one pass over the stream, about 0.5 ms a layer at 8,192 rows and about 0.6 s at 93K.

Each of the last three is under the 2 % that separates two runs of the same build.

### The experts' activation as its head alone (opt-in, not the default)

Under D117 an f32 activation enters the routed-expert MMA as two BF16 planes, its head and its
remainder, both multiplied through the same weight fragments into one f32 accumulator. That keeps
16 significant bits and costs two MMAs and two activation loads a K step, and under the row split
the SwiGLU rows cross the wire as both planes.

`SPARKFLASH_INKLING_PLANES=head` runs the head-only forms of the six expert passes
(`q4p_moe_gate_up_capped_head`, `q4p_moe_down_head`, and the `_mx8` and `_runs` forms of each):
`q4p_moe_body` with `PLANES` 1 stages the head plane alone, gate_up writes only the head of its
SwiGLU rows, and the row split gathers half the bytes. The MXFP8 form's head-only stage is
12.5 KB, so it keeps three stages, not two. Unset or `both` keeps both planes and every bit of the
build before: the teacher-forced logits are identical and the 93K prompt's digest is
`c6e0ef10fd49a8e0`. `head` named under any walk but `tensor` is refused, as is any other value.

The head alone is a different sum, since the activation keeps 8 significant bits, so it stays
opt-in until the maintainer decides.

| on the four ranks, row split | both planes (default) | head only |
|---|---|---|
| 6,000-token text, twelve teacher-forced cuts | 8.53 s | 7.74 s |
| 93K prompt | 81.0 s | 72.2 s |
| prompt tokens a second at 93K | 1,150 | 1,290 |

The distance, teacher-forced on twelve cuts of the 6,000-token text (the D117 method):

| | top-1 | top-5 overlap | mean abs logit difference a cut | max | KL max (mean), nats |
|---|---|---|---|---|---|
| head only against exact | 12 of 12 | 4-5 | 0.071-0.420 | 2.84 | 1.32e-2 (3.50e-3) |
| head only against both planes | 12 of 12 | 4-5 | 0.074-0.400 | 2.81 | 1.88e-2 (5.23e-3) |
| both planes against exact (D117, for scale) | 12 of 12 | 4-5 | 0.057-0.339 | 2.03 | 1.55e-2 (4.58e-3) |

So the head alone moves the logits about a quarter further from exact than both planes do, with
a KL to exact of the same size. After the 93K prompt the greedy ids agree for 15 tokens and part
at the 16th, as the intermediate split's do.

### Rewinding to the last prompt's end (D121)

A chat client sends the whole conversation every turn. When the reply comes back in the ids the
model wrote, or in ids the byte match above takes, the held sequence extends and only the new turn
runs. When it does not, because the client renders the reply again in a form the match cannot
take (a reasoning block dropped, a template that re-spells the turn) or regenerates the answer to
the same prompt, the prompt keeps the last prompt and parts from the reply after it, and the whole
conversation ran cold: 91.6 s at 93K.

A session now keeps a checkpoint at the end of every prompt (`Boundary` in `device.rs`): the state
its cache rows do not hold, which is each layer's four convolution histories and the DSpark
drafter's Markov feature, in one allocation. A position's keys and values stay in the cache until
a later position's write lands on their slot. A sliding ring holds its window and a widest chunk,
so the rows before the boundary survive while the positions written past it (the reply, and a
verify's drafts past its last fed token) number at most a widest chunk; under the switch the
drafter's local rings are opened as wide (window + widest chunk - 1 rows, 32 KiB a row).

A next prompt that starts with the last prompt, goes past it, and carries the last prompt's images
where it held them takes the session back there: the checkpoint is copied back, the history and
position are cut to the boundary, and the rest of the prompt is forwarded. The host driver offers
it (`ModelSession::rewind_prefix`) when the prompt neither extends the held ids nor matches them by
text, prints `prefix reuse by rewind`, and reports the boundary as the hit; the group sends the
one point to every rank, so they rewind together. A reply longer than a widest chunk past the
boundary makes the session offer no rewind, and the prompt runs as it would have.

`SPARKFLASH_REWIND=off` (named `SPARKFLASH_INKLING_REWIND` when this was measured) keeps no checkpoint and opens the drafter's rings at their window,
and the prompt then runs cold. Any value other than on or off is refused.

Four ranks, row split, batch arm (`--rewind 512`: a prompt, 24 greedy ids, then the prompt and
512 other ids):

| prompt held, then forwarded | cold | rewound |
|---|---|---|
| 6,356 ids, then 512 | 6.47 s | 1.25 s |
| the same under the served launch (DSpark depth 7, 4,096-row chunks) | 6.56 s | 1.25 s |
| 93,159 ids, then 512 | 91.6 s | 1.40 s |

The bits are those of a session that held the prompt and was then given the next one with no
reply in between. Under `--rewind` the batch arm runs that session too, and against it the
rewound session's logits are equal in every value, with the same decode digest
(`62483efcb3086f59`) and greedy ids, on the plain launch and on the served one. Against a cold
prefill of the whole second prompt the logits differ (largest 2.06 at 6K, 1.13 at 93K) and the
greedy ids agree: the rewound prompt's chunks start at the boundary rather than at row 0, and a
prompt row's bits depend on where its chunk starts, as they do for any prompt that extends a held
sequence (D66). Under `off` the second prompt is a cold prefill in every value.

### One sampler for both models; DSpark drafts with the target's noise (D123)

A sampled request used to draw from the engine sampler: it sorted the row, took the cut and drew
one uniform from a generator stepped in order. The drafter had no share in that draw. It drafted
the argmax, so under sampling a draft was accepted only when the sample happened to be the
argmax. Inkling now samples the way DeepSeek does: Gumbel-max with counter noise. The token at
sequence index `t` is the argmax of `logit / T + G(seed, t, id)` over the cut. `G` is a hash of
the seed, the index and the id, so it holds no state. Both models use one sampler,
`sparkflash_model::gumbel::GumbelSampler`: DeepSeek's `tp4_forward/sample.rs` now re-exports
it, a pure move, and DeepSeek's cubins match their pins.

* **The host pick.** `pick_row` takes the whole logit row after the request's penalties
  (`Sampler::penalized`). With temperature alone it takes the perturbed argmax over the tokens
  within the noise's span of the best. With top_k or top_p it bounds the candidate list with a
  512-bucket histogram of the scaled logits, takes DeepSeek's cut over that list, and widens to
  a full sort only when the list cannot decide the cut. On 150K-wide rows it takes 0.25-0.6 ms
  for a peaked row and 1.7 ms for a flat one, against 2.1-3.9 ms for the engine sampler's draw.
* **The draft.** The drafter's Markov head adds the same `G` at the index each draft would land
  on (`markov_logits`, `inv_t` and the seed). Each draft is then the sample of the drafter's own
  tempered distribution under the target's noise, so a draft is accepted when it equals the
  target's sample. The draft ignores the request's cut and penalties; a draft outside them is
  rejected at verify. The depth policy's confidence stays the drafter's p(argmax).

The emitted token at an index depends only on the target's logits, the cut and the seed. Plain
and speculative decoding therefore emit the same ids for one seed. A seed's ids differ from the
old draw's, because the random stream is a different one; greedy is unchanged.

On the four ranks, served files (intermediate split), adaptive depth 7, one window:

| run | plain | DSpark | ids |
|---|---|---|---|
| greedy, repeating 5K prompt, 96 tokens | 25.21 ms (digest `ec85a691b74bbc6b`, the deployed build's) | 12.76 ms, 81 of 82 drafts | equal |
| T 1, top_p 0.95, seed 12345, same prompt | 26.12 ms | 13.19 ms, 81 of 82 | equal |
| T 0.7, top_p 0.9, top_k 40, seed 777 | 26.18 ms | 13.03 ms, 81 of 81 | equal |
| T 1, top_p 0.95, seed 4242, 93,159-token text, 48 tokens | 29.55 ms | 26.70 ms, 26 of 42 in 14 rounds | equal |

On the repeating prompt the sampled ids are greedy's: every step's top token holds more than the
nucleus. The 93K text's sampled ids leave greedy's at the third token, so that row is the one that
tests a real draw. `inkling-tp4 --sample T,TOP_P,TOP_K,SEED` sets a run's decode sampling. The
served group sends the request's sampler in its request frame, so every rank draws the same noise.

### A verify's expert items, one pair a group

A stage ledger of the served build (rank 0, PDL off, so each kernel's span is its own work) puts
a depth-7 round on the repeating prompt at 14.6 ms of device time a token. The expert items walk
(`inkling_*_expert_items`, five rows and up) is 9.4 ms of that. Inside it, the down projection
took 611 µs a layer, nearly the 670 µs of gate/up, although it reads half the bytes. Its rows are
576 columns, so a block's work is short. It walked four pairs a group, and their accumulators held
it to 128 registers and two blocks an SM. An item holds one slot or a few, so the extra pairs
bought nothing. The items walk now takes one pair a group (`INKLING_ITEM_PAIRS`). Each pair's sum
is its own chain in either grouping, so no output word moves. The prefill pair walk keeps four.

| micro-bench, one layer, eight rows' 64 slots, L2 flushed | four pairs | one pair |
|---|---|---|
| NVFP4 gate/up, 1,152 x 4,608 a rank | 993 µs | 920 µs |
| NVFP4 down, 4,608 x 576 | 796 µs | 548 µs |
| MXFP8 down | 1,035 µs | 800 µs |

Every output word equalled the band's one-pair matvec. At one row the walk is the band's speed
within 5%, so plain decoding keeps the band.

| on the four ranks, served files, DSpark adaptive depth 7 | four pairs | one pair |
|---|---|---|
| repeating 5K prompt, 96 tokens (twice each) | 12.76 / 12.76 ms a token | 11.52 / 11.59 ms |
| 93,159-token text, 48 tokens (twice each, interleaved) | 26.23 / 26.00 ms | 24.89 / 25.47 ms |

The digests were the deployed build's (`ec85a691b74bbc6b` on the repeating prompt,
`2f09c281b606c98e` at 93K at 48 tokens), and DSpark's ids equalled plain decoding's.

The ledger's other lines bound what is left for the drafter. A round runs two head-sized GEMVs of
about 1.6 ms: the verify's eight rows and the drafter's base logits. Both are at the DRAM floor.
The seven Markov fold steps are 94 µs each. The drafter's five layers read its 649 MB of packed
projections once a round. E4M3 copies of those projections and of the fold would save about 1 ms
of a 105 ms round, and a copy of the head would cost 238 MB a rank, so the E4M3 drafter is not
built.

### The packed dense GEMVs' verify rows, split in token halves (neutral end to end)

A batched `q4e_gemv_packed` / `q4e_gemv4_packed` warp took two rows against every token, so half
the grid's warps sat idle and a lane held sixteen sums (`q4e_gemv4_packed` at 167 registers, one
block an SM). A work item is now a pair of rows and one half of the tokens (`q4e_batch_half`):
every warp works and a lane holds eight sums (127 registers). A (row, token) sum is the same walk,
so no output word moves. Only Inkling launches these kernels.

| on the four ranks, served files, DSpark adaptive depth 7 | before | after |
|---|---|---|
| repeating 5K prompt, 96 tokens (twice each, interleaved) | 11.57 / 11.91 ms a token | 11.63 / 11.46 ms |
| 93,159-token text, 48 tokens (twice each, interleaved) | 25.06 / 25.13 ms | 24.85 / 25.51 ms |
| 93K prefill | 67.2 / 67.7 s | 67.0 / 67.6 s |

Digests (`ec85a691b74bbc6b`, `2f09c281b606c98e`) and ids are unchanged, and DSpark's ids equal
plain decoding's. The difference is inside run-to-run noise: the dense projections were already
near the DRAM floor, and the idle warps had been hiding behind the weight stream. It is kept for
the lower register count and because it costs nothing.

### The drafter's full layer keeps the whole context (a window is a diagnostic, not a default)

Four of the drafter's five layers are local (a 1,024 window). Layer 2 reads the whole context,
2 KB a position a rank (two key/value heads in f32), so 186 MB a pass at 93K.
`SPARKFLASH_INKLING_DRAFT_WINDOW` (unset or `full`, the default; a positive count; anything else
refused) gives that layer a window as the local ones have. The drafter only proposes tokens, so
the window moves acceptance and never an emitted id: every run below equalled the unset run's ids.

| DSpark adaptive depth 7, 96 tokens | drafts accepted | ms a token |
|---|---|---|
| repeating 5K prompt, full | 81 of 82 | 11.51 |
| repeating 5K prompt, window 4,096 | 31 of 129 | 36.26 |
| repeating 5K prompt, window 1,024 | 34 of 120 | 33.91 |
| 93,159-token text, full (twice) | 65 of 100 | 22.98 / 23.11 |
| 93,159-token text, window 16,384 | 65 of 97 | 22.01 |
| 93,159-token text, window 4,096 | 64 of 98 | 22.58 |
| 93,159-token text, window 1,024 | 63 of 96 | 22.64 |

The full layer is what drafts a copy from far back. Without it the repeating prompt, where
speculation pays most, falls to a third of its speed. A window only wins where nothing is copied
from beyond it, so the default stays `full`. The 93K rows show what the layer's read costs at that
length: about 1 ms a token (4.5%), more than its 186 MB a pass at the DRAM floor would. The old
attention walk read each key span once per four queries, so twelve blocks read the same span.

### The drafter's attention reads each key span once

`q4d_attn_span_chunk` (in `dflash2_sm121.cu`) replaces that walk. The grid is (key head, chunk):
each block walks one contiguous run of keys for all of its key head's queries (6 heads × 8 block
rows = 48 a rank), one warp per eight queries. A batch of four keys gives a warp 32 dot products,
summed by a transpose reduction so lane L ends with pair L. The running softmax steps once per
query per batch, and each weight is broadcast from its pair's lane. `q4d_attn_span_merge` then
folds the 48 chunks (`ATTENTION_CHUNKS`). Contiguous chunks matter: giving the old walk more
blocks over strided spans made it slower. The drafter refuses to load a checkpoint with more than
256 queries per key head (`ATTENTION_MAX_QUERIES`). Only the Inkling drafter launches these
kernels.

| one full-layer pass, 8 block rows, one rank | old walk | span chunks |
|---|---|---|
| 93K keys | 2.70 ms | 1.06 ms |
| 5K keys | 110 µs | 49 µs |
| 1K keys | 28.5 µs | 20.5 µs |

The new walk adds in a different order, so the drafter's sums move in their last bits and
acceptance moves by a draft or two either way. Emitted ids never move: plain decoding's digest is
`ec85a691b74bbc6b` as before, and DSpark's ids equal plain decoding's on every text below,
greedy and sampled.

| DSpark adaptive depth 7, served files, ids equal in every row | old ms a token | new | ms a round old / new |
|---|---|---|---|
| repeating 5K prompt, 96 tokens (twice) | 11.38 / 11.42 | 11.53 / 11.48 | |
| 93,159-token text, 96 tokens (twice) | 22.99 / 23.21 | 23.46 / 23.75 | 87.4 / 85.7 |
| score text, 192 tokens | 21.86 | 21.99 | 67.4 / 64.6 |
| decode check text, 192 tokens | 12.97 | 12.88 | 88.5 / 87.9 |
| pf7 split text, 192 tokens | 11.84 | 11.78 | 83.7 / 83.3 |
| r2 split text, 192 tokens | 11.27 | 11.29 | 82.8 / 83.0 |
| 6K check text, 192 tokens | 17.17 | 16.94 | 96.5 / 95.1 |
| 46K text, 192 tokens | 22.71 | 21.99 | 66.7 / 70.0 |

A round is equal or faster wherever the round count holds; tokens a second move with acceptance,
which shifts both ways (93K: 65 then 63 of about 100 drafts; 46K: 65 then 60 rounds). Net it is
about 0.4% faster a token across the texts, inside run-to-run noise. The ~1 ms the full layer cost
at 93K is now about 0.4 ms. It is kept: it costs nothing, it never moves an id, and the gap grows
with context.

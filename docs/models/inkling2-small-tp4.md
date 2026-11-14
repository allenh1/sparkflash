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

**What the arm still does not do:** a chunked prefill (it loops the decode walk, which is correct
and slow), the DSpark drafter, the image path, and a serving path for the device arm (`serve`
still selects the TP1 f32 host reference for this pack).

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

The numpy implementation lives at `/home/allenh1/inkling-bin/hf_reference.py` on the cluster, with
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

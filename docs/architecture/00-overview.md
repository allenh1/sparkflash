# 00 — Overview: Hyperparameters, Parameter Arithmetic, Checkpoint Tensor Map

Sources of truth used in this document (nothing here is inferred from the spec):

- /home/user/dsv41-research/config.json — released HF config (the **real** hyperparameters).
- /home/user/dsv41-research/inference/config.json — the runtime config that generate.py feeds
  straight into ModelArgs(**json.load(...)) (inference/generate.py:121-123).
- /home/user/dsv41-research/inference/model.py — reference forward implementation.
- /home/user/dsv41-research/model.safetensors.index.json — 96,085 tensor names in 48 shards,
  metadata.total_size = 510286023000.
- /home/user/dsv41-research/tech_report.txt — the 51-page report (cited as tech_report.txt p.N).

---

## 1. Hyperparameter table (values read from the released config)

The HF config nests everything under text_config/vision_config; the runtime config is flat and
**renames** many keys. ModelArgs field names equal the *runtime* keys
(inference/model.py:44-136; the docstring on line 46 says "Field names are exactly the config JSON
keys").

| HF text_config key | Runtime / ModelArgs field | Value | Meaning |
| :--- | :--- | :--- | :--- |
| num_hidden_layers | n_layers | 40 | backbone blocks; compress_ratios has 43 entries |
| hidden_size | dim | 5120 | residual width d |
| vocab_size | vocab_size | 129280 | bos=0, eos=1, pad=2, image=129264 |
| num_attention_heads | n_heads | 64 | query heads |
| num_key_value_heads | (none) | 1 | present in config but **unused** by the reference (no GQA path) |
| head_dim | head_dim | 512 | MLA latent width; includes the RoPE slice |
| qk_rope_head_dim | rope_head_dim | 64 | last 64 dims of head_dim are rotated |
| q_lora_rank | q_lora_rank | 1280 | query LoRA rank |
| o_groups | o_groups | 8 | output projection groups |
| o_lora_rank | o_lora_rank | 1024 | per-group intermediate width |
| rms_norm_eps | norm_eps | 1e-20 | RMSNorm eps; also the engram coefficient eps |
| sliding_window | window_size | 128 | n_win |
| compress_ratios | compress_ratios | [0,0, 2x18, 1x20, 0,0,0] | per layer, MTP included |
| kv_source_layer_ids | kv_source_layers | [2,8,14,20] | layers that own main-KV + index-K |
| index_source_layer_ids | index_source_layers | [2,8,14,20,24,28,32,36] | layers that run an indexer |
| candidate_source_layer_id | candidate_source_layer | 20 | layer that builds the HSI pool |
| candidate_topk_blocks | candidate_topk_blocks | 2048 | blocks |
| candidate_block_size | candidate_block_size | 8 | positions per block |
| index_n_heads | index_n_heads | 32 | indexer query heads |
| index_head_dim | index_head_dim | 128 | indexer head width |
| index_topk | index_topk | 512 | sparse selections per query |
| n_routed_experts | n_routed_experts | 384 | plus 1 shared (n_shared_experts) |
| num_experts_per_tok | n_activated_experts | 6 | top-6 routed |
| moe_intermediate_size | moe_inter_dim | 2304 | per-expert width |
| scoring_func | score_func | sqrtsoftplus | gate score |
| topk_method | (none) | noaux_tc | realised as scores + bias for selection only |
| routed_scaling_factor | route_scale | 1.5 | post-normalisation multiplier |
| norm_topk_prob | (absent, default) | true | ModelArgs.norm_topk_prob = True |
| swiglu_limit | swiglu_limit | 10.0 | clamp in every Expert |
| rope_theta / rope_scaling.factor | rope_theta / rope_factor | 10000 / 16 | YaRN |
| rope_scaling.original_max_position_embeddings | original_seq_len | 65536 | YaRN is active when > 0 |
| rope_scaling.beta_fast / beta_slow | beta_fast / beta_slow | 32 / 1 | YaRN ramp |
| compress_rope_theta | compress_rope_theta | 160000 | RoPE base for compressed latents |
| max_position_embeddings | (max_seq_len) | 1048576 | 1M context |
| hc_mult / hc_sinkhorn_iters / hc_eps | same | 4 / 20 / 1e-06 | Hyper-Connections |
| engram_layer_ids | same | [1,14] | |
| engram_num_embeddings | same | [384006168, 384016682] | rows per table |
| engram_vocab_size | same | 16000000 | prime search start |
| engram_max_ngram_size | same | 4 | orders 2,3,4 |
| engram_n_heads / engram_head_dim | same | 8 / 256 | 24 x 256 = 6144 |
| engram_compressed_vocab_size | same | 99092 | asserted at boot (inference/engram.py:146) |
| engram_pad_token_id | engram_pad_id | 2 | |
| num_nextn_predict_layers | n_mtp_layers | 3 | DSpark stages |
| dspark_block_size | same | 5 | draft block |
| dspark_noise_token_id | same | 128799 | filler draft id |
| dspark_target_layer_ids | same | [37,38,39] | |
| dspark_markov_rank | same | 256 | |
| dspark_n_routed_experts | same | 128 | MTP MoE only |
| dspark_num_experts_per_tok | dspark_n_activated_experts | 3 | MTP MoE only |
| quantization_config.weight_block_size | (none) | [32,32] | fp8 block scale |
| quantization_config.scale_fmt | (none) | ue8m0 | torch.float8_e8m0fnu |
| quantization_config.expert_dtype | expert_dtype | fp4 | torch.float4_e2m1fn_x2 |
| dtype | dtype | bfloat16 (config) / fp8 (runtime) | runtime switches dense weights to fp8 |
| image_token_id (top level) | image_token_id | 129264 | |

Vision (vision_config mapped to runtime vision_*): num_hidden_layers=32, hidden_size=1024,
num_attention_heads=16, intermediate_size=2816, patch_size=14, downsample_ratio=3,
max_image_tokens=1024, min_pixels=295936 (=544^2), max_wh_ratio=null, rope_theta=10000.

**Key-name gulf.** 28 HF keys have no runtime counterpart and 32 runtime keys have no HF
counterpart, so a loader must translate (hidden_size->dim, sliding_window->window_size,
scoring_func->score_func, routed_scaling_factor->route_scale, engram_pad_token_id->engram_pad_id,
candidate_source_layer_id->candidate_source_layer, index_source_layer_ids->index_source_layers,
...). Deriving ModelArgs by feeding the HF file directly falls back to the small default model
compiled into ModelArgs (dim=1024, n_layers=5, n_heads=16, ...; inference/model.py:56-136).

---

## 2. Parameter-count arithmetic

Reconstructed from the shapes implied by config.json plus the tensor names in
model.safetensors.index.json, then cross-checked against metadata.total_size (section 2.4).

### 2.1 Backbone — 552B

Per layer (dim=5120, H=64, head_dim=512, q_lora=1280, o_groups=8, o_lora=1024, E=384, EI=2304):

| Block | Params |
| :--- | ---: |
| wq_a 1280x5120 | 6,553,600 |
| q_norm | 1,280 |
| wq_b 32768x1280 | 41,943,040 |
| wkv 512x5120 | 2,621,440 |
| kv_norm | 512 |
| wo_a 8192x4096 (block-diagonal over 8 groups) | 33,554,432 |
| wo_b 5120x8192 | 41,943,040 |
| attn_sink | 64 |
| attn_norm + ffn_norm | 10,240 |
| gate.weight 384x5120 | 1,966,080 |
| gate.bias + gate.bias_vl | 768 |
| 384 routed experts x 3 x (2304x5120) | 13,589,544,960 |
| 1 shared expert x 3 x (2304x5120) | 35,389,440 |
| hc_attn_fn + hc_ffn_fn (24x20480 each) | 983,040 |
| hc_*_base (24x2) + hc_*_scale (3x2) | 54 |
| **per layer** | **13,754,509,990** |
| x40 | 550,180,399,600 |
| embed.weight 129280x5120 | 662,118,400 |
| head.weight 129280x5120 | 662,118,400 |
| norm.weight | 5,120 |
| source-layer extras: compressor.wkv x4, compressor.wgate x3, compressor.norm x4, indexer.wk x4, indexer.k_norm x4, indexer.wq_b x8, indexer.weights_proj x8 | 15,792,896 |
| **backbone total** | **551,510,179,440 (about 551.5B)** |

The paper says 552B backbone parameters (tech_report.txt p.7, line 285); the released config
reproduces that to 0.09%. The 3 MTP/DSpark stages add about 7.6B (section 2.3).

### 2.2 Engram — 196B

(384,006,168 + 384,016,682) x 256 = 196,613,849,600, about **196.6B** table rows. The report's
"196B Engram parameters" (tech_report.txt p.7, line 285) and the model card (README.md:45) both
round this. Adding the per-layer projections (wkv 25600x6144, plus q_weight and k_weight of shape
4x5120, at both layers) gives **196,928,504,320 (about 196.9B)**. Because engram_layer_ids covers
only layers 1 and 14 (config.json), the Engram tables are 35.7% of all parameters but are touched
at only two layers.

### 2.3 Active parameters — 8B prefill / 16B decode

Per-token active set (dense attention + gate + 6 of 384 routed + 1 shared + HC predictors):

```
per layer = 6,553,600 + 41,943,040 + 2,621,440 + 33,554,432 + 41,943,040
          + 1,966,080 + 7 x 35,389,440 + 983,040 + 120       = 377,290,752
x40                                                          = 15,091,630,080
+ engram wkv at layers 1 and 14 (2 x 157,286,400)             = 15,406,202,880  (about 15.4B)
```

The report's "16B during decode" (tech_report.txt p.7, line 286; p.22 lines 1010-1011) is this
number rounded. "8B during prefill" is exactly half of it, a **direct consequence of CED** (only the
20 causal-encoder layers run during prefill; tech_report.txt p.9 lines 337-367 and p.20-22
lines 981-995). The **released reference forward does not implement CED**: Transformer.forward
iterates over all 40 layers for every token (inference/model.py:1261-1267) and generate.py prefills
whole chunks through all 40 layers (inference/generate.py:69-74). See section 4.

### 2.4 Checkpoint byte layout reconstructed from total_size

Assuming fp8 (1 byte/param plus one ue8m0 byte per 32x32 block) for dense Linears, packed MXFP4
(0.5 byte/param plus one ue8m0 byte per 32 elements along K) for routed and shared experts,
2 bytes for embeddings/norms/HC/Engram tables plus 1/32 byte per param for Engram scales, and
4 bytes for attn_sink:

| Group | Bytes |
| :--- | ---: |
| routed + shared experts (40 layers) | 290,194,790,400 |
| Engram tables (fp8 plus ue8m0 per 32) | 202,758,032,400 |
| dense attention (40 layers) | 5,069,721,600 |
| MoE gates + MTP MoE | 7,406,531,712 |
| embed + head | 2,647,654,400 |
| HC parameters (40 + 3 layers, fp32) | 169,092,168 |
| compressor + indexer | 60,947,520 |
| Engram projections | 315,207,680 |
| vision + aligner + image delimiters | 553,394,784 |
| **reconstructed total** | **509,820,148,088** |
| metadata.total_size | **510,286,023,000** |
| delta | 0.09% |

The residual sits in the packing details (per-tensor scale padding, and whether every fp8 tensor
really uses 32x32 rather than 128x128 blocks; inference/convert.py:164 accepts both).

---

## 3. Tensor-name map of the checkpoint

96,085 names across 48 shards. Top-level namespaces: layers.* (93,415), mtp.* (2,401), vision.*
(259), aligner.* (4), embed.weight, head.weight, norm.weight, image_start, image_end,
image_newline.

| Pattern | Count | Notes |
| :--- | ---: | :--- |
| layers.{0..39}.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale} | 40 each | scale only for fp8 tensors |
| layers.N.attn.attn_sink | 40 | no .weight suffix; fp32 |
| layers.N.attn.{q_norm,kv_norm}.weight, attn_norm, ffn_norm | 40 each | fp32 weights |
| layers.{2,8,14,20}.attn.compressor.{wkv,norm}.weight | 4 | main-KV compressor |
| layers.{2,8,14}.attn.compressor.wgate.weight | 3 | **absent for layer 20** (ratio 1) |
| layers.{2,8,14,20}.attn.indexer.{wk,k_norm}.weight | 4 | owns_k layers only |
| layers.{2,8,14,20,24,28,32,36}.attn.indexer.wq_b.{weight,scale}, .weights_proj.weight | 8 each | indexers |
| layers.N.ffn.gate.weight | 40 | 384x5120, fp8 |
| layers.N.ffn.gate.bias, layers.N.ffn.gate.bias_vl | 40 each | noaux correction bias plus the VL twin |
| layers.N.ffn.experts.{0..383}.{w1,w2,w3}.{weight,scale} | 15,360 each | fp4 packed (int8 view) plus scale |
| layers.N.ffn.shared_experts.{w1,w2,w3}.{weight,scale} | 40 each | 1 shared expert |
| layers.N.hc_attn_fn, hc_ffn_fn | 40 each | 24x20480, fp32 |
| layers.N.hc_attn_base, hc_ffn_base, hc_attn_scale, hc_ffn_scale | 40 each | fp32 |
| layers.{1,14}.engram.embed.weight | 2 | fp8 [rows, 256] |
| layers.{1,14}.engram.embed.scale | 2 | ue8m0 [rows, 8] |
| layers.{1,14}.engram.wkv.{weight,scale} | 2 each | fp8, 25600 x 6144 |
| layers.{1,14}.engram.q_weight, k_weight | 2 each | fp32 [4, 5120] |
| mtp.{0,1,2}.attn.*, mtp.N.*_norm.weight, mtp.N.hc_* | 3 each | one DSpark block per stage |
| mtp.N.ffn.experts.{0..127}.*, mtp.N.ffn.gate.* | 128 experts each | **128 routed, top-3** |
| mtp.N.ffn.shared_experts.* | 3 | |
| mtp.0.main_proj.{weight,scale}, mtp.0.main_norm.weight | 1 | stage 0 only |
| mtp.2.norm.weight, mtp.2.markov_head.embed.weight, mtp.2.markov_head.head.weight, mtp.2.confidence_head.proj.weight | 1 | last stage only |
| vision.blocks.{0..31}.{norm1,norm2}.weight | 32 each | |
| vision.blocks.N.attn.wqkv.{weight,bias}, .wo.{weight,bias} | 32 each | ViT attention **has bias** |
| vision.blocks.N.mlp.{w1,w2}.weight | 32 each | SwiGLU, **no bias**, w1 is 5632-wide |
| vision.patch_embed.proj.{weight,bias}, vision.norm.weight | 1 each | linear patch embed (no conv), 588 -> 1024 |
| aligner.{w1,w2}.{weight,bias} | 1 each | 9216 -> 5120 -> 5120 |
| embed.weight, head.weight | 1 each | untied (tie_word_embeddings: false) |
| image_start, image_end, image_newline | 1 each | [5120] learned span delimiters |

**Absent from this checkpoint:** any tid2eid / tie2eid / hash-table tensor, any ape positional
embedding, and any num_hash_layers config key. inference/convert.py:116-118 still carries rename
slots for tie2eid / tid2eid / ape because sibling checkpoints use them; a scan over all 96,085
names finds zero matches here.

**Conversion-time renames** (inference/convert.py:105-126): strip a leading "model.", drop
mtp.{i}.embed.weight and mtp.{i}.head.weight (tied to the backbone, inference/model.py:1212-1213),
self_attn -> attn, mlp -> ffn (skipped under a vision. prefix), weight_scale_inv -> scale,
e_score_correction_bias -> bias; wo_a is dequantised to bf16 (convert.py:156-173) and non-fp8
expert tensors are viewed as float4_e2m1fn_x2 (convert.py:174-181). Expert counts are inferred
from the names rather than the config (convert.py:67-77), yielding (384, 128).

---

## 4. Spec deltas

**D1 — The CED prefill bypass does not exist in the reference implementation.** The spec states
that "all N tokens pass through the 20 Causal Encoder layers ... activating only **8B parameters per
token**" and that "the decoder's single shared global KV cache (owned by Layer 20) is projected
directly from the final encoder layer's output hidden states H_19" (spec section 2.2). The tech
report does describe CED (tech_report.txt p.9 lines 337-367; p.20-22 lines 981-995), but the
released code has no branch, no layer-range gate and no second projection stage:
Transformer.forward unconditionally runs over all 40 blocks for every token
(inference/model.py:1261-1267), and generate.py prefills the whole prompt chunk through all 40
layers (inference/generate.py:69-74). "8B prefill" is a training-time property of the trained
weights, not a behaviour of the reference forward; an engine that skips decoder layers during
prefill cannot be validated against this implementation.

**D2 — "Z_20 = H_19 W_20^Z" is unsupported.** The spec's CED formula has a compression-weight
projection for layer 20. Layer 20 has compress_ratio == 1, and Compressor.forward returns
self.norm(self.wkv(x)) with **no gate** for ratio 1 (inference/model.py:461-462); the checkpoint has
no compressor.wgate tensor for layer 20 (only layers 2, 8 and 14 have one). Layer 20's main-KV
projection does implicitly act on H_19, because layer 20's input hidden state *is* the encoder's
last hidden state, but there is exactly one such projection rather than the per-layer W^KV_l,
W^Z_l family of tech-report Eq. (1) (tech_report.txt p.9 line 350).

**D3 — The spec calls layers 0..1 "SWA-only" and then calls layer 1 the carrier of the first Engram
module.** That is self-consistent, but the Engram contribution lands *before* the block
(inference/model.py:1262-1264), i.e. on the input of layer 1, so an engine cannot fold it into
layer 0's output.

**D4 — qk_rope_head_dim = 64 is double-counted in the spec's KV arithmetic.** See
08-kv-footprint.md section 5: the spec's per-entry sum ("256 bytes + 32 bytes ... plus
qk_rope_head_dim = 64 and Indexer K ...") implies 2.5 x 420 = 1050 B/token, not the 890 B/token
the same sentence claims. 64 is already inside head_dim = 512.

**D5 — The spec's "6 heads per rank" Engram split is a deployment choice, not a model property.**
The spec asserts that "Rank r in {0,1,2,3} owns exactly 6 heads (d_mem,local = 1536)" (section 3.2).
The reference shards Engram tables by **row**, not by head (inference/model.py:303-305;
inference/convert.py:135-141): every rank receives all 24 hash columns, the wkv GEMM is
column-parallel, and rows are split ceil(rows/world_size) with zero/one padding. Head-splitting the
lookup is possible, but it turns wkv's all-reduce into an all-gather of 1536-dim slices and is not
what the reference does.

**D6 — The spec has one coefficient predictor per block; the checkpoint has two.** See
02-mhc.md section 5: hc_attn_* and hc_ffn_* are separate fp32 tensors in all 40 layers, and the
attention input mix consumes the *previous* block's ffn_pre.

**D7 — The spec omits attn_sink, hc_scale and engram.q_weight/k_weight from every memory budget.**
They are small in bytes (attn_sink = 40 x 64 x 4 B = 10 KiB) but are required for numerical parity;
attn_sink participates in the softmax denominator (inference/kernel.py:382-383).

**D8 — "Decoder Groups 2-5 ... 12 Reuse layers" undercounts the model-wide Reuse population.**
Within layers 24..39 the count is right (4 Reindex + 12 Reuse), but across the whole model the Reuse
population is **30 layers** (38 CSA2 layers minus 3 Full, 1 Full+HSI and 4 Reindex); see
01-csa2-attention.md section 2.

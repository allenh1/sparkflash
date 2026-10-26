# 00 — Overview: Hyperparameters, Parameter Arithmetic, Checkpoint Tensor Map

"The spec" in these documents is the project's initial design specification, written before the
checkpoint was studied; the "Spec deltas" sections record where it was wrong. It was removed before
release and remains in the git history.

Sources of truth used in this document (nothing here is inferred from the spec):

- <checkpoint>/config.json — released HF config (the **real** hyperparameters).
- <checkpoint>/inference/config.json — the runtime config that generate.py feeds
  straight into ModelArgs(**json.load(...)) (inference/generate.py:121-123).
- <checkpoint>/inference/model.py — reference forward implementation.
- <checkpoint>/model.safetensors.index.json — 96,085 tensor names in 48 shards,
  metadata.total_size = 510286023000.
- <checkpoint>/tech_report.txt — the 51-page report (cited as tech_report.txt p.N).

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

### 2.1 Backbone — 551.6B (the report says 552B)

Per layer (dim=5120, H=64, head_dim=512, q_lora=1280, o_groups=8, o_lora=1024, E=384, EI=2304):
every row is a count from the module constructors in `inference/model.py` plus `config.json`, and the
rows are meant to be added up.

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
| **per layer: the 15 rows above, summed** | **13,754,511,990** |
| 40 layers = 13,754,511,990 x 40 | 550,180,479,600 |
| embed.weight 129280x5120 | 661,913,600 |
| head.weight 129280x5120 | 661,913,600 |
| norm.weight | 5,120 |
| source-layer extras: compressor.norm 4x512 + compressor.wkv 4x2,621,440 + compressor.wgate 3x2,621,440 + indexer.wk 4x65,536 + indexer.k_norm 4x128 + indexer.wq_b 8x5,242,880 + indexer.weights_proj 8x163,840 | 61,868,544 |
| **backbone total: 550,180,479,600 + 661,913,600 + 661,913,600 + 5,120 + 61,868,544** | **551,566,180,464 (about 551.6B)** |

The per-layer rows and the extras are exact counts, not estimates; the only rounding in this section
is the "551.6B". The 15-row per-layer sum is 13,754,511,990, and
13,754,511,990 x 40 + 661,913,600 + 661,913,600 + 5,120 + 61,868,544 = 551,566,180,464. The extras
row is the seven source-layer families the 15 rows leave out: the compressor (4 norms of 512, 4
`wkv` of 512x5120, 3 `wgate` of 512x5120) and the indexer (4 `wk` of 128x512, 4 `k_norm` of 128,
8 `wq_b` of 4096x1280, 8 `weights_proj` of 32x5120). The embed and head rows are 129,280 x 5120 =
661,913,600 each: V = 129,280 is `config.json`'s `text_config.vocab_size`, and the byte identity pins it
independently (section 2.4: embed + head = 2,647,654,400 bytes).

The paper says 552B backbone parameters (tech_report.txt p.7, line 285); the released config gives
551,566,180,464, within 0.08% (a parameter-count comparison, not the byte identity, which is exact:
section 2.4). The 3 MTP/DSpark stages add about 7.6B (section 2.3).

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

### 2.4 Checkpoint byte layout derived from the constructors, closed against total_size

Every row is derived from the module constructors, and the total closes **exactly**. Sizes:
fp8 1 byte/param plus one ue8m0 byte per 32x32 block; packed MXFP4 0.5 byte/param plus one ue8m0 byte
per 32 elements along K; bf16 2 bytes/param for the unquantised weights (embeddings, head, norms,
the gate weight, the Engram q/k weights, the compressor wkv/wgate, the DSpark confidence projection);
fp32 4 bytes/param for attn_sink, the HC parameters and the gate biases; ue8m0 1 byte per scale
element:

| Group | Bytes |
| :--- | ---: |
| routed experts, backbone (40 x 384 x 3 FP4 + 3 E8M0 scales) | 288,777,830,400 |
| shared experts, backbone (40 x 3 FP8 + 3 E8M0) | 1,416,960,000 |
| routed experts, MTP (3 x 128 x 3 + 3) | 7,219,445,760 |
| shared experts, MTP (3 x 3 + 3) | 106,272,000 |
| dense attention, backbone (40 layers; wo_a scale 32x32) | 5,069,578,240 |
| attention + FFN norms (43 layers: q_norm 1280, kv_norm 512, attn_norm/ffn_norm 5120) | 1,034,752 |
| dense attention, MTP (3 layers) | 380,218,368 |
| MoE gates, backbone + MTP (bf16 weight, fp32 biases) | 161,344,512 |
| embed + head | 2,647,654,400 |
| final norm.weight | 10,240 |
| HC parameters (40 + 3 layers, fp32) | 169,092,168 |
| compressor + indexer | 81,835,008 |
| Engram tables (FP8 plus ue8m0 per 32) | 202,758,032,400 |
| Engram projections (2 layers; q/k bf16) | 315,043,840 |
| MTP extras (main_proj, stage norms, markov head, confidence head) | 211,133,952 |
| image path (vision 823,685,120 + aligner 146,821,120 + 3 delimiters 30,720, bf16) | 970,536,960 |
| **total** | **510,286,023,000** |
| metadata.total_size | **510,286,023,000** |
| delta | **0 (exact)** |

The identity is what settles choices a safetensors header would otherwise have to settle:
`wo_a`'s scale grid is 32x32, i.e. `[256, 128]` — the 128x128 alternative `[64, 32]` would leave the
total 1,228,800 bytes short (40 layers x 30,720); the compressor `wkv`/`wgate`, the Engram
`q_weight`/`k_weight` and `mtp.2.confidence_head.proj.weight` are stored bf16 even where the
reference promotes the module to fp32 (inference/model.py:444-446, 932 with 940-946, 1092); and the
Engram `q_weight`/`k_weight` are the four tensors whose fp32-vs-bf16 choice is worth 163,840 bytes
(4 x 4 x 5120 x 2). The shard files are not on this machine, so a header read is still
the direct confirmation; this table is the corroboration.

---

## 3. Tensor-name map of the checkpoint

96,085 names across 48 shards. Top-level namespaces: layers.* (93,415), mtp.* (2,401), vision.*
(259), aligner.* (4), embed.weight, head.weight, norm.weight, image_start, image_end,
image_newline.

| Pattern | Count | Notes |
| :--- | ---: | :--- |
| layers.{0..39}.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale} | 40 each | scale only for fp8 tensors |
| layers.N.attn.attn_sink | 40 | no .weight suffix; fp32 |
| layers.N.attn.{q_norm,kv_norm}.weight, attn_norm, ffn_norm | 40 each | bf16 in storage; RMSNorm casts to fp32 internally (model.py:288-293) |
| layers.{2,8,14,20}.attn.compressor.{wkv,norm}.weight | 4 | main-KV compressor |
| layers.{2,8,14}.attn.compressor.wgate.weight | 3 | **absent for layer 20** (ratio 1) |
| layers.{2,8,14,20}.attn.indexer.{wk,k_norm}.weight | 4 | owns_k layers only |
| layers.{2,8,14,20,24,28,32,36}.attn.indexer.wq_b.{weight,scale} | 8 | the only quantised indexer projection |
| layers.{2,8,14,20,24,28,32,36}.attn.indexer.weights_proj.weight | 8 | bf16, no scale (model.py:515, 518) |
| layers.N.ffn.gate.weight | 40 | 384x5120, **not quantised**: the index has no `ffn.gate.scale` and the gate runs the plain fp32 linear path (model.py:811) |
| layers.N.ffn.gate.bias, layers.N.ffn.gate.bias_vl | 40 each | noaux correction bias plus the VL twin |
| layers.N.ffn.experts.{0..383}.{w1,w2,w3}.{weight,scale} | 15,360 each | fp4 packed (int8 view) plus scale |
| layers.N.ffn.shared_experts.{w1,w2,w3}.{weight,scale} | 40 each | 1 shared expert |
| layers.N.hc_attn_fn, hc_ffn_fn | 40 each | 24x20480, fp32 |
| layers.N.hc_attn_base, hc_ffn_base, hc_attn_scale, hc_ffn_scale | 40 each | fp32 |
| layers.{1,14}.engram.embed.weight | 2 | fp8 [rows, 256] |
| layers.{1,14}.engram.embed.scale | 2 | ue8m0 [rows, 8] |
| layers.{1,14}.engram.wkv.{weight,scale} | 2 each | fp8, 25600 x 6144 |
| layers.{1,14}.engram.q_weight, k_weight | 2 each | bf16 [4, 5120] (storage bf16; built outside the fp32 block, model.py:932 + 940-946) |
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

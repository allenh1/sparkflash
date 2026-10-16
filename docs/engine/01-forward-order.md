# 01 — Engine forward order and tensor map, from the reference implementation

This is the reconnaissance slice for the inference engine (milestone 6): the exact order of
operations the engine must reproduce, in the order the released reference executes them, plus every
checkpoint tensor that order consumes.

Sources, all read-only: `inference/model.py`, `inference/kernel.py`, `inference/engram.py`,
`inference/generate.py`, `inference/convert.py`, `inference/config.json`, the repository root
`config.json`, `model.safetensors.index.json` and `tech_report.txt` under
`/home/user/dsv41-research`. Line numbers are from those files as they exist on this machine.

Two conventions used throughout:

- **cited** means the sentence carries a `file:line` that I read.
- **derived** means the sentence is arithmetic over `config.json` and the module constructors in
  `model.py`, because the 48 shard files themselves are **not present on this machine** (section 1).
  Every derived shape is flagged in its row.

The prior architecture work in this repository (`docs/architecture/00-overview.md` …
`08-kv-footprint.md`) was read and cross-checked against the reference; section 9 lists the places
where this document corrects it.

---

## 1. The checkpoint, counted

The index is the whole authority on names. Its own metadata and the counts, from the exact command:

```bash
cd /home/user/dsv41-research && python3 -c "import json; wm=json.load(open('model.safetensors.index.json'))['weight_map']; print('tensors', len(wm)); print('shards', len(set(wm.values())))"
```

raw output:

```
tensors 96085
shards 48
```

and for the declared byte total:

```bash
cd /home/user/dsv41-research && python3 -c "import json; d=json.load(open('model.safetensors.index.json')); print('metadata.total_size', d['metadata']['total_size'])"
```

```
metadata.total_size 510286023000
```

**96,085 tensors in 48 shards**, declared total 510,286,023,000 bytes (≈510.3 GB). Top-level
namespaces, counted over the same index with a `Counter` on the first dot-separated field (the index
collapses to 94 distinct name patterns once every digit run is replaced with `N`; section 10 dumps
all 94):

| Namespace | Tensors | What it is |
| :--- | ---: | :--- |
| `layers.*` | 93,415 | 40 backbone blocks |
| `mtp.*` | 2,401 | 3 DSpark stages |
| `vision.*` + `aligner.*` | 259 + 4 | ViT and its projector; not read by the text forward |
| `embed.weight`, `head.weight`, `norm.weight` | 1 each | top level |
| `image_start`, `image_end`, `image_newline` | 1 each | learned span delimiters |

Arithmetic check of the three big groups (all derived, all exact):

- `layers.*` = 40 × 2,334 + 11 (compressor) + 32 (indexer) + 12 (engram) = 93,415.
  Per layer: 2,334 = 13 attention + 2 norms + 6 hc + 3 gate + 6 shared-expert + 384 experts × 6.
- `mtp.*` = 3 × 798 + 3 (stage 0) + 4 (stage 2) = 2,401.
- Everything else = 259 + 4 + 6 = 269; and 93,415 + 2,401 + 269 = **96,085**.

**Which checkpoint form this is.** The 48 `model-000NN-of-00048.safetensors` shards are the
HuggingFace export; `inference/convert.py` is what turns them into the reference's per-rank files
`model{rank}-mp{mp}.safetensors` (convert.py:154-182, inference/README.md:15-35), and
`generate.py:133` is what the reference actually loads. The names in the index are already in the
reference's own vocabulary (`attn`, `ffn`, `.scale`, `.bias`), so convert.py's renames
(convert.py:105-126) are no-ops for this checkpoint; what it does do is dequantise `wo_a` to bf16
(convert.py:156-173), view expert `int8` tensors as `float4_e2m1fn_x2` (convert.py:174-181) and drop
the tied MTP embedding/head names (convert.py:108-110 — they are not in the index either, which is
how the stage-level head stays aliased to the backbone, model.py:1212-1213).

**What that means for this engine.** Our loader should read the 48 HF shards directly, which means it
must implement the two conversions convert.py performs: packed-FP4 experts and FP8-pair
`wo_a`. Because the shard files are absent here, **no shape or dtype in section 2 was read from a
safetensors header**; shapes come from the module constructors plus the config, and storage dtypes
come from the presence of a `.scale` sibling (quantised) or from the parameter's dtype at its
construction site (unquantised), except where `convert.py` fixes them. Those derived dtypes are
corroborated by a byte identity: the checkpoint plan derived from this section sums to
`metadata.total_size` **exactly** (510,286,023,000, decomposition in
`docs/architecture/00-overview.md` section 2.4), and it only closes with the dtypes stated here.
A safetensors header read would still be the direct confirmation.

---

## 2. Tensor map

Symbols, all from the root `config.json` text_config: `d` = 5120, `h` = 64 heads, `hd` = 512,
`rd` = 64 (the rope slice), `q` = 1280 (q_lora_rank), `og` = 8, `ol` = 1024, `mi` = 2304,
`E` = 384 routed experts (`Em` = 128 for MTP), `V` = 129280, `hc` = 4 (`mix_hc` = (2+hc)·hc = 24,
`hc_dim` = hc·d = 20480), `idh` = 128, `inh` = 32, `eng_hd` = 256, `eng_cols` = 3·8 = 24.
`B` = 32 is the FP8 block (root config `weight_block_size: [32, 32]`); FP4 weights carry one scale
per 32 along K (kernel.py:582). `fp8` below means E4M3 weight + E8M0 (ue8m0) block scale; `fp4`
means packed E2M1 (two per byte, last dim halved) + E8M0 scale.

### 2.1 Top level and backbone, unconditioned

| Name pattern | Count | Logical shape | Storage dtype | Read by |
| :--- | ---: | :--- | :--- | :--- |
| `embed.weight` | 1 | [V, d] = [129280, 5120] | bf16 | `ParallelEmbedding.forward`, model.py:166, 174, 1253 |
| `head.weight` | 1 | [V, d] | bf16 in the file, fp32 in the module | `ParallelHead`, model.py:1005-1012, 1269 |
| `norm.weight` | 1 | [d] | bf16 | final `RMSNorm`, model.py:1205, 1269 |
| `image_start`, `image_end`, `image_newline` | 1 each | [d] | bf16 | VL span overwrite only, model.py:1220-1222, 1235-1237 |
| `layers.N.attn.attn_sink` | 40 | [h] = [64] | fp32 | `Attention.__init__`, model.py:639; added to the softmax denominator, kernel.py:382-383 |
| `layers.N.attn.wq_a.weight` | 40 | [q, d] = [1280, 5120] | fp8 | model.py:640, 770 |
| `layers.N.attn.wq_a.scale` | 40 | [40, 160] | E8M0 | fp8_gemm, kernel.py:231-232 |
| `layers.N.attn.q_norm.weight` | 40 | [q] | bf16 | model.py:641, 770 |
| `layers.N.attn.wq_b.weight` | 40 | [h·hd, q] = [32768, 1280] | fp8 | model.py:642, 771 |
| `layers.N.attn.wq_b.scale` | 40 | [1024, 40] | E8M0 | |
| `layers.N.attn.wkv.weight` | 40 | [hd, d] = [512, 5120] | fp8 | model.py:643, 705 |
| `layers.N.attn.wkv.scale` | 40 | [16, 160] | E8M0 | |
| `layers.N.attn.kv_norm.weight` | 40 | [hd] | bf16 | model.py:644, 705 |
| `layers.N.attn.wo_a.weight` | 40 | [og·ol, h·hd/og] = [8192, 4096] | fp8 in the file; convert.py dequantises to bf16 | model.py:645-649, 786-787 |
| `layers.N.attn.wo_a.scale` | 40 | [256, 128] **or** [64, 32] | E8M0 — see section 8 | convert.py:157-164 |
| `layers.N.attn.wo_b.weight` | 40 | [d, og·ol] = [5120, 8192] | fp8 | model.py:650, 788 |
| `layers.N.attn.wo_b.scale` | 40 | [160, 256] | E8M0 | |
| `layers.N.attn_norm.weight` | 40 | [d] | bf16 | pre-attention norm, model.py:933, 984 |
| `layers.N.ffn_norm.weight` | 40 | [d] | bf16 | pre-FFN norm, model.py:934, 991 |
| `layers.N.hc_attn_fn` | 40 | [mix_hc, hc_dim] = [24, 20480] | fp32 | `hc_mixes` for attention, model.py:941, 982 |
| `layers.N.hc_ffn_fn` | 40 | [24, 20480] | fp32 | `hc_mixes` for the FFN, model.py:942, 989 |
| `layers.N.hc_attn_base`, `layers.N.hc_ffn_base` | 40 each | [24] | fp32 | model.py:943-944, 955 |
| `layers.N.hc_attn_scale`, `layers.N.hc_ffn_scale` | 40 each | [3] | fp32 | model.py:945-946, 955 |
| `layers.N.ffn.gate.weight` | 40 | [E, d] = [384, 5120] | **unquantised** (no `.scale` in the index) | `Gate.forward` in fp32, model.py:805, 811 |
| `layers.N.ffn.gate.bias` | 40 | [E] | fp32 | noaux selection bias, model.py:806, 822 |
| `layers.N.ffn.gate.bias_vl` | 40 | [E] | fp32 | image-span bias, model.py:807, 819-820 |
| `layers.N.ffn.shared_experts.w1.weight` / `.w3.weight` | 40 each | [mi, d] = [2304, 5120] | fp8 | model.py:887, 836 |
| `layers.N.ffn.shared_experts.w1.scale` / `.w3.scale` | 40 each | [72, 160] | E8M0 | |
| `layers.N.ffn.shared_experts.w2.weight` | 40 | [d, mi] = [5120, 2304] | fp8 | |
| `layers.N.ffn.shared_experts.w2.scale` | 40 | [160, 72] | E8M0 | |
| `layers.N.ffn.experts.{0..383}.w1.weight` / `.w3.weight` | 15,360 each | [mi, d/2] = [2304, 2560] packed, logical [2304, 5120] | fp4 | model.py:836, 873-885; convert.py:174-181 |
| `layers.N.ffn.experts.{0..383}.w1.scale` / `.w3.scale` | 15,360 each | [mi, d/32] = [2304, 160] | E8M0 | |
| `layers.N.ffn.experts.{0..383}.w2.weight` | 15,360 | [d, mi/2] = [5120, 1152] packed, logical [5120, 2304] | fp4 | |
| `layers.N.ffn.experts.{0..383}.w2.scale` | 15,360 | [d, mi/32] = [5120, 72] | E8M0 | |

The expert count is the largest single group: 40 × 384 × 3 × 2 = 92,160 names, of which the index
confirms 46,080 `.weight` and 46,080 `.scale`.

### 2.2 Backbone, conditioned on the config lists

| Name pattern | Count | Layers | Logical shape | Storage dtype | Read by |
| :--- | ---: | :--- | :--- | :--- | :--- |
| `layers.N.attn.compressor.wkv.weight` | 4 | 2, 8, 14, 20 | [hd, d] = [512, 5120] | **bf16 in storage**; the module is promoted to fp32 for ratio > 1 (2, 8, 14) and stays bf16 at ratio 1 (20) | model.py:444-446, 462, 465 |
| `layers.N.attn.compressor.norm.weight` | 4 | 2, 8, 14, 20 | [hd] | bf16 in storage; not part of the fp32 promotion (only `wkv`/`wgate` are) | model.py:443, 462/485 |
| `layers.N.attn.compressor.wgate.weight` | 3 | 2, 8, 14 | [hd, d] | **bf16 in storage**; fp32 in the module (created only for ratio > 1) | model.py:447-448, 465 — **absent for layer 20** |
| `layers.N.attn.indexer.wk.weight` | 4 | 2, 8, 14, 20 | [idh, hd] = [128, 512] | bf16 | model.py:518, 544 |
| `layers.N.attn.indexer.k_norm.weight` | 4 | 2, 8, 14, 20 | [idh] | bf16 | model.py:519, 544 |
| `layers.N.attn.indexer.wq_b.weight` | 8 | 2, 8, 14, 20, 24, 28, 32, 36 | [inh·idh, q] = [4096, 1280] | fp8 | model.py:514, 550 |
| `layers.N.attn.indexer.wq_b.scale` | 8 | same | [128, 40] | E8M0 | |
| `layers.N.attn.indexer.weights_proj.weight` | 8 | same | [inh, d] = [32, 5120] | bf16 | model.py:515, 555 |
| `layers.{1,14}.engram.embed.weight` | 2 | 1, 14 | [rows, eng_hd]: [384,006,168, 256] and [384,016,682, 256] | fp8 | model.py:309, 317; rows from config `engram_num_embeddings` |
| `layers.{1,14}.engram.embed.scale` | 2 | 1, 14 | [rows, 8] | E8M0 | model.py:310, 318 |
| `layers.{1,14}.engram.wkv.weight` | 2 | 1, 14 | [d·(hc+1), eng_cols·eng_hd] = [25600, 6144] | fp8 | model.py:345, 353 |
| `layers.{1,14}.engram.wkv.scale` | 2 | 1, 14 | [800, 192] | E8M0 | |
| `layers.{1,14}.engram.q_weight`, `layers.{1,14}.engram.k_weight` | 2 each | 1, 14 | [hc, d] = [4, 5120] | bf16 in storage and in the module, cast fp32 in use | model.py:347-348, 356; `Engram` is built at model.py:932, outside the fp32 block at 940-946 |

### 2.3 MTP / DSpark stages (`N` ∈ {0, 1, 2})

| Name pattern | Count | Logical shape | Storage dtype | Read by |
| :--- | ---: | :--- | :--- | :--- |
| `mtp.N.attn.attn_sink`, `mtp.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}` | 3 each | identical to 2.1 | identical | `DSparkAttention`, model.py:1032-1074 |
| `mtp.N.attn.q_norm.weight`, `mtp.N.attn.kv_norm.weight`, `mtp.N.attn_norm.weight`, `mtp.N.ffn_norm.weight` | 3 each | as 2.1 | bf16 | |
| `mtp.N.hc_attn_fn`, `mtp.N.hc_ffn_fn` | 3 each | [24, 20480] | fp32 | `Block.forward` via `DSparkBlock`, model.py:1100-1126 |
| `mtp.N.hc_attn_base`, `mtp.N.hc_ffn_base` | 3 each | [24] | fp32 | |
| `mtp.N.hc_attn_scale`, `mtp.N.hc_ffn_scale` | 3 each | [3] | fp32 | |
| `mtp.N.ffn.gate.weight` | 3 | [Em, d] = [128, 5120] | unquantised | model.py:798, 805; counts from `get_moe_config`, model.py:142-149 |
| `mtp.N.ffn.gate.bias`, `mtp.N.ffn.gate.bias_vl` | 3 each | [128] | fp32 | **bias_vl is present but unused**: `DSparkBlock.forward` passes no image mask, model.py:1126 |
| `mtp.N.ffn.experts.{0..127}.w1/w3.weight` | 384 each | [2304, 2560] packed | fp4 | |
| `mtp.N.ffn.experts.{0..127}.w1/w3.scale` | 384 each | [2304, 160] | E8M0 | |
| `mtp.N.ffn.experts.{0..127}.w2.weight` | 384 | [5120, 1152] packed | fp4 | |
| `mtp.N.ffn.experts.{0..127}.w2.scale` | 384 | [5120, 72] | E8M0 | |
| `mtp.N.ffn.shared_experts.w{1,2,3}.{weight,scale}` | 9 each | as 2.1 | fp8 / E8M0 | model.py:887 |
| `mtp.0.main_proj.weight` | 1 | [d, d·3] = [5120, 15360] | fp8 | `forward_embed`, model.py:1113, 1130 |
| `mtp.0.main_proj.scale` | 1 | [160, 480] | E8M0 | |
| `mtp.0.main_norm.weight` | 1 | [d] | bf16 | model.py:1114, 1130 |
| `mtp.2.norm.weight` | 1 | [d] | bf16 | final stage norm, model.py:1116, 1145 |
| `mtp.2.markov_head.embed.weight` | 1 | [V, 256] | bf16 | `DSparkMarkovHead`, model.py:1080, 1084 |
| `mtp.2.markov_head.head.weight` | 1 | [V, 256] | bf16 in the file, fp32 in the module | model.py:1081, 1085 |
| `mtp.2.confidence_head.proj.weight` | 1 | [1, d + 256] = [1, 5376] | **bf16 in storage, fp32 in the module** (model.py:1092 states it) | model.py:1092-1093, 1097 |

### 2.4 Vision

`vision.*` (259 tensors) and `aligner.*` (4) are consumed only when `images` is not `None`
(model.py:1225-1239, 1254-1256). Their shapes were documented in `docs/architecture/06-vision.md` and
are **out of scope for this slice**; this document does not re-verify them.
---

## 3. The forward, in the order the reference runs it

### 3.1 Whole-model entry point (`Transformer.forward`, model.py:1241-1272)

One function serves both phases; `start_pos == 0` is prefill and `start_pos > 0` is one decode step
(model.py:1242-1243). The order is:

1. `image_mask = token_types >= 0` when VL inputs are present, and `engram_mask = ~image_mask`
   (model.py:1249-1251). Text-only passes `None` for both, so nothing is masked.
2. `engram_hashes = self.engram_hash(input_ids, start_pos, engram_mask)` — **one** hashing pass for the
   whole chunk, computed before any layer runs and shared by both Engram layers (model.py:1252;
   `NgramHashState.forward`, engram.py:159-184). Its output is `[b, s, n_engram_layers, 24]`.
3. `h = self.embed(input_ids)` (model.py:1253) — vocabulary-sharded embedding, bf16.
4. If `images` is not `None` and `start_pos == 0`, `merge_image_embeddings` overwrites the image spans
   (model.py:1254-1256, 1228-1239). Text-only skips this.
5. `h = h.unsqueeze(2).repeat(1, 1, hc_mult, 1)` → the stream becomes `[b, s, 4, 5120]`
   (model.py:1258).
6. `pre_mix = make_identity_pre_mix(h, hc_mult)`: a one-hot on copy 0, `[b, s, 4]` (model.py:1260,
   1159-1163).
7. Per layer `i` in ascending order (model.py:1261-1267):
   a. **Engram first, outside the block**: if `layer.engram is not None` (layers 1 and 14), the
      Engram write is `h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], engram_mask)`
      (model.py:1262-1263). It lands on the *input* of layer 1 and layer 14, before the block.
   b. **The DSpark tap**: if `i in target_layer_ids` (37, 38, 39) the block *input* is reduced over
      the hc copies with a plain mean and kept — `main_hiddens.append(h.mean(dim=2))`
      (model.py:1264-1266).
   c. `h, pre_mix = layer(h, start_pos, pre_mix, image_mask)` (model.py:1267).
8. `h = layer.hc_pre(h, pre_mix)` — collapse the 4 copies using the **last** block's `ffn_pre`
   (model.py:1268).
9. `logits = self.head(self.norm(h))` (model.py:1269): final RMSNorm, then the LM head, which slices
   to the last position unless `full_logits` (model.py:1008-1017).
10. `output_ids = sample(logits, temperature)` — Gumbel-max, not multinomial (model.py:1270,
    1285-1292).
11. `main_hidden = torch.cat(main_hiddens, dim=-1)` when the tap fired (model.py:1271).

Return `(output_ids, logits, main_hidden)` (model.py:1272). Nothing anywhere in this path skips a
decoder layer: all 40 blocks run for every token (model.py:1261-1267; `generate.py:67-80`).

### 3.2 Per-block order (`Block.forward`, model.py:968-994)

```text
residual = x
(attn_pre, attn_post, attn_comb) = hc_mixes(x, hc_attn_fn, hc_attn_scale, hc_attn_base)   # 982
x = hc_pre(x, pre_mix)          # the PREVIOUS sublayer's coefficient; line 983
x = attn_norm(x)                                                                            # 984
x = attn(x, start_pos, *attn_args)                                                          # 985
x = hc_post(x, residual, attn_post, attn_comb)                                              # 986
residual = x
(ffn_pre, ffn_post, ffn_comb) = hc_mixes(x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)          # 989
x = hc_pre(x, attn_pre)         # THIS block's attention coefficient                         # 990
x = ffn_norm(x)                                                                             # 991
x = ffn(x, image_mask)                                                                      # 992
x = hc_post(x, residual, ffn_post, ffn_comb)                                                # 993
return x, ffn_pre   # consumed by the next block                                             # 994
```

This is Single-Pass mHC: the input mix uses the *previous* sublayer's `A` (tech_report.txt:586-593),
which is what makes the shift in step 1.5 / step 7c observable in the numerics. The very first block
gets the identity one-hot, and layer 0's attention therefore sees copy 0 only.

- `hc_mixes` (model.py:948-955): flatten to `[b, s, 20480]` in fp32, `rsqrt = rsqrt(mean(x²) +
norm_eps)` (**norm_eps = 1e-20**, not hc_eps), `mixes = F.linear(x, hc_fn) * rsqrt`, then
`hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)`.
- `hc_split_sinkhorn` (kernel.py:406-474) splits `mixes` into the three coefficient sets and
  normalises `comb`:
  - `pre = sigmoid(mixes[0:hc]·scale[0] + base[0:hc]) + eps` (kernel.py:427),
  - `post = 2·sigmoid(mixes[hc:2hc]·scale[1] + base[hc:2hc])` — **no `+ eps`** (kernel.py:429),
  - `comb` raw = `mixes[2hc:]·scale[2] + base[2hc:]` reshaped `[hc, hc]` (kernel.py:430-431),
  - `comb = softmax(comb, -1) + eps` (kernel.py:436-443), then `comb /= (colsum + eps)`
    (kernel.py:445-448), then `sinkhorn_iters - 1 = 19` further alternating row/column
    normalisations (kernel.py:450-458). **Total: 20 row/column passes** = `hc_sinkhorn_iters` = 20,
    with `hc_eps = 1e-6` in every denominator.
- `hc_pre` = `Σ_hc pre_mix ⊙ x` → `[b, s, d]` (model.py:957-960); `hc_post` =
  `post ⊙ x + Σ_hc comb ⊙ residual` → `[b, s, hc, d]` (model.py:962-966).

### 3.3 Attention order (`Attention.forward`, model.py:765-789)

1. `freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]` (model.py:767).
2. `qr = q_norm(wq_a(x))` (model.py:770).
3. `q = wq_b(qr).unflatten(-1, (n_local_heads, 512))` (model.py:771).
4. `apply_rotary_emb(q[..., -64:], freqs_cis)` — **only the last 64 of 512 dims rotate**
   (model.py:772; the slice is `rope_head_dim`, model.py:632).
5. Window path `_window_kv` (model.py:700-720): `kv = kv_norm(wkv(x))`, rope the last 64, then
   `act_quant(kv, 32, "ue8m0", e8m0, inplace=True)` — an FP8 round trip in place (model.py:705-707).
   Prefill writes the chunk (or its last `window_size` tokens, wrapped) into the ring buffer and
   attends over the chunk itself; decode writes one row at `start_pos % 128` and attends over the
   whole ring, oldest first (model.py:708-719; `get_window_topk_idxs`, model.py:409-426).
6. Compressed path, only when `compress_ratio > 0` (model.py:775-778, 739-763): see 3.4 and 3.5.
7. `kv = cat([window_kv, compress_kv], dim=1)` and `topk_idxs = cat([window_idxs, compress_idxs], -1)`
   — one index space, compressed ids shifted by the window length (model.py:777-778).
8. `o = sparse_attn(q, kv, attn_sink, topk_idxs, head_dim^-0.5)` (model.py:780; kernel.py:392-403 and
   311-389): gather `kv` rows by index (index `-1` contributes a zero row and an `-inf` score),
   online softmax over blocks of 64 with a finite `-1e30` floor, `attn_sink` added **to the
   denominator only**, then divide (kernel.py:355, 360-365, 382-385).
9. `apply_rotary_emb(o[..., -64:], freqs_cis, inverse=True)` — the query's rotation is removed from the
   output so the cache stays in one rotated form (model.py:781, 392-406).
10. Grouped low-rank output: `o.view(b, s, 8, -1)`, `wo_a.weight.view(8, 1024, -1)`, einsum
    `"bsgd,grd->bsgr"`, then `wo_b(flatten)` (model.py:785-788). `wo_a` is applied by hand precisely
    because it is block-diagonal per group — it is not a plain `Linear` call (model.py:783-787).

### 3.4 The compressor (main KV), `Compressor.forward`, model.py:458-485

Run only by `is_kv_source` layers (2, 8, 14, 20). Order inside the step:

1. `ratio = compress_ratios[layer_id]` (model.py:439, 460).
2. Ratio 1 short-circuits: `return self.norm(self.wkv(x))` with **no gate and no fp32**
   (model.py:461-462). This is layer 20's path.
3. Ratio > 1 prefill: `kv = wkv(x)`, `score = wgate(x)` in fp32; the trailing partial group
   (`seqlen % ratio` tokens) is parked in `kv_state`/`score_state`; complete groups pool as
   `Σ ratio (kv ⊙ softmax(score, over the ratio axis))` (model.py:464-475). The softmax is over the
   ratio axis per channel, i.e. a learned weighted average of that group's KV latents.
4. Ratio > 1 decode: write this token into slot `start_pos % ratio` and only pool when the group just
   completed (`(start_pos + 1) % ratio == 0`), otherwise return `None` (model.py:476-485). The partial
   state is carried in the two buffers across steps.
5. The returned latent is `norm(kv)` **before RoPE** (model.py:485): the indexer needs the unrotated
   form, and the attention applies RoPE to it afterwards.

### 3.5 The indexer and the HSI pool, model.py:527-580 and 583-610

Run only by `is_index_source` layers (2, 8, 14, 20, 24, 28, 32, 36). The order has a hard dependency:

1. **Index-K publication, before the main KV is rotated**: if the layer `owns_k` (i.e. is a KV source)
   and the compressor produced a latent, then `k = k_norm(wk(latent))`, rope the last 64 at the
   *compressed* positions `j·ratio` (prefill: `freqs_cis[:seqlen - seqlen % ratio : ratio]`; decode:
   `freqs_cis[start_pos + 1 - ratio]`), then `fp4_act_quant(k, 32, inplace=True)` with **E8M0**
   scales, write it into `k_cache[start_pos // ratio : ...]` and publish `shared_attn.index_k`
   (model.py:537-548).
2. `q = wq_b(qr)` → `[b, s, 32, 128]`, rope the last 64 at the token positions, then the same FP4
   round trip (model.py:550-552).
3. `index_k = shared_attn.index_k[:b, :end_pos // ratio]` — this is layer 20's cache for layers 24, 28,
   32, 36, because they do not own K (model.py:500, 554).
4. `weights = weights_proj(x) * (index_head_dim^-0.5 * index_n_heads^-0.5)` (model.py:555).
5. `index_score = einsum("bshd,btd->bsht", q, index_k)`, then
   `(relu(index_score) * weights.unsqueeze(-1)).sum(dim=2)` — rectified, then head-mixed
   (model.py:556-557).
6. Reachability mask: prefill masks every compressed position the query cannot see yet
   (`arange(seqlen // ratio) >= arange(1, seqlen+1) // ratio`), decode just carries
   `end_pos // ratio` (model.py:561-567).
7. **HSI**: if this layer is the candidate source (layer 20), `select_candidate_blocks` keeps the
   `candidate_topk_blocks = 2048` best-scoring blocks of `candidate_block_size = 8` positions and
   publishes the boolean mask (model.py:569-572, 583-610). Otherwise, if the layer comes after the
   source, its scores outside the pool are set to `-inf` (model.py:573-575).
8. `topk = min(index_topk = 512, end_pos // ratio)`; take the top `topk` entries without sorting, then
   sort them into position order; return `where(idxs < compress_lens, idxs + offset, -1)`
   (model.py:578-580). The `offset` is the window KV length so both id sets live in one index space
   (model.py:776).

Two details an engine must preserve: the pool is **2048 × 8 = 16,384 candidate positions**
(tech_report.txt:541-547), and the indexer runs **before** the latent is rotated and written to the
compress cache, because the same storage is then overwritten with the rotated, FP4-quantised values
(model.py:531, 749-761).

### 3.6 MoE order (`MoE.forward`, model.py:889-904)

1. Flatten to `[n, d]`; `weights, indices = self.gate(x, image_mask)` (model.py:891-892).
2. `y = zeros(n, d, fp32)`; `counts = bincount(indices.flatten(), minlength=n_routed_experts)`
   (model.py:893-894).
3. For each **local** expert index `i` in ascending order, skip if `counts[i] == 0`, else gather its
   tokens and accumulate `y[idx] += expert(x[idx], weights[idx, top, None])` (model.py:895-900).
   The iteration order is an implementation detail of this reference (a Python loop); the operator
   order that matters is *sorted by global expert index*, which is also what makes the fp32
   accumulation deterministic for a fixed routing.
4. `all_reduce(y)` in fp32 when `world_size > 1` (model.py:901-902) — experts are **whole-expert
   sharded**, not width-sharded (model.py:863-885; convert.py:129-134).
5. `y += self.shared_experts(x)` — the shared expert always runs, on the same input, after the
   routed sum (model.py:903).
6. `return y.type_as(x).view(shape)` (model.py:904).

Gate (`Gate.forward`, model.py:809-827): `scores = linear(x.float(), gate.weight.float()) / gate_temp`;
`sqrtsoftplus` (`F.softplus(scores).sqrt()`, model.py:816-817) or softmax/sigmoid; the noaux bias is
added **only for selection** — `indices = (scores + bias).topk(topk).indices`, `weights =
scores.gather(indices)` from the unbiased scores (model.py:821-823); then `weights /= (sum + 1e-20)`
when `norm_topk_prob` and `topk > 1` (model.py:824-825), then `weights *= route_scale` (model.py:826).
Image-span tokens use `bias_vl` instead of `bias` (model.py:818-820).

Expert (`Expert.forward`, model.py:841-851): `gate = w1(x).float()`, `up = w3(x).float()`, clamp `up`
to `±swiglu_limit` and `gate` to `≤ swiglu_limit`, `x = silu(gate) * up`, scale by the routing weight if
given, then `w2(x.to(dtype))`. Note the activation for `w2` is cast to the *input* dtype before its own
quantiser runs; each of the three GEMMs quantises its own activation independently.

### 3.7 Engram order (`Engram.forward`, model.py:350-365)

Called from the layer loop, before the block, for layers 1 and 14 (model.py:1262-1263). Input `x` is
`[b, s, 4, d]` (all hc copies) and `hash_ids` is `[b, s, 24]` for that layer.

1. Lookup: `ParallelEngramEmbedding.forward` takes the fp8 table and the fp8 scale table by index,
   dequantises per 32-block (`values.unflatten(-1, (-1, 32)) * scales`), casts to bf16, zero-fills
   rows that belong to another rank, and all-reduces across ranks (model.py:312-325).
2. `flatten(-2)` → `[b, s, 24·256 = 6144]`, then `kv = wkv(...)` (fp8 GEMM) → `[b, s, 25600]` split
   into `key [b, s, 4, 5120]` and `value [b, s, 5120]` (model.py:353-354).
3. Gate (model.py:356-362): `weight = q_weight * k_weight` (an fp32 product, never used separately);
   `rstd = rsqrt(mean(h², -1) + eps) * rsqrt(mean(key², -1) + eps)` with `eps = norm_eps = 1e-20`,
   normalised **per (token, hc copy) over dim**, not jointly over copies;
   `dot = Σ_d (h · weight · key) · rstd · d^-0.5`;
   `gate = sigmoid(sign(dot) · sqrt(max(|dot|, 1e-6)))` — a signed square root before the sigmoid.
4. `gate = gate.masked_fill(~token_mask, 0)` (image positions get no Engram contribution)
   (model.py:363-364).
5. Write: `h + gate.unsqueeze(-1) * value.unsqueeze(-2)`, cast back to `x.dtype` (model.py:365). The
   addition is per hc copy and shares one value vector.

### 3.8 DSpark order (model.py:1100-1156, 1274-1282)

`Transformer.forward_spec(input_ids, main_hidden, start_pos)` is the entry point (model.py:1274-1282);
it is **not called by generate.py** (section 7).

1. `forward_embed` (model.py:1128-1135): `main_x = main_norm(main_proj(cat(main_hiddens)))` — the three
   taps of layers 37, 38, 39 concatenated to `[b, s, 15360]`; then `draft_input_ids = full([b, 5],
   noise_token_id = 128799)` with position 0 overwritten by the sampled token; `x = embed(...)` →
   `[b, 5, 5120]`, expanded to hc copies (`unsqueeze(2).repeat`) → `[b, 5, 4, 5120]`.
2. `pre_mix = make_identity_pre_mix(...)`, then the three stages in order (model.py:1277-1279).
3. `DSparkBlock.forward` (model.py:1122-1126): if `start_pos == 0` only the attention's prefill branch
   runs, which **seeds the window cache and returns `x` unchanged** — a prefill pass produces no draft
   logits. Otherwise the normal `Block.forward` runs with `main_x` as the attention's extra argument.
4. `DSparkAttention.forward` (model.py:1033-1074): assert `compress_ratio == 0`; build the main KV from
   `main_x` (`kv_norm(wkv(main_x))`, rope, FP8 round trip) and, in prefill, only write it into the ring
   buffer; in decode, build q and kv from the draft positions, rope them at
   `start_pos + 5 .. start_pos + 10` (model.py:1055), set `topk_idxs = [window slots ; 5 draft slots]`
   (`get_dspark_topk_idxs`, model.py:1021-1029 — the 5 draft positions see the whole 128-slot window
   plus all 5 draft keys), write the main KV into the ring, concatenate, run `sparse_attn`, undo the
   rotation, and apply the grouped `wo_a`/`wo_b` pair. There is **no intra-block causal mask**: the 5
   positions are computed fully connected.
5. `forward_head` (model.py:1137-1156) on the last stage only: `x = hc_pre(x, pre_mix)`;
   `logits = head(norm(x), full_logits=True)` → `[b, 5, V]`; then for `i` in 0..4:
   `logits_bias, markov_embed = markov_head(output_ids[:, i])`, `logits[:, i] += logits_bias`,
   `output_ids[:, i+1] = sample(logits[:, i], temperature)`, collecting the Markov embeddings; finally
   `confidence = confidence_head(x, stack(markov_embeds))` (model.py:1149-1155). Note the Markov bias is
   **added to the base logits**, and the sampled token at position `i` feeds the Markov head at
   position `i+1` — that is the only sequential dependency in the block.
6. `forward_spec` returns `None` when `start_pos == 0`, else `(output_ids, logits, confidence)` where
   `output_ids` is `[b, 6]` (the anchor plus 5 drafts) and `confidence` is `[b, 5]` raw fp32
   (model.py:1280-1282, 1146-1156).

### 3.9 Prefill versus decode, side by side

| Aspect | Prefill `start_pos == 0` | Decode `start_pos > 0` |
| :--- | :--- | :--- |
| Window cache | whole chunk written (wrapped if longer than 128); attention over the chunk | one row at `start_pos % 128`; attention over the whole 128-slot ring |
| Window indices | per-query causal window, `-1` before the sequence start (model.py:417-420) | all 128 slots oldest-first, `-1` for unfilled slots (model.py:421-424) |
| Compressor | pools complete groups, parks the remainder in state (model.py:466-475) | fills one slot, pools only on a completed group, else `None` (model.py:476-485) |
| Index-K positions | `freqs_cis[: seqlen - seqlen % ratio : ratio]` (model.py:540) | `freqs_cis[start_pos + 1 - ratio]` (model.py:542) |
| Reachability | per-query mask (model.py:563-565) | scalar `end_pos // ratio` (model.py:567) |
| Draft (DSpark) | attention seeds the cache, returns `x`; `forward_spec` returns `None` | full block + head; returns 5 drafts, 5 logits, confidence |

---

## 4. Per-layer branch selection

Predicates, straight from the code (model.py:653-661, 500-503):

| Predicate | Expression | Effect |
| :--- | :--- | :--- |
| `is_kv_source` | `layer_id in kv_source_layers and layer_id < n_layers` | owns a `Compressor` + `compress_kv_cache`; publishes the shared main KV |
| `is_index_source` | `layer_id in index_source_layers and layer_id < n_layers` | owns an `Indexer`; publishes `topk_idxs` |
| `owns_k` (indexer) | `layer_id in kv_source_layers` | builds index K from its own latent; every other indexer reads it |
| `is_candidate_source` | `layer_id == candidate_source_layer` | builds the HSI candidate pool |
| `uses_candidates` | `0 <= candidate_source_layer < layer_id` | masks its own index scores with that pool |

`kv_source_layers = [2, 8, 14, 20]` and `index_source_layers = [2, 8, 14, 20, 24, 28, 32, 36]`
(root config.json:106-121), `candidate_source_layer_id = 20` (line 125). The resulting map, matching
the tensor counts in section 2.2 exactly:

| Layers | ratio | Mode (report's name, tech_report.txt:465-524) | Owns | Reads from |
| :--- | ---: | :--- | :--- | :--- |
| 0, 1 | 0 | SWA only | — | — (layer 1 also carries Engram) |
| 2 | 2 | Full | compressor, indexer, index K, HSI pool input | — |
| 3..7 | 2 | Reuse | — | layer 2's KV and indices |
| 8 | 2 | Full | compressor, indexer, index K | — |
| 9..13 | 2 | Reuse | — | layer 8's |
| 14 | 2 | Full (plus the second Engram) | compressor, indexer, index K | — |
| 15..19 | 2 | Reuse | — | layer 14's |
| 20 | 1 | Full + HSI source | compressor (ratio 1, no gate), indexer, index K, candidate pool | — |
| 21..23 | 1 | Reuse | — | layer 20's |
| 24 | 1 | Reindex | indexer | layer 20's KV and index K; own Top-512 inside the pool |
| 25..27 | 1 | Reuse | — | layer 24's indices |
| 28, 32, 36 | 1 | Reindex | indexer | as layer 24, in their groups |
| 29..31, 33..35, 37..39 | 1 | Reuse | — | their group's Reindex layer |
| 40..42 (MTP) | 0 | SWA only (DSpark) | — | — (`layer_id >= n_layers` disqualifies them) |

Counts: 38 layers have ratio > 0 — 3 Full, 1 Full+HSI, 4 Reindex, 30 Reuse. This reproduces the
report's own figures (tech_report.txt:541-555 for the pool, 465-524 for the modes).

---

## 5. Constants that must be wired verbatim

| Constant | Value | Source |
| :--- | :--- | :--- |
| vocab_size / eos / pad | 129280 / 1 / 2 | config.json:24, 9-10 |
| image_token_id | 129264 | config.json:11 |
| max_position_embeddings | 1,048,576 | config.json:43 |
| routing function | `sqrtsoftplus`: `sqrt(softplus(x))` | config.json:55; model.py:816-817 |
| routed experts / activated / shared | 384 / 6 / 1 | config.json:52-54 |
| route_scale, norm_topk_prob | 1.5, true | config.json:57-58; model.py:824-826 |
| gate_temp | 1.0 (not a config key; `ModelArgs` default) | model.py:67, 811 |
| swiglu_limit | 10.0 | config.json:36; model.py:845-847 |
| MoE reduction epsilon | `1e-20` (the literal, **not** `norm_eps`) | model.py:825 |
| hc_mult, sinkhorn iters, hc_eps | 4, 20, 1e-6 | config.json:128-130; kernel.py:450 |
| norm_eps | 1e-20 | config.json:37; model.py:75 |
| heads / head_dim / rope slice | 64 / 512 / last 64 | config.json:28, 30-31; model.py:772 |
| window_size | 128 | config.json:59 |
| rope_theta (SWA layers) | 10000 | config.json:44 |
| compress_rope_theta, original_seq_len, factor, betas | 160000, 65536, 16, 32/1 | config.json:105, 50, 47-49 |
| YaRN selection | applied only when `compress_ratio > 0`; SWA layers set `original_seq_len = 0` | model.py:680-696 |
| compress_ratios | 43 entries: 0,0, then 18×2, then 20×1, then 3×0 | config.json:60-104 |
| kv_source / index_source / candidate source | [2,8,14,20] / [2,8,14,20,24,28,32,36] / 20 | config.json:106-125 |
| index_n_heads / index_head_dim / index_topk | 32 / 128 / 512 | config.json:122-124 |
| candidate_topk_blocks / candidate_block_size | 2048 / 8 → 16,384 positions | config.json:126-127 |
| softmax scales | main `512^-0.5`; indexer `128^-0.5 · 32^-0.5` | model.py:651, 513, 555 |
| Engram layers | 1, 14 | config.json:131-134 |
| Engram rows per layer | 384,006,168 and 384,016,682 | config.json:135-138 |
| Engram n-gram orders | 2..4 (`max_ngram_size` 4 → 3 orders × 8 heads = 24 columns) | config.json:139; engram.py:90-91 |
| Engram prime base / heads / head_dim | 16,000,000 / 8 / 256 | config.json:140-142 |
| Engram pad / compressed vocab | 2 / 99,092 | config.json:143-144; engram.py:146 |
| Engram gate clamp, coefficient | `1e-6`, `q_weight ⊙ k_weight`, `d^-0.5` | model.py:341, 356, 360-362 |
| Engram hash multipliers | from `default_rng(10007 · layer_id)`, forced odd, bounded by `i64max / 99092 / 2` | engram.py:64-83 |
| DSpark stages / block / noise token | 3 / 5 / 128799 | config.json:145-147 |
| DSpark target layers | 37, 38, 39 | config.json:148-152 |
| DSpark markov_rank / experts / activated | 256 / 128 / 3 | config.json:153-155 |
| DSpark draft positions | 5 drafts + 1 anchor; window 128 | model.py:1131-1132, 1021-1029 |

---

## 6. Every quantize–dequantize point, in the order the forward hits them

Weights: `linear()` dispatches on the weight dtype (model.py:181-207). An FP8 weight takes an FP8
activation; an FP4 weight takes **the same FP8 activation**, and the FP4 operand is cast to FP8
through FP32 inside the kernel (kernel.py:538-540). So in both cases the activation quantiser is
`act_quant(x, 32, "ue8m0", e8m0)` — E4M3, one power-of-two scale per 32 elements, amax floored at
`1e-4` (kernel.py:76-80, 98-124). Weight scales are per 32×32 block for FP8 (config.json:15-18) and
per 32 along K for FP4 (kernel.py:582).

Activations and caches, in forward order:

| # | Where | Call | Format | Notes |
| ---: | :--- | :--- | :--- | :--- |
| 1 | every `wq_a/wq_b/wkv/wo_b` GEMM | `act_quant(x, 32, "ue8m0", e8m0)` | FP8 E4M3, per-32 E8M0 | model.py:187-205 |
| 2 | every expert GEMM (w1, w3, w2) and the shared expert | same activation quant, then `fp4_gemm` | FP8 act × FP4 E2M1 weight, per-32 scales both sides, MMA in fp8 | model.py:186-195; kernel.py:486-498, 538-540 |
| 3 | window / SWA KV | `act_quant(..., inplace=True)` | FP8 round trip in place | model.py:707 |
| 4 | main compressed KV, **after** RoPE | `fp4_act_quant(latent, 16, True, scale_dtype=e4m3)` | FP4 E2M1, one **E4M3** scale per 16 channels | model.py:758-760; kernel.py:160-163 |
| 5 | indexer K, after RoPE | `fp4_act_quant(k, 32, True)` | FP4 E2M1, per-32 **E8M0** | model.py:545-546 |
| 6 | indexer Q, after RoPE | `fp4_act_quant(q, 32, True)` | same as 5 | model.py:551-552 |
| 7 | Engram table rows | stored FP8, dequantised per 32-block on lookup | FP8 + E8M0 scales | model.py:317-320 |
| 8 | DSpark main/draft KV | `act_quant(..., inplace=True)` | FP8 round trip | model.py:1042, 1062 |

Every one of 3-6 and 8 is `inplace=True`, so the reference *quantises and dequantises* and keeps bf16
values in its buffers: the FP4 layout is a training-time and memory-layout contract, not what the
reference buffers hold (kernel.py:81-86, 121-123, 167-172). An engine that stores real FP4 saves the
bytes and must reproduce the same round-off. The main KV is quantised **after** RoPE and includes the
rotated 64 dims in the same block structure (tech_report.txt:662-665).

---

## 7. The DSpark acceptance rule — what the reference does and does not contain

What exists: the confidence head emits a **raw fp32 logit per draft position** with no sigmoid and no
normalisation (model.py:1093-1097), from the collapsed hidden state concatenated with the Markov
embedding of the *input* token at that position (model.py:1155). The report's description of the
mechanism is: "a confidence head predicts per-position conditional acceptance probabilities, which
are used to estimate prefix survival probabilities. The scheduler combines these estimates with
profiled engine throughput curves to dynamically select the verification length for each request"
(tech_report.txt:625-631).

What does not exist anywhere in the released code: any accept/reject test, any survival-probability
computation, any verification-length scheduler, any KV rollback on rejection, and any confidence
threshold. `generate.py` never calls `forward_spec` (generate.py:67-80 only calls `model.forward`), the
only caller is the `__main__` demo (model.py:1295-1309) which ignores the return value's confidence,
and `model.py:129-130` says explicitly that the speculative-decoding loop is out of scope for this
repository. **The acceptance rule is therefore ours to design, and it cannot be validated against
this reference** — only its inputs (the 5 logits and the 5 confidence logits) can be.

---

## 8. What I could not determine from the reference

1. **Any shape or dtype read from a safetensors header.** The 48 shard files are not on this machine
   (only the index, config and tokenizer). Every shape in section 2 is derived from the module
   constructors plus the config, and every unquantised dtype comes from the parameter's dtype at its
   construction site (`"dtype": "bfloat16"`, config.json:6). The byte identity (section 1,
   `00-overview.md` section 2.4) corroborates them but does not replace a header read. Two classes
   were pinned only by that identity and should be checked first when the shards land: the Engram
   `q_weight`/`k_weight` and `mtp.2.confidence_head.proj.weight` are stored bf16 (model.py:932
   outside the fp32 block at 940-946; model.py:1092).
2. **`wo_a`'s scale grid — pinned by the byte identity, not by a header.** convert.py accepts
   either 32×32 or 128×128 and asserts the quotient (convert.py:157-168), and the index alone cannot
   tell them apart. The identity can: `[256, 128]` (32×32) makes the plan exact, while the 128×128
   alternative `[64, 32]` would leave the total 1,228,800 B short (40 layers × 30,720). Worth
   confirming against a header when the shards land.
3. **The MTP acceptance rule** (section 7) — the reference has the confidence logit only.
4. **The serving temperature / sampling policy** for drafts: `ModelArgs.temperature` defaults to 1.0
   (model.py:53) and is a runtime argument (generate.py:105, 123); the config does not fix it.
5. **Whether the released expert tensors really are `int8`-packed FP4.** convert.py asserts
   `dtype == torch.int8` and packs two per byte (convert.py:174-181), but I could not open a shard to
   confirm.
6. **Vision and the image-span protocol** (259 + 4 tensors): documented in
   `docs/architecture/06-vision.md`, not re-verified here.
7. **The exact 99,092-entry compressed token map**: `engram.py:17-61` builds it from the tokenizer's
   normalizer chain and `engram.py:146` asserts the size against the config, but reproducing the
   partition needs the Rust `tokenizers` normalizers; I did not run it.
8. **`max_seq_len`**, which sizes every cache, is a runtime setting (`ModelArgs.max_seq_len`,
   model.py:52; generate.py:126 sets 64K in interactive mode), not a checkpoint property. The config's
   `max_position_embeddings = 1,048,576` is the trained limit, not the allocated size.

---

## 9. Reconciliation with the prior architecture docs

While writing this document I re-checked the prior architecture work against the reference. Eight
claims were wrong; all eight are now fixed in the files named below, so `00-overview.md`,
`03-moe.md`, `DECISIONS.md` (D18) and this document agree on one byte identity.

| # | Claim as written | Correction | Status |
| :--- | :--- | :--- | :--- |
| 1 | `00-overview.md` section 3: `ffn.gate.weight` "384x5120, fp8" | **not quantised**: no `ffn.gate.scale` in the index, and `Gate.forward` runs the plain fp32 `F.linear` path (model.py:811, 206-207); quantising it changes routing | fixed in `00-overview.md` |
| 2 | `00-overview.md` section 3: the norm weights are "fp32" | bf16 in storage, fp32 in compute — `RMSNorm` casts internally (model.py:288-293) and the module takes the ambient bf16 default (generate.py:118) | fixed |
| 3 | `00-overview.md` section 3: `indexer.wq_b.{weight,scale}` and `weights_proj.weight` in one row | only `wq_b` is quantised (model.py:514); `wk` and `weights_proj` are bf16 (model.py:515, 518) | fixed |
| 4 | `00-overview.md` section 3 and its section 2.4 byte table: Engram `q_weight`/`k_weight` are fp32 | bf16 — `Engram` is built at model.py:932, outside the `set_dtype(torch.float32)` block at 940-946; as fp32 the plan overshoots by 4 × 4 × 5120 × 2 = 163,840 B | fixed |
| 5 | `00-overview.md` section 2.4: image path 553,394,784 B | **970,536,960** — vision 823,685,120 + aligner 146,821,120 + the three delimiters 30,720, all bf16 in storage; the old figure could not be reproduced at any uniform dtype | fixed |
| 6 | `03-moe.md`:18 and :121 list `ffn.gate.weight` as fp8 / among the fp8 Dense Linears | not quantised, as item 1 | fixed in `03-moe.md` |
| 7 | `00-overview.md` section 2.4: the reconstruction lands "within 0.09%" of `metadata.total_size` | exact — `metadata.total_size` is the sum of the stored tensor bytes (it contains no index JSON), and the 16 rows there total 510,286,023,000 | fixed |
| 8 | this file, section 2.3: `mtp.2.confidence_head.proj.weight` is fp32 in storage | bf16 in storage, fp32 in the module (model.py:1092); that 10,752 B is the last piece of the identity | fixed above |

Everything else I re-checked in the prior work — the mode assignment and count in
`01-csa2-attention.md:45-71`, the Sinkhorn schedule in `02-mhc.md:59-86`, the W4A8 activation story
in `03-moe.md:116-171` (after item 6), the hash layout in `04-engram.md:104-192`, and the DSpark
wiring gaps in `05-dspark.md:196-247` — still matches the reference as read here. The one prior
claim I cannot certify either way is `06-vision.md`'s vision-tower precision: the byte total it now
carries follows from the constructors and the identity, but I did not re-verify the ViT shapes or
`vision.py` itself (section 8 item 6).

---

## 10. Reproduction

```bash
# the byte identity: the plan derived from the module constructors closes against total_size exactly
python3 - <<'PY'
d=5120; E=384; E2=128; mi=2304; V=129280; R=384006168+384016682
attn=(1280*d+40*160)+(32768*1280+1024*40)+(512*d+16*160)+(8192*4096+256*128)+(d*8192+160*256)+64*4
exp=lambda n: n*((mi*(d//2))*2+(mi*(d//32))*2+(d*(mi//2))+(d*(mi//32)))
shr=lambda n: n*((mi*d+72*160)*2+(d*mi+160*72))
lm=(exp(40*E)+shr(40)+exp(3*E2)+shr(3)
    +40*attn+43*(1280*2+512*2+5120*2+5120*2)+3*attn
    +40*(E*d*2+E*4+E*4)+3*(E2*d*2+E2*4+E2*4)
    +2*(V*d*2)+d*2+43*(24*20480*4*2+24*4*2+3*4*2)
    +4*(512*2+512*d*2)+3*(512*d*2)+8*(4096*1280+128*40+32*d*2)+4*(128*512*2+128*2)
    +R*256+R*8+2*(25600*6144+800*192+4*d*2+4*d*2)
    +d*15360+160*480+d*2+d*2+V*256*2+V*256*2+1*5376*2)
img=32*((1024+1024)+(3072*1024)+3072+(1024*1024)+1024+(5632*1024)+(1024*2816))*2+(1024*588+1024+1024)*2+(5120*9216+5120+5120*5120+5120)*2+3*d*2
print('language model', lm); print('image path', img); print('total', lm+img)
PY
# language model 509315486040 / image path 970536960 / total 510286023000  (= metadata.total_size)

# tensor and shard counts (the numbers asserted in section 1)
cd /home/user/dsv41-research && python3 -c "import json; wm=json.load(open('model.safetensors.index.json'))['weight_map']; print('tensors', len(wm)); print('shards', len(set(wm.values())))"

# declared byte total
cd /home/user/dsv41-research && python3 -c "import json; print('total_size', json.load(open('model.safetensors.index.json'))['metadata']['total_size'])"

# every distinct name pattern in the index, with a count and an example (94 rows)
cd /home/user/dsv41-research && python3 -c "
import json,re,collections
wm=json.load(open('model.safetensors.index.json'))['weight_map']
g=collections.defaultdict(list)
for n in wm: g[re.sub(r'[0-9]+','N',n)].append(n)
for p in sorted(g): print(len(g[p]), p, 'e.g.', sorted(g[p])[0])
"
```

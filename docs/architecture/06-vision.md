# 06 — Vision: DeepSeek-ViT, the 3x3 aligner, and the image-token span protocol

References: /home/user/dsv41-research/inference/vision.py (119 lines),
image_processor.py (173 lines), model.py:1224-1239 and 1253-1256.

**Status (2026-10-18).** This document specifies the reference vision path. The vision forward (the
ViT tower, the 3x3 aligner and the image-token derivation) is not implemented in this tree, and
`TensorPlan::language_model` excludes the 263 `vision.*`/`aligner.*` tensors and the three image
delimiters, so no forward module consumes them. Only the placeholder normalisation and the
image-source/span parsing exist (`crates/sparkflash-api/src/multimodal.rs` and the encoding
`case.rs` image-span handling); they name and count image parts but produce no visual-token ids.
`GET /v1/models` advertises `multimodal: false`, and a request carrying an image part is refused at
the engine seam with `unsupported_capability` / `vision` instead of being charged a fabricated token
count (DECISIONS.md D18 and D27). A raw-body scan (`prompt::find_image_carrier`) backstops the typed
parsers: an image carrier whose spelling the parser does not know (for example the documented
`computer_screenshot` output) is refused with the same 400 rather than ignored.

Config (config.json text_config.vision_config / runtime config.json):
vision_n_layers 32, vision_dim 1024, vision_n_heads 16, vision_inter_dim 2816, vision_patch_size 14,
vision_downsample_ratio 3, vision_max_n_token 1024, vision_min_pixels 295936,
vision_max_wh_ratio null, vision_rope_theta 10000, image_token_id 129264, dim 5120.

---

## 1. Shape inventory

| Component | Shapes | Notes |
| :--- | :--- | :--- |
| PatchEmbed | Linear(3 * 14 * 14 = 588 `->` 1024), with bias | a linear projection, not a conv (vision.py:37-43) |
| Attention | wqkv Linear(1024 `->` 3072) with bias; wo Linear(1024 `->` 1024) with bias | head_dim = 1024/16 = 64, all dims rotated |
| MLP | w1 Linear(1024 `->` 2*2816 = 5632, bias=False); w2 Linear(2816 `->` 1024, bias=False) | w1 output chunked into (gate, up), SiLU(gate) * up (vision.py:63-71) |
| Block | pre-norm RMSNorm(1024) x 2, eps 1e-6, fp32 weight | residual around both sublayers (vision.py:74-84) |
| ViT | 32 blocks + final RMSNorm(1024) | vision.py:87-103 |
| Aligner | w1 Linear(1024 * 9 = 9216 `->` 5120) with bias; w2 Linear(5120 `->` 5120) with bias | gelu between (vision.py:106-119) |
| Max ViT sequence | up to 1024 * 9 = 9216 patches | from vision_max_n_token and the 3x3 downsample |

---

## 2. Image preprocessing, exactly

plan_image_grid (image_processor.py:101-112):

1. If vision_max_wh_ratio were set and width `>` height * ratio, clamp width. **It is null in this
   checkpoint, so this step and the matching branch in load_image are inert** (image_processor.py:104-105,
   122-123).
2. If 0 `<` width * height `<` vision_min_pixels = 295,936 (= 544x544), upscale by
   sqrt(min_pixels / (width * height)) (image_processor.py:106-109).
3. Round the pixel size **up** to a multiple of the patch size, independently per axis
   (image_processor.py:110-111).
4. safe_resize (image_processor.py:60-67) checks whether the token grid fits the 1024-token budget;
   if not, solve_resize_ratio (image_processor.py:46-57) computes the largest aspect-preserving size
   whose grid satisfies n_llm_h * (n_llm_w + 1) + 2 `<=` max_n_token in closed form from
   max_w = sqrt((max_n_token - 2) / r + 0.25) - 0.5, with explicit single-row/single-column
   fallbacks.
5. load_image resizes to exactly (best_width, best_height): a plain resize when the wh-ratio branch
   fires (never here), otherwise ImageOps.pad with mid-grey (127, 127, 127)
   (image_processor.py:122-125).
6. Normalisation is (x / 255 - 0.5) / 0.5 in **float32 then cast to bfloat16**
   (image_processor.py:126-127) -- no mean/std per channel.
7. Patches are cut as [3, n_vit_h, 14, n_vit_w, 14] `->` [n_vit_h * n_vit_w, 3, 14, 14]
   (image_processor.py:128). Row-major patch order.

---

## 3. ViT internals

### 2D RoPE (vision.py:8-21, 92-93, 100)

```
rope_dim = vision_dim // vision_n_heads // 2 = 1024 // 16 // 2 = 32
inv_freq[k] = theta^(-2k / rope_dim),  k = 0..15, theta = 10000
freqs[N, 2k]   = hpos * inv_freq[k]          # N = n_h * n_w, positional grid
freqs[N, 2k+1] = wpos * inv_freq[k]
apply_rotary(x): x1, x2 = chunk(x, 2, -1); out = cat(x1*cos - x2*sin, x2*cos + x1*sin)
```

So the rotary is a **half-split** (not interleaved-pair) rotation applied to the whole 64-dim head,
and the height/width positions alternate across the 16 frequency pairs (vision.py:13). The cos/sin
table is cached by (n_h, n_w, dim, theta) and is shared by all 32 blocks (vision.py:8).

### Attention (vision.py:46-60)

Single-image, fully bidirectional: F.scaled_dot_product_attention with no mask, over
[N, heads, head_dim] after transposing the sequence to the front. There is no KV cache, no sliding
window and no GQA: q, k, v all have 16 heads of dimension 64 produced by one fused wqkv.

### Precision

The tower is bf16 end to end except the RMSNorm weights, which are fp32 and computed in fp32 before
casting back (vision.py:24-34). There is no FP8 in vision.py -- the spec's "BF16 / FP8" for the
vision encoder is a SparkFlash-RS optimisation, not a reference behaviour. The vision parameters are
also **replicated** on every rank: convert.py's shard mapping (convert.py:56-64) contains no vision
entries, so nothing is partitioned.

In the released shards the whole image path is bf16 and its byte total is derived rather than read:
vision 823,685,120 bytes (32 blocks of 25,702,400 plus patch_embed and the tower norm at 1,208,320),
aligner 146,821,120, and the three image delimiters 30,720 - **970,536,960** in total. Those are the
bytes that make the language-model plan plus the image path equal `metadata.total_size` exactly
(00-overview.md section 2.4); the identity is the corroboration, and a safetensors header read remains
the direct confirmation of the per-tensor dtypes.

---

## 4. The aligner: 3x3 pixel-unshuffle (vision.py:106-119)

```
x: [n_h, n_w, 1024] -> permute to [1024, n_h, n_w]
pad right/bottom to a multiple of 3 (F.pad with zeros)
F.unfold(x, 3, stride=3) -> [1024 * 9, L] -> transpose -> [L, 9216]
y = w2(gelu(w1(v)))            # 9216 -> 5120 -> 5120
```

Two details that a reimplementation must match:

- The 9216-vector ordering is **channel-major within each 3x3 block**: F.unfold emits
  index c * 9 + kh * 3 + kw (vision.py:118). A "pixel-unshuffle" written as
  reshape(..., 3, 3, C).flatten() would produce a spatial-major order and score differently.
- Padding is applied before the unfold, and the padded rows/columns are real zero-valued ViT outputs
  only if the patch grid was not already a multiple of 3 -- which it always is here, because
  plan_image_grid rounds the pixel size up to a multiple of patch_size and the aligner's own pad uses
  ceil-division in llm_grid (image_processor.py:41-42). In practice the pad is a no-op for the
  reference's own planner, but not for arbitrary callers of encode_image.

The upstream statement "a pixel-unshuffle operation with 3x3 downsampling to reduce the visual token
count by a factor of nine" (tech_report.txt p.8 lines 322-324) matches this exactly.

---

## 5. Image-span token protocol

```
TEXT = -1;  IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = 0, 1, 2, 3
                                                                 image_processor.py:22-23
types = [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
num_image_tokens(n_h, n_w) = n_h * (n_w + 1) + 2
                                                                 image_processor.py:35-36, 132-137
```

prepare_vl_inputs (image_processor.py:140-173):

1. Tokenize the prompt, count occurrences of image_token_id = 129264 and require them to equal the
   number of supplied images (image_processor.py:154-157); error if the model has no vision tower.
2. Replace each placeholder with the full type span; the token ids of the span are all the
   placeholder id, and the layout lives only in token_types (image_processor.py:168-172).
3. IMAGE_NEW_LINE is one extra token per row, which is exactly the +n_llm_h accounted for in
   num_image_tokens.

Merging into the residual stream (model.py:1224-1239):

- image_mask = token_types `>=` 0; engram_mask = ~image_mask (model.py:1249-1251).
- merge_image_embeddings overwrites the span in place: IMAGE_START / IMAGE_END / IMAGE_NEW_LINE rows
  take the three learned vectors image_start / image_end / image_newline (model.py:1220-1222,
  1235-1237), and the IMAGE rows take the aligner output **in row-major order**
  (model.py:1238-1239).
- The image span must lie inside a single prefill chunk: model.py:1255 asserts start_pos == 0
  whenever images are passed, and generate.py asserts the span fits within the chunk it is prefilling.

Downstream consequences:

- MoE routing for every image token uses gate.bias_vl instead of gate.bias (model.py:818-820); the
  weighting still uses the raw scores, so the two modalities only differ in expert selection.
- Engram turns off for image tokens: the hash cache stores DEAD and the gate is zeroed, so image
  rows pass through the Engram block untouched (model.py:363-364, 1251-1253; see 04-engram.md).
- The image span is part of the *causal* context for later text tokens: nothing in the attention path
  treats image tokens specially. Only the SWA window (128 slots) limits how far back a text token can
  see after the image.

---

## 6. Spec deltas

**D1 — The spec lists the vision encoder as "BF16 / FP8".** The reference vision tower is bf16 with
fp32 norm weights only; there is no FP8 quantiser anywhere in vision.py (vision.py:24-34, 46-71). An
FP8 ViT is a new optimisation, and it needs its own accuracy case.

**D2 — The spec's "replicated or TP4 (4 heads/rank)" is not the reference partition.** convert.py
shards nothing in vision.* (convert.py:56-64), so the released converter produces a fully replicated
tower on every rank; a 4-head-per-rank split would require new weight slicing of wqkv (which is a
fused 3-in-1 projection, so per-head slicing also has to split its rows into q/k/v blocks).

**D3 — The spec describes the aligner as a pixel-unshuffle without fixing the element order.** The
reference's order is channel-major within each 3x3 block (F.unfold, vision.py:118). Any
reshape/permute-based implementation must reproduce c * 9 + kh * 3 + kw.

**D4 — The spec does not mention the image-span token protocol's +1 token per row.** The span is
n_h * (n_w + 1) + 2 tokens (image_processor.py:35-36) because every visual row is terminated by an
IMAGE_NEW_LINE token and the span is bracketed by IMAGE_START/IMAGE_END -- all of which take learned
embeddings, and all of which are concatenated into the prefill sequence. A planner that budgets
n_h * n_w image tokens will overflow the 1024-token budget and misalign every following position.

**D5 — The spec's min-pixel behaviour is unstated.** Images below 295,936 pixels (544x544) are
**upscaled**, then rounded up to a multiple of 14, then reduced again only if the token budget is
exceeded (image_processor.py:106-112). "Resize down to at most 1024 tokens" is not what the reference
does.

**D6 — The spec asserts the VL bias is applied to the MoE "scores".** It is applied to the
**selection** only: bias = where(image_mask, gate.bias_vl, gate.bias) then topk(scores + bias), while
weights come from the unbiased scores (model.py:813-822).

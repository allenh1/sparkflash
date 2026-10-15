# 01 — CSA2: Compressed Sparse Attention 2, the Indexer, and the Hierarchical Sparse Indexer

All line numbers refer to /home/user/dsv41-research/inference/model.py unless stated otherwise.
Constants come from config.json / inference/config.json.

---

## 1. Module and shape map

| Object | Shape / value | Source |
| :--- | :--- | :--- |
| q path | wq_a (dim=5120 `->` q_lora_rank=1280), q_norm(1280), wq_b (1280 `->` H*head_dim = 64*512 = 32768) | model.py:640-642, 770-771 |
| attn_sink | [n_local_heads] fp32, one learned logit per head | model.py:639 |
| wkv (window KV) | 5120 `->` head_dim=512 | model.py:643 |
| kv_norm | RMSNorm(512), then RoPE on the last 64 dims | model.py:644, 705-706 |
| window_kv_cache | [max_batch_size, window_size=128, 512], non-persistent buffer | model.py:663-668 |
| compressor (only 4 layers) | wkv 5120`->`512, norm(512); plus wgate 5120`->`512 when ratio > 1 | model.py:437-456 |
| compress_kv_cache | [max_batch, max_seq_len // compress_ratio, 512] | model.py:669-679 |
| indexer (only 8 layers) | wq_b 1280 `->` 32*128 = 4096; weights_proj 5120 `->` 32 heads; wk (512`->`128) and k_norm(128) only when it owns K | model.py:514-519 |
| indexer k_cache | [max_batch, max_seq_len // compress_ratio, 128] | model.py:520-525 |
| softmax_scale (main) | head_dim^-0.5 = 512^-0.5 | model.py:651 |
| softmax_scale (indexer) | index_head_dim^-0.5 = 128^-0.5 | model.py:513 |
| indexer weight scale | weights_proj(x) * (softmax_scale * index_n_heads^-0.5) | model.py:555 |
| o path | wo_a per group: (64*512/8 = 4096) `->` (8*1024 = 8192), bf16; wo_b 8192 `->` 5120 | model.py:645-650 |
| sparse_attn | gathers topk indices, online softmax, attn_sink in the denominator, d = 512 | kernel.py:311-389 |

---

## 2. The three modes, derived from the id lists

Three config lists drive everything. The implementation's predicates are:

| Predicate | Expression | Meaning |
| :--- | :--- | :--- |
| is_kv_source | layer_id in kv_source_layers and layer_id < n_layers | owns a Compressor and a compress_kv_cache, publishes shared_attn.compress_kv (model.py:654, 746-748) |
| is_index_source | layer_id in index_source_layers and layer_id < n_layers | owns an Indexer and publishes shared_attn.topk_idxs (model.py:655, 736) |
| owns_k (indexer) | layer_id in kv_source_layers | may build index keys; every other indexer reads the published k_cache (model.py:500, 517-525, 548) |
| is_candidate_source | layer_id == candidate_source_layer | builds the HSI pool (model.py:502, 569-572) |
| uses_candidates | 0 `<=` candidate_source_layer `<` layer_id | masks its own index scores with the pool (model.py:503, 573-575) |

Because kv_source_layers = [2,8,14,20] is a subset of index_source_layers =
[2,8,14,20,24,28,32,36], layer 20 is the only decoder layer that both compresses KV *and* runs an
indexer; layers 24, 28, 32 and 36 run an indexer over layer 20's index-K cache.

Complete layer assignment (compress_ratios has 43 entries: index 0..42):

| Layers | ratio | Mode | Why |
| :--- | ---: | :--- | :--- |
| 0, 1 | 0 | SWA only | ratio 0; no compressor, no indexer (model.py:656-661, 775) |
| 2 | 2 | **Full** | kv_source, index_source, owns_k; ratio 2 so wgate exists |
| 3..7 | 2 | **Reuse** | neither source: reads shared compress_kv and shared topk_idxs |
| 8 | 2 | **Full** | kv_source + index_source |
| 9..13 | 2 | Reuse | |
| 14 | 2 | **Full** (plus the second Engram layer) | kv_source + index_source |
| 15..19 | 2 | Reuse | |
| 20 | 1 | **Full + HSI source** | kv_source + index_source + candidate_source; ratio 1 so no wgate |
| 21..23 | 1 | Reuse | |
| 24 | 1 | **Reindex** | index_source, uses_candidates (24 `>` 20), does **not** own K |
| 25..27 | 1 | Reuse | |
| 28 | 1 | **Reindex** | |
| 29..31 | 1 | Reuse | |
| 32 | 1 | **Reindex** | |
| 33..35 | 1 | Reuse | |
| 36 | 1 | **Reindex** | |
| 37, 38, 39 | 1 | Reuse | |
| 40..42 (MTP) | 0 | SWA only (DSpark) | layer_id `>=` n_layers disqualifies them from both source sets (model.py:653-655) |

Counts: 38 layers have ratio `>` 0, of which 3 Full, 1 Full+HSI, 4 Reindex and **30 Reuse**
(5+5+5 in the ratio-2 groups, then 3 + 3+3+3+3 in the ratio-1 groups). This matches tech_report.txt
p.10 lines 442-460 (encoder: 3 groups of six, first layer Full, remaining five Reuse; decoder: layer
20 Full, then four groups of four, first layer of each Reindex, remaining three Reuse).

The sharing is implemented through one process-global slot per artefact, which is safe because layers
always run in order and each source writes before its consumers read (model.py:1166-1180):

```
SharedAttentionRuntime: compress_kv, index_k, topk_idxs, candidates    (model.py:1166-1176)
```

Consequences an engine must preserve:

1. A non-source layer with ratio r has no compressor at all; it slices the shared cache to
   (start_pos + seqlen) // r positions (model.py:744, 763). Its ratio always equals its group
   leader's, so the slice is well-formed.
2. Reuse layers never run the indexer; they inherit the index positions verbatim
   (model.py:725-726).
3. An index source publishes topk_idxs even when compress_len == 0, with shape [b, s, 0]
   (model.py:729-731), so consumers must tolerate an empty index set.
4. Reindex layers keep the source's main KV and index K but recompute their own Top-K
   (model.py:735; tech_report.txt p.11 lines 513-516).

### 2.1 The offset convention

Window ids and compressed ids are concatenated into one index space (model.py:774-778). The
compressed ids are shifted by offset = window KV length (model.py:776), which is seqlen during
prefill and window_size = 128 during decode (model.py:708-719). sparse_attn then sees
kv = [window_kv ; compress_kv] with rows [0, offset) and [offset, offset + compress_len)
(model.py:777).

---

## 3. Window KV path

- kv = kv_norm(wkv(x)); RoPE the last 64 dims; then act_quant(kv, 32, "ue8m0", e8m0fnu, True)
  (model.py:705-707). True is the inplace flag: the reference quantises and dequantises in one
  kernel and keeps bf16 in memory (see 08-kv-footprint.md section 6).
- Prefill: if seqlen `<=` 128 the chunk is copied to the front of the ring; otherwise only the last
  128 rows are kept and rotated into the ring (model.py:708-716).
- Decode: one row is written to slot start_pos % 128 and the query attends over the whole ring
  (model.py:717-719).
- get_window_topk_idxs (model.py:409-426) builds the per-query index list. Prefill rows use
  (t - 128 + 1).clamp(0) + arange(min(seqlen, 128)) with entries beyond t set to -1; decode builds
  cat(arange(oldest, 128), arange(oldest)) with oldest = start_pos % 128 + 1 and slots beyond
  start_pos set to -1. Order inside a row does not matter to the kernel.

---

## 4. Compressor: how a main-KV entry is produced

```
ratio == 1:          latent = norm(wkv(x))                                 model.py:461-462
ratio > 1, prefill:  kv, score = wkv(x), wgate(x)            (both fp32)
                     a trailing partial group is stashed in kv_state/score_state
                     latent = norm( sum_j softmax(score)_j * kv_j ) over groups of ratio
                                                                           model.py:464-475
ratio > 1, decode:   write one row into slot start_pos % ratio
                     if (start_pos + 1) % ratio == 0 pool the whole group, else return None
                                                                           model.py:476-485
```

Only a compressor with ratio 1 avoids the softmax pooling and the fp32 promotion; the checkpoint is
consistent with this (compressor.wgate exists for layers 2, 8 and 14 only, and layer 20's
compressor.wkv is bf16).

The returned latent is **pre-RoPE** by design: the indexer needs the unrotated form
(model.py:430-435, 749-750), and Attention rotates it afterwards with the group-indexed position
j * ratio (model.py:752-758), using compress_rope_theta = 160000 and YaRN (model.py:680-696).
Then it is quantised with fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
(model.py:760) and written to compress_kv_cache at start_pos // ratio (model.py:761).

compress_len = (start_pos + seqlen) // ratio (model.py:744); a group contributes only once it is
complete, which is exactly what the index-score reachability mask encodes.

---

## 5. Indexer

```
k side (owns_k only):  k = k_norm(wk(latent))              [512 -> 128]     model.py:544-545
                       RoPE the last 64 dims at position j*ratio            model.py:539-545
                       fp4_act_quant(k, 32, True)  -> E8M0 scales           model.py:546
                       k_cache[start_pos//ratio : ...] = k                  model.py:547
                       shared_attn.index_k = k_cache                        model.py:548
q side:                q = wq_b(qr) -> [b, s, 32, 128]; RoPE; fp4_act_quant(q, 32, True)
                                                                            model.py:550-552
                       weights = weights_proj(x) * (128^-0.5 * 32^-0.5)     model.py:555
score:                 index_score[b,s,h,t] = <q[b,s,h,:], index_k[b,t,:]>
                       index_score = (relu(index_score) * weights[...,None]).sum(heads)
                                                                            model.py:556-557
```

The score is a **rectified** dot product (relu before the weighted head sum, model.py:557).

Reachability (model.py:561-567): during prefill compress_lens = (arange(1, seqlen+1) // ratio) is a
per-query tensor and every position `>=` compress_lens is masked to -inf; during decode
compress_lens = end_pos // ratio is a plain integer.

HSI gating (model.py:569-575):

- the candidate source computes the pool from its own index_score and publishes it;
- every later index source masks its index_score with ~candidates before top-k.

Selection (model.py:577-580):

```
topk = min(index_topk=512, end_pos // ratio)
idxs = index_score.topk(topk, sorted=False).indices.sort(-1).values
return where(idxs < compress_lens, idxs + offset, -1)      # int32
```

The index list is ascending; the kernel does not need order, but sorting makes the reachability
comparison well-defined.

---

## 6. HSI candidate pool: select_candidate_blocks (model.py:583-610)

Input logits is the source layer's index_score, already -inf wherever unreachable; compress_lens is
an int in decode and a [seqlen, 1] tensor in prefill.

| Step | Code | Effect |
| :--- | :--- | :--- |
| 1 | F.pad(logits, (0, -width % block_size), value=-inf) | right-pad to a multiple of 8 |
| 2 | unflatten(-1, (-1, 8)).amax(-1) | block score = max index score in the block (model.py:598-599) |
| 3 | scores[..., last] = +inf with last = (compress_lens - 1) // block_size | pin the newest, partially-filled block so it cannot be outscored (model.py:602-605) |
| 4 | scores.topk(min(2048, num_blocks)) | keep the highest-scoring blocks (model.py:607) |
| 5 | keep = scatter(zeros, top.indices, top.values `>` -inf) | drop picks that came back -inf, i.e. padded or unreachable blocks (model.py:609) |
| 6 | keep.repeat_interleave(8, -1)[..., :width] | expand the block mask back to one bool per compressed position (model.py:610) |

With the released settings that is 2048 blocks x 8 positions = **16,384 candidate positions**
(tech_report.txt p.12 lines 542-547 gives the same example), against which each Reindex layer's
32 x 128 indexer scores.

---

## 7. sparse_attn: top-k attention with attn_sink

Kernel at kernel.py:311-389, applied to kv = [window_kv (offset rows) ; compress_kv] with
topk_idxs = [window ids ; shifted compressed ids].

1. For each query row the kernel walks the topk index list in blocks of 64, loading kv rows by
   index; index -1 loads zeros and contributes weight exp(-inf) = 0 (kernel.py:360-364).
2. scores = q @ k^T * scale with scale = head_dim^-0.5 (model.py:651; kernel.py:319-320, 365-367).
3. The running max starts at the finite value -1e30, deliberately not -inf, so a row whose indices
   are all -1 produces an all-zero output instead of NaN (kernel.py:352-355).
4. Online softmax exactly as FlashAttention (kernel.py:368-380).
5. **attn_sink**: after the loop, sum_exp[i] += exp(attn_sink[i] - scores_max[i])
   (kernel.py:382-383), and the output is divided by that augmented denominator. attn_sink is a
   per-head fp32 learned logit that adds probability mass with no value vector, i.e. a learned
   "attend to nothing" sink.

Because window rows and compressed rows share one attention call, the two distributions are
normalised jointly; there is no separate window-vs-compressed weighting.

---

## 8. Quantisation layout of the three caches

| Cache | Format | Scales | Bytes per entry | Where |
| :--- | :--- | :--- | ---: | :--- |
| Main compressed KV (512) | FP4 E2M1 | one **E4M3** scale per 16 channels `->` 32 scales | 256 + 32 = 288 | model.py:760; kernel.py:160-163 |
| Indexer K (128) | FP4 E2M1 | one **E8M0 (ue8m0)** power-of-two scale per 32 channels `->` 4 scales | 64 + 4 = 68 | model.py:546; kernel.py:164-166 |
| Window KV (512) | FP8 E4M3 | one ue8m0 scale per 32 channels `->` 16 scales | 512 + 16 = 528 | model.py:707; kernel.py:77-80 |
| Dense weight GEMMs | FP8 E4M3, weight_block_size [32,32] | E8M0 block scales | -- | model.py:27-30, 196-205 |

Two different FP4 scale conventions coexist in one quantiser (kernel.py:128-181):

- E4M3 scales: amax = max(amax, 6 * 2^-9), s = cast_e4m3(amax / 6) -- no power-of-two rounding
  (kernel.py:161-163). This is the NVFP4-style layout of the main KV, exactly as the report says
  (tech_report.txt p.14 lines 642-650: E2M1 with one E4M3 scale per 16 channels, no second-level
  global scale).
- E8M0 scales: amax = max(amax, 6 * 2^-126), s = 2^ceil(log2(amax/6)) (kernel.py:164-166, 36-37).
  This is the indexer's MXFP4 layout.

Every one of these paths is called with inplace=True from the attention module, so the reference
quantises, clamps and dequantises in one kernel and stores the dequantised bf16 values
(kernel.py:81-86, 121-123, 167-172, 201-203). The FP4/E4M3 byte counts above are the logical layout
the model was trained with, and the layout a memory-saving engine should implement -- not what
these reference buffers hold. See 08-kv-footprint.md.

---

## 9. qk_rope_head_dim = 64 inside head_dim = 512

There is no separate 64-dim rope slot. wkv and wq_b both produce head_dim = 512 vectors and
apply_rotary_emb(x[..., -64:], freqs) rotates only the trailing 64 dims (model.py:706, 758, 772).
nope_head_dim = head_dim - rope_head_dim = 448 is computed but used only as documentation
(model.py:632). The output rotation is applied with inverse=True, i.e. the conjugate, so the cache
can stay in one shared rotated form (model.py:392-406, 781).

RoPE configuration differs by layer type (model.py:680-696):

- ratio `>` 0 (CSA2 layers): compress_rope_theta = 160000 with YaRN, original_seq_len = 65536,
  factor = 16, beta_fast = 32, beta_slow = 1.
- ratio == 0 (layers 0 and 1, plus the MTP layers): original_seq_len = 0 disables YaRN and
  rope_theta = 10000 is used.

Compressed positions use the group index j, i.e. position j * ratio, not the token position
(model.py:539-543, 752-757).

---

## 10. Grouped low-rank output (the o_groups / o_lora_rank twin)

```
o:    [b, s, 64, 512] -> view(b, s, 8, 4096)             # 8 groups x 8 heads each
wo_a: [8192, 4096]    -> view(8, 1024, 4096)             # per-group block, no cross-group mixing
o = einsum("bsgd,grd->bsgr", o, wo_a) -> [b, s, 8, 1024]                    model.py:785-787
x = wo_b(o.flatten(2))                -> [b, s, 5120]    # 8192 in, 5120 out    model.py:788
```

wo_a is stored bf16 (model.py:645-649) and convert.py dequantises the checkpoint's fp8 copy to bf16
(convert.py:156-173); an fp8 grouped GEMM would halve its bytes (comment at model.py:783-784). A
tensor-parallel engine shards groups across ranks (n_local_groups = 8 // world_size, model.py:634)
and wo_b is a row-parallel layer with an fp32 all-reduce (model.py:260-278).

---

## 11. Spec deltas

**D1 — The spec's indexer formula omits the rectifier.** The spec writes
"index_score = (q^T k) * w". The implementation is
(relu(einsum('bshd,btd->bsht', q, index_k)) * weights[..., None]).sum(heads) (model.py:556-557).
Negative similarities are clipped to zero, which changes any reimplementation that follows the spec
literally.

**D2 — The spec describes a main KV entry as head_dim plus qk_rope_head_dim plus Indexer K.** head_dim
= 512 already contains the 64 rotated dims, and the spec's own 890 B/token figure only closes if that
"+64" is dropped (08-kv-footprint.md section 5). The real per-entry sizes are 288 bytes (main) and
68 bytes (indexer).

**D3 — The spec never mentions attn_sink.** It is a per-head fp32 parameter present in all 40 layers
plus the 3 MTP layers (layers.N.attn.attn_sink: 40 tensors) and it enters the softmax denominator
(kernel.py:382-383). An engine that drops it will not reproduce the reference logits.

**D4 — The spec swaps the two FP4 scale conventions in places.** Main compressed KV is per-16 E4M3
(model.py:760); indexer K is per-32 E8M0/MXFP4 (model.py:546); window KV is per-32 E8M0 FP8
(model.py:707). The spec's section 5.5 calls the indexer "MXFP4" (correct) while section 2.3.4
attributes E4M3 to it and MXFP4 to main KV.

**D5 — The spec's mode table misses that index_source_layer_ids is a superset of the KV owners and
that only 4 of the 8 index sources own index K.** In the reference, layers 24/28/32/36 run an indexer
but read layer 20's k_cache (model.py:500, 548). An engine that allocates an index-K cache on all 8
index-source layers would both waste 4 x max_seq_len x 68 bytes per sequence and compute keys that
the reference never uses.

**D6 — The spec claims Z_20 = H_19 W_20^Z for the layer-20 compressor.** Layer 20 has ratio 1 and no
wgate at all (model.py:461-462); compressor.wgate appears in exactly 3 checkpoint tensors.

**D7 — The spec's candidate-pool description omits the pinned newest block and the -inf filter.**
select_candidate_blocks forces the block holding the query's newest reachable position to +inf
(model.py:602-605) and drops selected blocks whose score is -inf (model.py:607-609). Without both, a
reimplementation diverges from the reference exactly at the context frontier.

**D8 — The spec treats the top-512 set as the whole story, without the prefill reachability mask.**
In prefill each query sees only the compressed positions whose group has completed at that query
(model.py:563-565); decode uses the scalar end_pos // ratio (model.py:567). Prefill and decode
therefore produce different index sets for the same absolute position.

# 05 — DSpark: the three-stage MTP head, its attention, and the confidence path

References: <checkpoint>/inference/model.py (DSparkAttention 1032-1074,
DSparkMarkovHead 1077-1086, DSparkConfidenceHead 1089-1097, DSparkBlock 1100-1156,
get_dspark_topk_idxs 1020-1029, Transformer.forward_spec 1274-1282, sample 1285-1292).

Config: dspark_block_size 5, dspark_noise_token_id 128799, dspark_target_layer_ids [37,38,39],
dspark_markov_rank 256, dspark_n_routed_experts 128, dspark_n_activated_experts 3,
num_nextn_predict_layers 3.

---

## 1. Stage inventory

The checkpoint has three MTP stages under the mtp.* namespace (convert.py:105-110 strips a leading
"model." and the loader renames self_attn -> attn, mlp -> ffn for them as well). Each becomes a
DSparkBlock with layer_id 40, 41, 42 (model.py:1210-1213), which selects compress_ratios[40..42] = 0
and therefore a window-only attention with no compressor and no indexer.

| Stage | id | Owns |
| :--- | ---: | :--- |
| mtp.0 | 40 | main_proj: Linear(3 * 5120 -> 5120), main_norm: RMSNorm(5120) |
| mtp.1 | 41 | attention + MoE only |
| mtp.2 | 42 | norm: RMSNorm(5120), markov_head, confidence_head, and the shared head |

Both the input embedding and the output head are **shared with the backbone**, not stored per stage:
Transformer.__init__ assigns self.mtp[-1].embed = self.embed and .head = self.head
(model.py:1212-1213), and convert.py drops mtp.*.embed.weight / mtp.*.head.weight when loading
(convert.py:108-110). The spec's "3 MTP stages ... FP8 draft head" is therefore one head, shared, not
three.

MoE in the MTP stages uses 128 routed experts with top-3 (model.py:142-149), i.e. the same code path
with a per-stage expert count; the expert tensors are mtp.N.ffn.experts.*.

---

## 2. Target-layer tap (the input to the draft)

Inside the backbone loop (model.py:1264-1266):

```
for i, layer in enumerate(self.layers):
    if i in self.target_layer_ids:            # {37, 38, 39}
        main_hiddens.append(h.mean(dim=2))    # mean over the 4 hc copies
    h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
main_hidden = cat(main_hiddens, dim=-1)       # [b, s, 3 * 5120 = 15360]
```

Three points worth flagging:

1. The tap is the block **input** h of layers 37, 38 and 39, not their output (comment at
   model.py:1264). Since h at iteration i is the residual stream entering block i, the draft sees
   three consecutive pre-attention states.
2. The tap collapses the 4 hc copies with a **plain mean** (model.py:1266), unlike the final logits
   which use hc_pre with the last block's A_ffn (model.py:1268). Two different reductions in the same
   function.
3. The three taps are concatenated (not summed) and projected once by stage 0's main_proj
   (15360 -> 5120) then normalised (model.py:1113-1114, 1130). Stages 1 and 2 have no projection: they
   receive the same main_x (model.py:1276-1279).

---

## 3. Draft embedding: noise tokens and the anchor

```
main_x = main_norm(main_proj(main_hidden))          # [b, s, 5120], computed ONCE per step
draft_input_ids = full([b, block_size], noise_token_id = 128799)
draft_input_ids[:, 0] = input_ids                   # position 0 is the real anchor token
x = embed(draft_input_ids)                          # [b, 5, 5120]
x = x.unsqueeze(2).repeat(1, 1, hc_mult, 1)         # [b, 5, 4, 5120]
                                                                          model.py:1128-1135
```

So positions 1..4 of the draft block are the single noise token 128799, and only the first position
carries the just-sampled real token. All 5 positions are processed in one batched forward per stage;
there is no autoregressive re-embedding inside a stage (the Markov head supplies the sequential
dependency, section 6).

---

## 4. Window attention over [history ; draft]

DSparkAttention (model.py:1032-1074) inherits Attention with compress_ratio = 0 and asserts it
(model.py:1034). Prefill and decode behave differently:

**Prefill (start_pos == 0)** -- the stage only seeds its own window KV cache from main_x and returns
x unchanged (model.py:1044-1052):

```
main_kv = act_quant(kv_norm(wkv(main_x)), fp8, 32, ue8m0, inplace)   # [b, s, 512]
if s <= 128:  cache[:b, :s] = main_kv
else:         rotate the LAST 128 rows into the ring with cutoff = s % 128
return x
```

Note this caches main_x, **not** the draft tokens: during prefill the draft heads run but produce
nothing (Transformer.forward_spec returns None, model.py:1280-1281).

**Decode (start_pos > 0)** -- main_x is a single position (seqlen = 1) and x is the 5-token draft
block (model.py:1054-1074):

```
freqs_cis = self.freqs_cis[start_pos + 1 : start_pos + 1 + 5]     # draft positions
q  = wq_b(q_norm(wq_a(x))) unflattened to [b, 5, n_local_heads, 512]; RoPE the last 64 dims
kv = kv_norm(wkv(x)); RoPE; act_quant(fp8 ue8m0 inplace)           # the 5 draft keys
topk_idxs = cat(arange(min(128, start_pos + 1)), 128 + arange(5))  # window rows, then draft rows
window_kv_cache[:b, start_pos % 128] = main_kv.squeeze(1)
kv = cat([window_kv_cache[:b], kv], dim=1)                        # [b, 133, 512]
o  = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
apply_rotary_emb(o[..., -64:], freqs_cis, inverse=True)
o  = einsum("bsgd,grd->bsgr", o.view(b, 5, n_local_groups, -1), wo_a.view(...))
x  = wo_b(o.flatten(2))
```

Two subtleties:

- The draft block attends to **its own 5 keys** as well as the 128-slot history; the offset 128 in
  topk_idxs is the usual window/compressed offset convention (section 2.1 of 01-csa2-attention.md),
  and the draft keys occupy rows 128..132.
- The ring slot written is start_pos % 128 for **every** stage: each stage writes its own main_x into
  its own cache at the same slot, so the three stages' caches stay independent (model.py:1065).
- get_dspark_topk_idxs is lru_cached and independent of the query row: every draft position attends
  to exactly the same 128 + 5 keys (model.py:1020-1029). There is no causal mask inside the draft
  block, so draft position 0 can attend to draft position 4. A draft block whose positions were
  intended to be causal would need an extra mask; the reference has none.

---

## 5. Stage forward and the shared head

DSparkBlock.forward (model.py:1122-1126) dispatches:

- start_pos == 0: run attention only to seed the cache, return (x, pre_mix) unchanged.
- else: super().forward(x, start_pos, pre_mix, image_mask=None, main_x), i.e. the ordinary Block
  path with the mHC machinery, MoE with top-3 of 128, and no VL routing bias.

---

## 6. Markov head and the sequential block

```
DSparkMarkovHead:
    embed: ParallelEmbedding(vocab_size, 256)          # [129280, 256] rows per rank
    head:  ParallelHead(vocab_size, 256)               # [129280, 256], all-gathered with world_size > 1
    forward(ids): (head(embed(ids), full_logits=True), embed(ids))
                                                                          model.py:1077-1086

forward_head(x, pre_mix, input_ids):
    x = hc_pre(x, pre_mix)                       # [b, 5, 5120]
    logits = self.head(self.norm(x), full_logits=True)     # [b, 5, 129280]
    output_ids[:, 0] = input_ids
    for i in 0..4:
        (bias, markov_embed) = markov_head(output_ids[:, i])   # [b, 129280] and [b, 256]
        logits[:, i] += bias                                   # in-place bias on the shared logits
        markov_embeds.append(markov_embed)
        output_ids[:, i + 1] = sample(logits[:, i], temperature)
    confidence = confidence_head(x, stack(markov_embeds, 1))   # [b, 5]
    return output_ids, logits, confidence
                                                                          model.py:1137-1156
```

Notes:

- The Markov head is a rank-256 factorisation of a vocabulary-to-vocabulary transition: it adds a
  previous-token-dependent bias to the logits of the next position. It is applied to the **anchor**
  token for position 0 and to the just-sampled token for positions 1..4.
- Sampling is sequential and uses the *mutated* logits: logits[:, i] already includes the Markov
  bias when it is sampled (model.py:1153). The Gumbel-max helper is
  probs.div_(exponential_(1)).argmax() with temperature == 0 short-circuiting to argmax
  (model.py:1285-1292).
- The returned logits are the backbone head's output **with the Markov bias added in place**. A
  verifier that wants the raw target logits for acceptance must not use this tensor.

---

## 7. Confidence head

```
DSparkConfidenceHead:
    proj: Linear(dim + markov_rank, 1, dtype=fp32)   # 5120 + 256 -> 1, fp32 parameter
    forward(hidden, markov_embed) = proj(cat([hidden, markov_embed], -1).float()).squeeze(-1)
                                                                          model.py:1089-1097, 1155
```

- Input is the **collapsed** [b, 5, 5120] x from hc_pre, concatenated with the Markov embedding of the
  *input* token at each position.
- The output is a raw per-position score: there is **no sigmoid and no normalisation** in the
  reference (model.py:1097). The report describes the confidence head as predicting "per-position
  conditional acceptance probabilities" and feeding a scheduler that chooses the verification length
  (tech_report.txt p.14); the released code stops at emitting the logit.
- The parallel head for the confidence is stored bf16 in the checkpoint while the parameter is fp32
  (comment at model.py:1092), so it must be upcast at load time.

---

## 8. Where DSpark is (and is not) wired up

- Transformer.forward computes the backbone logits, samples, and returns (output_ids, logits,
  main_hidden) -- the tap is produced unconditionally on every decode step (model.py:1265-1272).
- Transformer.forward_spec consumes it: forward_embed on mtp[0], then the three stages in order, then
  forward_head on the last stage (model.py:1274-1282).
- **Nothing in the serving path calls forward_spec.** generate.py (batch generation, KV tracking,
  prefill chunking) has no reference to forward_spec, confidence or target layers; only the __main__
  demo in model.py:1295-1309 drives it, and that demo has no acceptance/rejection step either. The
  spec's DSpark acceptance loop is therefore a design for SparkFlash-RS, not a port of released code.

---

## 9. Spec deltas

**D1 — The spec presents DSpark acceptance as if it were implemented; the reference never verifies a
draft.** Transformer.forward_spec only returns (output_ids, logits, confidence) and generate.py has
no caller (model.py:1274-1292; no forward_spec in generate.py). The acceptance policy, the
"confidence threshold -> verification length" scheduler, and the KV rollback on rejection must all be
designed from scratch.

**D2 — The spec calls DSpark a decode-time MTP over the final layer's hidden state.** The reference
taps **three** layers (37, 38, 39), takes the block *input* h rather than the output, and reduces the
4 mHC copies with a plain mean (model.py:1264-1266). Using layer 39's output as the spec implies
shifts every draft.

**D3 — The spec's "128 experts (32/rank)" repeats the whole-expert-versus-width confusion.** With
whole-expert sharding (convert.py:129-134) each rank holds 32 of the 128 MTP experts at full width
2304, and top-3 selection is global across all 128 -- it is not a per-rank top-3.

**Implementation note (DECISIONS.md D30, 2026-10-18).** The paragraph above is the *reference's*
rule, and it is the evidence for the spec confusion. It is not the serving layout this tree records
for routed experts: D30 adopts TP=4 over the intermediate width (every rank holds every expert's
576-value slice, padded to 640 - the only routed-expert layout the device has run), and the MTP
blocks use the same code path and kernel ABI, so their experts are sliced and padded the same way
when they land. D6's layout half is corrected by D30; D3's functional facts stand.

**D4 — The spec treats the confidence output as a probability.** It is an unbounded fp32 logit with
no sigmoid (model.py:1097). A threshold comparison written against "probability in [0,1]" will
misbehave.

**D5 — The spec omits the noise token.** Positions 1..4 of every draft block are the single token
128799 (model.py:1131-1132). That token never appears in user text and must be handled explicitly by
any speculative loop that reconstructs the draft input.

**D6 — The spec does not mention that the draft attention has no intra-block causal mask.** Every
draft position attends to all 5 draft keys plus the 128-slot window (model.py:1020-1029, 1064-1067),
so the 5 logits are computed from a fully-connected block, not autoregressively. This is what makes
the block parallel; it also means the draft block cannot be truncated arbitrarily without changing
the remaining positions.

**D7 — The spec says the MTP stages have their own draft head ("FP8 draft head").** The head and the
input embedding are the **backbone's** tensors, aliased into all three stages
(model.py:1212-1213) and dropped from the checkpoint by the converter (convert.py:108-110). Only the
Markov head's embedding/projection and the confidence projection are new parameters.

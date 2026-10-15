# 04 — Engram: tokenizer compression, prime-modulus hashing, layout, and the lookup path

References: /home/user/dsv41-research/inference/engram.py (184 lines) and model.py:296-365.
Config (config.json text_config): engram_layer_ids [1, 14], engram_num_embeddings
[384006168, 384016682], engram_vocab_size 16000000, engram_max_ngram_size 4, engram_n_heads 8,
engram_head_dim 256, engram_pad_token_id 2, engram_compressed_vocab_size 99092.

---

## 1. What Engram is, in shapes

| Quantity | Value | Source |
| :--- | :--- | :--- |
| tables | one per engram layer, rows [384,006,168 ; 384,016,682] | config.json |
| row width | engram_head_dim = 256, fp8 e4m3, plus 8 ue8m0 scale bytes (256/32) | model.py:307-310 |
| bytes per row | 256 + 8 = **264** | model.py:309-310 |
| hash columns (lookups) per token | (max_ngram_size - 1) * n_heads = 3 * 8 = **24** | model.py:344 |
| fetched vector per token | 24 * 256 = **6144** | model.py:345, 353 |
| projection | wkv: 6144 `->` dim * (hc_mult + 1) = 5120 * 5 = 25600 | model.py:345 |
| split | key = first 20480 = 4 x 5120, value = last 5120 | model.py:354 |
| gate weights | q_weight, k_weight: [hc_mult, dim] = [4, 5120], fp32 | model.py:347-348 |
| table size on disk | 768,022,850 rows x 264 B = 202,758,032,400 B = **188.8 GiB** | derived; matches 196.6B params |
| Engram blocks in the stack | layers 1 and 14 only | config.json; tech_report.txt p.13 lines 605-621 |

The report says the 196B parameters are split evenly across two Engram modules (tech_report.txt
p.13 lines 605-611); the config's two row counts differ by 10,514 rows (0.003%), which is the
prime-count bookkeeping of section 4, not a real imbalance.

---

## 2. build_compressed_token_map (engram.py:17-61)

Purpose: collapse tokens that normalise alike so that " The", "the" and "THE" hash to the same bucket
(engram.py:18-23). It is **not** a simple lowercase: the exact normalizer chain is

| Order | Normalizer | Source |
| ---: | :--- | :--- |
| 1 | NFKC | engram.py:31 |
| 2 | NFD | engram.py:32 |
| 3 | StripAccents | engram.py:33 |
| 4 | Lowercase | engram.py:34 |
| 5 | Replace regex [ \t\r\n]+ with a single space | engram.py:35 |
| 6 | Replace regex "^ $" (a lone space) with the private-use sentinel U+E000 | engram.py:36 |
| 7 | Strip | engram.py:37 |
| 8 | Replace U+E000 back with a plain space | engram.py:38 |

Step 6 exists so that a token that is exactly one space survives Strip instead of collapsing to the
empty string and merging with unrelated tokens (comment at engram.py:26-27).

Key derivation per token id (engram.py:42-59):

1. Decode the id with the **raw Rust backend tokenizer**
   (`backend.decode([token_id], skip_special_tokens=False)`), matching training with no
   clean_up_tokenization_spaces (engram.py:42-47).
2. If the decoded text contains U+FFFD (a partial UTF-8 byte token) it has nothing to normalise, so
   the key is the raw token string from id_to_token (engram.py:48-50).
3. Otherwise the key is the normalized text, falling back to the raw text when normalisation yields
   the empty string (engram.py:51-53).
4. Keys are interned into a dense id space in first-seen order; lookup[token_id] = new id
   (engram.py:55-59).

The returned size is the compressed vocabulary size, and NgramHashState asserts it equals
engram_compressed_vocab_size = 99092 at boot (engram.py:145-146) -- the comment explains why: every
hash multiplier is derived from that size, so a mismatch silently rehashes the whole table.
The map is 129,280 u32 entries (one per tokenizer id).

---

## 3. find_next_prime and the hash multipliers

```
find_next_prime(start, seen):  smallest prime > start not already in seen     engram.py:9-14
compute_hash_multipliers(layer_ids, max_ngram_size, tokenizer_vocab_size):
    multiplier_bound = max(1, (int64_max // tokenizer_vocab_size) // 2)
                     = (9223372036854775807 // 99092) // 2 = 46539438283891
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(0, multiplier_bound, size=(max_ngram_size,), dtype=int64)
        row = values * 2 + 1                     # forced odd
    return stack(rows)                            # [n_engram_layers, max_ngram_size]
                                                                          engram.py:64-83
```

The bound exists so that token_id * multiplier cannot overflow int64 (engram.py:67-72). Reproducing
these four small integers bit-exactly requires numpy's PCG64 generator with Lemire bounded-integer
draws; a Rust port should either implement that exactly or -- far better -- hardcode the generated
constants, which are:

| Layer | multipliers[0] | multipliers[1] | multipliers[2] | multipliers[3] |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 76632096046245 | 4839876093313 | 35959672319349 | 73987337458391 |
| 14 | 67716810739261 | 51510806800915 | 30921347202721 | 82619226485591 |

(Numerically reproduced with `uv run --with numpy,sympy`; the seed and the algorithm are exactly
engram.py:74-82. Note the `tokenizer_vocab_size` argument is the **compressed** vocabulary size:
engram.py:150 calls `compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)`
where `vocab_size` is the value returned by `build_compressed_token_map` and asserted equal to
engram_compressed_vocab_size = 99092 on the preceding line, and the comment at engram.py:20-22 is
explicit that the multiplier bound derives from the compressed size. Using engram_vocab_size
(16,000,000) here instead shrinks every multiplier by a factor of 161.466.)

---

## 4. EngramLayout.from_args and the disjoint prime buckets (engram.py:86-126)

```
primes, seen = [], set()
for each engram layer:                       # 2 layers
    for each n-gram order in 0..max_ngram_size-2:      # 3 orders: 2-, 3-, 4-gram
        current = engram_vocab_size - 1 = 15,999,999
        for each head in 0..n_heads-1:                 # 8 heads
            current = find_next_prime(current, seen); seen.add(current)
                                                                          engram.py:108-118
```

So the layout is primes[layer][order][head], 2 x 3 x 8 = 48 **distinct** primes drawn in increasing
order from above 16,000,000, never reused across (layer, order, head). That is what makes the bucket
ranges disjoint (docstring at engram.py:88-93).

The generated table (reproduced for this document) is small enough to embed verbatim, and its sums
**exactly** reproduce config.json's engram_num_embeddings:

| Engram layer | order | primes (head 0..7) | order sum |
| ---: | ---: | :--- | ---: |
| 1 (layers[0]) | 2 | 16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183 | 128000880 |
| 1 | 3 | 16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321 | 128002054 |
| 1 | 4 | 16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463 | 128003234 |
| 14 (layers[1]) | 2 | 16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627 | 128004290 |
| 14 | 3 | 16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769 | 128005694 |
| 14 | 4 | 16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889 | 128006698 |

Layer 1 total = 384,006,168 and layer 14 total = 384,016,682, exactly the two numbers in
config.json. **This is a useful boot-time assertion for a Rust port: regenerate the 48 primes, sum
them per layer, and compare against engram_num_embeddings.**

Bucket offsets within a layer's table (cumulative prefix over the 24 primes in
[order][head] order, NgramHashState at engram.py:148-152):

| Engram layer | offset of order 2 | offset of order 3 | offset of order 4 | end |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 128,000,880 | 256,002,934 | 384,006,168 |
| 14 | 0 | 128,004,290 | 256,009,984 | 384,016,682 |

Hash id 24c + h (c = order index 0..2, h = head 0..7) therefore indexes the range
[offset[order=c] , offset[order=c] + primes[layer][c][h]) of that layer's table.

---

## 5. NgramHashState: hashing a position (engram.py:129-184)

```
DEAD = -1                                     # image spans are stored as DEAD in the cache
compressed = token_map[input_ids]             # [B, L]
if token_mask is not None: compressed = where(token_mask, compressed, DEAD)
cache[:, start_pos : start_pos+L] = compressed                # carries across prefill/decode

positions = arange(start_pos, start_pos+L).expand(B, L)
tokens, blocked = [], zeros(bool)
for shift in range(max_ngram_size):
    source  = cache.gather(1, (positions - shift).clamp_min(0))
    blocked = blocked | (positions < shift) | (source == DEAD)
    tokens.append(where(blocked, pad_id, source))              # pad_id = token_map[2]

tokens   = stack(tokens, -1)                        # [B, L, max_ngram_size], shift-major
products = tokens.unsqueeze(2) * multipliers        # [B, L, n_layers, max_ngram_size]
rolling  = products[..., 0]
hashes   = []
for i in range(1, max_ngram_size):
    rolling = rolling XOR products[..., i]
    hashes.append((rolling.unsqueeze(-1) % primes[:, i-1]))
return cat(hashes, -1) + offsets                    # [B, L, n_layers, 24]
                                                                          engram.py:160-184
```

Important properties:

- **Look-back stops at image spans and at the sequence start**, and once blocked, every longer shift
  stays blocked (engram.py:173) -- an n-gram never spans an image (docstring at engram.py:132-135).
- The n-gram hash is **not** a polynomial rolling hash: it is
  h_n = XOR over i in [0, n) of ( token_{t-i} * multiplier_i ), then reduced modulo the bucket prime.
  products are computed without an intermediate modulo, relying on the bound from section 3.
- Column order inside the 24 is shift-major: columns 0..7 are the 2-gram heads, 8..15 the 3-gram
  heads, 16..23 the 4-gram heads (hashes are appended order by order).
- The hash **cache is int64 and full-resolution over max_seq_len** (engram.py:155-157); it is not
  sharded. In a serving engine this is 8 bytes/token of extra state per engram layer, independent of
  TP.
- engram_mask comes from the same token types as the VL gate: `engram_mask = ~image_mask`
  (model.py:1249-1253), i.e. all four image token types (start, patch, newline, end) are dead.

---

## 6. From 24 lookups to the residual: the gate equation

```
value_rows = ParallelEngramEmbedding(hash_ids)          # [B, L, 24, 256], rows dequantised on lookup
kv    = wkv(value_rows.flatten(-2))                     # 6144 -> 25600            model.py:353
key, value = kv.split([hc_mult * dim, dim], -1)         # 20480 key, 5120 value    model.py:354
key   = key.unflatten(-1, (hc_mult, dim))               # [B, L, 4, 5120]          model.py:355
w     = q_weight * k_weight                             # [4, 5120] fp32, folded    model.py:356
rstd  = rsqrt(mean(h^2, -1) + eps) * rsqrt(mean(key^2, -1) + eps)                model.py:359
dot   = sum_dim(h * w * key) * rstd * dim^-0.5                                    model.py:360
gate  = sigmoid( copysign( sqrt(max(|dot|, 1e-6)), dot ) )                        model.py:362
h     = h + gate.unsqueeze(-1) * value.unsqueeze(-2)                              model.py:365
```

Precision points, all of which change the result if missed:

- The two normalisations are **per (token, hc copy) over dim = 5120**, not jointly over the copies
  (comment at model.py:358).
- q_weight and k_weight are only ever used as their product (model.py:356); an engine can fold them
  offline, exactly as the report says the normalization weights are folded into the projection
  (tech_report.txt p.13 lines 575-578).
- eps is norm_eps = 1e-20 (model.py:346).
- The clamped **signed** square root before the sigmoid is deliberate ("matching the training
  kernel", model.py:361-362): sqrt(|dot|) preserves the sign of dot, so the gate is a smooth
  signed-similarity squashing.
- token_mask is applied by zeroing the gate, not the value (model.py:363-364), so an image token gets
  its original h back exactly.
- The concrete layers: layer 1 runs Engram before block 1; layer 14 before block 14
  (model.py:1262-1264). The block inputs are the 4 hc copies, so Engram writes into all 4 copies with
  one shared value vector and 4 different gate scalars.

---

## 7. Storage layout and parallelism

| Aspect | Reference behaviour | Source |
| :--- | :--- | :--- |
| On-disk tensors | two separate tensors per layer: embed.weight fp8 [rows, 256] and embed.scale ue8m0 [rows, 8] | index.json; model.py:309-310 |
| Row bytes | 264 total, but **not interleaved**: weight and scale are distinct tensors in the checkpoint | index.json |
| Lookup | F.embedding on both, then `values.float().unflatten(-1, (-1, 32)) * scales.float().unsqueeze(-1)` | model.py:317-320 |
| Sharding | by **rows**: part_num_embeddings = ceil(rows / world_size) | model.py:303-305; convert.py:135-141 |
| Padding of the tail shard | weights padded with 0, scales padded with **1** | convert.py:138-141 |
| Out-of-range rows | masked to 0 locally, then all_reduce sums the one real row | model.py:313-324 |
| Precision | the table stays fp8 in memory; rows are dequantised per lookup and cast to bf16 | model.py:297, 320 |

Because rows are sharded and every rank must be able to answer any hash id, each lookup ends in an
all_reduce of a [B, L, 24, 256] bf16 tensor (model.py:323-324). This is the single largest
communication in the model per token (24 * 256 * 2 B = 12,288 B of all_reduce payload per token per
engram layer, or 24,576 B for two layers) and is the reason the spec's "prefetch" design matters.

---

## 8. Spec deltas

**D1 — The spec's gate equation drops the signed square root.** The spec gives
`sigma( RMSNorm(h)^T RMSNorm(k) / sqrt(5120) )`. The implementation is
sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)) where dot = sum(h * q_weight * k_weight * key) * rstd
* dim^-0.5 (model.py:360-362). A literal sigmoid(dot) is wrong for every negative dot: it would
produce a gate below 0.5 instead of above it.

**D2 — The spec says "Each lookup yields a 256-dim FP8 vector (plus 8 ue8m0 scale bytes for
32-element blocks)", which is right, but then claims a 512-byte packed row stride.** 264 bytes per
row is what the tensors imply; the 512-byte stride is a packer's choice added by the spec
(ENGRAM_PACKED_ROW_STRIDE), not a property of the reference, and it costs 94% extra storage
(512/264).

**D3 — The spec's prime derivation is stated loosely ("the i-th prime above 16M", "head seed").** The
reference derives the first prime from 16,000,000 - 1 (engram.py:112), draws primes strictly
increasing across the whole (layer, order, head) space with a global `seen` set (engram.py:108,
114-116), and has no head seed at all. Its multipliers come from
np.random.default_rng(10007 * layer_id) (engram.py:75-82), not from a head seed. The generated sums
reproduce config.json exactly (section 4), so a port must match this order, not the spec's.

**D4 — The spec's hash formula is not the reference's.** The spec's `hash_ngram` multiplies a running
value by COMPRESSED_VOCAB_SIZE and adds tok * multiplier, then XORs a rotation. The reference XORs the
per-position products and reduces once at the end (engram.py:179-184). Different bucket assignment,
different ids, different memory reads.

**D5 — The spec's multiplier "derived from COMPRESSED_VOCAB_SIZE and head seed" is a different
construction.** See section 3: bound = (int64_max // 99092) // 2, using the compressed vocabulary
rather than the config's engram_vocab_size, with the values forced odd and one RNG per layer; there
is no head component.

**D6 — The spec shards Engram by head ("6 heads per rank, d_mem,local = 1536"); the reference shards
by row** (model.py:303-305; convert.py:135-141). See 00-overview.md D5 for the consequence.

**D7 — The spec's image masking is described only as "masked to pad"**; the reference stores DEAD =
-1 in the hash cache, propagates blockage to all longer shifts (engram.py:173) and zeroes the gate
(model.py:363-364), which is *not* the same as substituting the pad id: substituting pad changes the
n-grams of following positions, while DEAD blocks them.

**D8 — The spec asserts a 1e-6 clamp in the gate but calls it a "cosine similarity"**; the reference
normalises h and key separately and never renormalises the product by the norms of h and k after
multiplication, so it is a bounded signed similarity rather than a cosine.

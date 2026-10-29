# 08 — KV footprint: the 890 B/token derivation, SWA replay, and what the reference actually allocates

Sources: model.py:613-763 (cache allocation), kernel.py:40-204 (quantisers), config.json,
tech_report.txt p.1 lines 17-23 and p.19-20 lines 858-925, and the spec's own budget table
(spec lines 73-76, 152, 174-176).

---

## 1. The three cache formats

| Cache | Fields | Bytes | Source |
| :--- | :--- | :--- | :--- |
| Main compressed KV (per entry) | FP4 E2M1, 512 dims `->` 256 B; E4M3 scale per 16 channels `->` 32 scales | **288** | model.py:760; kernel.py:158-163 |
| Indexer K (per entry) | FP4 E2M1, 128 dims `->` 64 B; E8M0 scale per 32 channels `->` 4 scales | **68** | model.py:546; kernel.py:164-166 |
| SWA window KV (per slot) | FP8 E4M3, 512 dims `->` 512 B; E8M0 scale per 32 channels `->` 16 scales | **528** | model.py:707; kernel.py:74-80 |

The main-KV scale is E4M3, not a power of two: the quantiser floors amax at 6 * 2^-9 and stores
amax/6 cast to E4M3 (kernel.py:161-163). The other two use fast_round_scale, i.e. 2^ceil(log2(amax/6))
(kernel.py:36-37, 78, 166).

---

## 2. Entries per token: the 2.5 factor

Only layers `[2, 8, 14, 20]` allocate a compressed cache (is_kv_source, model.py:654, 669-679), and
only those same layers allocate an index-K cache (owns_k = layer_id in kv_source_layers,
model.py:500, 517-525). Layers 24, 28, 32 and 36 run an indexer but read layer 20's cache, so they add
no KV.

| Layer | ratio r | entries/token = 1/r | Main KV B/token | Index K B/token |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 0.5 | 144 | 34 |
| 8 | 2 | 0.5 | 144 | 34 |
| 14 | 2 | 0.5 | 144 | 34 |
| 20 | 1 | 1.0 | 288 | 68 |
| 24, 28, 32, 36 | 1 | 0 (read layer 20) | 0 | 0 |
| everything else | -- | 0 | 0 | 0 |
| **total** | | **2.5** | **720** | **170** |

**720 + 170 = 890 bytes per token**, matching the report's "890 bytes per token, roughly 1/4 of the
corresponding footprint of DeepSeek-V4-Flash" (tech_report.txt p.1 lines 19-20).

Note this is 2.5 entries for main KV *and* 2.5 entries for index K -- the same four layers, so one
multiplier covers both. The spec arrives at the same total but itemises it as
"256 + 32 + qk_rope_head_dim 64 + (64 + 4)" per entry (spec line 76), which sums to 1050 B/token
after scaling by 2.5. The 890 figure only closes if the `+64` is dropped: head_dim = 512 already
includes the 64 rotated dimensions (see 01-csa2-attention.md section 9).

---

## 3. How the footprint scales with context

Per sequence, excluding fixed overheads and any paging slack (1 GiB = 2^30 B):

| Context | Global KV (890 B/token) | SWA ring (40 layers x 128 x 528 B) | Reference resident (bf16, section 5) |
| ---: | ---: | ---: | ---: |
| 1 | 890 B | 2,703,360 B = 2.578 MiB | 5,242,880 B = 5.0 MiB |
| 128K | 116,654,080 B = 0.109 GiB | 2.578 MiB | 400 MiB |
| 1M | 933,232,640 B = **0.869 GiB** | 2.578 MiB | **3.125 GiB** |
| 6.5M | 5,785,000,000 B = 5.388 GiB | 2.578 MiB | 20.3 GiB |

The SWA ring is **per sequence and constant in context length**: 128 slots x 40 layers x 528 B, i.e.
2.578 MiB for every sequence, no matter how long it is. It does not scale with batch or context, only
with the number of concurrent sequences.

### 3.1 The spec's "13.4 GiB holds 3.67M-6.50M tokens" is inconsistent with 890 B/token

890 B/token gives 3.67M tokens in 3.04 GiB and 6.50M tokens in 5.39 GiB. Conversely
13.4 GiB = 14,388,140,442 B holds **16,166,449 tokens** at 890 B/token -- 2.5x the top of the spec's
own range (spec line 174). Either the pool figure or the token range is wrong; they cannot both be
right, and the spec uses both in the same table row.

---

## 4. Persistent KV, bounded replay, and the 1/8 claim

The report's decomposition (tech_report.txt p.19 lines 859-863): the persistent (SSD/host) KV
footprint falls to 1/8 of V4's as the product of two factors --

1. SWA KV is removed from the persistent cache (about halves it), because it is moved to a
   distributed host-DRAM pool with a minutes-long TTL (tech_report.txt p.19 lines 881-886);
2. global KV shrinks to 1/4 of V4's through CSA2 cross-layer reuse plus FP4 (the 890 B/token).

Exact SWA recovery would need L x n_win = 40 x 128 = 5,120 token-layer forwards (the spec quotes the
decoder half, 20 x 128 = 2,560, spec line 57). SWA Bounded Replay instead replays only the most recent
n_win = 128 tokens and truncates SWA to the replay segment, accepting approximate states: for a replay
starting at s, a query at i attends to keys in [max(s, i-W+1), i] (tech_report.txt p.20 lines 896-904).
Encoder replay regenerates only SWA KV while reusing cached global KV; decoder replay feeds the last
128 tokens' encoder outputs through the decoder layers under the same truncation and uses the result
for decoding only, never for prefix caching (tech_report.txt p.20 lines 905-925).

Two engine consequences the spec gets right in spirit and wrong in arithmetic:

- **Approximate states break equality with a cold prefill.** The report states the replayed state is
  not mathematically identical across cache-hit positions (tech_report.txt p.20 lines 911-914); any
  determinism guarantee in the Rust engine must be scoped to "same cache state", not "same prompt".
- **The spec's SWA ring budget of 1.1 GiB cannot come from 128 tails.** One sequence tail is 2.578 MiB
  at 528 B/slot, so 1.1 GiB is ~437 sequences' tails (spec line 175 says "128 tokens x 40 layers x
  128 slots" for a 1.1 GiB row). Even at bf16 (1024 B/slot) the same product is 5.0 MiB per sequence.

---

## 5. What the reference actually allocates (and why 890 B/token is not what it stores)

Every quantiser call in the attention path passes `inplace=True`:

```
self.window_kv_cache = torch.zeros(max_batch_size, 128, 512)      # default dtype = bf16
act_quant(kv, 32, "ue8m0", e8m0fnu, True)      # fused quant+dequant back to bf16  model.py:707
fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)                  model.py:760
fp4_act_quant(k, 32, True)                                                        model.py:546
```

With inplace=True both kernels set out_dtype = in_dtype, compute `cast(fp4/fp8, clamp(x/s)) * s` and
write that back (kernel.py:81-86 and 167-172), and the scales tensor they allocate is **dropped on
return** (kernel.py:121-124, 201-204). The buffer dtype comes from the module's default dtype, i.e.
bfloat16 (generate.py:118).

So the released reference stores:

| Cache | Logical (design/training) | Reference resident per token/sequence |
| :--- | :--- | :--- |
| Main compressed KV | 288 B/entry | 512 x 2 = 1024 B/entry -> 2,560 B/token |
| Indexer K | 68 B/entry | 128 x 2 = 256 B/entry -> 640 B/token |
| **global total** | **890 B/token** | **3,200 B/token (3.60x)** |
| SWA window | 528 B/slot | 1024 B/slot -> 5.0 MiB per sequence |

The docstring at model.py:701-702 says the window K "stays fp8, quantized over the whole post-RoPE
vector", but the inplace=True call at model.py:707 contradicts it: the values are dequantised before
they reach the cache. This is the single largest gap between the spec's memory plan and the reference's
behaviour, and it is exactly where SparkFlash-RS can win -- provided it stores the scales too, which
the reference never does.

Additionally, the shipped entry points size the caches far below 1M:

- model.py:51-52 defaults max_batch_size = 4, max_seq_len = 4096.
- inference/config.json contains no max_seq_len / max_batch_size keys at all.
- generate.py:124-126 overrides to max_batch_size = 1, max_seq_len = 64 * 1024 only in interactive
  mode.

A faithful port therefore starts from 4 x 4096 (non-interactive) or 1 x 64K (interactive), not 1M.
At 1M x the full batch of 4, the global KV alone would be 4 x 0.869 = 3.48 GiB logical, 12.5 GiB
resident.

---

## 6. Fixed overheads the spec's budget does not itemise

| Item | Size | Source |
| :--- | :--- | :--- |
| RoPE tables (freqs_cis) | 2 distinct tables x max_seq_len x 32 complex64 x 8 B = **512 MiB at 1M**, 32 MiB at 64K | model.py:368-389, 688-698 (lru_cache(2): one YaRN table for ratio > 0, one plain table for ratio == 0) |
| Engram hash cache | int64 [max_batch_size, max_seq_len] = 8 B/token = 8 MiB per sequence at 1M | engram.py:155-157 |
| Engram tables | 768,022,850 rows x 264 B = 202,758,032,400 B = **188.83 GiB** (47.2 GiB/rank at world_size 4) | index.json + model.py:309-310; spec line 103 agrees (189 GiB) |
| DSpark stage rings | 3 x 128 x 528 B = 202,752 B (bf16: 393,216 B) | model.py:1065-1066 |
| Candidate block mask | [b, s, 2048] bool + the pool's scatter temporaries | model.py:607-610 |
| Indexer score transient | b x s x 32 x t, bf16; at s = 4096 and 1M visible positions = 4096 x 32 x 1,048,576 x 2 B = **256 GiB**, and the `* weights.unsqueeze(-1)` broadcasting doubles the peak | model.py:550-557 |
| Indexer all_reduce | [b, s, t] bf16 per index-source layer = 8 GiB per layer at the same shape, x 8 index sources | model.py:558-559 |

Two of these deserve emphasis:

- The indexer's score tensor is the dominant transient in the whole model, and the spec's "~60 GiB"
  estimate (spec line 902) is not reproducible from the reference's shapes: the einsum alone is
  256 GiB in bf16 before the weights broadcast. Any prefill plan must tile query rows *and* key
  positions; tiling only the queries is not enough.
- The index_score is all-reduced (not all-gathered) across ranks because each rank computes
  n_local_heads = 8 of the 32 indexer heads (model.py:550-559, 507-508). That collective is over the
  full [b, s, t] tensor, so its bandwidth cost is the same order as the KV cache it replaces.

---

## 7. Spec deltas

**D1 — The spec's itemisation of a main KV entry double-counts the RoPE dimensions.** "head_dim = 512
... plus qk_rope_head_dim = 64 and Indexer K" (spec line 76) implies 1050 B/token, not the 890 it
claims; the 64 rotated dims are inside head_dim (01-csa2-attention.md section 9).

**D2 — "13.4 GiB holds 3.67M-6.50M tokens" is wrong by a factor of 2.5.** 13.4 GiB at 890 B/token is
16,166,449 tokens; 6.5M tokens is 5.39 GiB (section 3.1). The spec's own KV table is internally
inconsistent.

**D3 — The spec's "890 B/token" describes a layout the reference never stores.** Every cache write in
the reference goes through inplace=True quant+dequant, so the resident caches are bf16 at 3,200 B/token
(3.60x the spec figure) and the FP4/E4M3 scale tensors are discarded (kernel.py:121-124, 201-204).
"Already implemented at 890 B/token" is a target, not a description.

**D4 — The spec's SWA cache sizing is off by two orders of magnitude.** "128 tokens x 40 layers x 128
slots" at FP8 is 0.31 GiB; one sequence's ring is 2.578 MiB at 528 B/slot. The spec's 1.1 GiB row
(spec line 175) matches neither.

**D5 — The spec never mentions the RoPE tables, the Engram hash cache, or the indexer score
transient.** At 1M context the two freqs_cis tables alone are 512 MiB per rank, and the indexer
transient is 256 GiB at a 4096-row chunk -- both larger than most of the rows in the spec's budget
table (section 6).

**D6 — The spec's budget assumes a 1M-token cache pool but the reference allocates 4 x 4096 by default
and 1 x 64K interactively** (model.py:51-52; generate.py:124-126; the runtime config has no
max_seq_len key). Any "measured" number attributed to the reference must state which sizing was in
force.

**D7 — The spec says 12 Reuse layers; the correct count is 30** (spec line 71). With
compress_ratios = [0, 0, 2 x 18, 1 x 20, 0, 0, 0], the 38 compressed layers are 3 Full + 1 Full+HSI +
4 Reindex + **30 Reuse** (01-csa2-attention.md section 2). This matters for the spec's own kernel-count
argument: reuse layers still run a sparse attention over the shared cache, so they are not free.

**D8 — The spec's persistent-KV story omits that bounded replay yields approximate states.** The
report is explicit that replayed SWA KV is not mathematically equivalent across cache-hit positions
(tech_report.txt p.20 lines 911-914). A Rust engine that advertises bit-exact prefix-cache reuse while
using replay would be overclaiming.

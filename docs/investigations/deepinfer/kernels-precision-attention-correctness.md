# DeepInfer as a reference: kernels, precision, and attention correctness

Slice report. Author: exploration agent (kernels / precision / attention-correctness slice).
Repo under analysis: `/home/user/deepinfer` (commit `774ce24`, 6.4 MB, all six vendor
submodules populated). Our engine: `/home/user/deepseek-v4.1-flash-spark`.
Nothing in this report was measured on a GPU; all claims are read from source. Every place
where I am inferring rather than reading is marked **[inference]**.

---

## 0. What I actually read

| Area | Files (line counts are real) |
| :--- | :--- |
| Attention runtime | `deepinfer/model_executor/models/v4_modules/attn.py` (1008), `mla_attn.py`, `index_attn.py` (575), `compress.py` (167), `cross_layer.py` (383) |
| KV cache / publication | `deepinfer/worker/cache_engine.py` (978), `input_metadata/input_metadata.py` (490), `model_executor/model_metadata_utils.py` |
| Metadata kernels | `tilelang_ops/cuda/{prepare_decode_metadata,get_kv_index,swa_index,compress,store_kv_cache,load_paged_kv_cache,kv_index_cat,flash_mla_utils}.py` |
| FlashMLA | `vendor/FlashMLA/setup.py`, `csrc/api/sparse_decode.cpp`, `csrc/cuda_kernels/sm100/decode/sparse/head64/config.h`, `kernel.cuh` |
| GEMM / MoE / EP | `deepinfer/model_executor/layers/linear.py`, `models/v4_modules/{moe,mlp}.py`, `eplb/__init__.py`, `vendor/DeepGEMM/{README,csrc}`, `vendor/DeepEP`, `vendor/DeepSelect` |
| Formats | `tilelang_ops/cache_common.py` (V41_FORMAT / V41_FP4_FORMAT / FP4_INDEX_KV_FORMAT), `fused_transform.py` |
| Scheduling | `csrc/engine/scheduler.hpp` (1329), `environ.py` |
| Ground truth for our own bug | `/home/user/dsv41-research/inference/model.py` (`_compress_kv`, `_window_kv`, `SharedAttentionRuntime`), `reference/{hf,inference}_config.json` |

Not available / not done: no GPU run, no profiling; no DeepInfer runtime config beyond the HF
checkpoint config we already hold locally.

---

## 1. Verdicts at a glance

| # | Candidate | Usable on sm_121a? | Value to us |
| :- | :--- | :--- | :--- |
| 1 | CSA shared-channel publication semantics (`cache_engine.py` + `attn.py` + `model.py`) | n/a (semantics) | **Decisive**: our fix is exactly right, both halves, proven twice |
| 2 | Decode launch contract: `topk + extra_topk <= 640` -> no split-KV; FP4 **only** as extra KV | n/a (policy) | **High**: batch-invariance for free + format validation |
| 3 | `klen` / `positions_c` / `slot_mapping_store_ca` / `kv_ca_index` formulas | n/a (formulas) | **High**: bit-exact spec for the code that produced our holes bug |
| 4 | Compressor state ring (`ratio + backtracking - 1` slots) and chunked-prefill tail | n/a (semantics) | **High** for our new chunked path and for MTP |
| 5 | State-block `clone_states` / `zero_states`; `load_segments` `src_block_id=-1` -> zero-fill | n/a | **Medium-high**: prevents silent prefix-cache corruption |
| 6 | FlashMLA sparse decode / fused norm+rope+attn+cast | **No** (SM100f runtime gate + `sm_100a/103a` only + tcgen05) | Design reference only |
| 7 | DeepGEMM (`fp8_fp4_gemm_nt`, `recipe=(1,1,32)`) | **No** (SM90/SM100 only) | Scale-factor **layout** rules only |
| 8 | DeepEP (all-to-all) + EPLB | **No** (symmetric memory, EP>=2) | Nothing for TP=4 / 1 GPU per node |
| 9 | DeepSelect `topk` | **No** (`sm_100a/103a`) | Tie / `-1` / `end` semantics only |
| 10 | `tilelang_ops/*` (TileLang 0.1.15 + foreign `tile-kernels` 2.1.0) | Source-only, not drop-in | Index-expansion formulas as a spec |
| 11 | YOCO / SWA bounded replay (`compact`, `KEEP_TOKENS=140`) | n/a | **Dead for this checkpoint** - README is wrong (7.2) |
| 12 | Hadamard before KV cast | never enabled | No numerics surprise |

---

## 2. THE CRITICAL CROSS-CHECK - the shared compressed-KV channel

### 2.1 What DeepInfer does (mechanism, not policy)

The channel is **not a publication at all; it is a cache view resolved once, at allocation
time, from static config.** `deepinfer/worker/cache_engine.py`:

```python
# :160-172, inside CacheEngine.get_block_shape
mapped_layer_id = _get_shared_mapped_layer_id(layer_id, self.shared_layer_topology) if use_shared_kv else layer_id
assert mapped_layer_id <= layer_id, "the KV layer of share-kv must precede the layer that uses it"

# :431 / :449, per layer, no per-forward work
ica_cache = cache_list[ica_mapped_layer_id].ica_cache
ca_cache  = cache_list[kv_mapped_layer_id].ca_cache
```

So a consumer layer literally holds *the same `torch.Tensor` object* as its owner. There is no
copy, no publish call, no length bookkeeping. Consequences, all of which the code gets for free:

* the consumer sees the owner's **whole** cache, including rows written by earlier forwards;
* a forward that adds **no** new compressed row (a partial compression group) writes nothing
  (`slot_mapping_store_ca = -1`) and needs to write nothing - the owner's buffer already
  contains the history;
* an owner change cannot create holes, because the map is static: layer 8's rows live in
  layer 8's slice of the paged pool, addressed by the *same* `block_tables` as every other
  layer. Each owner writes its own full history, in its own slice.

The prefill path is the same idea in a temporary buffer (`attn.py:612-626`): the **KV source
loads `total_k_compressed` rows from `ca_cache` into the segment buffer**; consumers
`assert loaded_kv.shape == shape` and reuse it, overwriting **only the SWA region at the front**
("a share-kv consumer ... does not recreate or load the main KV", `cross_layer.py`).
`total_k_compressed` comes from `(start_pos+seqlen)//ratio`, i.e. the whole history, never "the
rows this forward added".

### 2.2 What the released reference does (verbatim, `model.py`)

`/home/user/dsv41-research/inference/model.py`, `_compress_kv`:

```python
if self.is_kv_source:
    latent = self.compressor(x, start_pos)
    shared_attn.compress_kv = self.compress_kv_cache        # <- the pointer, EVERY source forward
...
if latent is not None:
    ... fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn) ...
    self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent
# read after the write, so this does not depend on the slice aliasing the cache
return shared_attn.compress_kv[:bsz, :compress_len], idxs   # compress_len = (start_pos+seqlen)//ratio
```

Three facts, each of which is one half of our bug:

1. `shared_attn.compress_kv = self.compress_kv_cache` is executed **unconditionally**, above the
   `if latent is not None` - so a source publishes on **every** forward, including one that adds
   no row. Our "publish even when `latent_rows == 0`" half is **exactly** the reference.
2. The published object is the owner's **persistent cache array**, not this forward's rows - so the
   consumer reads rows 0..compress_len, i.e. the owner's **whole** cache. Our "publish the whole
   cache" half is **exactly** the reference.
3. The consumer's read length is derived from **position**, not from the publication
   (`(start_pos+seqlen)//ratio`), and the source's `SharedAttentionRuntime` is a process singleton
   that is *never reset between forwards* (its docstring says so explicitly).

### 2.3 Verdict on our fix (`0f0b895`, branch `batch/2026-10-20-17`)

**Match, on both halves, and the emulation is the correct host equivalent of moving the pointer.**
Not "one of two valid behaviours" - the reference has only one behaviour, and the fix reproduces
both of its load-bearing properties. The commit message's diagnosis ("rows 0-2 zero for the
consumer that runs after layer 5") is also the right failure mode.

Differences that remain, in order of importance:

| Aspect | Reference | Ours (after fix) | Assessment |
| :--- | :--- | :--- | :--- |
| Storage | alias (same buffer) | `Vec<f32>` copy of the owner's whole cache | Observationally equal **iff** every consumer runs after its owner and the copy source is the owner's own accumulated cache (it is: `&state.compressed` / `&state.indexer`). O(L) per source per forward; see below. |
| Writer identity | static map, `assert mapped <= layer` at allocation; consumer resolves its source from config | last-writer-wins slot with `owner: Option<usize>`, **no assertion** that the writer is this consumer's planned source | **Weakest point.** A layer-order regression would silently make a consumer read the previous segment's owner. One-line fix: assert `published_owner == planned_source(layer)`. |
| Length | consumer derives `compress_len` from position; no length in the channel | channel carries `compressed_rows`; consumers refuse if too short | Fail-closed where the reference is silent. Better for us, but it means a rewind/preempt fixture will *diverge in behaviour* (we error, the reference reads whatever is in the buffer - which is why the reference has `zero_states`, section 5). |
| Read extent | prefix `[:compress_len]` | whole published cache | Harmless (superset), but the literal equivalent of "moving the pointer" is a *borrowed slice*, not a copy. |

Cost of the copy: the compressed caches are `compress_len` rows of 512 f32 (2 KB) for four
owners -> about 8 KB x L per forward on the host path. At L=4096 that is about 32 MB/step of
memcpy; on the device path (288 B/row FP4) about 4.6 MB. Small against a 926 ms step, so this is
**not** a performance argument - it is an argument for making the channel a view so the invariant
"the channel *is* the owner's cache" is structural rather than maintained by code that can rot.

### 2.4 Additional reference invariants worth copying

* `_validate_shared_layer_cfg` (`cache_engine.py`) requires source and consumer to agree on
  `kv_compress_ratio`, `index_topk` and `use_swa` - the topology check we should mirror in
  `layer_plan.rs`.
* Topology validity is **segmented**: a consumer's source is the *nearest preceding* source, and an
  unrelated layer sandwiched inside a segment is an error, not a silent self-map.
* The consumer chain `index-topk subset index-K subset KV` segment granularity is validated at
  construction.

---

## 3. Top-K and compressed rows: the exact contract

### 3.1 What the rows *are*

* **Prefill.** One gathered KV buffer per share-KV segment. Layout is `[cached SWA | current SWA]`
  then the compressed block; `kv_index_cat` (`tilelang_ops/cuda/kv_index_cat.py:60-80`) rewrites the
  compressed ids in place as `R += total_k + total_k_swa` while **leaving `-1` untouched** (its
  `if R != -1`). The indexer's top-K ids are produced with `output_idx_offset=ks[chunk]` because
  prefill top-K is chunked over queries. `-1` is the only invalid marker; there is no separate
  length for the sparse path.
* **Decode.** Two caches, two index arrays, one kernel: `key_cache=swa_cache` +
  `indices_in_kvcache=kv_swa_index` (window rows, **physical ring slots**), and
  `key_cache_extra=ca_cache` + `indices_in_kvcache_extra=topk` (compressed rows, **physical paged
  slots**). `attn.py:628-670`. The top-K tensor *replaces* the dense `kv_ca_index` for sparse
  layers (`attn.py:568-578`); the dense path uses `kv_ca_index` + `kv_ca_index_len` instead.
* Physical slot conversion is `convert_kv_index` (`flash_mla_utils.py`), applied **in place** to the
  topk tensor, again skipping `-1`. So after conversion the topk tensor is *not* sorted by
  position even though `topk(sorted_index=True)` produced it sorted; the kernel must not assume
  monotonicity (it doesn't - `causal=False`).
* Candidate blocks: `block_candidate_idxs` published at layer 20, consumed by 24/28/32/36, and
  **rebased whenever the query row layout changes** (`yoco_boundary_compact` gathers the topk and
  candidate rows at the compaction boundary) - dead for our checkpoint (7.2), but it is a reminder
  that any row-layout change must rebase *published* topk rows too, not just the current layer's.

### 3.2 The hard kernel contract (this is the part I would assert in our launcher)

`vendor/FlashMLA/csrc/api/sparse_decode.cpp`:

```cpp
:190  TORCH_CHECK(arch.is_sm100f(), "... only supported on SM100f architectures");
:223  bool enable_split_kv = !enable_batch_invariant && !(topk + extra_topk <= 640);
:287  TORCH_CHECK(model_type != ModelType::V41_FP4, "The fp4 KV cache is only supported as extra_kv");
:230  TORCH_CHECK(h_kv == 1, "Currently only MQA ...");
:231  TORCH_CHECK(d_qk == 512, ...);  :232 TORCH_CHECK(d_v == 512, ...);
```

Three things fall out, all directly useful:

1. **`topk + extra_topk <= 640` is the "one CTA handles it" threshold.** With our topology
   512 + 128 = 640, split-KV is **always off**, therefore decode is **batch-invariant by
   construction** (the only reason the flag exists is to disable split-KV,
   `flash_mla_interface.py:110-111`). If our kernel ever splits the 640 rows across CTAs it buys
   nothing and costs determinism. Our `CSA_UNION_MAX_ATTENDED = 640` is the same number by an
   independent route - a good sign, and worth keeping as a compile-time assert.
2. **FP4 is legal only for the compressed (extra) cache; the window cache must be FP8.**
   `ModelType::V41_FP4` as `kv` is a hard error, and the kernel statically asserts
   `ExtraKVFormat::D_FP8 + ExtraKVFormat::D_FP4 == D_FP8` (`head64/config.h`) because both are
   dequantized into **one bf16 tile** by the same warpgroup before one MMA. That is an independent
   confirmation of our `layout.rs` (288 B FP4 main + 528 B FP8 SWA) and of our CSA-union kernel's
   design (window rows + selected rows -> one attention).
3. Byte strides are deliberate: FP8 row = 576 B in SMEM (512 raw + 16 scales + **48 padding** so
   adjacent rows are 64 B apart mod 128 to avoid LDS bank conflicts), FP4 row = 288 B. If we ever
   tune our SMEM staging, that padding trick is the reference's answer.

### 3.3 Bookkeeping formulas (decode)

From `tilelang_ops/cuda/prepare_decode_metadata.py` and the `_write_kv_*` helpers:

```
klen[ratio]                = (pos + 1) // ratio                     # visible compressed rows
positions_c[ratio]         = (pos // ratio) * ratio   if (pos+1) % ratio == 0 else 0
slot_mapping_store_ca      = block_tables[cn // cbs] * cbs + cn % cbs   if (pos+1) % ratio == 0 else -1
                             cbs = block_size // ratio          # compressed rows per page
kv_ca_index_offset_len     = [(pos + 1) // ratio, (pos + 1) // ratio]     # (offset, len) per row
swa_off                    = state_id * swa_block_size + window_start_pos % swa_block_size
klen_swa                   = min(pos + 1, window_size)
swa_block_size             = window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS
```

Cross-checked against `model.py`: `compress_len = (start_pos+seqlen)//ratio` (identical),
`freqs_cis[start_pos + 1 - ratio]` for the new row's RoPE (identical to `(pos//ratio)*ratio` when
a group completes), ring slot `start_pos % win` (identical). **Our compressed-token bookkeeping
should be diffable against these three formulas line by line**; they are the exact seam where our
holes bug lived.

---

## 4. Compressor state ring and "no new row"

`tilelang_ops/cuda/compress.py` (decode) and `models/v4_modules/compress.py`:

* ring `state_cache[state_id, cache_pos, 2, 512]`, `cache_pos = (klen-1) % state_cache_size`,
  `state_cache_size = compress_ratio + max(0, MAX_BACKTRACKING_TOKENS - 1)` -> **7 for ratio 2, 6
  for ratio 1** with the default `MAX_BACKTRACKING_TOKENS = nextn+1 = 6`.
* The token's `(score, latent)` is always written to the ring; the group result is computed **only**
  when `(klen-1) % ratio == ratio-1`; otherwise the kernel writes a **zero** row to `kv_compressed`
  and the store is masked out by `slot_mapping_store_ca = -1`. So "a forward that adds no row"
  is represented by a dummy row + a `-1` slot, and nothing reaches the cache.
* Group pooling is a **per-channel softmax over the `ratio` rows** then a weighted sum
  (`reduce_max` / `exp` / `reduce_sum` along the ratio axis, `compress.py:278-290`) - matches our
  `softmax(score) * latent` reading.
* **Prefill / chunking:** the prefill kernel reads the *previous chunk's* tail from the same ring
  before a second kernel overwrites it ("read the previous chunk's tail before store overwrites
  state_cache", `cuda/compress.py:76`). Partial groups therefore span chunk boundaries correctly,
  and the ring is the only cross-chunk state. Any chunked-prefill implementation of ours that does
  not persist this ring per sequence will silently mis-compress at chunk boundaries - exactly the
  class of bug that looks like "the TTFT path is fine but the tokens produced after the chunk are
  wrong".

---

## 5. State-block lifecycle (a hole-adjacent hazard we should copy)

`SchedulerOutputs` carries `clone_states` and `zero_states` (`csrc/engine/scheduler.hpp:76`), and
`load_paged_kv_cache` documents `src_block_id == -1` -> "writes zeros" for a
`[src_block, begin, end, dst_offset]` segment row. Two rules:

* a **forked** sequence (prefix share) gets its state block **cloned**;
* a **reused** state block is **zeroed** before use, because the reference's channel is a live view
  and would otherwise read stale rows from a previous request.

Our `sparkflash-kv` tier reuses SWA tails by hash; if a reused ring slot is not zeroed, a short
sequence can read another request's window rows. Worth a targeted fixture: two requests, the
second shorter than the first, same reused state slot, assert equality against a cold run.

---

## 6. Precision and formats

### 6.1 The KV formats (read off `cache_common.py`, cross-checked with `layout.rs`)

| Cache | Format | Per token | Where |
| :--- | :--- | ---: | :--- |
| SWA / window (`swa_cache`) | FP8 E4M3, per-32, **UE8M0** scales | 528 B (512 + 16) | `V41_FORMAT`, `derive_main_kv_cache_num_per_channels(is_swa=True) -> 32` |
| Compressed main (`ca_cache`) | **FP4 E2M1**, per-16, **E4M3** scales | 288 B (256 + 32) | `V41_FP4_FORMAT`, when `use_fp4_sparse_kv` |
| Index K (`ica_cache`) | FP4 E2M1, per-32, UE8M0 | 68 B (64 + 4) | `FP4_INDEX_KV_FORMAT` (`index_head_dim//2 + //32`) |
| Compressed state (`ca_state` / `ica_state`) | **f32** `[slots, ring, 2, 512]` | - | `_state_bytes` |

Two independent confirmations that our choice is right and complete:
`TORCH_CHECK(model_type != ModelType::V41_FP4, "the fp4 KV cache is only supported as extra_kv")`
(`sparse_decode.cpp:287`) and the static assert
`ExtraKVFormat::D_FP8 + ExtraKVFormat::D_FP4 == D_FP8` (`head64/config.h`). DeepInfer and `model.py`
(`fp4_act_quant(latent, 16, True, scale_dtype=E4M3)`) agree, and so do we. `cache_engine` also asserts
`kv_lora_rank + rope_head_dim == layout.num_elements` and `per_n_cast == layout.elements_per_sf` -
a cheap structural check we can mirror.

**No Hadamard.** `fused_norm_rope_hadamard_cast_store_kvcache` has a `hadamard_args` parameter that is
`None` on every V4.1 call site (`attn.py`, `compress.py`, `index_attn.py`). The transform exists for the
Ascend path / other models. So there is no hidden rotation in front of the FP8/FP4 cast, and our
quantiser does not need one to be bit-identical.

### 6.2 Weight formats

`layers/linear.py:17-31`: activations + weights are E4M3 with `num_per_channels=32`, `round_sf=True`,
`use_packed_ue8m0=True`, `use_tma_aligned_col_major_sf=True`, and the GEMM is
`deep_gemm.fp8_fp4_gemm_nt(..., recipe=(1, 1, 32))` - i.e. **F8_E4M3 Block32 with UE8M0
power-of-two scales**, exactly our attention-weight format. Experts are `use_fp4_weight`
(`cfg.mxfp4`) built with `per_token_cast(..., "e2m1", 32, ...)` from an FP4-QAT checkpoint (there is
an `assert ((w.view(int32) & 0x03030303) == 0), "Only support FP4-QAT checkpoint"`) - i.e.
**F4_E2M1 with E8M0 per-32**, exactly ours. No contradiction anywhere on the quantisation axis.

One reusable idea from `ReplicatedLinear`: every GEMM output is allocated with a **32-byte-aligned
row stride** (`_allocate_output`) and `out_acc` lets a second GEMM accumulate into an existing
buffer. If our `wo_a`/`wo_b` chain or the low-rank q path currently re-allocates and copies to
satisfy an alignment rule, this is the reference's cheaper pattern.

### 6.3 Batch invariance, honestly

The *only* mechanism in the released kernels is the split-KV disable above
(`sparse_decode.cpp:223,329` -> a distinct `BATCH_INVARIANT` feature instantiation). For our
topology it costs nothing because split-KV is already off at 640 rows. Whatever else the README's
"batch-invariant inference" covers (GEMM tiling, sampler reductions) lives in DeepGEMM/TileKernels,
not in the sources shipped here, so I cannot verify it. **[inference]** Our equivalent claim should
be scoped as "decode attention is batch-invariant because 640 <= 640 disables split-KV; MoE and
reductions are invariant because of fixed-order accumulation" - and tested, not asserted.

---

## 7. The things that CONTRADICT our design

### 7.1 Nothing contradicts our CSA / KV-reuse model (verified, not assumed)

I checked the one thing that could have: a **shared hidden** across layers. DeepInfer has a whole
YoCo machinery for it (`yoco.py`, `shared_hidden_kv_proj`, `consumer_full_hidden`, `_store_yoco_full_kv`),
and if it were active, layers 20..39 would project `kv_swa` from **layer 20's** attn-norm hidden
instead of their own, and would share layer 20's compressed KV even more tightly than we model.
It is **not active for this checkpoint**: `yoco_enabled = cfg.get("n_prefill_layer",
cfg.num_hidden_layers) < cfg.num_hidden_layers` and neither `reference/inference_config.json` nor
`reference/hf_config.json` contains `n_prefill_layer`. Hence 40 < 40 is false, there are no consumer
layers, `yoco_swa_share_hidden` is unreachable, and each layer's window KV comes from its own hidden -
exactly what our engine does. Our `LayerMode` set (`SwaOnly/Full/Reindex/Reuse/Mtp`) is complete; no
`Consumer` mode is missing.

### 7.2 The README's headline recipe is unrunnable on this checkpoint

`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` + `KEEP_TOKENS=140` - advertised in the README as
"the default V4.1 settings" - hits

```python
# model_metadata_utils.py:88-92
self.yoco_enabled = n_prefill_layer < self.n_layer
if envs.DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE != "off":
    assert self.yoco_enabled and n_prefill_layer > 0
```

and raises `AssertionError` at startup, because the open checkpoint has no `n_prefill_layer`. The
deployment Q&A's "For V4.1, the stored layers are the first 20" (`doc/inference_deployment_qa.md:163`)
and "a maximum hit length of L - 134" (`doc/inference_deployment_qa.md:191`) describe the
**production** V4.1 (which plausibly has `n_prefill_layer=20`), not `DeepSeek-V4.1-Flash-1017`.
**Action: drop SWA bounded replay / YoCo compaction from our TTFT roadmap.** It cannot help us, and
we have already implemented the equivalent (tail replay in `sparkflash-kv/src/swa.rs`). Anyone who
reads the README and budgets a week for this feature would waste it.

### 7.3 Our per-token cost is not where the kernels are

**[inference, flagged]** Prefill is about 0.9 s/token and decode about 0.93 s/step. A single
640-row attention over 40 layers moves about 640 KB x 40 = 25 MB/token; the weight bytes a rank must
touch are single-digit GB/token at worst; at LPDDR5X-class bandwidth neither is 900 ms. Two numbers
that close say the step is dominated by something **per token, not per byte**: 40 sequential layer
dispatches with a cross-node collective each is the prime suspect (40 x ~20 ms = 800 ms), or
per-layer host gaps. Cheap diagnostic: time one layer with 1 vs 16 query rows in the *chunked*
path. If 16 rows is about 1 row, we are bandwidth-bound and kernels matter; if 16 rows is about 16x,
we are dispatch/serialisation-bound and **no kernel from any reference can help**. This is the
single most important measurement before investing in kernel work, and it is about 30 minutes of
work with the existing `chain_timing` instrumentation.

### 7.4 Prefill scheduling

The reference never loops per token: prefill is one varlen launch per layer over `total_q` rows,
with the gathered KV built once per share-KV segment, chunked over **queries** with the indexer's
top-K computed per chunk. Its scheduler budgets by tokens *and* by `sum_seqlen_qxk` (the quadratic
attention term), with `attn_tflops`/`total_tflops` estimates and a 0.9 cache watermark
(`scheduler.hpp:76-100`). If our chunked prefill sizes chunks by token count only, we will
over-admit long prompts; the `sum_seqlen_k` budget is the right knob and it is the one line I would
lift from `scheduler.hpp` (`prefill_max_sum_seqlen_k`, `DEFAULT_PREFILL_MAX_SUM_SEQLEN_K`).

### 7.5 MTP / DSpark prerequisites (hand-off to the MTP slice)

Not my slice, but two numerical preconditions that live in *my* files and must land first:

* `swa_block_size = window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS` (`cache_engine.py:88-93`), with
  `MAX_BACKTRACKING = max(v, NEXTN+1) = 6` by default (`environ.py:129-132`) -> the window ring is
  **134 slots, not 128**, in the reference *even with MTP disabled*. Ours is 128 (`swa.rs`). It must
  become 134 (`window_size + nextn + 1`) before any draft token is written, or the draft will
  overwrite the rows the verifier is about to re-read.
* compressor ring `ratio + nextn` (7 for ratio 2, 6 for ratio 1), and `positions_c` must be computed
  per draft row (`max_seqlen_q = DEEPINFER_USE_NEXTN_DECODE + 1` is passed to the decode compressor
  explicitly, `models/v4_modules/compress.py:117`), i.e. the compressor runs over
  `total_q = b*(nextn+1)` tokens with per-row `context_lens`, and a rejected draft must rewind both
  rings.
* The DSpark blocks are a **different, smaller MoE** (`dspark_n_routed_experts=128`,
  `dspark_n_activated_experts=3`, `dspark_block_size=5`, `dspark_target_layer_ids=[37,38,39]`,
  `dspark_markov_rank=256` per `reference/inference_config.json`). Any plan that budgets the MTP
  layers as "3 more 384-expert layers" is 3x too pessimistic.

---

## 8. Kernel-by-kernel: can a Rust/CUDA sm_121a engine use it?

**Licence is not the obstacle.** DeepInfer is Apache-2.0; FlashMLA, DeepGEMM, DeepEP and DeepSelect
are all MIT (Copyright 2025 DeepSeek). Porting with attribution is fine. **Architecture is the
obstacle**, and it is categorical, not a matter of tuning.

### 8.1 FlashMLA (sparse decode / sparse prefill / fused norm+rope+attn+cast)

* **(a) Path.** `vendor/FlashMLA/csrc/cuda_kernels/sm100/{decode/sparse/head64,prefill/sparse,fused_norm_rope_attn_rope_cast_fwd}`,
  API in `csrc/api/{sparse_decode,sparse_prefill,fused_norm_rope_attn_rope_cast_fwd}.cpp`.
* **(b) What it does.** Decode: one kernel, two KV pointers (FP8 window ring + FP4-or-FP8 compressed
  paged cache), top-K indices as physical slots, `-1` masked, `MMA_M=64` = h_q, `D_V=512`,
  `B_TOPK=64` rows per step, optional split-KV, both cache formats dequantized into one bf16 tile and
  fed to one MMA, with an attention sink in the denominator. Prefill: varlen sparse over a gathered
  `[SWA | compressed]` buffer. Fused: RMSNorm + RoPE of q **inside** core_attn and RoPE + FP8 cast of
  the output, with `permute_q_b_proj`/`permute_wv_proj` pre-permuting weights.
* **(c) Our hardware.** **No.** Three independent gates: (i) compile targets are `sm_100a`/`sm_103a`
  only, with the comment that `sm_100f` is avoided because of `cvt ... .scaled::n2::ue8m0`
  (`setup.py:24-27`); (ii) a runtime `TORCH_CHECK(arch.is_sm100f())` (`sparse_decode.cpp:190`);
  (iii) the bodies use **tcgen05** (8 uses in the head64 decode kernel), SM100 2-CTA cluster TMA and
  `kerutils/device/cuda/sm100` intrinsics. sm_121a has no tcgen05 and cannot run `compute_100a` SASS
  or JIT its PTX. CUDA 13 is not the issue; the ISA is.
* **(d) Mapping to our engine.** Only as a **specification**: our `flash_mla_csa2_{seq_,}sm121.cu`
  already implements the same union (640 rows, window format 2, FP4 code+scale rows for the
  compressed part, `-1` masking). What is worth copying is *policy*, not code: the 640 threshold, the
  "FP4 only for extra_kv" rule, and the 48-byte SMEM padding trick.
* **(e) Payoff.** About 0 for tok/s and TTFT at our numbers (7.3). The fusion ideas are real but the
  tensors are small: q_b output is 64x512 per token.
* **(f) Risk.** None if read-only. Any attempt to compile it for sm_121 fails at ptxas, not at
  runtime - do not spend time on a "port".

### 8.2 DeepGEMM

* **(a)** `vendor/DeepGEMM/{deep_gemm,csrc/jit_kernels}`; API
  `fp8_fp4_gemm_nt(x, (w, w_sf), out, out_acc, recipe=(1,1,32))`.
* **(b)** JIT-compiled grouped/contiguous GEMMs, SM90 (Hopper wgmma) and SM100 (tcgen05 UMMA), with
  MXFP8/MXFP4 operands and packed-UE8M0 TMA-aligned column-major scale factors;
  `transform_weights_for_mega_moe` repacks the expert pair into the mega-MoE layout.
* **(c)** **No.** README: "SM90 or SM100 architecture GPU"; layout helpers
  `DG_HOST_ASSERT(jit->device.get_arch_major() == 10 ...)`. Arch major 12 is not in the table.
* **(d)** Our `b12x_*_sm121.cu` kernels and `sparkflash-engine/src/linalg.rs` already implement the
  same arithmetic. The one genuinely portable asset is the **scale-factor layout convention**
  (`get_mn_major_tma_aligned_packed_ue8m0_tensor`: 4 UE8M0 bytes packed into one int32, TMA-aligned,
  column-major) - relevant only if we ever consume `weight.sf` from the checkpoint instead of our own
  layout.
* **(e)** 0 (our GEMMs are ours). **(f)** Low; format compatibility only.

### 8.3 DeepEP / EPLB

* **(a)** `vendor/DeepEP`, `deepinfer/model_executor/eplb/{__init__,policy,shuffle}.py`.
* **(b)** Expert-parallel all-to-all over NVSHMEM-style symmetric memory with IBGDA, plus redundant
  experts (`n_redundant_experts`, `resolve_num_extra_experts`) and an in-place global expert
  permutation (`expert_model_parallel_shuffle_inplace` uses `torch.distributed._symmetric_memory`).
* **(c)** **No.** Requires a symmetric-memory-capable interconnect and EP >= 2; we have 1 GPU/node
  and RoCE/TCP.
* **(d)** Nothing to map. Worth stating why explicitly: with **intermediate-width** sharding every
  rank reads `(activated experts x expert bytes)/4`; with EP=4 every rank reads
  `(activated experts / 4) x expert bytes`. **The bytes are identical**; EP only buys load balance
  and pays an all-to-all plus a permutation table. Our choice is right for 4 nodes without NVLink,
  and EPLB (which exists to rebalance EP shards) is idle weight for us.
* **(e)** 0. **(f)** none.

### 8.4 DeepSelect `topk`

* **(a)** `vendor/DeepSelect/deep_select/interface.py` (+ `deep_select_cuda`).
* **(b)** Batched top-K over bf16/fp32 rows with `end` (per-row valid length), `output_idx_offset`,
  `sorted`/`sorted_index`, and `idx_oob_fill_value`/`value_oob_fill_value` for the "fewer than k
  valid" case. Used for the per-query top-512 and for HSI block ranking.
* **(c)** **No.** `setup.py:125-126` builds only `compute_100a`/`compute_103a`.
* **(d)** Reimplement, but **copy the semantics exactly**, because they decide tie and boundary
  behaviour our fixtures must match: invalid slots get `2147483647` (int32 max) and `-inf`, *not*
  `-1` - DeepInfer converts to `-1` afterwards for the attention path. Our engine breaks ties
  towards the lower index; `torch.topk` leaves ties undefined, so this is a documented deviation, not
  a bug (worth an explicit note in the fixture).
* **(e)** 0 for tok/s; the cost is host-side index selection. **(f)** Tie-breaking is the only real
  risk; it must be labelled.

### 8.5 The TileLang ops

* **(a)** `deepinfer/model_executor/layers/tilelang_ops/{cuda,}/*.py` (about 25 kernels: index
  expansion, compressor, KV store/load, norm, sampling, fused transform, candidate blocks).
* **(b)** Mostly **metadata kernels** - expand `(offset, len)` rows into paged/ring index arrays
  (`get_kv_index.py:20-66` has three modes: paged via `block_tables`, contiguous, and in-block ring),
  build SWA pivots (`swa_index.py`), build the decode/spec metadata, and the fused
  norm+rope+cast+store. These are the *cheapest* thing in the reference to reimplement and the *most
  likely* to hide a bug if we reimplement differently.
* **(c)** The sources are TileLang 0.1.15 (portable in principle: it generates CUDA for the local
  arch) but they are not self-contained: they import `tile_kernels` 2.1.0 (an external package,
  **not** in the repo) and use TMA / `T.pdl_sync` (programmatic dependent launch). Nothing here is
  drop-in for a Rust engine, and TileLang 0.1.x may not emit sm_121 at all. Treat as
  **specification**.
* **(d)** Our equivalents already exist (`sparkflash-kernels` metadata builders, `sparkflash-kv`
  paged/SWA layout). The mapping that matters is *index formula*: `kv_index_cat`'s
  `+ total_k + total_k_swa` for the compressed block, and `-1` preservation.
* **(e)** Possibly a real TTFT item: if our prefill spends host time rebuilding 37-token index arrays
  per layer per token, that is per-token overhead (7.3), not attention. Incremental index
  maintenance per layer would be the fix, and the reference's answer is "one device kernel per
  layer, launched once for the whole `total_q`".
* **(f)** Bit-identical only if we reproduce `-1` filtering and the exact ring/paged formulas; these
  are cheap to unit-test against the docstrings, which are precise.

### 8.6 `csrc/` (C++ host logic) - the most portable asset in the repo

`prepare_tensor_input_v4.cpp` (655), `scheduler.hpp` (1329), `block_manager.hpp` (242),
`mtp_varlen.hpp` (275), `prepare_sample.cpp` (284). Plain C++ that computes metadata and schedules;
no device code, no arch assumptions. This is the part I would read line by line when hardening our
paged decode metadata, because it implements the formulas in 3.3 for the real (CP>1, YoCo,
MTP-varlen) cases.

---

## 9. Ranked recommendations

1. **Keep the CSA publication fix as it is; add the owner assertion** (3 lines). `publish_rows`
   already copies the owner's whole cache and fires on no-new-row forwards, which is exactly the
   reference (2). Add `debug_assert_eq!(published_owner, spec.kv_source)` at every consumer,
   mirroring `assert mapped_layer_id <= layer_id`. *Payoff*: turns a silent cross-segment read into a
   fail-fast. *Risk*: none. *Effort*: 1 h.
2. **Make the channel a borrowed view instead of a copy** (the literal reference mechanism).
   *Payoff*: removes O(L) memcpy per owner per forward; makes the invariant structural. *Risk*: Rust
   lifetimes across the channel; hold an index + range into the owner's `AttnState` rather than a
   slice. *Effort*: 0.5-1 day. Do it only if the copy shows up in a profile - cleanliness, not speed.
3. **Assert the decode kernel contract in the launcher** (3.2): `h_kv==1`, `d==512`, heads in
   {64,128}, topk+window = 640, FP4 only for the compressed cache, `-1` as the sole sentinel,
   split-KV disabled at 640. *Payoff*: prevents a class of silent corruption and gives us a
   defensible "batch-invariant by construction" claim. *Risk*: none. *Effort*: 2-4 h.
4. **Diff our compressed-token bookkeeping against 3.3 line by line**, especially `positions_c`,
   `slot_mapping_store_ca` (`block_size//ratio` compressed slots per page, `-1` for no-store) and
   `klen = (pos+1)//ratio`. This is where our holes bug lived; a table-driven unit test over
   `(ratio, pos, block_tables)` against these formulas is the cheapest insurance we can buy.
   *Effort*: 2 h. *Risk*: none.
5. **Size the rings for speculation now**: window ring 128 -> `window_size + nextn + 1 = 134`,
   compressor ring `ratio + nextn` (7 / 6). *Payoff*: unblocks MTP, the only lever that divides the
   926 ms step. *Risk*: SWA footprint grows 4.7% (134/128). *Effort*: 2 h + fixtures.
6. **Read `csrc/engine/{prepare_tensor_input_v4,scheduler}.hpp` before hardening our paged metadata**
   and copy the `sum_seqlen_k` chunk budget and the `clone_states`/`zero_states` lifecycle rules.
   *Effort*: 1 day of reading, then small patches.

**The single highest-leverage item in this slice** is not a kernel: speculative decode is the only
available mechanism that divides the per-step cost, and the two ring sizes in item 5 plus the
`positions_c`/`-1` conventions in item 4 are its hard prerequisites. Everything else here is either
already correct (the publication fix - verified twice) or worth well under 1% of 926 ms. Corollary:
**do not budget time for porting DeepInfer kernels** - none can execute on sm_121a, and the
arithmetic they implement is already in `kernels/*_sm121.cu`.

---

## 10. Appendix - evidence index

| Claim | Evidence |
| :--- | :--- |
| Channel is a static cache **alias**, resolved at allocation | `cache_engine.py:160-172` (`assert mapped_layer_id <= layer_id`), `:431` (`ica_cache = cache_list[ica_mapped_layer_id].ica_cache`), `:449` (`ca_cache = cache_list[kv_mapped_layer_id].ca_cache`) |
| Consumer never writes the shared compressed cache | `attn.py`: `_store_main_kv` gated on `is_kv_source`; `store_index_kv` on `has_local_index` |
| Prefill consumer reuses the source's gathered buffer, overwrites only SWA | `attn.py:612-626` + `_prepare_prefill_kv` docstring; `cross_layer.py` `CrossLayerRuntime` docstring |
| Publication = whole cache, every forward, even with no new row | `model.py _compress_kv`: assignment above `if latent is not None`; returns `[:compress_len]` |
| Topk rows replace the dense index for sparse layers; `-1` only sentinel | `attn.py:568-587`, `index_attn.py:448-575` |
| Physical-slot conversion preserves `-1` | `tilelang_ops/cuda/flash_mla_utils.py` (`convert_kv_index`, decode topk path) |
| Compressed block shifted by `total_k + total_k_swa`, `-1` untouched | `tilelang_ops/cuda/kv_index_cat.py:60-80` |
| `klen` / `positions_c` / `slot_mapping_store_ca` / SWA offset formulas | `tilelang_ops/cuda/prepare_decode_metadata.py` (`_write_kv_*`), `cuda/get_kv_index.py:20-66`, `cuda/swa_index.py:14-32` |
| Compressor ring size and no-store case | `cuda/compress.py` (decode kernel: `(klen-1) % state_cache_size`, `-1` slot), `models/v4_modules/compress.py:110-119` (`max_seqlen_q=NEXTN+1`) |
| Chunked prefill reads the previous chunk's tail from the ring | `cuda/compress.py:76` comment + the second store kernel |
| FP4 only as extra KV; window must be FP8 | `sparse_decode.cpp:287`; `head64/config.h` `static_assert(ExtraKVFormat::D_FP8 + ExtraKVFormat::D_FP4 == D_FP8)` |
| Split-KV only when `topk+extra_topk > 640`; batch invariance = no split-KV | `sparse_decode.cpp:223`, `:329-331`, `flash_mla_interface.py:110-111` |
| SM100f runtime gate | `sparse_decode.cpp:190` |
| SM100-only compile targets + `cvt.scaled::n2::ue8m0` rationale | `vendor/FlashMLA/setup.py:24-27` |
| DeepSelect SM100-only | `vendor/DeepSelect/setup.py:121-126` |
| DeepGEMM SM90/SM100 only | `vendor/DeepGEMM/README.md:31`; `csrc/jit_kernels/impls/smxx_layout.hpp:224` |
| Licences | DeepInfer Apache-2.0; FlashMLA/DeepGEMM/DeepEP/DeepSelect MIT |
| YoCo gated off for this checkpoint | `yoco.py:14-16`; `model_metadata_utils.py:88-92`; no `n_prefill_layer` in `reference/{inference,hf}_config.json` |
| Window ring 134, not 128 | `cache_engine.py:88-93` + `environ.py:131-132` |
| Hadamard never enabled | `fused_transform.py:189-320`; `hadamard_args=None` at every V4.1 call site |
| State clone/zero and `-1` zero-fill | `scheduler.hpp:76`, `load_paged_kv_cache.py:36-52` |

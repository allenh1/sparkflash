# AttnState residency for the whole-layer device arm (2026-10-22)

The design of record for where the attention seam's sequence state lives once the layer runs on
device. Companion to `docs/build/ws-whole-layer-device-port-20261022.md`; this file is the only
place the residency ruled here is written, so the port doc stays a merge record.

## 1. Ownership ruling: one window, one owner

There are two window copies in the tree today:

1. the engine's `AttnState::window` ring, f32 rows, written by `WindowRing::write`
   (`crates/sparkflash-engine/src/forward/attention.rs:457-470`), and
2. the four-rank lean loop's fp8 window cache, `Vec<Vec<(Vec<u8>, Vec<u8>)>>`, appended in
   `crates/sparkflash-cli/src/tp4_forward.rs` (declaration at :12372, append at :12870
   `cache.len() == position` check, window_start at :12898) and staged as the attention body's
   `window_codes`/`window_scales` operands.

**Ruling: the device arm owns exactly one window, and it is the loop's fp8 cache, moved to the
arm's device buffer.** `WindowRing` stays read-only and host-arm only. The arm does not mirror it,
does not write both, and does not read it. Two mutable caches is how a row becomes unexplainable;
one owner is what makes a stage record complete.

The fp8 window form is also the attention body's own operand form (`WINDOW_CODE_BYTES` +
`WINDOW_SCALE_BYTES` per row, engine-format E4M3FN codes and E8M0 scales,
`crates/sparkflash-cli/src/tp4_prefill.rs:40, staged by stage_attention at :294`), so a device-owned window needs no repack for
the body: the append writes exactly the bytes the launch reads.

## 2. What moves, and its size

Per layer, allocated by the arm **at open, never per position** (`head_dim = 512`, `dim = 5120`;
positions bound `MAX_POSITIONS`, refused past it rather than grown silently):

| channel | form | bytes | bound |
| :-- | :-- | --: | :-- |
| window rows | fp8 codes + E8M0 scales | 264,192 B | `window_size * (WINDOW_CODE_BYTES + WINDOW_SCALE_BYTES)`, window_size 129 |
| compressed rows (FP4-native, see 3) | E2M1 codes + scales | `(MAX_POSITIONS/ratio) * STORE_CODE_BYTES` + scales | `MAX_POSITIONS/ratio` rows |
| indexer keys | f32 | `128 * MAX_POSITIONS * 4` | `index_head_dim * MAX_POSITIONS` |
| compressor `kv_state` | f32 | `ratio * head_dim * 4` | one partial group |
| compressor `score_state` | f32 | `ratio * head_dim * 4` | one partial group |
| step `latent` | bf16 rows (declared deviation) | `ratio * head_dim * 2` | one call's rows |
| `index_weights` | f32 | `index_n_heads * 4` = 128 B | one query |
| compressor weights | f32 (ratio > 1) or bf16 (ratio == 1) | 20.98 MB / 5.24 MB | `wkv` (+ `wgate` above ratio 1) + `norm` |
| host-weight GEMV staging | f32 / bf16 | 10.49 MB / 5.24 MB | the same weights, staged once per layer per run |

The compressor weight staging is the only large load: 10.49 MB f32 (layers 2, 8, 14: `wkv` +
`wgate`) or 5.24 MB bf16 (layer 20), a pure permutation of the checkpoint's own values, once per
layer per run, not per position. Nothing here is per-position traffic.

**Memory accounting.** Charge all of it to the layer-arm slab and print the sum at open; the
residency figure is a number, not an impression. The accounting line must separate
(a) sequence-state bytes that grow with `MAX_POSITIONS`, (b) per-layer constants, and (c) the
staging permutation, so a later `MAX_POSITIONS` change is a one-line diff with a known byte
consequence.

## 3. The compressed store: FP4-native, adopted

**Chosen form: the expanded FP4 store is the cache's native form.** Today the seam publishes f32
rows (`AttnState::compressed`, `attention.rs:599-645`) and `stage_attention` re-quantises each
selected row with `quantize_main_kv` into a zero-filled `EXPANDED_ROWS`-row store on every call
(`tp4_prefill.rs:332`; 147,456 B zero-filled per call). Making the store native removes both
the per-selected-row quantiser and the per-call zero-fill: the compressor appends into
`EXPANDED_ROWS * STORE_CODE_BYTES` (plus scale bytes) once, and `stage_attention`'s contract
becomes a byte copy of selected rows instead of a repack.

Byte consequence: the compressed cache grows by `STORE_CODE_BYTES + STORE_SCALE_BYTES` per
compressed position instead of `head_dim * 4` f32 (the store is narrower than the f32 row it
replaces), and the per-call 147,456 B transient disappears. The exact constant should be read off
`tp4_prefill.rs`'s `STORE_CODE_BYTES`/`STORE_SCALE_BYTES` when the stage is wired; the ruling
here is the form, not the constant.

## 4. Process-global channels: unreachable from the arm

`reset_shared_attention()` (`attention.rs`, called from `ReferenceAttention::open`) owns the
process-global candidate/index/topk channels the reference path shares between layers. The arm must
not read or write them: its channels are its own slab. If the arm needs a reset, it is an explicit
arm call, and the host arm's globals stay exactly as they are for the reference path. A shared
mutable cache or channel between the two arms is the bug this design exists to prevent.

## 5. What the arm does not do

It does not keep a host copy of any of the above for convenience, does not read back the window or
the compressed store in the steady state, and does not fall back to the host cache on a short span:
a span that does not fit the bound is refused with the stage's error, not silently regrown. The
only host sync in the steady state is the position boundary.

## Second section: the device arm as the resident version of attention_device.rs

The m20 backend already allocates every operand the CSA2 body dereferences
(`crates/sparkflash-cli/src/attention_device.rs`, 2,905 lines); it fills them with per-position host
uploads. The port is that layout made resident, not a new layout. This section is the mapping and
the capacity rules; the typed form is `crates/sparkflash-kernels/src/cache_plan.rs`
(`CachePlan`, `Geometry`, `Span`, `LayerSpans`, `WorkspaceSpans`, `HsiMode`, `PlanError`).

### Per-layer bytes and capacity rules

Ring capacities are **max_positions** (the refusal bound, 4,096), not the window: the seq entry
reads the last 128 rows contiguously from a base offset, so the ring must never wrap under a read,
and a position past the bound is refused rather than grown.

| span | rule | bytes at max_positions = 4,096 |
|---|---|---|
| latent ring | 512 f32 pre-RoPE latent rows; RoPE applied at group completion from the deterministic position (attention.rs:2966-2974), so no rope table is resident | 8,388,608 |
| window codes | 512 E4M3 code bytes per row, capacity max_positions | 2,097,152 |
| window scales | 16 E8M0 scale bytes per row (block 32) | 65,536 |
| FP4 store codes | fixed 512 expanded rows x 256 B | 131,072 |
| FP4 store scales | fixed 512 expanded rows x 32 B | 16,384 |
| append cursor | one u32, append-only | 4 |
| selected | 512 u32 expanded row ids | 2,048 |
| HSI operands + sink, direct mode | four declared words plus 256 B sink | 272 |
| HSI operands + sink, block-mask mode | 4,194,304 + 16,384 + 8,192 + 8,192 + 256 | 4,227,328 |
| **per-layer stride, direct** | | **10,701,076** |
| **per-layer stride, block mask** | | **14,928,132** |

Window row cost is **528 B** (512 code + 16 scale), the number the port's window ring is priced at.
Four ratio layers: direct **42,804,304 B** of layer slab; one-copy workspace q + out_pre_rope +
out_post_wo_b = **196,608 B**; total **43,000,912 B (~41.0 MiB)**. In block-mask mode the layer slab
is 59,712,528 B and the total 59,909,136 B (~57.1 MiB) - the 16 MiB difference is HSI operands the
arm only needs when the body computes the mask itself.

### Which attention_device.rs buffers survive, and which disappear

| buffer (attention_device.rs) | m20 bytes | decision and reason |
|---|---|---|
| q, MLA_Q_BYTES :47 | 65,536 | survives as workspace; written once per step by ref_seam, no longer read back and re-uploaded |
| a_codes/a_scales/a_bf16/w/w_scales/out, :358-363 | up to 5,120x8,192 | survive as per-step operands plus per-layer resident weights: the weight H2D happens once per run, the activation traffic is the projection chain's own |
| kv_fp4 :52, kv_scales :53 | 131,072 + 16,384 | survives as the FP4 expanded store, but **append-written** (compressed_append_sm121): the per-call 147,456 B zero fill and the per-row quantize_main_kv disappear |
| window_codes :49, window_scales :50 | 65,536 + 2,048 | survives at **capacity max_positions** (528 B/row), so the last 128 rows are contiguous with no wrap; the per-position upload disappears |
| selected :54 | 2,048 | survives and is written on device by compressed_selected_fill_sm121 from the ref_select ids; no host upload |
| index_key :56 | 4,194,304 | kept only in block-mask mode. With ref_select ids the body does not read it, so direct mode allocates one word and **declares** that state - a zero-filled or null operand is a declared wiring state, never a neutral default |
| index_query :57 | 16,384 | same rule as index_key |
| block_scores :58, block_best :59 | 8,192 each | same rule as index_key |
| sink :60 | 256 | survives per layer, uploaded once at open: a per-layer constant, not per-position traffic |
| mla_out :376 and out :363 | 65,536 each | both survive as **distinct** workspace rows: the pre-inverse-RoPE output and the post-wo_b output are never the same span |

What the port deletes is traffic, not buffers: q, kv_fp4, kv_scales, window codes/scales, selected
and the sink are no longer packed, quantised and copied per position; the store is written by the
append, the selected list by the fill symbol, the window rows by the KV path, and the two out rows
stay separate because aliasing them once changed a result in the pre-port backend.

### Correction: per-layer kinds, not one stride times a count

The earlier totals in this section (43,000,912 B direct, 59,909,136 B block-mask) multiplied one
uniform stride by four layers. That is wrong and would have bitten at open: **every one of the 40
layers does window attention, so every layer needs its own codes/scales ring**, and the seq body
dereferences each layer's. Only the compressor channels (pre-RoPE latent parents, FP4 store, cursor)
belong to the ratio layers; the selected list and sink are per layer as well, because the body takes
them per launch and aliasing them across layers is a sequencing assumption we do not need. The plan
now iterates kinds ("CachePlan::new" takes "num_hidden_layers", "compress_ratios",
"kv_source_layer_ids", "index_source_layer_ids" from the config) and layer offsets are the running sum
of the per-kind byte counts; a config that does not expose a ratio for every layer is refused by name
(PlanError::RatiosShort) rather than assumed 4 or 40.

The released text_config: 40 layers; compress_ratios = [0, 0, 2 x 18, 1 x 20, 0 x 3 padding];
kv_source_layer_ids [2, 8, 14, 20]; index_source_layer_ids [2, 8, 14, 20, 24, 28, 32, 36].

| layer kind | layers | channels | bytes per layer, Direct |
|---|---|---|---|
| window only, ratio 0 (0, 1) | 2 | window codes/scales, selected, sink | 2,164,992 |
| window only, ratio 2 consumers (3-7, 9-13, 15-19) | 15 | window codes/scales, selected, sink | 2,164,992 |
| window only, ratio 1 consumers (21-39) | 19 | window codes/scales, selected, sink | 2,164,992 |
| compressor owner, ratio 2 (2, 8, 14) | 3 | + latent ring, store, cursor | 10,701,060 |
| compressor owner, ratio 1 (20) | 1 | + latent ring, store, cursor | 10,701,060 |

Channel totals at max_positions = 4,096 (Direct): window channels on 40 layers 86,507,520; latent
rings on 4 owners 33,554,432; FP4 store 589,824; cursors 16; selected on 40 layers 81,920; sinks
10,240 -> layers 120,743,952. Workspace (q, out_pre_rope, out_post_wo_b, index query/scores/best,
index weights rows, Direct key word) 745,476 -> **grand total 121,489,428 B (~115.9 MiB)**. Block-mask
mode adds one 4 MiB index key per key owner (4 x 4,194,304 = 16,777,216) with consumers aliasing the
preceding owner's span: **138,266,644 B (~131.9 MiB)**. The order-120 MB figure this replaces the
41 MiB claim with is fine; the point is that the offset math now allocates all 40 window rings, so
no layer can refuse or silently read another layer's stale ring.

The per-layer tiling assertion now runs for every layer (last span ends exactly at the next layer's
base), and the tests parse the released table from the config numbers above.

### The 8-vs-4 index split

Index-source layers are [2, 8, 14, 20, 24, 28, 32, 36] (8: they have "indexer.weights_proj");
key-owner layers are the kv sources [2, 8, 14, 20] (4: they run "write_index_keys"). Layers 24, 28,
32 and 36 are index sources that own no keys - a real distinction the body can get wrong, so the plan
carries both flags ("Kind::index_source", "Kind::key_owner") and the weight rows live in the
one-copy workspace (steps are sequential) while the key span is owned once and aliased.

### Deviant 2 resolved: the index query is 32 heads, not 64

The pinned kernel static_asserts "SPARK_M4_INDEX_HEADS == 32u" (flash_mla_csa2_ring_sm121.cu:73) and
addresses "index_query + index_head * SPARK_M4_INDEX_HEAD_DIM" with
index_head = head / SPARK_M4_MLA_HEADS_PER_INDEX_HEAD (64 query heads over 2 = 32 index heads);
mla_ref::INDEX_HEADS documents itself as mirroring SPARK_M4_INDEX_HEADS and is 32; and the host
producer in tp4_forward.rs allocates exactly "mla_ref::INDEX_HEADS * mla_ref::INDEX_HEAD_DIM * 2" =
8,192 B. So the operand width is **32 heads / 8,192 B**, and the 64-head constant in
attention_device.rs:57 (16,384 B) is a stale 2x over-allocation the m20 kernel never read. The plan
uses 8,192 B; the cost of the correction is 8,192 B less workspace and no numeric change.

### W1 correction, as amended: one convention for the shared word

The engine rule is filled = min(p + 1, window) with p the 0-based token index (window_indices decode,
attention.rs:945; m20's window_count counts the current row; the port's reference staging is one row
at position 0). The module's old arithmetic min(position, 128) was right for a **count** argument
(position meaning the token count p + 1); the hazard was that the body launch's position word is the
index p, so a caller passing p with ring_rows = p + 1 passed every check and silently got
window_rows = p - no window at position 0 and one row short everywhere else, a quietly-wrong
attention output that the stage exists to prevent. Fixed by taking p everywhere: assemble derives
window_rows = (p + 1).min(128) and window_start = p + 1 - window_rows through one public helper
(window_span), so the body's position and window_rows words cannot disagree; the refusal is p + 1 >
ring_rows, and a caller that passes the count is now refused (assembling at ring_rows = 41, position
41 returns PositionPastRing) instead of short. append-before-read is a **requirement, not an
equivalence**: the normal ring state at a stage is ring_rows == p + 1, a stage run before its append
is refused rather than assembled sink-only. The arm plan's equivalence sentence (single-writer port
doc, append visibility section) needs the same wording; this doc states it and leaves that file to its
writer.

Tests: window_span is pinned against the engine's window_indices row for tiny's window 8 - the decode
at p = 10 is the engine's [3, 4, 5, 6, 7, 0, 1, 2] and our (start, rows) = (3, 8), p = 0 is exactly
one row at slot 0, a short sequence at p = 2 gives (0, 3) with the unreached slots NO_INDEX - and the
count-style call is a refusal test. The stage tests keep the p convention: p = 0 -> (1 row, start 0);
p = 40 -> (41, 0); p = 300 -> (128, 173). Open question for the wiring slice, not silently changed:
selected_rows keeps its base convention (ids offset by window_rows); if gemv passes absolute token
positions, the base must be window_start.

### W2 declaration: milestone A's indexer is declared-zero, not missing

The seq entry's indexer stage strides by SPARK_M4_HEADS = 64 while a per-rank launch covers 16 heads
(64 query heads over TP=4), so the HSI block pool is not complete at world 4, and index_key /
index_query have no device producer among the 27 pinned kernels. Both facts are declared, not
papered over: the HSI operands are allocated as **zero-filled spans** and nothing consumes them in
the format-2 selected path, where the candidate list comes from ref_select (that is exactly what
HsiMode::Direct models, and its wording now says the block arrays are zero-filled and unconsumed). A
truthful indexer is a later slice with its own pin. A reader who assumes the indexer ran would
mis-read every later number in this document, so this paragraph is the qualifier for all of them.

### kv_path re-point: the append's inputs, the norm gap, and the refusal side

**The latent ring's row is 512 bf16 = 1,024 B, not the f32 pre-RoPE 2,048 B first planned.** The
append consumer is compressed_append_sm121, which takes the normed and roped latent rows (kv_rotated,
bf16) and consumes ratio of them per completed row, contiguous in slot order from
first_row = ((p + 1) / ratio - 1) * ratio at a boundary p where (p + 1) % ratio == 0. So the ring
holds post-norm post-rope bf16 rows, its capacity must cover one append (a plan with max_positions <
ratio is refused as PositionsShortOfRatio), and the rows must be written before the append that reads
them - the same append-then-read requirement as the window ring. Per owner: 4,096 x 1,024 =
4,194,304 B; owner layer 6,506,756 B; layers 103,966,736 B; workspace 745,476 B; grand total
104,712,212 B (~99.9 MiB), and 121,489,428 B (~115.9 MiB) in block-mask mode with the four 4 MiB index
keys. The earlier 2,048 B row also made the totals in the correction table stale; these are the
current numbers.

**The norm is a declared gap, not an approximation.** The host fuses the norm on f32 and rounds to
bf16 once (tp4_forward.rs:3032-3038), the device kernel kv_norm_rope_bf16 is not pinned yet, and
NORM_PIN is None. NormSource::HostReferenceOnly is refused with PlanError::NormUnpinned, so the
re-point cannot append the unnormalised latent and no green body can hide it: until the fused kernel
lands, the ring write is refused by name or the deviation is recorded next to it.

**The two selected-id refusals live in the host reference path, by construction, not on the device.**
Option (a): ref_select_sm121 writes position + offset with offset = window_rows over compressed
positions bounded by the plan (compressed_rows = 512 / ratio), so id >= window_rows and
(id - window_rows) * ratio < 512 hold by construction; the fill kernel assumes a valid plan and its
arithmetic is out = (id - window_rows) * ratio. The below-window and past-store refusals are
compressed_cache_ref::selected_rows checks and its tests - a reference-path check, never a
per-position readback - and a twin range test pins ids in [window_rows, window_rows +
compressed_positions) mapping below 512 rows. No kernel or ABI change, no new pin.

**The selected count's owner is the device.** The fill kernel writes out_count, and the seq entry
reads selected_count from that word; the host's plan-derived count is a twin check and never the
launch's n_ids, because a host formula that disagreed at a masked position would make the entry read a
stale tail. One number, one writer.

### The probe row's refusal is a missing resident class, not a kernel defect: mhc_seam, ten stored tensors per layer

The row refused at layer 0's first body with `no hc_attn_fn for layer 0 in the resolved set`, identically on four
ranks, after the bootstrap agreed. The census says why: the attention bind binds 208 tensors (200 backbone B12x
projections + 8 indexer wq_b) and structurally excludes 19 host seams, and there is no `hc_` name in the bind at all.
The six mHC tensors and the four attention norm weights are plan tensors of the `plain` (stored) layout - no B12x
payload, no scale grid - so the attention bind's table never contained them, the arena never uploaded them, and
`collect_layer_weight_pairs` could only resolve keys the bind carried. `WEIGHT_NAMES` already lists all thirteen keys
per layer; the bind, not the loop, was the gap.

**Decision: (a) a new resident class `mhc_seam`, ten stored tensors per layer, bound as a second pointer family
over the same already-uploaded plan arena - a bind-only change.** The ten, by consumer: `mhc_entry`'s four - `hc_attn_fn` (F32 [24, 20480] = 1,966,080 B),
`hc_attn_base` (`[24]` F32 = 96 B), `hc_attn_scale` (`[3]` F32 = 12 B) and `attn_norm.weight` (BF16 [5120] =
10,240 B); `ffn_seams`' four - `hc_ffn_fn` (1,966,080 B), `hc_ffn_base` (96 B), `hc_ffn_scale` (12 B) and
`ffn_norm.weight` (10,240 B); the query and KV path's two - `q_norm.weight` (BF16 [1280] = 2,560 B) and
`kv_norm.weight` (BF16 [512] = 1,024 B). The mhc_entry declaration already takes `fn_w`, `hc_base`, `hc_scale` and
`norm_w` as device words, so the class is what the ABIs were written for. Per layer 3,956,440 B; 400 tensors over 40
layers = **158,257,600 B (150.92 MiB)**, trivial against the 72 GiB arena - and not new bytes at all: the probe's
census reads resident_tensors 95,818 with `mhc: 240`, `attention_aux_host_read: 416` (160 of those are these four
norms) and `attention_projection: 208`, and the streaming upload already carries every resident shard, so the class
adds no plan row, no byte, no read, no copy and no allocation - only the pointer family and its coverage line. Not (b) open-time staging: the loop must not
carry a host decode, and staging would put the same bytes through a copy the class does not need.

**Counters.** The attention projection bind stays 208 expected / 208 bound (it keeps meaning "B12x device weights:
200 backbone projections plus 8 indexer wq_b"), `host_seams_excluded` stays 19, and the new class reports
`mhc_seam: 400 expected, 400 bound` in its own `mhc_seam` block of each rank's `tp4-bind-r<rank>.json`, with its 400
names in `tp4-bind-names-r<rank>.txt` - the next row's diagnosis is a counter (`grep -c hc_` = 240, `norm.weight` =
162), not a failed row - and the summary's `device_bound_names` goes 92,731 -> 93,131. The class is additive: it
reads no checkpoint byte, allocates nothing and copies nothing (the plan's arena already holds the shards), and the
host arm's per-step path never reads it - only its open-time bookkeeping sees 400 extra pointer entries and 400
extra manifest lines, so its 179.81 ms/step reference is untouched. One class for the whole 608 is the alternative; the separate report is chosen so
the 208 sentence stays a statement about B12x projections and the stored seam weights get their own sentence.

**Resolve.** `LayerWeights::resolve` gains the mhc_seam bind beside the attention bind, merged into the same
`(layer, WEIGHT_NAMES-index)` map; then all thirteen keys resolve for every layer and layer 0's first body resolves
`hc_attn_fn`. The 400 tensors are `Resident` and replicated (`AttentionReplicated`; the arm has no EP/TP split for
attention weights) and the bind's stored branch checks `stored_bytes`, no B12x and no scale grid.

**Refusal if the class is absent.** The bind fails before any position runs, with the coverage sentence naming every
expected-but-not-bound tensor (`mhc_seam bind coverage: 400 expected, 0 bound, ...`), and because the set is
rank-invariant the message is identical on all four ranks - the same deterministic-on-four-ranks invariant the probe
row exercised, now on a class that is supposed to exist. It stays exit 2 at open, so no stage counters, no run record
and no outer timeout path: a clean refusal is still the failure mode, and it names the class.

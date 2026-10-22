# attention_body / attention_output launch-sequence contract (device arm)

Status: **review bar, not an implementation**. Written before gemv's `attention_body`/`attention_output`
sha exists, so the body is reviewed against a pinned sequence instead of against whatever it happens to
do. No code in this document is new; every claim is tied to a file and a line of the integration tip
`54784974` or to the recon, and the places where this contract disagrees with what is landed are called
out as findings.

Sources read at `54784974`: `crates/sparkflash-cli/src/attention_device.rs` (the m20 backend, 2,905 lines),
`crates/sparkflash-cli/src/tp4_forward.rs` (`MlaAttentionBody::run` ~3369, `run_layer_kv_path` ~3029,
`run_layer_attention_output` ~4034), `crates/sparkflash-cli/src/tp4_prefill.rs` (the reference staging,
`stage` ~465), `crates/sparkflash-kernels/src/{attention_stage,compressed_cache_ref,compressed_append_launch,
kv_path_launch,flash_mla_csa2_seq_launch,flash_mla_csa2_ring_launch,bf16a_mxfp8w_wo_a_launch,
mxfp8_wo_a_launch,mxfp8_gemv_launch}.rs`, `kernels/flash_mla_csa2_seq_sm121.cu`, and
`crates/sparkflash-engine/src/forward/attention.rs` (`window_indices` 945, decode branch 3519).

## What correct means here

Correct is **bit identity against the port's own host reference per rank**, not against a world-1 run.
`run_layer_attention_output` at the tip already runs the released per-rank decomposition: `groups =
o_groups / world` (1 at world 4), `heads_per_rank = 16`, `group_inner = heads_per_group * head_width =
16 * 512 = 8,192`, `o_lora_rank = 1,024`, `wo_b_inner = groups * o_lora_rank = 1,024`, wo_a geometry
`(groups * o_lora_rank, group_inner) = (1024, 8192)`, wo_b geometry `(HIDDEN, 1024) = (5120, 1024)`. The
wo_b inner axis is TP4-sharded and the cross-rank f32 reduce order is a designed non-associativity, so
the device arm must reproduce **each rank's partial** and then the canonical rank-ascending reduce; it
must not claim a world-1 full-inner equality.

## Stage boundary and the rows it needs

The reference body stage ends with the **inverse RoPE**: `MlaAttentionBody::run` launches the sequence
entry, reads `out` back, rotates each head's last `rope_head_dim = 64` back with
`linalg::rope_rotate_inverse`, and returns that as `AttentionStep.activation_bf16`, which is what wo_a
consumes (tp4_forward.rs ~3524-3540). So the boundary is:

* `attention_body` -> the body row after the inverse rotation (per rank: `heads_per_rank * head_width`
  bf16 = 16 * 512 * 2 = 16,384 B). **New row**, name it `attn_body_bf16`. It cannot be
  `workspace.attn_out_bf16`: that row is `dim * 2` = 10,240 B and is already the wo_b output /
  all-reduce input by contract (layer_device.rs workspace table). Two rows, distinct names, and the
  contract says which is which.
* `attention_output` -> `workspace.attn_out_bf16` (the wo_b partial, dim bf16) -> `workspace.reduced_bf16`
  (the all-reduce output, consumed by `mhc_attn_residual`). Both rows exist.

Rows the two stages read or write, with provenance:

| row | bytes (per rank) | producer | consumer | exists at 54784974 |
| --- | --- | --- | --- | --- |
| `workspace.q` | 16,384 | `query_path` (wq_b + forward rope tail in place) | body seq entry | yes |
| `kv_ring` codes/scales | 270,336 per layer (512 rows x (512 + 16)) | `kv_path` `window_quant_fp8_bf16` per position, slot `position % rows` | body window read | yes (`WindowRingLayout::new(capacity_rows)`) |
| store codes/scales | 147,456 (512 x (256 + 32)) | `compressed_append_sm121` (device) | body seq entry | **no row**; launcher exists, no caller |
| `selected` ids | 2,048 (512 x 4) | `ref_select_sm121` + `compressed_selected_fill_sm121` (device), or host-staged ids in the interim | body `selected` word | **no row** |
| `selected_count` | 4 (u32) | host-computed (see below), never a D2H | body `selected_count` word | no word |
| `index_key` pool | 4,194,304 (2048 blocks x 8 tokens x 128 bf16) | **none on device**; host `write_index_keys` today, m20 zero-fills | body indexer stage | **no row** |
| `index_query` | 256 (128 bf16, index head 0) for the seq entry; 2,048 if ref_select wants all 8 per-rank index heads | indexer wq_b projection (not wired) | body indexer + ref_select | **no row** |
| `sink` | 64 (16 heads x f32) | staged once per layer at open from the arena | body `sink` word | no row |
| `block_scores` / `block_best` | 8,192 + 8,192 (2048 f32 + 2048 u32) | the seq entry writes them every launch | ref_select / candidate select (not wired) | no row |
| `attn_body_bf16` | 16,384 | `attention_body` (seq out then inverse rope, may alias itself) | `attention_output` wo_a | **no row** |
| `workspace.act_codes` / `act_scales` | 1,024 + 32 used of the existing spans | `w8a8_prologue_bf16_e4m3` in `attention_output` | wo_b GEMV | yes (shared, stage order makes it safe) |
| `workspace.attn_out_bf16` | 10,240 | wo_b GEMV | all_reduce | yes |
| `workspace.reduced_bf16` | 10,240 | all_reduce | `mhc_attn_residual` | yes |

## attention_body: the ordered sequence

**Step 0 - preconditions, all decided on the host, no device readback.** `run_layer` order is load-bearing
here: `kv_path` runs before the body (tp4_forward.rs 11687 region), the ring row for `position` is already
written, and the body reads it. State the rule as:

```
window_rows   = min(position + 1, 128)
window_start  = position + 1 - window_rows
ring slot p   = p % ring_rows, ring_rows = max_positions (512), so a run of <= 512 positions never wraps
```

Here `position` is the loop's 0-based token index p (that is the body's `position` word). The stage's
`position` argument is the count p + 1 (see W1); one of the two names must change before wiring.

**FINDING W1 (a naming hazard, now stated correctly).** The landed `attention_stage::assemble` takes a
`position` that is **the token count, p + 1, not the 0-based loop position**: its own doc says "position ==
ring_rows is the normal case, the append lands before the stage", and its tests pin that reading
(`assemble(ring_rows=40, position=40)` -> `(window_rows, window_start) = (40, 0)`; `(300, 300)` ->
`(128, 172)`; `(4, 5)` -> `PositionPastRing`). Under that reading its `window_rows = min(position, 128)` and
`window_start = position - window_rows` **are** the released rule including the current token
(`window_indices` decode: `filled = window.min(position + 1)`, engine attention.rs 965; m20 `window_count`
counts the current row; the port's reference staging is 1 row at position 0, tp4_forward.rs 13462-13464).
So the formula is not an off-by-one; the hazard is that the **body's `position` word is the 0-based p**
(`FlashMlaSeqParams` gets the loop's `position`, tp4_forward.rs ~3493) while the stage's argument of the
same name is p + 1. A caller that passes the loop's p into `assemble` with `ring_rows = p + 1` passes every
check (`p <= p + 1`) and silently gets `window_rows = p`: no rows at position 0, one row short everywhere
else. Fix: either rename the stage's argument to say it is the count (and pin it with a test derived from
`window_indices`), or take p and derive both words inside the module (`window_rows = (p + 1).min(128)`,
`window_start = p + 1 - window_rows`), which is the version this contract's step 0 states and the one that
cannot disagree with the launch. The 54784974 layer_device.rs module doc's sentence that the window is
`[position - window_width ..= position - 1]` is only true when its `position` is the count; it should state
the convention next to the formula. The append-before-read order remains a **requirement** for decode, not
the equivalence that doc claims: the ring must hold the current row before the body reads.

Refusals this stage owns, all host-decidable (the reference `cache::Store` is the host mirror, not a
device read): `ring_rows > max_positions` refused; `position > ring_rows` refused (no clamp); the append for
this position must have landed, so the count the stage is given equals `ring_rows` (`CursorMismatch` covers
the store side); `window_rows + selected_count <= 640`;
`selected_count <= 512`; `format == 2`; ratio > 0 for a compressed layer.

**Step 1 - compressed store append (device), when this position completes a row.**
`compressed_append_sm121(latent, codes, scales, first_row, row_count, ratio)` (6 words;
compressed_append_launch.rs). `latent` is the bf16-narrowed, normed, roped latent row (`kv_rotated`),
`first_row = local * ratio` where `local` is the compressed row index `(position + 1) / ratio - 1`,
`row_count = 1`, `ratio` is the layer's. Refusals: `first_row + row_count <= EXPANDED_ROWS (512)`; the
store spans are zeroed once at open because odd `ratio = 2` rows must stay zero, the kernel writes only
the rows it is given. This is what replaces the reference's host `quantize_main_kv` staging; it must be
bit-identical to it (the launcher documents it as the native twin of the pinned codec).

**Step 2 - selected ids, device path.** `ref_select_sm121` (18 words: `q_index, keys, weights, out_ids,
out_scores, mask_in, mask_out, n_pos, heads, head_dim, reach, topk, offset, mask_mode, ...`) writes exactly
`keep = min(topk, n)` ids, every slot written: `position < reach ? position + offset : no_index`
(ref_select_sm121.cu, the rank loop fills all `keep` chosen slots). The id space and the refusals are then
settled **by caller contract, not by a check**: the caller passes `offset = window_rows` and
`reach <= plan.capacity_rows()`, so every non-NO_INDEX id is in `[window_rows, window_rows +
compressed_positions)` and `(id - window_rows) * ratio < 512` holds by construction. The two refusals - an
id below the window offset, an expanded row past the store - are therefore the **host reference path's**
(`compressed_cache_ref::selected_rows`); the production path never runs them, which is acceptable only
because the construction above is enforced where the select request is built, and because
`compressed_selected_fill_sm121` is unchecked by design (it assumes a valid plan). State both facts next
to the launch; a silently unvalidated underflow would wrap to a huge expanded row.

The mapping still must not run on the host: `compressed_selected_fill_sm121(ids, n_ids, window_rows, ratio,
out, out_count)` (6 words) maps each id to the **expanded store row** `(id - window_rows) * ratio`.

**The count: ruling (iv), keep-count padding.** The valid count is score-dependent (`out_count` counts
ids `!= NO_INDEX`), so the host cannot derive it without a readback - and `selected_count` is a **host
word** of the seq launch. Two options were rejected for the record: (i) read `out_count` back, 4 B + a
sync per (layer, position), a declared deviation that puts a host round trip back in the timed region;
(ii) make `selected_count` a device pointer in a new seq symbol, a new pin and an ABI change to a pinned
released kernel. **Lead ruled (iv):** `selected_count = keep = min(topk, n)`, a host scalar true by
construction because `ref_select` fills every slot; `no_index` slots stay in place; and the fill maps
**in place, no compaction**: `out[i] = ids[i] == no_index ? no_index : (ids[i] - window_rows) * ratio`
for every slot. The reference's compacted count (`staged.selected.len()`) is a **twin-only** number and
must not appear in the body doc. (iv) is bit-identical to the reference's compacted list because the union path excludes
any candidate `> position` with the exact sentinel `-3.402823466e38`: probability `exp(sentinel - s_max)`
underflows to `+0.0`, `bf16(0) = 0`, `local_sum += 0.0` is exact, the value pass skips the slot through
`spark_mla_seq_union_attends`, and `s_max` is finite while `window_rows >= 1` (append-then-read). The
sizing the launch must **assert, not assume**: `no_index > window_rows + keep - 1` for every position the
body runs (`no_index = 0xFFFFFFFF` is far above the 640 bound, but the guarantee is contractual), and
`window_rows <= 128`, `keep <= 512` so `window_rows + keep <= 640 = SPARK_M4_MLA_MAX_ATTENDED`, which is
the size of `s_scores` in the kernel. Under (iv) the body's `selected` is the fixed-length `keep` array
with sentinels in it - not the compacted list - and `selected_count = keep` is true by construction, not
by agreement. The host reference keeps its compacting path (`staged.selected.len()`); the equivalence is
the argument above plus a kernel-level test (sentinels appended, output bit-equal to the compacted list).
**Status:** ruling (iv) is the contract, but its one code change - the fill kernel's in-place map and its
re-pin - is the **first item of the next iteration**. This iteration ships `ref_select`, `moe_router`, the
arm-gated collectives and the producers, not the attention body (Lead's feature freeze until the MR is
open and the device row has run). Until it lands, milestone A stages ids host->device as below.

The code-side quotations live in `compressed_append_launch.rs` and are the ones this section cites:
`SELECTED_COUNT_OWNER` (present-vs-ruled: `keep` is the count **once** the in-place fill lands; the shipped
fill still compacts and skips `NO_INDEX`, so on today's tree the count is `out_count` and no caller may
pass `keep`), `SELECT_VALIDATION` (the reference-path ruling plus "the seq entry is sized by
`MAX_ATTENDED = 640`; the ring entry's 646 `static_assert` is a cross-check, not this launch's bound"),
`MAX_ATTENDED = 640`, `RING_MAX_ATTENDED = 646` (ring entry only), `CANDIDATE_SENTINEL = f32::MIN` (the
body's exact `-3.402823466e38`, whose `exp(sentinel - s_max)` underflows to `+0.0`), and
`CANDIDATE_GUARD = candidate <= position` - all pinned by `the_union_guards_quote_the_body_numbers`. When
the fill change lands, `SELECTED_COUNT_OWNER` loses its present-vs-ruled split and this paragraph shrinks
to a citation.

Interim (milestone A, if the device select is not wired yet): stage the ids host->device (<= 2,048 B) and
record that crossing as the only per-position host upload this slice has.

**Step 3 - the sequence entry.** `flash_mla_csa2_seq_sm121`, 19 words, grid `(heads_per_rank, 1, 1)`, block
`(256, 1, 1)` (the reference launches exactly this: `(self.heads as u32, 1, 1)` at tp4_forward.rs ~3493;
m20 runs the same kernel at world 1 with 64). Word by word:

| word | operand | provenance |
| --- | --- | --- |
| `q` | body query row | `workspace.q`, `query_path` (forward rope already applied) |
| `kv_fp4`, `kv_scales` | expanded store | step 1, device, zeroed at open |
| `window_codes`, `window_scales` | ring base + `codes_offset(window_start)` / `scales_offset(window_start)` | `kv_ring`, no gather, one pointer offset |
| `window_kv_format` | 2 (FORMAT_CSA_UNION) | constant, never a span |
| `window_rows` | `min(position + 1, 128)` | step 0, host |
| `selected` | expanded ids from step 2, sentinel-padded to `keep` (ruling (iv)) | never recomputed here |
| `selected_count` | `keep = min(topk, n)`, host-known, true by construction | host word, never a device read |
| `index_key`, `index_query` | HSI pool and index-head-0 query | **milestone A: zeroed spans**; no device producer exists |
| `sink` | per-rank head shard | staged once at open |
| `block_scores`, `block_best` | scratch | the entry writes them every launch |
| `out` | body row | `attn_body_bf16` |
| `position`, `layer_id`, `step_seq` | p, layer, 0 | host |
| `trace_ring` | `self.seam_trace_ring()` | the stage-timing ring, 0 by default |

**FINDING W2 (indexer geometry and the world-1 assumption).** The indexer stage uses
`SPARK_M4_HEADS = 64` for its worker stride (`block_stride = HEADS * WARPS`, kernel ~256) while the union
attention stage uses `blockIdx.x` only. A per-rank launch (`grid.x = 16`) is therefore valid for the
attention arithmetic but covers only 16/64 of the HSI blocks in the indexer scoring round (blocks are
strided by 512 while only 128 workers run), so `block_scores`/`block_best` are **not** a complete pool at
world 4. And `index_key` has no device writer in the pinned 26 at all: the host `write_index_keys` fills a
f32 shared pool, and m20 passes a zero-filled 4 MB pool. Consequence for the contract: milestone A passes
zeroed `index_key`/`index_query` spans and **nothing may consume the block arrays** (they are scratch); a
truthful indexer needs a device key producer and a sharded or corrected indexer stage, which is a later
slice and a new pin, not a body change.

Refusals the entry guards itself: `format == 2`; `window_rows + selected_count <= 640`; `selected_count <=
512`; a null ring or an inactive scope writes nothing (pinned by the seam-image tests).

**Step 4 - inverse RoPE (the body's last launch).** `rope_rotate_rows_bf16` with
`RopeRowsParams { src: body out, cos: plain table + position * pairs * 4, sin: plain table +
max_positions * pairs * 4 + position * pairs * 4, out: body row (aliasing allowed), rows: 16, head_width:
512, rope_head_dim: 64, sin_sign: -1, trace_ring, layer_id, step_seq }` (11 words;
layer_seams_launch.rs). Grid `(rows, 1, 1)`. It must be this kernel and not a host rotate: the reference
does decode-bf16 -> f32 rotate -> one bf16 narrow, which is exactly the seam kernel's arithmetic. Placing
the rotate at the start of `attention_output` instead is arithmetically identical, but then the body's out
row is pre-rotation and the stage record must say so; the reference's boundary is post-rotation, so this
contract pins it at the **end of attention_body**.

## attention_output: the ordered sequence

All five steps are device-to-device; the per-position H2D/D2H of the reference
(`copy_htod` of the activation, `copy_dtoh` of lo and of the wo_b partial) disappear.

1. `b12x_bf16a_mxfp8w_wo_a_sm121` (8 words: `attn_bf16, w_fp8, w_scales, lo_out, groups, trace_ring,
   layer_id, step_seq`), grid `(groups, O_LORA_RANK / TILE, 1) = (1, 4, 1)`, block 256. `groups =
   o_groups / world = 1`; `lo_out` = `groups * o_lora_rank` bf16 = 1,024. Refusals: the weight pair must
   match `(groups * o_lora_rank, group_inner) = (1024, 8192)` from the resident plan, both pointers
   non-null; `groups >= 1`.
2. `w8a8_prologue_bf16_e4m3` over `lo_out` (1,024 values, `rows = 1`, `cols = 1,024`): E4M3 codes + one
   E8M0 scale per 32 (32 scales) into `workspace.act_codes`/`act_scales`. Refusals: `cols` a multiple of
   32; the code/scale spans long enough for `cols * rows` and `ceil(cols / 32) * rows`.
3. `b12x_mxfp8_gemv_m1_sm121` (11 words: `a_fp8, a_scales, w_fp8, w_scales, out, rows, cols, inner,
   trace_ring, layer_id, step_seq`) with `rows = 1`, `cols = HIDDEN = 5,120`, `inner = groups *
   o_lora_rank = 1,024`; `out` = `workspace.attn_out_bf16`. Refusals: the wo_b weight geometry
   `(HIDDEN, inner)` from the plan; `inner` a multiple of the 32-value block; spans long enough.
4. `all_reduce` (comm's DeviceReduce): input `workspace.attn_out_bf16`, output `workspace.reduced_bf16`,
   **5,120 elements** (not bytes), stream 0, one `reduce_bf16` enqueue, rank-ascending f32 accumulation
   as the canonical layout defines; no sync and no readback in the stage. This is a per-layer collective
   and the only cross-rank step in the attention sublayer.
5. `mhc_attn_residual`: `hc_rows` (entry's copy), `mix_table` (post+comb), `reduced_bf16`, the f32
   `residual.stream` -> stream, 23-word ABI, trace ring = the stage-timing ring. Refusals the kernel
   owns: `native_bf16` word, `dim = 5,120`, `hc_mult = 4`, `rows = 24`; it never reads a null ring.

## The two red flags, stated as review findings

1. **No gather of the sequence window.** The window is a pointer offset into `kv_ring` at
   `window_start = position + 1 - window_rows`, with row strides 512 (codes) and 16 (scales). A body that
   copies, gathers, rotates or reorders window rows before the launch is a finding even if the numbers
   match: the traffic and the ring's reason for existing are the point. The only per-launch host work is
   the pointer arithmetic and the `window_rows` word.
2. **`selected` is never recomputed on the host.** The expanded-id mapping is `selected_fill` on the
   device; `attention_stage::assemble`/`compressed_cache_ref::selected_rows` are the validation reference
   (the same formula, two refusals, NO_INDEX skipped). A body that computes `(id - window_rows) * ratio`
   on the host, or reads ids back to map them, is a round trip in the steady state and a finding even when
   the values agree.

## m20 cross-check: keep, drop, repurpose

m20 (`attention_device.rs`) is the validated geometry this port comes from, but it is a world-1 backend
that stages every operand per position. Its 15 allocations against the device arm:

| m20 buffer | m20 bytes | port decision | why |
| --- | --- | --- | --- |
| `q` | 65,536 | **keep**, per-rank `workspace.q` 16,384 | same role, head-sharded |
| `kv_fp4` + `kv_scales` | 131,072 + 16,384 | **keep the sizes, new rows** | the expanded store is now produced by `compressed_append`; no upload ever again |
| `window_codes` + `window_scales` | 65,536 + 2,048 (128 rows) | **repurpose/grow** to `kv_ring` 270,336/layer | 128 rows became `max_positions` rows because the body reads contiguously at a base offset and a run of <= 512 never wraps; the producer is kq_path's quantiser, not a host upload |
| `selected` | 2,048 | **keep** | producer moves to `ref_select` + `selected_fill`; interim host-staged |
| `index_key` | 4,194,304 | **keep as one shared zeroed pool** (milestone A), per-layer only if a producer lands | m20 allocates it per backend (one layer) and zero-fills; no device producer exists; do not pay 4 MB x 40 layers for a pool nothing writes |
| `index_query` | 16,384 | **shrink/repurpose** | m20 allocates 64 heads' worth; the seq entry reads index head 0's 128 values = 256 B, and per-rank there are 8 index heads = 2,048 B if ref_select wants all |
| `block_scores` + `block_best` | 8,192 + 8,192 | **keep as scratch** | written every launch, consumed only by a device select (and incomplete at world 4 until W2 is fixed) |
| `sink` | 256 (world-1, 64 heads) | **keep, shard** | per rank 16 heads x 4 B = 64 B, staged once per layer at open, never per position |
| `a_codes`, `a_scales`, `a_bf16` | 32,768, 1,024, 65,536 | **repurpose the existing `act_codes`/`act_scales`**; no separate `a_bf16` | native wo_a consumes the bf16 body row directly; wo_b's prologue needs 1,024 + 32 |
| `w`, `w_scales` | 41,943,040 + 40,960 | **drop** | the resident weight arena holds the coded pairs; the reference binds them by plan geometry |
| `out` (generic GEMM out) | 65,536 | **split** | body row 16,384 + `attn_out_bf16` 10,240 |
| `mla_out` | 65,536 | **keep, shard** = `attn_body_bf16` 16,384 | per-rank heads |

Per-position traffic the port removes at a full window/store (m20): H2D `q` 65,536 + window 65,536 +
2,048 + store 131,072 + 16,384 + selected 2,048 + sink 256 = 282,880 B, plus a D2H readback of `out`
65,536 B, i.e. up to ~348 KB crossing per layer per position. After this contract: nothing per position
on the device-select path; on the interim host-select path, the selected ids (<= 2,048 B) only. The sink
and (milestone A) the zeroed index pool are per-run/per-layer, not per position.

## What must exist before the body can be truthful

1. The new rows: `attn_body_bf16` (16,384 B/rank/layer), store codes+scales (147,456 B/layer), `selected`
   (2,048 B/layer) and a count word; `index_key` (one shared 4,194,304 B pool, zeroed once),
   `index_query` (256 B, or 2,048 B if ref_select consumes all per-rank index heads), `sink` (64 B/layer,
   staged from the arena at open), `block_scores`/`block_best` (8,192 B each, shared scratch).
2. The window convention settled (W1): one of the two `position` names changes, and the winning side pins
   `window_rows = (p + 1).min(128)`, `window_start = p + 1 - window_rows` against the released
   `window_indices` rather than against its own arithmetic. The module doc and the layer_device window
   paragraph must state the convention next to the formula, and the append-before-read sentence becomes a
   requirement (the ring holds the current row before the body reads it).
3. The store producer wired: `compressed_append_sm121` at the ratio boundary, per layer, with the store
   zeroed at open.
4. The select wiring: device `ref_select` + `selected_fill` (the port's target) or host-staged ids
   (milestone A, one documented per-position upload). Under ruling (iv) the fill maps in place and preserves
   `no_index`, `selected_count = keep` - that is the port's own fill kernel plus one baseline pin line, not
   the seq ABI, and it is the first item after the feature freeze.
5. The indexer decision (W2): milestone A zero spans with no consumer, or a device key producer plus a
   sharded indexer stage (new kernel, new pin).

## Review bar for the sha (a)-(g), mapped

| item | where this contract pins it |
| --- | --- |
(a) window pointer, not gather | attention_body step 3 word table, red flag 1 |
(b) `min(position+1,128)` and the two refusals, refuse not clamp | step 0 + W1 |
(c) `selected`/`selected_count` passed through, mapped on device | step 2, red flag 2 |
(d) inverse rope `sin_sign = -1` over the 64-wide tail before wo_a, out may alias | attention_body step 4 |
(e) wo_a grid `(1, 4, 1)` 8 words, prologue over `groups * o_lora_rank = 1,024` values | attention_output 1-2 |
(f) wo_b GEMV `rows = 1, cols = 5,120, inner = 1,024` | attention_output 3 |
(g) all_reduce input is the wo_b output, output consumed by `mhc_attn_residual` | attention_output 4-5 |

Additional evidence to demand with the sha: the launch counts per position (4 in the body with a
compressed row, 3 without; 5 in the output), no `copy_htod`/`copy_dtoh` inside either stage, the trace
words on every kernel that takes one, and the two rows named exactly as above.
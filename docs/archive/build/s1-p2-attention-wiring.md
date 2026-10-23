# S1 P2 attention wiring: layers 0/1 now, layers 2..39 blocked by design

**Owner:** gemv-f32-impl (task-2)
**Coded against:** branch `integration/s1-layer-device` at `d330e5b8` (working tree shared with
device-lead and qa-owner)
**Review bar:** `docs/archive/build/attention-sequence-contract-20261022.md` (the rulings below are that
document's, made implementable; where this document disagrees with it the disagreement is named in
§0, and this document wins because it is derived from the code as it is on this tip).
**Status:** Part A (layers 0 and 1, `compress_ratio == 0`) is wireable on this tree with no new
kernel. Part B (layers 2..39) is blocked by design, not by a missing launch: the 19
`AttentionHostSeam` tensors the seam needs are structurally outside the device bind. Nothing in
this document was run on a device: there is no GPU in this environment (see §5).

This is the wiring spec for the five attention stages in
`impl LayerStages for DeviceLayer` (`crates/sparkflash-cli/src/tp4_forward.rs` 11795+), currently
five `Err(Self::refusal(...))` stubs at 12238-12256:

| stage | line | this document |
| --- | --- | --- |
| `reference_seam` | 12238 | §A.1, no-op on ratio 0, named refusal above |
| `ref_select` | 12242 | §A.2, no-op on ratio 0 |
| `stage_attention` | 12246 | §A.3, validation only, never a launch |
| `attention_body` | 12250 | §A.4, one pinned launch + one inverse-rope launch |
| `attention_output` | 12254 | §A.5, three launches |

The operands are stated once in `crates/sparkflash-kernels/src/attention_operands.rs` (new on this
branch, 20 tests). Stage bodies call that module instead of re-deriving numbers; the constants and
the refusal wording there are the ones this document quotes.

---

## 0. Source of truth, and four corrections to the earlier documents

The numbers below were read off the code on this tip. The sequence contract's per-rank table has
one arithmetic slip and one append-word slip; both are corrected here, and the wiring must use the
code numbers.

| quantity | value at world 4 | source |
| --- | --- | --- |
| `groups = o_groups / world` | **2** (not 1) | `tp4_forward.rs:4072-4073`, `resident_weights.rs:242-255` |
| `heads_per_rank` | 16 (= 64 / 4) | `open`: `q_cols = n_heads / world * head_dim` |
| `heads_per_group = heads_per_rank / groups` | 8 | `tp4_forward.rs:4093` |
| `group_inner = heads_per_group * head_width` | **4,096** (not 8,192) | `tp4_forward.rs:4094` |
| wo_a shard `(cols, inner)` | **(2,048, 4,096)** | `resident_weights.rs:244-247`; host twin asserts it |
| wo_b shard `(cols, inner)` | **(5,120, 2,048)** | `resident_weights.rs:249-255` |
| `o_lora_rank` | 1,024 | config |
| wo_a output `lo` (`groups * o_lora_rank`) | 2,048 bf16 = 4,096 B | §A.5 |
| wo_b activation codes / scales | 2,048 B / 64 B | `OutputGeometry::lo_code_bytes` |
| body row `heads_per_rank * head_width` bf16 | 16,384 B | `OutputGeometry::activation_bytes` |
| wo_b output `attn_out_bf16` | 5,120 bf16 = 10,240 B | `workspace.attn_out_bf16` |

Corrections:

1. **Groups.** The contract's "`groups = o_groups / world` (1 at world 4)" divides by world twice
   (8/4 = 2). The host twin and the plan geometry are authoritative: `groups = 2`, so wo_a is
   `(2,048, 4,096)` and wo_b is `(5,120, 2,048)`. `attention_operands::OutputGeometry` reproduces
   this and its test pins the released numbers.
2. **`compressed_append_sm121`'s `first_row` is the compressed row index, not the expanded store
   row.** The kernel computes `store_row = (first_row + row) * ratio` itself
   (`kernels/compressed_append_sm121.cu:129`). The contract's "`first_row = local * ratio`" would
   double-apply the ratio. The correct call at a completing position is
   `first_row = position / ratio`, `row_count = 1`
   (`attention_operands::append_row(ratio, position) -> Option<usize>` returns exactly this and
   `None` while the group is filling). Ratio-0 layers never reach it.
3. **The id-space offset is a label, not the reference's number.** The host reference's
   `state.window_rows` is 1 at `start_pos == 0` and `config.window_size` after that
   (`engine/forward/attention.rs:2091-2096`); the device arm's offset is
   `compressed_offset = window_rows = min(position + 1, 128)` (contract step 0,
   `attention_operands::WindowOperands`). Both are internal labels for the window half of the
   shared id space; the only consumer is the fill's `(id - window_rows) * ratio`, so as long as
   `ref_select`'s `offset` word and the fill's/`assemble`'s `window_rows` are the *same* value the
   expanded rows are identical. The device select is relaunched (the host ids are not reused), so
   nothing needs to match the host's label. Do not "correct" the offset to 128.
4. **Position convention, and a correction to the contract's FINDING W1.** The loop's `position`
   argument of every stage is the 0-based token index `p` (proof: `kv_path` asserts
   `appended_rows[layer] == position` before its append and writes slot `position % rows`,
   `tp4_forward.rs:12117-12126`), and the body's `position` word is also `p` (`MlaAttentionBody`,
   3493). The landed `attention_stage::window_span(position, window)` takes the **same** 0-based
   `p`: its doc says "for the 0-based token index p ... includes the current row", and its tests pin
   it (`assemble(..., position = 0)` -> 1 row; `position = 40` -> 41 rows; `position = 300` ->
   `(128, 173)`). So FINDING W1 in the sequence contract is a misreading of the landed module (it
   read the old doc example and concluded the argument is `p + 1`); there is **no** convention
   mismatch to fix, and no rename is needed. `attention_operands::WindowOperands::new` delegates to
   `attention_stage::window_span`, so the device arm and the host staging share one formula. Use
   `WindowOperands::new(p, ring_rows)`; do not add a `p + 1` anywhere.

Also corrected: the contract's `kv_ring` row size for a 512-row ring is 270,336 B per layer
(`512 * (512 + 16)`), and the store bound (see §1.4) is not a window-size effect but the expanded
store's own 512 rows.

---

## 1. The operands, once

All helpers named here are in `sparkflash_kernels::attention_operands` unless another module is
named. They are pure: no device pointer is dereferenced by any of them.

### 1.1 Window (every stage, every layer)

```
p              = the loop's 0-based position
ring_rows      = DeviceLayer::capacity_rows  (== max_positions at open)
window_rows    = min(p + 1, 128)                  = WindowOperands::window_rows
window_start   = p + 1 - window_rows              = WindowOperands::window_start
slot(p)        = p % ring_rows                    = WindowOperands::slot
compressed_offset = window_rows                   = WindowOperands::compressed_offset
attended       = window_rows + selected_count <= 640 = MAX_ATTENDED
```

`WindowOperands::new` refuses `ring_rows == 0` (`ZeroRing`) and `p >= ring_rows`
(`PositionPastRing`, text "position {p} needs its row appended first: the ring holds {n} rows").
`WindowOperands::checked(p, ring_rows, max_positions)` adds `RingPastCapacity` when
`ring_rows > max_positions`. `attended(selected)` refuses `selected > 512`
(`SelectedPastTopk`) and the sum over 640.

### 1.2 Window ring addresses (the body's two window words)

```
ring base    = self.kv_ring_base + self.kv_ring.layer_offset(layer)
window_codes = ring base + self.kv_ring.codes_offset(slot(window_start))
window_scales= ring base + self.kv_ring.scales_offset(slot(window_start))
```

`attention_operands::window_ring_addresses(&self.kv_ring, self.kv_ring_base, n_layers, layer,
&window)` returns exactly those two `u64`s and refuses a layer outside the run
(`LayerOutside`) or a window the ring cannot hold contiguously (`WindowPastRing`,
`WindowWrapsRing`). The read is contiguous: `window_rows` rows of 512 code bytes then 512 scale
bytes at `codes_offset`/`scales_offset`, no gather and no per-position upload. It is safe because
`ring_rows == max_positions` and `window_start + window_rows = p + 1 <= max_positions`; the
`kv_path` append for `p` has already written slot `p % ring_rows` (stage order, §A.8) and does not
overwrite a row the window still reads until `p + ring_rows`, which a `ring_rows >= 128` run never
reaches inside a window.

### 1.3 Rope pair addresses (the body's inverse rope)

The open uploads one table per rope class, all positions' cos pairs then all positions' sin pairs
(`open`, the `rope_tables_range` loop):

```
pairs          = rope_head_dim / 2 = 32
position_bytes = pairs * 4 = 128
table          = buffers.workspace.rope_tables_plain   (class 0, compress_ratio == 0)
                 buffers.workspace.rope_tables_compressed (class 1)
cos            = table + p * 128
sin            = table + max_positions * 128 + p * 128
```

`attention_operands::rope_table_addresses(table, max_positions, rope_head_dim, head_width)` plus
`.at(p, max_positions)` returns `(cos, sin)`; refusals are `BadRopeTail` (zero/odd/too-wide tail)
and `PositionPastTable`. Class selection is `self.compress_ratio[layer] > 0`, the same expression
`kv_path` already uses (`tp4_forward.rs:12161-12166`); layers 0 and 1 are class 0.

### 1.4 Compressed store and the id space (ratio > 0 only; §B)

```
rows_written(p)   = (p + 1) / ratio            = StoreOperands::rows_written
store_zero        = rows_written == 0          = StoreOperands::store_zero
compressed row j  -> expanded store row j * ratio, codes at row*256, scales at row*32
selected id       = window_rows + j            = SelectedIdSpace::id
expanded row      = (id - window_rows) * ratio  = SelectedIdSpace::expanded_row
```

`StoreOperands::new(ratio, p)` refuses `RatioZero` and `RowsPastStore`. The store bound is the
expanded store's 512 rows: the last published row's expanded row is
`(rows_written - 1) * ratio < 512`, which for the released ratios allows ratio 1 up to `p = 511`
and ratio 2 up to `p = 512` (ratio 2: `(p+1)/2 <= 256`). This is the S1 decode-length bound at
ratio > 0 and it is a property of `compressed_cache_ref::EXPANDED_ROWS = 512`, not of the ring;
it does not apply to layers 0/1 (no store) and it is already the host twin's bound
(`tp4_prefill::stage_attention` sizes the expanded store the same way).

### 1.5 Span sizes for the body and output stages

From `attention_operands` and the launchers:

| row | bytes | notes |
| --- | --- | --- |
| `attn_body_bf16` | 16,384 | body out, inverse rope in place, wo_a input |
| `selected` ids | 2,048 | 512 × u32; count is a host word, never a read-back |
| store codes / scales | 131,072 / 16,384 | expanded store, zeroed at open |
| `index_key` pool | 4,194,304 | `index_key_pool_bytes()`; run-wide zero pool in milestone A |
| `index_query` | 8,192 (ABI) / 256 read | the seq entry only reads index head 0: `index_query` word is used as `const __nv_bfloat16 *query_row = index_query;` (`flash_mla_csa2_seq_sm121.cu:249`) |
| `block_scores` / `block_best` | 8,192 / 8,192 | scratch, also written by every seq launch |
| sink shard | 64 per layer (16 f32) | run pool `n_layers * heads_per_rank * 4 = 2,560` |
| `attn_lo` (wo_a out) | 4,096 | 2,048 bf16 |
| wo_b act codes / scales | 2,048 / 64 | `workspace.act_codes`/`act_scales` are big enough (8,192 / 160 B) and stage order makes reuse safe |
| candidate mask | 512 | `candidate_mask_bytes()`, run-wide, ratio>0 only |

---

## 2. Part A — layers 0 and 1 (`compress_ratio == 0`): wire these now

Released ratios are `[0, 0, 2×18, 1×20, 0×3 pad]` (`cache_plan.rs` RATIOS); layer 0 and layer 1
are window-only. `attention_seam_step_with_backend` returns early for them (`ratio == 0` skips
compressor, index keys, index weights, `compressed_indices`, `write_compressed_cache`;
`attention.rs:2084-2144`), returning `ratio = 0`, empty `compressed_ids`/`compressed_rows`, and
`window_width = window_indices(...) != NO_INDEX count = min(p + 1, 128)`. Nothing in the seam for
these layers needs a weight outside the bind, so the whole attention half can be device-only for
them. **This is the S1-a gate path: layer 0 device-only, one explicit residual copy in and one
out (`begin_position`/run tail), `SPARKFLASH_LAYER_DEVICE=1`.**

The stage bodies take the loop's `p`. What each does:

### A.1 `reference_seam`

**Host twin replaced:** `engine::forward::attention::attention_seam_step_with_backend` (the
compressor/index/select/store sequence and the `window_width` count).

Pure decision, no launch, no device read, no copy:

```rust
fn reference_seam(&mut self, layer: usize, position: usize) -> Result<(), String> {
    // ratio must be 0 on this branch of the port; a compressed layer is the Part B refusal.
    let ratio = *self.compress_ratio.get(layer).ok_or_else(...)?;
    if ratio != 0 { return Err(Self::refusal("reference_seam")); }   // the Part B text, §B
    let window = WindowOperands::checked(position, self.capacity_rows, self.max_positions)
        .map_err(|e| format!("layer device stage reference_seam: {e}"))?;
    self.attention_state[layer] = AttentionStepState {
        window,
        selected_count: 0,
        store_rows: 0,
        store_zero: true,
    };
    probe::count("reference_seam");
    Ok(())
}
```

`window.window_rows` **is** the reference's `window_width` (the count of non-`NO_INDEX` window
ids, engine `attention.rs:2141-2143`): `min(p + 1, 128)`. Assert the identity in a unit test
rather than at every position. `ids` and `rows` stay empty by construction; there is no store row
to append and no compressor to run. `self.compress_ratio[layer]` is already resolved at open.

### A.2 `ref_select`

**Host twin replaced:** `compressed_indices_with_backend` (ratio 0 returns early on the host too).

```rust
fn ref_select(&mut self, layer: usize, position: usize) -> Result<(), String> {
    let ratio = self.compress_ratio.get(layer).copied().ok_or(...)?;
    if ratio != 0 { return Err(Self::refusal("ref_select")); }      // never reached: A.1 refused
    // ratio 0 has no compressed positions: selected_count stays 0, no launch, no pointer.
    probe::count("ref_select");
    Ok(())
}
```

Do **not** launch `ref_select_sm121` with `n_pos = 0`: the kernel is O(n²) over `n_pos` and writes
`min(topk, n_pos)` ids; zero ids is a zero-row launch that still needs valid `q_index`/`keys`
pointers and buys nothing. The stage is a documented no-op, not a skipped check.

### A.3 `stage_attention`

**Host twin replaced:** `tp4_prefill::stage_attention` (the 147 KB host store + window/staging
struct).

Validation only: no launch, no copy, no host bytes. The reference `cache::Store`/`cache::Plan`
(`attention_stage::assemble`'s inputs) are not materialised on device, so the cursor is derived
from the position (§1.4) instead of read back, and the validation is the device-guarded subset:

| check | refusal |
| --- | --- |
| `ring_rows = self.capacity_rows <= self.max_positions` | `RingPastCapacity` |
| `position < ring_rows` | `PositionPastRing` |
| `store_rows == StoreOperands::rows_written` (ratio > 0; on ratio 0 both 0) | "the {ratio}-ratio store reports {got} rows at position {p}, the position publishes {want}" |
| `store_zero == (store_rows == 0)` | consistency of the derived pair |
| `window_rows + selected_count <= 640` | `AttendedPastBound` |
| `selected_count <= 512` | `SelectedPastTopk` |
| `window_rows <= 128` and `window_rows >= 1` | geometry |
| `format == 2` (constant) | never a span |

On ratio 0 the stage therefore accepts `store_rows = 0`, `selected_count = 0`, `window_rows =
min(p + 1, 128)` and records those in `AttentionStepState`. Do not call
`attention_stage::assemble`: it is semantically compatible (it takes the same 0-based `p`, §0.4)
but its inputs are the host mirrors (`cache::Plan`, `cache::Store`, `cursor`, the host id vector)
that the device arm does not have. The formula is shared through
`attention_operands::WindowOperands`, which delegates to `attention_stage::window_span`; the store
side is derived, not read back, and the store spans are allocated zeroed once at open. The
`selected_count = 0` body then reads none of the store.

The device analogue of `CursorMismatch` is the `store_rows[layer]` host counter (incremented by the
Part B append when it lands; today it stays 0 for every layer that can run). Keep it a `Vec<usize>`
next to `appended_rows` so the check has the same "append landed exactly once" meaning.

### A.4 `attention_body`

**Host twins replaced:** `MlaAttentionBody::run` (3375) — its four H2Ds, the launch, the D2H, the
host inverse rope at 3562 and the host finite scan — and the whole `AttentionStep` host struct.

Two launches. The inverse-rope kernel is already resident (`self.kernels.rope_rows`, the seam
image's eighth symbol); the sequence entry is a **new** open-time load (§A.6). Nothing is read
back, and the host `AttentionStep` struct disappears.

**Launch 1 — the sequence entry, 19 words, grid `(heads_per_rank, 1, 1) = (16, 1, 1)`, block
`(256, 1, 1)`.** The kernel is **not** loaded by `DeviceLayer::open` today: the host arm reads
`kernel_dir/flash_mla_csa2_seq_sm121.cubin` in `MlaAttentionBody::open` (3275-3283) and resolves it
with `flash_mla_csa2_seq_launch::load`; the device arm needs the same read at open and a new
`(Module, Kernel)` field (freed in `Drop`). Build the words with
`sparkflash_kernels::flash_mla_csa2_seq_launch::FlashMlaSeqParams::new` and launch with
`cuda.launch(kernel, (heads_per_rank, 1, 1), (THREADS, 1, 1), params.as_slice())`.
Do **not** call `flash_mla_csa2_seq_launch::launch`: it hard-codes `GRID = (64, 1, 1)`; and not
`launch_batched` either (that is the six-consecutive-positions entry, also fixed at `GRID`). The
one-line `cuda.launch` is what the host arm does (`tp4_forward.rs:3506-3519`); re-assert the two
things `launch_batched` would have caught - word 5 is `FORMAT_CSA_UNION` and `heads_per_rank <= 64`
- in the new helper.

The words are **not** in declaration order in the struct builder: `FlashMlaSeqParams::new` places
`selected_count` after `position` (word 15), even though the kernel declares it right after
`position` too - the two agree, and that is the order pinned below. Do not "fix" it to a grouping
that reads nicer.

| # | word | operand at layer 0 (`p`) |
| --- | --- | --- |
| 0 | `q` | `buffers.workspace.q.device` (query_path's wq_b + forward rope, 16,384 B) |
| 1 | `kv_fp4` | store codes span (§1.5), valid pointer; unread while `selected_count == 0` |
| 2 | `kv_scales` | store scales span |
| 3 | `window_codes` | §1.2, ring base + `codes_offset(slot(window_start))` |
| 4 | `window_scales` | §1.2, ring base + `scales_offset(slot(window_start))` |
| 5 | `window_kv_format` | `FORMAT_CSA_UNION = 2` |
| 6 | `window_rows` | `window.window_rows` (`min(p+1,128)`) |
| 7 | `selected` | `self.selected` span (2,048 B), valid pointer, unread while count 0 |
| 8 | `index_key` | run-wide zero pool, 4,194,304 B (§A.6) |
| 9 | `index_query` | 8,192 B zeroed span (entry reads index head 0 only) |
| 10 | `sink` | run sink pool + `layer * 64` |
| 11 | `block_scores` | per-layer 8,192 B scratch (written every launch) |
| 12 | `block_best` | per-layer 8,192 B scratch |
| 13 | `out` | `buffers.workspace.attn_body_bf16.device` (16,384 B) |
| 14 | `position` | `u64::try_from(p)` |
| 15 | `selected_count` | `0` (Part B: `keep = min(topk, n_pos)`) |
| 16 | `trace_ring` | `self.seam_trace_ring()` (0 by default) |
| 17 | `layer_id` | `u32(layer)` |
| 18 | `step_seq` | `u64(p)` (the landed stages' convention) |

Span checks before the launch, each naming its row: `q.len >= heads*head_width*2`,
`attn_body_bf16.len >= heads*head_width*2`, `window_rows <= 128`, `window_rows + selected_count <=
640`, `selected_count <= 512`, `position <= u32::MAX`, the ring addresses non-null and inside the
ring allocation. The entry itself guards format 2, `window_rows <= 128`, `selected_count <= 512`,
the 640 union and the candidate guard (`kernels/flash_mla_csa2_seq_sm121.cu`, `SELECT_VALIDATION`).

**Launch 2 — the inverse rope, 11 words, grid `(16, 1, 1)`, block `(256, 1, 1)`.** The kernel is
already resident: `self.kernels.rope_rows` is `rope_rotate_rows_bf16`, loaded at open from the seam
image as its eighth symbol (`LayerSeamKernels`, `tp4_forward.rs:10320-10329`), and its module is
freed by `Drop`. Use
`sparkflash_kernels::layer_seams_launch::{RopeRowsParams, launch_rope_rows}`
(`ROPE_ROWS_KERNEL = "rope_rotate_rows_bf16"`, `ROPE_ROWS_ARG_WORDS = 11`):

```rust
RopeRowsParams {
    src:  Deviceptr(attn_body_bf16),      // the launch-1 out row
    cos:  Deviceptr(cos),                 // §1.3, class 0 table for layer 0/1
    sin:  Deviceptr(sin),
    out:  Deviceptr(attn_body_bf16),      // alias is legal: each pair reads before it writes
    rows: heads_per_rank,                 // 16
    head_width,                           // 512
    rope_head_dim,                        // 64
    sin_sign: -1,                         // rope_rotate_inverse
    trace_ring, layer_id, step_seq,
}
```

`launch_rope_rows` re-validates the geometry (zero rows/width, odd tail, tail wider than the row,
sign not ±1). The arithmetic is the host's bit-for-bit: `layer_seams_ref::rope_rows_ref` is tested
against `engine::linalg::rope_rotate_inverse` (`layer_seams_ref.rs:916`), and the reference's host
sequence is decode-bf16 → f32 rotate → one bf16 narrow, which is this kernel. **Placement:** the
contract pins the rotate at the **end of `attention_body`** because the reference stage boundary is
post-rotation (the body returns `activation_bf16` = rotated). Putting it first in
`attention_output` is arithmetically identical (no host read between) and was the earlier plan; this
document keeps the contract's placement so the stage record names the reference boundary. The
seam-image doc comment at `tp4_forward.rs:10320-10324` still says "the `attention_output` stage owns
the launch"; whoever wires this should repoint that sentence to `attention_body` (one-line comment
change, same file as the wiring).

Row-count checks: `attn_body_bf16.len >= heads_per_rank * head_width * 2`,
`heads_per_rank == self.heads_per_rank`, `head_width == self.head_width`,
`rope_head_dim == self.rope_head_dim`, `p < self.max_positions`.

### A.5 `attention_output`

**Host twin replaced:** `run_layer_attention_output` (4046): its `copy_htod` of the activation, the
`copy_dtoh` of `lo`, the host `quantise_native_activation`, the H2D of the codes/scales, the
`copy_dtoh` of the wo_b partial, and the host all-reduce. On device the three launches are
device-to-device; **the collective is not part of this stage** (stage 12 `all_reduce` is landed and
owns `workspace.attn_out_bf16 → workspace.reduced_bf16`, stage 13 owns the residual).

All three kernels must be resident at open; two already are and one is new:

* **wo_a** (`b12x_bf16a_mxfp8w_wo_a_sm121`): **new**. The host arm loads it through its per-run
  `KernelCache::wo_a` (a digest-keyed cache, `tp4_forward.rs:1806-1824`), which `DeviceLayer` does
  not have; at open, `load_pinned_addition(kernel_dir, bf16a_mxfp8w_wo_a_launch::KERNEL, "layer
  device attention output")` + `bf16a_mxfp8w_wo_a_launch::load`, then free the module in `Drop`.
* **prologue_bf16** (`w8a8_prologue_bf16_e4m3`): already `self.kernels.prologue_bf16` (loaded at
  open, currently `#[allow(dead_code)]` - this stage is its first consumer).
* **wo_b** (`b12x_mxfp8_gemv_m1_sm121`): already `self.gemv` (loaded at open from the pinned gemv
  cubin). Its params struct is `mxfp8_dense_launch::Mxfp8DenseParams` and its launcher is
  `mxfp8_gemv_launch::launch` - exactly the pairing the host twin uses
  (`tp4_forward.rs:4220-4233`: `Mxfp8DenseParams::new` then `mxfp8_gemv_launch::launch`). Do not
  load `b12x_mxfp8_dense_sm121`; that symbol exists but is not the decode path's kernel.

1. **`b12x_bf16a_mxfp8w_wo_a_sm121`** — `Bf16aMxfp8wWoAParams::new(attn_bf16, w_fp8, w_scales,
   lo_out, groups, trace_ring, layer_id, step_seq)` (8 words), `wo_a_launch::launch` → grid
   `(groups, GRID_Y, 1) = (2, 4, 1)`, block 256. `attn_bf16` = the rotated
   `attn_body_bf16`; `lo_out` = new span `workspace.attn_lo` (2,048 bf16 = 4,096 B). Refusals: both
   weight pointers non-null, `groups >= 1` (the launcher's own `grid` refuses zero), plan wo_a
   `(groups * o_lora_rank, group_inner) == (2,048, 4,096)` (`OutputGeometry::check_wo_a`),
   `attn_bf16.len >= groups * group_inner * 2`, `lo_out.len >= groups * o_lora_rank * 2`.
2. **`w8a8_prologue_bf16_e4m3`** — `PrologueBf16Params { normed: attn_lo, codes:
   workspace.act_codes, scales: workspace.act_scales, cols: groups * o_lora_rank = 2,048, rows: 1,
   trace_ring, layer_id, step_seq }` (8 words), `seams::launch_prologue_bf16` (kernel
   `self.kernels.prologue_bf16`) → `prologue_grid(2048, 1) = (1, 1, 1)`, block 256. Refusals:
   `cols` a multiple of 32 and non-zero (the launcher checks it), `act_codes.len >= cols * rows`,
   `act_scales.len >= ceil(cols/32) * rows`.
3. **`b12x_mxfp8_gemv_m1_sm121`** — `mxfp8_dense_launch::Mxfp8DenseParams::new(act_codes,
   act_scales, w_fp8, w_scales, out, 1, HIDDEN, groups * o_lora_rank, trace_ring, layer_id,
   step_seq)` (11 words), `mxfp8_gemv_launch::launch(self.cuda, self.gemv.1, &params)` →
   `grid(1, 5,120)`, block 32 (the launcher derives both; the host arm uses the identical
   construction). `out` = `workspace.attn_out_bf16` (10,240 B). Refusals: plan wo_b `(HIDDEN,
   groups * o_lora_rank) == (5,120, 2,048)` (`OutputGeometry::check_wo_b`), `inner` a multiple of
   32 and `<= mxfp8_gemv_launch::MAX_INNER`, both weight pointers non-null.

Weight pairs: use `self.attention.pointers` (`RankBind`) with the real plan names
`layers.{layer}.attn.wo_a.weight` / `layers.{layer}.attn.wo_b.weight`. **`WEIGHT_NAMES` has no
wo_a/wo_b keys** (`layer_device.rs:176-194`), so `self.weights.pair(layer, "wo_a")` refuses; that is
a deliberate exclusion today and the device stage should read the bind directly, exactly as the host
twin does (`bind.attention.pointers.get`, `tp4_forward.rs:4097-4112`).

### A.6 Spans to add, and the open/`Drop` changes

New rows (workspace is carved per layer, `layer_device.rs` `LayerWorkspace`/`carve`/
`assert_contract_widths`); sizes per §1.5:

| span | dtype | bytes | init | producer → consumer |
| --- | --- | --- | --- | --- |
| `workspace.attn_body_bf16` | bf16 | 16,384 | any | body seq out → inverse rope (in place) → wo_a |
| `workspace.attn_lo` | bf16 | 4,096 | any | wo_a → prologue_bf16 |
| `workspace.selected` | u32 | 2,048 | any (count is a host word) | ref_select fill → body (Part B) |
| `workspace.store_codes` | u8 | 131,072 | **zeroed at open** | compressed_append → body (Part B) |
| `workspace.store_scales` | u8 | 16,384 | **zeroed at open** | compressed_append → body (Part B) |
| `workspace.index_query` | bf16 | 8,192 | **zeroed at open** | none in milestone A (index head 0 read only) |
| `workspace.block_scores` | f32 | 8,192 | any | seq entry writes |
| `workspace.block_best` | u32 | 8,192 | any | seq entry writes |
| `workspace.candidate_mask` | u32 | 512 | **zeroed at open** | ref_select Produce (layer 20) → Consume (21..39) |

Run-wide (beside `kv_cache`/`kv_ring`, not per layer), allocated once at open:

| allocation | bytes | init | why run-wide |
| --- | --- | --- | --- |
| `index_key` zero pool | 4,194,304 | **zeroed at open** | no device writer in milestone A; read-only for every layer, so one pool serves all 40 (per-owner allocation becomes necessary when the index-key producer lands, W2) |
| sink pool | `n_layers * heads_per_rank * 4` = 2,560 | uploaded once at open from `layers.{layer}.attn.attn_sink` (arena, `heads_per_rank * 4` per layer) | body reads `base + layer*64`; removes the host arm's 64-B H2D per position |
| `candidate_mask` (ratio>0 only) | 512 | zero | produced at layer 20, consumed by 21..39 in the same position (lives across layer stages, so it cannot be per layer) |

Open-time work: carve the nine workspace rows; grow `LayerWorkspace::bytes`/`carve`/
`assert_contract_widths` (the checks join the `25`-row table: `attn_body_bf16` bf16 16,384,
`attn_lo` bf16 4,096, `selected` u32 2,048/4 words, `store_codes` u8 131,072, `store_scales` u8
16,384, `index_query` bf16 8,192, `block_scores` f32 8,192, `block_best` u32 8,192,
`candidate_mask` u32 512); allocate the zero pool and upload the sink pool (two `cuda.alloc` +
`copy_htod` in `open`, freed in `Drop` beside `kv_cache`/`kv_ring`); load the **two new kernel
images** at open - the sequence entry from `flash_mla_csa2_seq_sm121.cubin`
(`flash_mla_csa2_seq_launch::load`) and wo_a (`bf16a_mxfp8w_wo_a_launch::load` after
`load_pinned_addition`) - each a new `(Module, Kernel)` field freed in `Drop`. The other three
kernels this half needs (`self.gemv`, `self.kernels.prologue_bf16`, `self.kernels.rope_rows`) are
already resident. The per-layer slab grows by ~190,464 B (≈7.4 MiB over 40 layers); the run
allocations by ≈4.2 MiB.

Note `attn_body_bf16` (16,384 B) cannot alias `attn_out_bf16` (10,240 B) — the contract's row table
already says so; the body row is bigger than the wo_b output row.

### A.7 Refusal text and contract entries to replace

Five stage bodies above stop returning `Err(Self::refusal(...))` for ratio 0. `Self::refusal`
(`tp4_forward.rs:11700`) builds its text from `STAGE_BLOCKERS`; update those entries so a Part B
refusal still reads as the real reason, and update the five `STAGE_CONTRACTS` rows, which are
placeholders that name kernels that do not exist. Suggested replacements (Lead-owned files,
`layer_device.rs`):

`STAGE_BLOCKERS`:

| stage | today | replace with |
| --- | --- | --- |
| `reference_seam` | "the seam's window selection is device work already; the stage is blocked on the kv_path row it consumes" | "ratio 0 (layers 0,1): none - the window width is min(position+1, 128), decided from the position with no device read; ratio > 0 is blocked on the compressor producer (attention.compressor.wkv/wgate are AttentionHostSeam tensors outside the device bind): the pooled latent, its norm/rope and the compressed store have no device path" |
| `ref_select` | "the kernel, its CPU twin and its cubin pin are landed; the body is refused because its operands (the window keys and the reference indexer query) have no device spans until the kv_path re-point lands the window ring" | "ratio 0: none - no compressed positions exist, so the stage is a documented no-op; ratio > 0: the kernel, twin and pin are landed, but its operands (indexer q_index/keys/weights) have no device producer - indexer wk/weights_proj are AttentionHostSeam tensors outside the device bind" |
| `stage_attention` | "the row staging is a host copy through StagedAttention today; the kernel takes device rows" | "none: the stage is validation only - the reference store is not materialised on device, the corpus is the ring the kv_path stage already wrote, and the store row count is derived from the position" |
| `attention_body` | "the flash_mla_csa2_seq_sm121 launch exists, but its query row arrives through the host structs the stages above build" | "ratio 0: none - the launch, its operands and the inverse-rope symbol all exist on device; ratio > 0 additionally needs the selected ids from ref_select and the compressed store from reference_seam, which are blocked above" |
| `attention_output` | "the wo_a/wo_b launches exist; the host-side combine of their outputs is the step" | "none - the wo_a, activation-quantise and wo_b launches are the stage, with no readback and no host quantise" |

`STAGE_CONTRACTS` (the five placeholder rows: `compressor_select` does not exist, `attention_stage`
is not a kernel, `csa2_sequence_attn`/`wo_a_wo_b_gemv` are not symbols):

| stage | kernel (suggested) | signature (suggested) |
| --- | --- | --- |
| `reference_seam` | "no kernel: a host-decided window width on ratio 0; blocked on the compressor producer above ratio 0" | "(host-decided: position -> window_width = min(position + 1, 128); no pointer is passed)" |
| `ref_select` | "ref_select_sm121 then compressed_selected_fill_sm121 (sequence: select, then the expanded-row map)" | the existing 18-word signature is correct - keep it, and append the fill: "then (const uint32_t* ids, uint32_t n_ids, uint32_t window_rows, uint32_t ratio, uint32_t* out, uint32_t* out_count)" |
| `stage_attention` | "attention_stage (validation only: no launch)" | "(no device operand: ring_rows, position, ratio, store_rows, selected_count are host-decided; the refusals are ring/capacity, position-past-ring, cursor, union bound)" |
| `attention_body` | "flash_mla_csa2_seq_sm121 then rope_rotate_rows_bf16 (sin_sign = -1)" | the 19-word seq signature plus "(const __nv_bfloat16* src, const float* cos, const float* sin, __nv_bfloat16* out, uint32_t rows, uint32_t head_width, uint32_t rope_head_dim, int32_t sin_sign, GpuTraceRing* trace_ring, uint32_t layer_id, uint64_t step_seq)" |
| `attention_output` | "b12x_bf16a_mxfp8w_wo_a_sm121 then w8a8_prologue_bf16_e4m3 then b12x_mxfp8_gemv_m1_sm121" | the 8-word wo_a, 8-word bf16 prologue and 11-word GEMV signatures (already in `KERNEL_CONTRACTS`/the module docs) |

### A.8 Ordering and preconditions inside a position (ratio 0)

The loop already runs the stages in this order; the device attention half adds only these
dependencies:

1. `kv_path` (landed) must have written the ring slot for `p` (kv_norm_rope → window_quant →
   cache append), so the body's window read is the reading of a row that exists. This is the
   "append-then-read" rule and it is what makes `window_rows >= 1` and the reference's
   `s_max` finite.
2. `query_path` (landed) wrote `workspace.q` with the forward rope applied; the body consumes it
   as-is (the reference rotates q forward and the body does not re-rotate).
3. `reference_seam` → `ref_select` → `stage_attention` decide the words; `attention_body` reads
   nothing host-side; `attention_output` writes `workspace.attn_out_bf16`.
4. Stage 12 `all_reduce` (landed) reduces `attn_out_bf16` → `reduced_bf16`; stage 13
   `attention_residual` (landed) consumes it. The attention_output stage must not reduce.
5. `probe::count("<stage>")` once per successful stage, like the landed stages, and
   `runs.record` in `run_layer` already records the frontier.

No H2D/D2H may appear in stages 7-11 for the S1-a gate. The only copies in a position are
`begin_position`'s residual in and the run tail's out (the gate's "single explicit residual copy
each way"). If an evidence readback is wanted, it must sit behind `layer_evidence_readbacks()` /
`--forward-lean` like the host path's, never in the numeric path.

### A.9 What layers 0/1 do not need

`store_codes`/`store_scales`/`selected`/`candidate_mask`/`index_key`/`index_query`/`block_*` are
passed as the valid spans above but read only in the ways §1.5 states: with `selected_count == 0`
the entry reads no store and no selected id; the indexer stage still runs (it is inside the pinned
kernel) and reads `index_key` zeros up to `index_key_block_bytes(p)` (≤ 512×8×128×2 = 1,048,576 B
at `p = 4095` per `attention_operands::index_key_block_bytes`; the full 4 MiB pool covers it), and
writes the block pairs as scratch (nothing consumes them in milestone A — W2). `kv_fp4`,
`kv_scales`, `selected`, `index_key`, `index_query`, `block_*` must still be **valid device
pointers**: the ABI has no null branch for them.

---

## 3. Part B — layers 2..39 (`compress_ratio > 0`): blocked by design, no S1 implementation

The blocker is not the launch; it is the **weights**. The seam's device work needs
`layers.N.attn.compressor.wkv.weight`, `.compressor.wgate.weight`, `.compressor.norm.weight`,
`.indexer.wk.weight` and `.indexer.weights_proj.weight`. On this tree all of those are
`PlanClass::AttentionHostSeam`:

* `classify_plan_name` puts them there (`resident_weights.rs:1145-1160`) and
  `bind_rank_attention` **skips** host-seam tensors before the B12x payload check
  (`resident_weights.rs:850-860`), recording them as `host_seams` in the coverage report;
* the count is 19: compressor wkv for 4 kv-source layers (2,8,14,20), wgate for the 3 of those
  above ratio 1 (2,8,14), indexer wk for the 4, weights_proj for the 8 index-source layers
  (2,8,14,20,24,28,32,36) - `resident_weights.rs` test `count(PlanClass::AttentionHostSeam) == 19`;
* `LayerWeights`/`WEIGHT_NAMES` therefore cannot resolve them, and the device arm has no pointers.

So the Part B sequence is stated for when those tensors move into the bind, and **nothing in S1
implements it** unless the Lead says so:

1. **Bind rule change:** classify compressor wkv/wgate/norm and indexer wk/weights_proj as
   bindable for the layers that own them (kv-source / index-source), with the dtype the released
   checkpoint stores (they are the release's bf16/f32 host-seam tensors today; the bind's B12x
   payload rule only applies if they are staged in the B12x layout, so a stored-bytes
   `MatrixBind`-like path or a staging step is the actual change, not a one-line class edit). This
   is a `resident_weights.rs` change and the reason (a) is not blocked.
2. **reference_seam (device):**
   compressor `wkv` (and `wgate` above ratio 1) host-weight GEMV (the same `hostw_gemv_launch`
   family the host arm's `SPOOKFLASH_SEAM_REF_HOST` arm already uses, `tp4_forward.rs:6069+`) →
   `pool_group` softmax weighting (new kernel; the host rule is `engine/forward/attention.rs:985`,
   per-element softmax over the ratio axis, empty groups contribute zero) → norm (the ratio's
   `compressor.norm`) → rope forward (`rope_rotate_rows_bf16` with `sin_sign = +1` over
   `ratio`-pooled rows) → `compressed_append_sm121` (`append_row(ratio, p)`, §0.2; only on a
   completing position). `kv_rotated` is **not** the store operand: it is the window KV latent
   (mla `wkv`), a different weight from the compressor's.
3. **ref_select (device):** needs the indexer's q (wq_b over q_norm; `indexer.wq_b` is already
   bound as an attention projection, `rank_attention_plan_shapes`), the index keys
   (`indexer.wk` + host `write_index_keys` equivalent: k_norm, rope, fp4 round trip on device),
   the per-query weights (`weights_proj`), then `ref_select_sm121` (18 words:
   `n_pos = (p+1)/ratio`, `heads = index_n_heads` (32 or the per-rank 8, see the cross-rank note
   below), `head_dim = index_head_dim` (128), `reach = n_pos`, `topk = min(index_topk, n_pos)`,
   `offset = window_rows`, `mask_mode = Produce` on the candidate-source layer (config
   `candidate_source_layer_id = 20`) / `Consume` above it / `None` below, `topk_blocks =
   candidate_topk_blocks = 2048`, `block_size = candidate_block_size = 8`, `no_index =
   NO_INDEX`, `flags = 0`), then `compressed_selected_fill_sm121` (6 words). For a decode step
   `reach == n_pos`, so every emitted id is `position + offset` and the list has
   `keep = min(topk, n_pos)` entries; the shipped fill compacts/skips `NO_INDEX` and its
   `out_count` equals `keep` - but the launcher's `SELECTED_COUNT_OWNER` still forbids passing
   `keep` to the body until the ruled in-place fill and its re-pin land. Treat that as the blocking
   item for the body's `selected_count` word on ratio>0 layers: either land the ruled fill, or add
   an ignored device test that asserts `out_count == keep` in the qa harness before the arm passes
   `keep`.
4. **Cross-rank identity (open design item, +1 collective or a replicated indexer):** the selected
   set must be identical on all four ranks (the union attention is per-rank-head but the candidate
   set is global). With `weights_proj` sharded over 8 index heads per rank and the index keys
   replicated, comparing scores needs an all-reduce of the per-rank partial score, or a replicated
   indexer (full 32 heads per rank) - that is a design decision with a kernel consequence, not a
   wiring detail. The contract does not settle it.
5. **Body:** the same 19-word launch, with `selected_count = keep`, `selected` = the expanded ids
   and window/selected union ≤ 640; store spans written by step 2. `index_query`/`index_key`
   become real only with step 3; W2's incompleteness of `block_scores` at `grid.x = 16` remains a
   milestone-A caveat even then (the kernel strides by `SPARK_M4_HEADS = 64`).

Part B refusals: every stage keeps a deterministic refusal naming the excluded class (same text on
all ranks), so a 40-layer run reports the frontier at `reference_seam` for layer 2 and never falls
back to the host attention path.

---

## 4. Implementation split: the new file (landed) and the `tp4_forward.rs` wiring (yours)

The split was approved and the new file is **written and green**:
`crates/sparkflash-cli/src/layer_attention_stages.rs`, declared by one line in
`crates/sparkflash-cli/src/lib.rs` (`pub mod layer_attention_stages;`). It is gated
`#![cfg(any(feature = "cuda", test))]`, so `cargo check -p sparkflash-cli --lib` without cuda sees
an empty module and stays clean; the device-facing items are compiled in the cuda build and the pure
word builders/tests also under `cfg(test)`. It touches nothing outside itself plus that one line.

As-built API (this is what the five trait bodies call; it is `pub` in the module and the module is
`pub`, so the build has no dead-code warnings until the wiring calls it):

```rust
pub struct AttentionStepState {                     // stage 7 builds it; 8/9 check it; 10/11 read it
    pub ratio: usize,                               // 0 for layers 0/1
    pub window: WindowOperands,                     // attention_operands, 0-based p
    pub selected_count: u32,                        // 0 on ratio 0
    pub store_rows: usize,                          // 0 on ratio 0
    pub store_zero: bool,
}
impl AttentionStepState { pub fn attended(&self) -> Result<usize, String>; }

pub fn reference_seam(ratio: usize, position: usize, ring_rows: usize,
                      max_positions: usize) -> Result<AttentionStepState, String>;
pub fn ref_select(ratio: usize, state: &AttentionStepState) -> Result<(), String>;
pub fn stage_attention(state: &AttentionStepState, capacity_rows: usize,
                       max_positions: usize) -> Result<(), String>;
pub fn store_rows_at(ratio: usize, position: usize) -> Result<Option<usize>, String>;  // wiring helper

pub struct BodyOperands {                           // every span carries its byte length
    pub q: DeviceSpan, pub kv_fp4: DeviceSpan, pub kv_scales: DeviceSpan,
    pub window_codes: Deviceptr, pub window_scales: Deviceptr,  // resolved ring addresses
    pub window_rows: u32, pub selected: DeviceSpan, pub selected_count: u32,
    pub index_key: DeviceSpan, pub index_query: DeviceSpan, pub sink: DeviceSpan,
    pub block_scores: DeviceSpan, pub block_best: DeviceSpan, pub out: DeviceSpan,
    pub position: u32, pub heads_per_rank: u32, pub head_width: u32,
    pub trace_ring: u64, pub layer_id: u32, pub step_seq: u64,
}
pub fn body_params(operands: &BodyOperands) -> Result<FlashMlaSeqParams, String>;  // testable words
pub fn launch_attention_body(cuda: &Cuda, kernel: Kernel, operands: &BodyOperands)
    -> Result<(), String>;                          // grid (heads_per_rank, 1, 1), block (256, 1, 1)

pub fn inverse_rope_params(row: Deviceptr, cos: Deviceptr, sin: Deviceptr, rows: u32,
    head_width: u32, rope_head_dim: u32, trace_ring: u64, layer_id: u32, step_seq: u64)
    -> Result<RopeRowsParams, String>;              // sin_sign = -1, out aliases src
pub fn launch_inverse_rope(cuda: &Cuda, kernel: Kernel, /* same, then */ )
    -> Result<(), String>;                          // through seams::launch_rope_rows

pub struct OutputOperands {
    pub attn_bf16: DeviceSpan, pub lo_out: DeviceSpan, pub act_codes: DeviceSpan,
    pub act_scales: DeviceSpan, pub wo_b_out: DeviceSpan,
    pub wo_a: (Deviceptr, Deviceptr), pub wo_b: (Deviceptr, Deviceptr),
    pub wo_a_shape: (usize, usize), pub wo_b_shape: (usize, usize),   // the bind's stored shapes
    pub geometry: OutputGeometry,
    pub trace_ring: u64, pub layer_id: u32, pub step_seq: u64,
}
pub fn output_checks(operands: &OutputOperands) -> Result<(), String>;
pub fn wo_a_params(operands: &OutputOperands) -> Result<Bf16aMxfp8wWoAParams, String>;
pub fn prologue_bf16_params(operands: &OutputOperands) -> Result<PrologueBf16Params, String>;
pub fn gemv_params(operands: &OutputOperands) -> Result<Mxfp8DenseParams, String>;
pub fn launch_attention_output(cuda: &Cuda, wo_a_kernel: Kernel, prologue_kernel: Kernel,
    gemv_kernel: Kernel, operands: &OutputOperands) -> Result<(), String>;
```

Two signature corrections against the plan above, both for the better and both already compiled in:

* the spans are `crate::layer_device::DeviceSpan` (pointer **and** length) instead of bare
  `Deviceptr`, so the "span too short names its row" refusals are inside this module and unit-tested
  without a device. `window_codes`/`window_scales` stay raw `Deviceptr`s because they are addresses
  *inside* the run-wide ring, not spans;
* `AttentionStepState` carries `ratio`, and `ref_select`/`stage_attention` take it from there, so the
  store-cursor check (`(p + 1) / ratio`) and the ratio-0 invariant are both expressed without a
  second parameter that could disagree.

Tests in the file (27, no `Cuda`, all pass): the 19-word body array including `selected_count` as
word 15 and the trace trio; the 11-word inverse-rope array with `sin_sign` as `u64::MAX`; the
8-word wo_a with `groups = 2`; the prologue's `cols = 2048, rows = 1`; the GEMV's
`rows = 1, cols = 5120, inner = 2048`; the ratio-0 window at 0/127/300; the ring/position/capacity
refusals; the store-cursor and non-empty-store refusals; the union/top-k/window/head-count bounds;
every thin-span refusal row by row; the index-pool position bound; the null-operand refusal; the
inverse-rope geometry refusals; the wo_a/wo_b plan-shard refusals (including the world-1 shard
`(1024, 8192)`/`(5120, 1024)`); and the GEMV `MAX_INNER` refusal on a crafted 6,208-value reduction.

**Yours in `tp4_forward.rs`** (unchanged from the approved split): the five `LayerStages` method
bodies (gather spans from `self.buffers`/`self.kv_ring`/`self.attention.pointers`/
`self.compress_ratio`, call these functions, `probe::count`), the `attention_state: Vec<AttentionStepState>`
and `store_rows: Vec<usize>` fields, the `open` allocations (zero pool, sink pool + upload, the two
new kernel loads) and their `Drop` frees, and the S1-a mixed arm. **Yours in `layer_device.rs`**: the
nine `LayerWorkspace` rows, the `assert_contract_widths` entries, the `STAGE_CONTRACTS`/
`STAGE_BLOCKERS` text (§A.7).

One known tree state to clear before a `-D warnings` clippy gate can pass: `crates/sparkflash-cli/src/tp4_forward.rs:10750`
defines `AttentionWindow` with no constructor yet (it is in `HEAD` already, not from this branch's
wiring), so `cargo clippy -p sparkflash-cli --lib --all-targets -- -D warnings` fails on
`dead_code` in your file. The new module itself is clippy-clean in both feature configurations
(see §5).

---

## 5. Verification status of this document's claims

Run in this environment (CPU only, no GPU, no `/dev/nvidia`, no node):

| command | result |
| --- | --- |
| `cargo check -p sparkflash-cli --lib` | **ok** - clean; the new module is `cfg(any(feature = "cuda", test))` and compiles to nothing without the feature, so the cuda-less library is unchanged |
| `cargo check -p sparkflash-cli --lib --features cuda` | **ok** - clean (the module compiles with the device-facing items) |
| `cargo test -p sparkflash-cli --lib layer_attention_stages` | **ok, 27 passed** (every word array and every refusal above) |
| `cargo test -p sparkflash-cli --lib layer_attention_stages --features cuda` | **ok, 27 passed** |
| `cargo test -p sparkflash-cli --lib` | **ok, 433 passed, 1 ignored** (the 27 included; no regression in the rest of the crate) |
| `cargo fmt -p sparkflash-cli -- --check` | **clean** |
| `cargo clippy -p sparkflash-cli --lib --all-targets` and `... --features cuda` (no `-D warnings`) | **no finding in `layer_attention_stages.rs` in either configuration**; the tree-wide `-D warnings` gate currently fails on the pre-existing `dead_code` of `AttentionWindow` (`tp4_forward.rs:10750`, already in `HEAD`) and on the 30-vs-27 pin-manifest mismatch in `sparkflash-kernels`' `routed_gate_up_swiglu_splitk_abi` (the pin job's in-progress manifest), neither of which this module touches |
| `cargo test -p sparkflash-kernels --lib` | **583 passed, 1 failed, 2 ignored** - the failure is `routed_gate_up_swiglu_splitk_abi::tests::the_pin_is_recorded_before_the_manifest_names_it` ("before the pin job the manifest holds 27 entries, found 30"), the pin job's own in-progress state; every `attention_operands` test (20) and every other kernel test passes |
| `/home/user/cuda-tk/bin/nvcc --version` | CUDA 13.0.88 (frontend only here) |

Read-only inspections this document is derived from: `kernels/layer_seams_sm121.cu:894-963`
(`rope_rotate_rows_bf16`), `kernels/flash_mla_csa2_seq_sm121.cu` (the 19-word body and its guards),
`kernels/compressed_append_sm121.cu:112-187` (the append's `store_row` and the fill), the launchers
(`layer_seams_launch.rs`, `flash_mla_csa2_seq_launch.rs`, `ref_select_launch.rs`,
`compressed_append_launch.rs`, `bf16a_mxfp8w_wo_a_launch.rs`, `mxfp8_gemv_launch.rs`,
`mxfp8_dense_launch.rs`), the host twins (`run_layer_query_path`, `run_layer_kv_path`,
`MlaAttentionBody::run`, `run_layer_attention_output`, `tp4_prefill::stage_attention`), the engine
seam (`attention_seam_step_with_backend`, `compressor_step`, `pool_group`,
`write_compressed_cache`), `layer_device.rs`, `resident_weights.rs`, and
`docs/archive/build/attention-sequence-contract-20261022.md`.

**UNRUN, and it cannot be run here:** any device launch (so any claim that the body/wo_a/wo_b
arithmetic is bit-equal on hardware), the D54 numerical gate, any timing (≤120 ms/step), the
`SPARKFLASH_LAYER_DEVICE=1` end-to-end layer-0 run, and any nvcc *cubin* build (the local toolkit
is a frontend; `build.rs` used clang `-fsyntax-only` for the 27-kernel probe, which is not PTX and
not numerics). Those belong to the qa-owner's harness on the GPU node and to the Lead's gate run.

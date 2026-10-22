# The device layer's per-position launch inventory (frontier probe, 2026-10-22)

Why this file exists: the graph-capture work will consume this list, and reading it is also the
cheapest way to spot a launch that is per-position but should be hoisted. Numbers are the released
geometry (dim 5,120; n_heads 64 over world 4 = 16 per rank; head_dim 512; rope_head_dim 64;
q_lora_rank 1,280; hc_mult 4). **No entry here is host-validated and none of it is a speed result**:
the device arm has not run against the host reference yet, and the row's frontier document says so.

Scope: one device-arm layer at one position, steady state. Everything below is enqueued on the
default stream; there is **no host wait, no readback and no host-side arithmetic** between the first
and the last launch of the position. The loop's only crossings are the position's start
(`begin_position`, one H2D of `residual.stream` = hc_mult * dim * 4 = 81,920 B) and its end
(`residual_out`, one D2H), both outside the layer loop.

## Launches, in `run_layer` order

| # | stage | kernel | payload in | payload out | span written |
| --- | --- | --- | --- | --- | --- |
| 1 | `quantise_prologue` | `w8a8_prologue_e4m3` (8-word ABI) | `residual.normed` 5,120 values | codes 5,120 B, scales 160 B | `workspace.act_codes`, `workspace.act_scales` |
| 2 | `attention_projections` | `b12x_mxfp8_gemv_m1_sm121` (11-word ABI), wq_a | codes/scales (cols 1,280) + wq_a fp8 + its scale grid | 1,280 bf16 = 2,560 B | `workspace.q_lora` |
| 3 | `attention_projections` | `b12x_mxfp8_gemv_m1_sm121`, wkv | codes/scales (cols 1,280) + wkv fp8 + scales | 512 bf16 = 1,024 B | `workspace.kv_latent` |
| 4 | `query_path` | `rmsnorm_sm121` | `q_lora` 1,280 bf16 + q_norm weight 1,280 bf16 | 1,280 bf16 = 2,560 B | `workspace.q_normed` |
| 5 | `query_path` | `w8a8_prologue_e4m3` (cols 1,280) | `q_normed` 1,280 values | codes 1,280 B, scales 40 B (reuses the spans, cols 1,280 <= 5,120 and 40 <= 160) | `workspace.act_codes`, `workspace.act_scales` |
| 6 | `query_path` | `b12x_mxfp8_gemv_m1_sm121`, wq_b | codes/scales (cols 1,280 = inner) + wq_b fp8 + scales | 8,192 bf16 = 16,384 B | `workspace.q` |
| 7-22 | `query_path` | `rope_rotate_bf16` x **16** (one per head) | one head row 1,024 B + cos/sin 128 B each | same row (out aliases row) | `workspace.q` (head h at h * 1,024 B) |
| 23 | `kv_path` | `rope_rotate_bf16` | `kv_latent` 1,024 B + tables 128 B each | 1,024 B (out of place) | `workspace.kv_rotated` |
| 24 | `kv_path` | `window_quant_fp8_bf16` | `kv_rotated` 512 values | codes 512 B, scales 16 B | the layer's window ring slot (`WindowRingLayout::codes_offset/scales_offset(position % capacity)`) |
| 25 | `kv_path` | `kv_append_bf16` | k row 1,024 B, v row 1,024 B (both `kv_rotated`) | slot's 2,048 B | the device cache slot (`DeviceKvCache::layer_span(layer)`, slot `position % capacity`) |

**Total: 25 launches per layer-position** (1 prologue + 2 projection GEMVs + 1 rmsnorm + 1 latent
prologue + 1 wq_b GEMV + 16 q-rope + 1 kv-rope + 1 window quantise + 1 append).

## Hoist candidates, in the order the graph work should look

1. **The 16 q-rope launches.** They exist because `rope_rotate_bf16` takes one `head_width`-wide row
   and rotates its last `rope_head_dim` values; the q span is 16 contiguous heads, so a kernel that
   walked the head count (or took `rows = 16` with a row stride) would be one launch. This is the
   first thing to fix if launch count matters; it is 64% of the position's launches.
2. **The two prologues per position** (stages 1 and 5). They quantise different widths (5,120 and
   1,280) into the same spans at different times, so they cannot merge as written; if the GEMV later
   takes a bf16 A operand the latent's prologue disappears entirely.
3. **The rmsnorm + latent prologue pair** (4 and 5) reads and writes 2,560 B each; a fused
   norm+quantise would save one launch and one span pass, but changes the reference arithmetic's
   ordering only in the rounding it already does, so it needs a host-reference diff, not just a
   launch count.

## Per-run work (outside the loop, at `DeviceLayer::open`)

* Kernel loads: `rmsnorm_sm121`; the seam image (prologue + the two symbols driver-fixer owns); the
  decode GEMV; the kv image (rope, window quantise, append) - six pinned cubins, all digest-checked.
* Rope table H2D: **two uploads**, one per class (`max_positions * 256 B` each at rope_head_dim 64),
  never per layer and never per position. Position p's slice is
  `base + p * pairs * 4` (cos) and `base + max_positions * pairs * 4 + p * pairs * 4` (sin).
* Allocations beside the slab: the device KV cache
  (`DeviceKvCache::bytes(n_layers, max_positions, 2 * head_dim)`) and the window ring
  (`WindowRingLayout::total_bytes(n_layers)`). Both sizes are in the run record's
  `layer_device_memory` next to `slab_bytes`, `max_positions` and `rope_table_bytes`.

## What is not here yet, deliberately

`reference_seam`, `stage_attention`, `attention_body`, `attention_output`, `all_reduce`,
`attention_residual`, `moe_router`, `moe_routed`, `moe_shared`, `all_reduce_moe`, `ffn_epilogue` -
their stages still refuse by name. `moe_shared` is the contract amendment of this commit: the shared
expert is a dense w1/w3/swiglu/w2 MLP on the normed row at `seam_row` base, it overwrites that base
with the shared row (f32 by default, bf16 only under native-rounding shared), and it adds **no**
collective because its weights are gathered once at open. It is not in the launch table above
because no body is wired yet.

The frontier the probe row shows today, **from the stage order and not from a run**, is
`reference_seam`: `engram_write` is a no-op for layer 0 (it refuses by design on layers 1 and 14),
`mhc_entry` and the four stages after it are landed, and `reference_seam` is the first refusing
stage. If a row ever reports `kv_path` instead, that is not the stage order - it means the kv body
refused at run time and needs looking at.

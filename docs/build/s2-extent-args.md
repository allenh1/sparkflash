# S2 extent-args audit: what each per-step launch still takes from the host

**Status: audit (S2 scope item 2, capture-safety prep for S3). No launch shape changed in S2.**
This is the line-precise list S3 (graph capture) starts from: every per-step launch of the device arm
that still takes a host `position` / `n_pos`-derived argument, and every step-wide hazard a captured
graph cannot contain. Line numbers are against this branch's tip
(`5-s2-whole-decoder-layer-on-device-nccl-all-reduce-60-ms-step`); `tp4_forward.rs` means
`crates/sparkflash-cli/src/tp4_forward.rs`.

The S1 step invariant (`d2h=1 / alloc=0 / sync=0`, enforced in-run, recorded per head position as
`position_records[].device_step_invariants`) counts **device** crossings: device-to-host copies,
device allocations and host waits. It does not count host-to-device copies, host `Vec` allocations or
launch parameters, and those are most of what is below. The invariant holding is necessary for
capture, not sufficient.

## 1. Step-wide hazards (not tied to one kernel)

| # | Hazard | Where | Why it blocks capture | S3 remedy |
| --: | :-- | :-- | :-- | :-- |
| H1 | **Legacy default stream.** Outside a capture segment, every launch, every device-to-device copy and both per-layer NCCL collectives use the null stream (stream 0) | direct launches and `cuMemcpyDtoDAsync_v2` pass `std::ptr::null_mut()` (`crates/sparkflash-kernels/src/cuda_driver.rs:2104`, `2829-2831`); `reduce_bf16(0, ...)` (`tp4_forward.rs:12241`) | The legacy stream cannot be captured, and it implicitly synchronises with every blocking stream in the context. The driver's existing capture segment (`graph_capture_begin`, `cuda_driver.rs:1814`) moves **kernel launches** onto a private capture stream and refuses copies, allocations and synchronises, but the NCCL enqueue is handed stream 0 explicitly and does not go through it: a capture today would record the kernels and **miss both collectives** | One run-owned `CU_STREAM_NON_BLOCKING` stream that the launchers, the capture segment and `ncclAllReduce` all take. **Not done in S2:** it touches every launcher's signature and is S3's first item |
| H2 | **Synchronous host-to-device copies at position start.** `begin_position` builds the hc-widened residual (4 x 5,120 f32 = 81,920 B) and the identity `pre_mix` seed (16 B) in host `Vec`s and copies both with synchronous `cuMemcpyHtoD_v2` from pageable memory | `tp4_forward.rs:13379-13392` (`copy_in`, `tp4_forward.rs:13152-13169`); the hidden row itself is cut from the host-side `prefill_hidden` embedding at `tp4_forward.rs:15857-15862` | A synchronous copy is illegal inside capture; pageable memory cannot be a memcpy node | A device embedding gather from a device token-id word plus a seed kernel, so the step's only input is one 4 B id in a fixed pinned word |
| H3 | **The head's readback and gather.** `screen_device` allocates a 129,280 B host `Vec` and does a synchronous `cuMemcpyDtoH` (the declared `d2h=1`); `gather_logits` runs over the TCP control transport with a sequence number derived from `position` | `crates/sparkflash-cli/src/tp4_tail.rs:456-457`; `tp4_forward.rs:16662-16670` | The D2H is the graph's end; the TCP gather is host work | The graph ends at the LM-head launch; the D2H becomes an async copy into a pinned buffer plus an event, and the gather stays a host step after the graph (or moves to an NCCL all-gather) |
| H4 | **`step_seq = position` in every launch's trace header.** All twenty stages pass `step_seq` by value (for example `tp4_forward.rs:12082`, `13427`, `13581`, `14395`) | every launcher's params | A captured graph bakes kernel params, so ~2,030 nodes would need `cuGraphExecKernelNodeSetParams` per step | A device step counter word the trace macros read; the host stops passing it |
| H5 | **Per-step host bookkeeping.** `SPARKFLASH_DEVICE_DIFF` is read from the environment at every head position; the per-position JSON record is built every position; the Engram n-gram hashes are computed on the host every position | `tp4_forward.rs:16692`, `16762`; `tp4_forward.rs:15833-15856` | Not a capture hazard, but host time on the critical path between graphs | Read the env once at open; build records after the graph's event; hashes only where the host arm needs them |
| H6 | **Per-layer weight lookups by formatted name.** `attention_output` formats `layers.{layer}.attn.wo_a.weight` and `wo_b` and looks them up every layer of every step (80 `String` allocations per step) | `tp4_forward.rs:14538-14539` | Host time only | Resolve both pointers per layer at open, as the other stages do |

## 2. Per-stage table: launches that take host position-derived arguments

"Step-only" means the stage's only per-step host argument is the H4 trace word: the launch is
otherwise position-free and captures as is once H1/H4 are fixed.

| # | Stage | Host argument(s) that change per position | Lines | Kind |
| --: | :-- | :-- | :-- | :-- |
| 1 | `engram_write` (Engram layers only) | host branch `is_skipped(slot, position)` decides whether the six launches run; the arena entry address `slot * positions + position` moves the input pointer every position | `tp4_forward.rs:13413-13420`, `11244-11253` | branch + pointer |
| 2 | `mhc_entry` | step-only | `tp4_forward.rs:13581`, `13653` | trace |
| 3 | `quantise_prologue` | none (`step_seq: 0`) | `tp4_forward.rs:13689` | position-free |
| 4 | `attention_projections` (`wq_a`, `wkv`) | step-only | `tp4_forward.rs:12082-12096` | trace |
| 5 | `query_path` | `wq_b` step-only; `rotate_q`: `heads_per_rank` launches per layer, each with cos/sin addresses offset by `position * pairs * 4` and `position: u32` by value; host bound check `position >= max_positions` | `tp4_forward.rs:12108-12160` | pointer + word |
| 6 | `kv_path` | `kv_norm_rope` table offset `position * pairs * 4` and `position_u32`; the window quantiser's ring slot `kv_ring.slot_for(position)`; `kv_append` `position: position_u32`; host checks `kv_append_rule(appended, position, ..)` and `position >= max_positions` | `tp4_forward.rs:13801-13909` | pointer + word + host check |
| 7 | `reference_seam` (kv-source layers, ratio > 0) | ring slot `position % ratio` moves both GEMV output addresses; **host branch `(position + 1) % ratio == 0` decides whether the compress launches run at all**; `first = (position + 1 - ratio) % ring.rows`; `first_row = position / ratio`; the index-key rope offset `rope_pos * pairs * 4` with `rope_pos = first_row * ratio`; the index-key destination `(owner_idx * max_positions + first_row) * index_head_dim * 4`; `seam_latent_rope`'s table offset; `seam_append_group`'s `first_row` word | `tp4_forward.rs:13932-14105`, `12598-12712` | **topology** + pointer + word |
| 7' | `reference_seam` (every layer) | publishes `window_rows = min(position + 1, window)` and `window_start = position + 1 - window_rows` as host state that stages 8-10 read | `tp4_forward.rs:14109-14119` | host state |
| 8 | `ref_select` (index-source layers) | `n_pos = (position + 1) / ratio` passed twice as a `u32` word; `keep = min(index_topk, n_pos)`; **host branch `n_pos == 0` skips the launch**; rope offset `position * pairs * 4`; `window_rows` word; `selected_count` published as host state | `tp4_forward.rs:14139-14315` | **topology** + word + host state |
| 9 | `stage_attention` | no launch; host validation only (`appended == position + 1`, the window inside the ring, ratio-0 `selected_count == 0`) | `tp4_forward.rs:14320-14360` | host check |
| 10 | `attention_body` | `window_rows`, `selected_count`, `position_u32` words; the ring address `kv_ring.slot_for(window_start)`; the inverse rope's table offset `position * pairs * 4` | `tp4_forward.rs:14378-14516` | pointer + word |
| 11 | `attention_output` | step-only (plus H6) | `tp4_forward.rs:14538-14561` | trace |
| 12 | `all_reduce` | none (the stream is H1; the buffers are registered once at open, S2-a) | `tp4_forward.rs:14575-14583` | position-free |
| 13 | `attention_residual` | step-only | `tp4_forward.rs:14605`, `14654` | trace |
| 14 | `ffn_seams` | step-only | `tp4_forward.rs:14674`, `14758` | trace |
| 15 | `moe_router` | step-only; the host probe-count guard is a check, not an argument | `tp4_forward.rs:14771-14833` | trace |
| 16 | `moe_routed` | step-only, `tokens = 1` (the extent S5's multi-token verify step changes) | `tp4_forward.rs:14881` | trace + extent |
| 17 | `all_reduce_moe` | none (as #12) | `tp4_forward.rs:15088-15091` | position-free |
| 18 | `moe_shared` | step-only | `tp4_forward.rs:14903-15080` | trace |
| 19 | `ffn_epilogue` | step-only | `tp4_forward.rs:15112`, `15159` | trace |
| H | head (`collapse_for_head`, `screen_device`, `gather_logits`) | `screen_device` step word `position as u64`; the gather sequence `HEAD_GATHER_SEQ - 2 * position`; H3 | `tp4_forward.rs:16655-16670` | trace + host |

(The contract's `all_reduce_rdma_arm` and `all_gather` rows make no launch on the NCCL arm.)

### What the kinds mean for S3

- **trace**: fixed by H4 alone.
- **pointer + word**: the kernel must read `position` (or `n_pos`, `window_rows`, `selected_count`)
  from a device word the step writes once, and compute its own table and ring offsets. The tables and
  rings are already sized at open (`max_positions` rope rows, the KV ring), so no allocation moves.
- **host state**: `window_rows` / `window_start` / `selected_count` live in `AttentionWindow` on the
  host and flow from `reference_seam` and `ref_select` into `attention_body`. They must become device
  words written by the producing stage.
- **host check**: the bound and ordering checks (`kv_append_rule`, `position >= max_positions`,
  `stage_attention`'s validations) cannot run inside a replayed graph. They move to a per-step
  preflight before the graph launches (they are pure functions of `position` and the open-time sizes),
  so the refusal text is kept.
- **topology**: `reference_seam`'s compress branch and `ref_select`'s `n_pos == 0` skip change *which
  launches exist* from one position to the next. A single captured graph cannot express that.
  The options are: kernels that early-exit on a device predicate (one graph, a few idle launches);
  conditional graph nodes; or one graph per phase of `position mod lcm(ratios)`. The first keeps one
  graph and is the recommended starting point.

## 3. The DSpark / Eagle3 taps on layers 37, 38, 39

The released forward keeps the block **input** of layers 37, 38 and 39, averaged over the hc copies
(`main_hiddens.append(h.mean(dim=2))`, model.py:1264-1266; [`../engine/01-forward-order.md`](../engine/01-forward-order.md)
step 7b), and concatenates the three per position into the drafter's `[b, s, 15360]` input. Today:

| Arm | Tap | Where |
| :-- | :-- | :-- |
| CPU engine (`sparkflash-engine`) | implemented: `main_hidden` holds `taps * dim` values per position | `crates/sparkflash-engine/src/forward/mod.rs:1288-1305` |
| TP4 host arm | none | - |
| TP4 device arm | **fixed span carved in S2 (`workspace.dspark_aux_taps`, 61,440 B = 3 × 5,120 f32)**; kernel launch deferred to S5 | `crates/sparkflash-cli/src/layer_device.rs` (`LayerBuffers::dspark_aux_tap_span` for layers 37, 38, 39) |

The tap is position-free arithmetic (a mean over four copies of `residual.stream` before
`engram_write` / `mhc_entry` of the tapped layer), so for capture it needs only **fixed spans**:
three 5,120-wide rows (f32 20,480 B each = 61,440 B contiguous), carved in the slab at open
(`workspace.dspark_aux_taps`), and one small launch per tapped layer writing its row, at a fixed
address the drafter reads. No host `position` enters it. Two extents do:

- S5's verify step taps `n_tokens` rows per layer. S2 carves one row per tap, the decode step's, so
  S5 widens the carve to the verify block at open (not per step).
- The drafter's own input position is `position`, which takes the same device word as section 2.

## 4. What S2 changed that S3 inherits

- The four collective spans (`attn_out_bf16`, `reduced_bf16`, `moe_delta_bf16`, `moe_reduced_bf16`)
  are registered with the communicator once at open (`ncclCommRegister`), and released in
  `DeviceLayer`'s Drop after a synchronise. Their addresses are fixed for the run, which is what a
  captured NCCL node needs.
- The run header refuses a run whose `SPARK_COMM_REDUCE` is unset or does not match the arm, so a
  captured step can never be silently replaced by a TCP-reduced one.
- The compressed KV store of each kv-source owner is sized at open to `max_positions` rows (never
  fewer than the 512 of `EXPANDED_ROWS`), by one helper (`seam_store_owner_bytes`) that the
  allocation, the seam append and the attention body share. The store's addresses and owner strides
  are therefore fixed for the run. Before this, a run refused at the first append whose group reached
  past expanded row 512, which is why the S2-e soak needed the change.

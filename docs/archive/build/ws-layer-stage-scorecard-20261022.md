# Layer-device stage scorecard (2026-10-22, integration tip `6ccd5172` + in-flight branches)

Twenty stages when `ref_select` lands (contract is 19 today; the split is decided and queued with its kernel).
"Body" means the stage has a `run_layer` body that either launches or refuses by name - a refusal is a known
gap, not a claim. Nothing here is a measured result: **the arm has never run on a node**, and the first probe
row stops at `reference_seam`.

| # | stage | kernel(s) | body | blocker / next |
| --- | --- | --- | --- | --- |
| 1 | `engram_write` | `engram_embed/wkv/gate_write_sm121` (pinned) | no-op on layer 0 only | non-layer-0 body unwritten; layers that have an engram table |
| 2 | `mhc_entry` | `mhc_entry_bf16` (20-word ABI) | **landed** | - |
| 3 | `quantise_prologue` | `w8a8_prologue_e4m3` (f32 row) | **landed** | - |
| 4 | `attention_projections` | `b12x_mxfp8_gemv_m1_sm121` | **landed** | - |
| 5 | `query_path` | rmsnorm, `w8a8_prologue_bf16_e4m3`, wq_b GEMV, 16 rope launches | **landed** | 16 rope launches/layer is a hoist candidate, not a defect |
| 6 | `kv_path` | `kv_norm_rope_bf16`, `window_quant_fp8_bf16`, `kv_append_bf16` | **landed** | declared bf16-latent widening (D54 item, measured 2 % bound) |
| 7 | `reference_seam` | `b12x_hostw_gemv_m1_sm121` + index writes | **refuses** | compressor weights are host-dtype (bf16/f32), staged per layer at open; compressed-cache + index writes; `AttnState` residency |
| 8 | `ref_select` *(new, index 7)* | new symbol | **not started** | top-k over <= 16,384 candidates, ascending-id tie-break; twin must be bit-equal on the index vector |
| 9 | `stage_attention` | native FP4 expanded store | **refuses** | chosen form: expanded store is the cache's native form (removes `quantize_main_kv` per selected row and the 147,456 B/call zero-fill) |
| 10 | `attention_body` | `flash_mla_csa2_seq_sm121` (pinned, no new kernel) | **not started** | wiring only - window pointer offset into the ring, `window_rows = min(position,128)`, `selected_count`; assert `ring_rows >= max_positions` and `position < ring_rows` |
| 11 | `attention_output` | inverse rope (new symbol), `b12x_bf16a_mxfp8w_wo_a_sm121`, `w8a8_prologue_bf16_e4m3`, GEMV | **not started** | inverse-rope symbol (rows-parameterised, sign word) is the only new kernel; the rest is wiring |
| 12 | `all_reduce` | `comm_all_reduce_bf16` (NCCL) | **refuses** | NCCL bootstrap: id over the loop transport, one `RunReduce` per run, `warm_up` outside the timed region |
| 13 | `attention_residual` | `mhc_attn_residual` | **landed** | exact per element vs `hc_post`/`hc_post_bf16`, both arms twin-pinned |
| 14 | `ffn_seams` | `mhc_ffn_seams` | **landed** | 24-word ABI + `ffn_x_bf16` out row in flight (one pin line) |
| 15 | `moe_router` | `moe_router_sm121` (10 words), `moe_router_topk_sm121` (8 words) | **not started** | reads the **un-normalised** bf16 pre-FFN row - needs `ffn_x_bf16` from `ffn_seams`; refuses while unwritten |
| 16 | `moe_routed` | `b12x_mxfp4_moe_stored_sm121` + 3 | **plumbing landed** | per-expert launch loop over 6 routed rows and per-rank shards; f32 delta -> bf16 pre-reduce row |
| 17 | `all_reduce_moe` | `comm_all_reduce_bf16` | **refuses** | same handle as #12, same sequence on every rank |
| 18 | `all_reduce_rdma_arm` | (arm reservation) | by design | `GpuRdma` refuses typed unless built; not a stage that "runs" |
| 19 | `moe_shared` | 3 GEMVs + `shared_swiglu_f32` (f32 arm) / coded arm | **landed, both arms** | `shared_present` flip on the FFN side (one commit) |
| 20 | `ffn_epilogue` | `mhc_ffn_epilogue` | **landed** | `shared_present=0` until #19's row is declared; then the arm has a shared row |

## What is genuinely new work

Four new kernels (unchanged count): `ref_select` (the select), the inverse-rope symbol, the compressed-store
producer pieces of #7/#9 (one or two symbols depending on the native-store choice), and that is it - the
compressor, the body and the output projections are all pinned symbols or wiring. Everything else on the list
is a body that already exists or a launch sequence.

## Reading the scorecard honestly

- **Landed** means "launches, or refuses by name with a blocker", twin-pinned where the arithmetic is testable
  without a node. It does not mean "numerically validated": nothing on this branch has been compared to the
  frozen reference on a device.
- **The frontier today is `reference_seam`** by stage order. The first probe row will confirm or contradict
  that; if it contradicts, the row wins and the tool/contract gets a reason, not a quiet edit.
- The three collectives/arms (#12, #17, #18) are the only places where a stage cannot refuse without knowing
  about the other ranks, which is why the frontier document checks rank agreement before anything else is
  believed.

# Layer-device dtype audit (2026-10-22)

Question: for every launch the device arm can reach from `run_layer`, does the operand's declared
dtype match the span it is passed? Method: read each kernel's parameter declaration in
`kernels/*.cu` and each call site's span in `tp4_forward.rs`; compare against the span docs in
`layer_device.rs` and the open-time widths that `LayerBuffers::assert_contract_widths` now enforces.

## Findings

**F1 - bug, FIXED: `query_path`'s latent quantise read f32 from a bf16 row.** (Fixed by the
bf16-input prologue `w8a8_prologue_bf16_e4m3` and driver-fixer's `PrologueBf16Params` + `cols*2` check
in `query_path`, merged on `a7cef896`; the f32 constructor split in `9b95043d`/this commit keeps the
two f32 call sites honest.)
`w8a8_prologue_e4m3` takes `const float* normed` (`kernels/layer_seams_sm121.cu:529`), but
`query_path` passes `workspace.q_normed`, which `rmsnorm_sm121` writes as BF16
(`kernels/rmsnorm_sm121.cu:100-103`; `q_normed` is `2 * Q_LORA_VALUES = 2,560 B` for 1,280 values,
`layer_device.rs:293`). `PrologueLaunch::new` only demands `row.len >= values * 2`
(`layer_device.rs:1083`), so the launch is authorised and the kernel reads 5,120 B out of a 2,560 B
span. Reference-equivalent fix: a bf16-input prologue variant (`const __nv_bfloat16* normed`, widen
then quantise), because the host's quantise operand is the f32 row **rounded to bf16**
(`attention.rs:1111` rms_norm, `1318-1341` bf16 storage). An f32-output rmsnorm is NOT equivalent:
it quantises the unrounded row the reference never sees. driver-fixer owns the symbol; wiring is one
call site plus the span-width assert when it lands.

**F2 - guard, open: the prologue span check is half-width.** `PrologueLaunch::new` accepts a span
`values * 2` long for a kernel that reads f32; a half-width check is worse than none because it
authorises the launch. Fix requested (driver-fixer, per Lead): parameterise the check by the row's
dtype (`values * 4` f32, `values * 2` bf16).

**F3 - deliberate, documented: the coded arm's BF16 shared row sits in the f32 `seam_row` span.**
`moe_shared`'s coded arm passes `seam_row`'s base as the GEMV's `void* out_bf16` and the epilogue
widens it under `shared_bf16 = 1`. The span stays f32-sized (still checked by F3's neighbouring
assert); there is no second span and no copy, and the f32 arm writes a real f32 row at the same base.

## Checked, matching

- `quantise_prologue`: f32 row, u8 codes, u8 scales / `residual.normed` f32, `act_codes`+`act_scales` u8. OK
- `attention_projections` (wq_a, wkv): u8 a/w codes and scales, bf16 out / `act_codes`+`act_scales`, pair spans, `q_lora` and `kv_latent` bf16. OK
- `query_path` rmsnorm: bf16 x, bf16 weight, bf16 out / `q_lora`, q_norm weight span (`cols * 2`), `q_normed`. OK
- `query_path` wq_b: u8/u8/bf16 / `act_codes`, `act_scales`, `q` bf16. OK (latent prologue before it is F1)
- `query_path` q rope tail, in place: bf16 row, f32 cos/sin tables, bf16 out aliasing the row / `q`, `rope_tables_{plain,compressed}` f32. OK
- `kv_path`: refuses at the missing `kv_norm_rope_bf16`; nothing runs (the pinned rope takes stored bf16, the host norms f32 first).
- `mhc_entry` / `ffn_seams` / `attention_residual` / `ffn_epilogue`: f32 streams, hc rows, mix tables, and `void* reduced_bf16` / matching spans. Pinned signatures in `layer_device.rs`; bodies are driver-fixer's.
- `moe_shared` coded arm: u8 codes/scales/B12x pairs, bf16 intermediate rows, BF16 final store over f32 base (F3). f32-reference arm: not yet reachable (device SwiGLU symbol pending).
- Stages that refuse by name today (`reference_seam`, `stage_attention`, `attention_body`, `attention_output`, `all_reduce`, `moe_router`, `moe_routed`, `all_reduce_moe`) launch nothing; their recorded spans are in `layer_device.rs` and are checked at open.

## Open-time half

`LayerBuffers::assert_contract_widths` runs immediately after `carve` in `DeviceLayer::open` and
requires every contract-named span to be exactly its width times its dtype rounded to `ALIGN`. It
covers the residual, workspace, rope-table and fixed-size spans; a length that disagrees refuses at
open with the span name, its length, the contract width and the rounded expectation (`layer_device.rs`,
test `every_carved_span_matches_its_contract_width`).
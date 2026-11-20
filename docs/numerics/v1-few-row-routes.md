<!--
Copyright (c) 2026 Hunter L. Allen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# The packed BF16 form's layout, as a type: every few-row and token-body route

**Captured 2026-11-19.** This is the follow-up to the Inkling head defect recorded in
[Inkling's v1 few-row record](inkling2/v1-few-row-decode.md): the few-row body
(`q4p_dense_rows`) decodes only the **tiled** packed layout, while Inkling packs in the **row
walk's** layout, and `Shared::dense_rows` admitted the copy by shape. That refusal was a runtime
patch; this is the lasting fix, so the same hand-off is now a compile error.

## The type

`sparkflash_ops::packed` now carries the layout in the packed matrix's type:

- `MatrixLayout` (crates/sparkflash-ops/src/packed.rs:110) is a marker trait with two
  implementations, `Walk` and `Tiled`, each naming the runtime `Layout` the kernels still take.
- `Packed<L: MatrixLayout = Walk>` (packed.rs:218) is one packed matrix in `L`'s layout.
  `Packed<Walk>` and `Packed<Tiled>` are distinct types, and the fields are private, so a copy
  of one layout cannot be minted as the other.
- `PackedSet<L: MatrixLayout = Walk>` (packed.rs:408) is a set packed in one layout;
  `PackedSet::new` is the walk's, and `PackedSet::tiled` consumes an empty walk set and
  returns a tiled one. `get` returns `Packed<L>`, so the set's type follows every copy it
  holds.
- `gemv` (packed.rs:315) and `gemv4` (packed.rs:346) take `Packed<Walk>` only.
- `Packed<Tiled>::tiled_planes` (packed.rs:291) is the four arguments a tiled body takes, and
  exists only on the tiled type. A walk copy has no such method.
- `PackedSet<Tiled>::body_copy` (packed.rs:464) returns the few-row body's
  `Option<Packed<Tiled>>`; `PackedSet<Walk>::body_copy` (packed.rs:473) is always `None`, and
  is the compile-time statement that a walk set has nothing the tiled body can read.

The runtime `Layout` enum, the packing formats (kernels' byte orders), and every kernel are
unchanged. `Packed::admits` (packed.rs:260) still checks shape, row count and activation
alignment: those are launch facts, not layout.

## The routes

Every few-row and token-body route on every arm, and what its operand is now: the packed BF16
form's layout where the body reads a packed copy, and the code-plane kind where it reads one of
those.

| Arm | Route (site) | Body / kernel it launches | Operand it now requires, and how | Test |
| :-- | :-- | :-- | :-- | :-- |
| Inkling-2 | Decode head, 1-8 rows: `device/chunk.rs:2161`, `device.rs:6369` | `q4p_dense_rows` | Tiled. The copy comes from `Shared::dense`, a `PackedSet<Walk>`, so `body_copy` is `None`; a walk copy is refused with a clean message (`device.rs:1663`) and the head falls back to the row GEMV. | CPU: `sparkflash-model-inkling2 device::tests::the_few_row_head_body_takes_only_the_tiled_packed_copy`; `sparkflash-ops packed::tests::the_body_copy_is_the_tiled_sets_only` |
| Inkling-2 | Dense decode row GEMV / fused four: `device/packed.rs:114`, `:136` | `q4e_gemv_packed`, `q4e_gemv4_packed` | Walk. `packed::gemv`/`gemv4` take `Packed<Walk>`; `Packed<Tiled>` is a type error. | CPU: `sparkflash-ops packed::tests::the_fused_call_lays_out_every_slot`, doctest `packed::gemv` (compile-fail) |
| Inkling-2 | NVFP4/MXFP8 expert rows, 1-8: `device/chunk.rs:400` (`expert_rows_body_fits`), band at `:483` | `q4p_moe_rows_gate_up_capped`, `q4p_moe_rows_down` (and the `_mx8` pair) | Not the packed form: the operand is `ExpertCodes` (NVFP4 or MXFP8), made only by `ExpertBody::codes`, which is `None` for a BF16 band. The same value picks the body's kernel pair, so the kind admitted and the kind launched are one value. Shape still decides the band and the block tiling. | CPU: `sparkflash-model-inkling2 chunk::tests::the_decode_band_takes_the_expert_rows_body_from_one_to_eight_rows` |
| GLM-5-next | Decode BF16 part, 1-8 rows: `device.rs:4149` (`gemv_small`), packed filter `:4172` | `q4e_gemv_tiled_packed_parts` | Tiled. The set is `PackedSet<Tiled>`, its copy is `Packed<Tiled>`, and the launch reads `tiled_planes()` (`:4177`, `:4204`). | GPU: `sparkflash-model-qwen4-exp tests/glm_prefill.rs::the_packed_tiled_gemv_is_the_bf16_ones_bits` (skipped here); type: `Packed<Tiled>::tiled_planes` compile-fail doctest |
| GLM-5-next | FP8 decode part, 1-8 rows: `device.rs:1310` (`fp8_token_body_fits`), launch `:1379` | `q4p_dense_fp8` | Not the packed form: the operand is `CodePlanes<Fp8E4m3>`, made only by `DeviceLinear::fp8_planes` from an E4M3 `Mat` with its scale plane. `TokenRoute::TokenBody` carries the operand to the launch, so a BF16 part has none and keeps the tiled GEMV. Shape still decides the eight-row band, the 128-column K and the part starts. | CPU: `sparkflash-model-glm5-next device::tests::the_fp8_token_body_takes_one_to_eight_rows`, `device::tests::the_decode_switch_picks_the_token_body_or_the_tiled_gemv`; `sparkflash-ops formats::tests::only_a_code_format_offers_its_planes`; doctests `CodePlanes::of` |
| Qwen4-exp | Few-row tile, 1-16 rows: `device/prefill.rs:786` (`dense_rows_launch`), copy at `:804` | `q4p_dense_rows` | Tiled. `self.backend.packed` is `PackedSet<Tiled>`; the copy is `body_copy(...)` -> `Packed<Tiled>` and its arguments come from `tiled_planes()` (`:807`). | GPU: `sparkflash-model-qwen4-exp tests/few_rows.rs::a_packed_copy_gives_the_bf16_rows_bits` (skipped here); CPU: `packed::tests::the_body_copy_is_the_tiled_sets_only` |
| Qwen4-exp | Tiled BF16 decode GEMV: `device.rs:4706` | `q4e_gemv_tiled_bf16` | Not a packed copy: the operand is the `Mat<Bf16>` itself, the only kind that implements `TiledOperand`, so `tiled_operands(matrix, input)` is written on the kind the launch reads. A `Packed` copy and a `Mat<Fp8E4m3>` are not a `TiledOperand` (compile errors, doctests). Shape still decides the 16-byte rows and the whole eight-value K. | CPU: `sparkflash-ops gemv::tests::the_tiled_admission_is_the_typed_operand`; doctests `gemv::TiledOperand` |
| Qwen4-exp | Hyper down/up tiled: `device.rs:5063` | `q4e_hyper_down_tiled`, `q4e_hyper_up_tiled` | Not a packed copy: each weight is its own `Mat<Bf16>` operand, checked against its own K run (`tiled_operand(&mix.down, width)`, `tiled_operand(&mix.up, config.hc_lowrank)`, the inject weight against `width`), so the predicate names the kind at the launch. Shape still decides. | CPU: `sparkflash-ops gemv::tests::the_tiled_admission_is_the_typed_operand` |
| Dory-10 | Few-row body, 1-8 rows: `device.rs:2470` (`Bf16Rows::of`), `dense_rows` `:2496` | `q4p_dense_rows` | The matrix's own BF16 rows, as a type: the body takes `Bf16Rows`, which only `Bf16Rows::of` makes, and that asks `PackedSet::keeps_bf16_rows`. Only a set that packs `Placement::Beside` the rows answers yes; an in-place set offers no operand, and a `Packed` copy is not a `Bf16Rows` at all, so the body can no longer be chosen by shape. Shape still decides the band and the alignment. | CPU: `sparkflash-model-dory device::tests::the_few_row_body_takes_only_rows_a_beside_set_kept`; `sparkflash-ops packed::tests::only_the_beside_set_keeps_the_bf16_rows`; `device::tests::the_decode_band_takes_the_wide_tiles_few_row_body_from_one_to_eight` |
| Dory-10 | Token row GEMV / fused four: `device.rs:2548`, `:2656` | `q4e_gemv_packed`, `q4e_gemv4_packed` | Walk. `packed_for` returns the `DeviceMat.packed` built by a `PackedSet<Walk>`, and `packed::gemv`/`gemv4` take only `Packed<Walk>`. | CPU: `sparkflash-ops packed::tests::the_fused_call_lays_out_every_slot`; doctests |
| Dense | Token projection: `lib.rs:2630` (`project_bf16`), packed call `:2646` | `q4e_gemv_packed` | Walk. `self.packed` is `PackedSet<Walk>`; the call is `packed::gemv`, which takes only `Packed<Walk>`. The arm declares no tensor-core token body. | CPU: ops packed tests; doctests |

## What remains shape-only, and why

No route now hands a packed copy to a body whose layout it has not proved. The remaining
shape-only decisions are:

- **`Packed::admits`** (crates/sparkflash-ops/src/packed.rs:260), called by the GLM tiled GEMV
  (`gemv_small`, device.rs:4172) and the Qwen few-row body (`dense_rows_launch`,
  device/prefill.rs:805). It filters columns, row count and 16-byte activation alignment. Shape
  is a launch fact; the layout is already a type, so `admits` choosing a packed launch can no
  longer choose the wrong layout. It decides only packed-vs-BF16.
- **Dory's few-row route** (device.rs:2483 `dense_rows_fits`, used at `:2581`, `:2642`): the
  body's operand is now `Bf16Rows`, made only by `Bf16Rows::of`, which asks
  `PackedSet::keeps_bf16_rows`. The shape rule then decides a band and an alignment over rows the
  set proved it left in place; an in-place Dory, or one handing `DeviceMat.packed` to the body,
  has no operand to pass and keeps the row GEMV. What remains shape-only is the eight-row band
  and the 8/32-column alignment, which are launch facts.
- **Inkling's body-vs-GEMV switch** (device.rs:1653-1670) still reads the runtime `Layout`
  (`dense_rows_input`, device.rs:757) to choose between the body, the BF16 rows and the row
  GEMV. That is a routing record, not an admission of a copy: the copy the body takes is typed
  `Option<Packed<Tiled>>` and comes only from `body_copy`.
- **FP8/NVFP4/MXFP8 few-row bodies** (GLM's FP8 decode part, Qwen's routed-expert tables,
  Inkling's expert rows): their operands are code planes plus scales, never the packed BF16 form,
  so the packed layout does not apply. The operand kind is a type instead: `CodePlanes<F>` in
  `sparkflash_ops::formats` (GLM's FP8 part), `ExpertCodes` (Inkling's band), and Qwen's
  `ExpertOperand`, which carries the table with its code format. What remains shape-only is the
  band and the block tiling, which are launch facts of a known-format operand.
- **`tiled_operand` / `tiled_operands`** (Qwen device.rs:4706, 5063): now a predicate over a
  typed operand (`TiledOperand`: `Mat<Bf16>`, `GpuMatE4m3`, `GpuMatE4m3Tile`), so the kind is
  named where the launch is written and a packed copy cannot be presented. What remains shape-only
  is the 16-byte alignment and the whole eight-value K run, which are launch facts of a
  known-format operand.

## Tests

The wrong-layout hand-offs are refused at compile time, and the refusal is pinned:

- `cargo test -p sparkflash-ops` runs
  `packed::tests::the_body_copy_is_the_tiled_sets_only`, which builds a walk set and a tiled
  set holding the **same shape at the same address**, checks that only the tiled set offers the
  body a copy, and that the walk set's copy is a `Packed<Walk>`.
- Its doctests are the compile-fail cases: `packed::gemv` (`Packed<Tiled>` where
  `Packed<Walk>` is required) and `Packed<Tiled>::tiled_planes` (`Packed<Walk>` has no such
  method). A build that loses the type parameter makes one of them compile and fail the doctest.
- `cargo test -p sparkflash-model-inkling2 --lib` keeps
  `device::tests::the_few_row_head_body_takes_only_the_tiled_packed_copy`, which pins the
  three-way routing record.
- `cargo test -p sparkflash-model-dory --lib` runs
  `device::tests::the_few_row_body_takes_only_rows_a_beside_set_kept`: a matrix under a
  `Placement::Beside` set has a `Bf16Rows` operand and takes the body, and the same matrix under
  a `Placement::InPlace` set has none, so the route cannot reach the body by shape.
  `packed::tests::only_the_beside_set_keeps_the_bf16_rows` pins the set's answer in ops.
- `cargo test -p sparkflash-model-glm5-next --lib` keeps
  `device::tests::the_fp8_token_body_takes_one_to_eight_rows` (a BF16 weight has no FP8 operand,
  so the body's rule is not reached) and
  `device::tests::the_decode_switch_picks_the_token_body_or_the_tiled_gemv` (the BF16 part keeps
  the tiled GEMV under both switch states). `sparkflash-ops
  formats::tests::only_a_code_format_offers_its_planes` pins the plane pair, and the doctests on
  `CodePlanes::of` are the compile-fail cases: `CodePlanes<Bf16>` does not exist and a
  `Packed<Walk>` is not a matrix.
- `cargo test -p sparkflash-model-inkling2 --lib` keeps
  `chunk::tests::the_decode_band_takes_the_expert_rows_body_from_one_to_eight_rows`, extended to
  check `ExpertBody::Bf16.codes()` is `None` (a BF16 band has no operand, so the body is not
  reachable by shape) and that NVFP4 and MXFP8 produce their own.
- `cargo test -p sparkflash-model-qwen4-exp --lib` keeps
  `device::tests::the_expert_operand_carries_its_code_format`: an NVFP4 table yields the
  per-tensor scale pair and an FP8 table yields only its table, so the token body's launch reads
  the pointers of the format it was admitted for.
- `cargo test -p sparkflash-ops --lib` runs
  `gemv::tests::the_tiled_admission_is_the_typed_operand`, which pins the 16-byte and eight-value
  rule over one typed operand. The doctests on `gemv::TiledOperand` are the compile-fail cases: a
  `Packed<Tiled>` copy, and a `Mat<Fp8E4m3>` where the BF16 tiled GEMV's operand is required.

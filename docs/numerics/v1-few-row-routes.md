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

Every route that offers a packed BF16 copy to a few-row or token body, on every arm:

| Arm | Route (site) | Body / kernel it launches | Layout it now requires, and how | Test |
| :-- | :-- | :-- | :-- | :-- |
| Inkling-2 | Decode head, 1-8 rows: `device/chunk.rs:2161`, `device.rs:6369` | `q4p_dense_rows` | Tiled. The copy comes from `Shared::dense`, a `PackedSet<Walk>`, so `body_copy` is `None`; a walk copy is refused with a clean message (`device.rs:1663`) and the head falls back to the row GEMV. | CPU: `sparkflash-model-inkling2 device::tests::the_few_row_head_body_takes_only_the_tiled_packed_copy`; `sparkflash-ops packed::tests::the_body_copy_is_the_tiled_sets_only` |
| Inkling-2 | Dense decode row GEMV / fused four: `device/packed.rs:114`, `:136` | `q4e_gemv_packed`, `q4e_gemv4_packed` | Walk. `packed::gemv`/`gemv4` take `Packed<Walk>`; `Packed<Tiled>` is a type error. | CPU: `sparkflash-ops packed::tests::the_fused_call_lays_out_every_slot`, doctest `packed::gemv` (compile-fail) |
| Inkling-2 | NVFP4/MXFP8 expert rows, 1-8: `device/chunk.rs` band | `q4p_moe_rows_gate_up_capped`, `q4p_moe_rows_down` (and the `_mx8` pair) | Not this form: the operands are NVFP4/MXFP8 code planes, not packed BF16. The band is chosen by shape and precision, as before. | Existing device tests (GPU, not run here) |
| GLM-5-next | Decode BF16 part, 1-8 rows: `device.rs:4149` (`gemv_small`), packed filter `:4172` | `q4e_gemv_tiled_packed_parts` | Tiled. The set is `PackedSet<Tiled>`, its copy is `Packed<Tiled>`, and the launch reads `tiled_planes()` (`:4177`, `:4204`). | GPU: `sparkflash-model-qwen4-exp tests/glm_prefill.rs::the_packed_tiled_gemv_is_the_bf16_ones_bits` (skipped here); type: `Packed<Tiled>::tiled_planes` compile-fail doctest |
| GLM-5-next | FP8 decode part, 1-8 rows | `q4p_dense_fp8` | Not this form: E4M3 codes plus scales, no packed BF16 copy. | Existing device coverage |
| Qwen4-exp | Few-row tile, 1-16 rows: `device/prefill.rs:786` (`dense_rows_launch`), copy at `:804` | `q4p_dense_rows` | Tiled. `self.backend.packed` is `PackedSet<Tiled>`; the copy is `body_copy(...)` -> `Packed<Tiled>` and its arguments come from `tiled_planes()` (`:807`). | GPU: `sparkflash-model-qwen4-exp tests/few_rows.rs::a_packed_copy_gives_the_bf16_rows_bits` (skipped here); CPU: `packed::tests::the_body_copy_is_the_tiled_sets_only` |
| Qwen4-exp | Tiled BF16 decode GEMV: `device.rs:4665` | `q4e_gemv_tiled_bf16` | No packed copy: the set is beside the BF16 rows, and this launch reads `matrix.data()`. Shape decides it, and shape is all it needs. | Existing coverage |
| Qwen4-exp | Hyper down/up tiled: `device.rs:5020` | `q4e_hyper_down_tiled`, `q4e_hyper_up_tiled` | No packed copy (BF16/FP8 rows), shape decides. | Existing coverage |
| Dory-10 | Few-row body, 1-8 rows: `device.rs:2544` -> `dense_rows` `:2462` | `q4p_dense_rows` | No packed copy is handed: the launch leaves the packed argument slots zero, so the body reads the matrix's BF16 rows. Dory packs `Placement::Beside`, so those rows are intact. Shape alone decides; see "What remains shape-only". | Existing device path; `dense_rows_fits` CPU-reachable shape rule |
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
- **Dory's few-row route** (device.rs:2450 `dense_rows_fits`, used at `:2544`): the body is
  chosen by shape alone, with no packed copy handed, so it reads the BF16 rows. This is safe
  only because Dory packs `Placement::Beside`; an in-place Dory, or one that starts handing
  `DeviceMat.packed` to the body, must route through `PackedSet::body_copy` (and so prove the
  tiled layout). Recorded as the one route whose admission is still shape-only and cannot be
  typed until it has a tiled set.
- **Inkling's body-vs-GEMV switch** (device.rs:1653-1670) still reads the runtime `Layout`
  (`dense_rows_input`, device.rs:757) to choose between the body, the BF16 rows and the row
  GEMV. That is a routing record, not an admission of a copy: the copy the body takes is typed
  `Option<Packed<Tiled>>` and comes only from `body_copy`.
- **FP8/NVFP4/MXFP8 few-row bodies** (GLM `q4p_dense_fp8`, Qwen's `q4p_moe_rows_*` FP8
  pair, Inkling's expert rows): their operands are code planes plus scales, never the packed
  BF16 form, so there is no layout to type here.
- **`tiled_operands`** (Qwen device.rs:4665, 5020): an alignment/shape predicate over BF16 and
  FP8 matrix pointers, not over packed copies.

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

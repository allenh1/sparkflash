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

# The parts catalogue

`sparkflash_model::parts::built` answers which parts a *build* claims. The catalogue in
`sparkflash_ops::parts` answers what that claim means: for every model-side
[`Part`](../crates/sparkflash-model/src/parts.rs) it names a target, a shared kernel image and
the image symbols that implement the part.

| item | meaning |
| :-- | :-- |
| `PartKernel` | one `(part, target, image)` implementation |
| `CATALOGUE` | every entry this tree declares |
| `kernels(part, target)` | the entries for one part on one target, in catalogue order |
| `carried(target)` | the parts with at least one entry for the target |
| `image_available(part, target)` | whether any entry's image is embedded in this build |
| `missing(architecture, target)` | `architecture.parts()` minus `carried(target)` |
| `loadable(target, lookup)` | the parts with an image this process can load |
| `PartKernel::symbols` | one `KernelSymbol` per exported name, with its own precision and pass |

## One entry per `(part, image)`

There is exactly one catalogue entry per `(part, target, image)`. An operation whose shape
rule is different from the block it belongs to gets its own `Part` identity rather than a second
entry for the same image: `DenseGemm` beside `DenseMlp`, `MoeRouter`,
`MoeCombine` and `GroupedMoeGemm` beside `Moe`, `Activation`,
`Rope`, `Quantize` and `LayerNorm` beside the block that launches them, and
`Collective` and `MoePlan` for the shared plans every arm loads. This is what makes the
shared token combine reachable at a GLM or DeepSeek width: the MoE block's 2560/512/10 rule no
longer applies to a combine that sums a runtime top-k over any width.

## Per-symbol precision and pass

An image is not one arithmetic and one pass. The token image exports a bf16 GEMM beside an FP8
quantiser, and the decode image exports a decode body beside a prefill projection. Every
`KernelSymbol` therefore carries:

* `name` - the exported symbol the driver loads;
* `precisions` - a `PrecisionMask` of the operand classes that symbol accepts;
* `phase` - `Both`, `Prefill` or `Decode`.

The resolver selects a symbol only when that symbol's own mask contains the requested precision
and its own phase serves the requested pass. `q4e_gemm_bf16` cannot satisfy an NVFP4 request
just because `q4e_quantize_e4m3_rows` lives in the same image, and a prefill projection cannot
satisfy a decode request. The catalogue's tests require every symbol to declare a non-empty,
non-universal precision mask and to appear in its image's exported-name registry.

## Canonical implementations

`PartKernel::canonical` marks the implementation a target brings up first. There is exactly
one canonical entry per `(part, target)`, it is listed first, and the resolver scores it ahead
of the others rather than merely trusting catalogue order. **DeepSeek remains canonical for every
part it implements**: MLA and CSA2 attention, compression, the sparse indexer, MoE and routed
experts, hyper-connection, Engram, MTP, embedding, norm and the LM head. A Qwen or GLM entry for
the same part is an additional implementation, never the default. A part DeepSeek does not
implement names its own canonical image.

## Resolution

`sparkflash_ops::KernelResolver` maps a `PartRequirement` to one catalogue entry and one
of its symbols as a pure function of `Part`, `Target`, `PartGeometry`,
`Precision`, `KernelPhase` and the availability policy. It never reads `model_type`
or any other model name. A request may name the symbols its arithmetic needs; those are a
priority list, and the selected symbol is always one of them.

`ShapeConstraint` is the real rule, not a hint. A rule that names a field is a proof
obligation: if the geometry leaves that field at zero, the entry is refused. Equality is not
skipped when a plan does not know a dimension; a plan that does not know the dimension cannot
resolve the specific entry. The `engine_plan` tests mutate every `model_type` to an
unregistered string and require the resolved kernel list to be identical.

## Availability

"the catalogue names a symbol" and "this process can load it" are different facts, so the policy
is explicit:

| policy | accepts |
| :-- | :-- |
| `Catalogue` | every entry (planning and CPU tests) |
| `Embedded` | only images this binary embedded |
| `Runtime { cubin_dir }` | embedded images, the RoCEnante runtime image, and the DeepSeek PTX set when the cubin directory is present |

`Availability::loadable` delegates to the same `parts::image_loadable` predicate the
catalogue's `loadable` uses, so the two answers cannot disagree about an image.

## Images

Every entry uses one of the existing `sparkflash_kernels::images::Image` variants:

| image | contents | embedded? |
| :-- | :-- | :-- |
| `Qwen4Exp` | Qwen3.8 token, norm, dense and hyper kernels, and the shared tiled decode GEMV body | yes, when nvcc built the cubin |
| `Qwen4Attn` | Qwen3.8 attention, sparse indexer and KV-cache token kernels | yes, when nvcc built the cubin |
| `Qwen4Moe` | Qwen3.8 decode/token routed-expert (MoE) kernels | yes, when nvcc built the cubin |
| `Qwen4Prefill` | Qwen3.8 tensor-core bulk prefill | yes, when nvcc built the cubin |
| `Qwen4Scan` | Qwen3.8 Gated DeltaNet and Kimi Delta Attention scan | yes, when nvcc built the cubin |
| `Dflash2` | DFlash 2 block drafter | yes, when nvcc built the cubin |
| `GlmPrefill` | GLM-5.3 block-scaled dense prefill and skinny projections | yes, when nvcc built the cubin |
| `GlmMoe` | GLM-5.3 bulk routed-expert (MoE) plan, MMA, SwiGLU and q8 kernels | yes, when nvcc built the cubin |
| `GlmDecode` | GLM-5.3 decode/token kernels: mHC mixers, router, indexer, LayerNorm, absorbed MLA and the tiled decode GEMV wrappers | yes, when nvcc built the cubin |
| `GlmAttention` | GLM-5.3 sparse MLA attention and per-head projections | yes, when nvcc built the cubin |
| `MoePlan` | the routed-expert planning kernel every arm shares | yes, when nvcc built the cubin |
| `Deepseek` | the shared PTX set the DeepSeek arm loads at run time | **no** |
| `Rocenante` | the one-shot collective | compiled at run time |

`Image::Deepseek` reports no embedded bytes: the DeepSeek arm loads those kernels from the
shared PTX set rather than from an `include_bytes!` image. It is still a real kernel set and a
canonical DeepSeek entry still names real symbols. Because of this, `image_available` can be
false for a part that `carried` reports, which is why the two functions are separate.

The image enum keeps its historical variant names for symbol loading. The catalogue and the
shared plans name the shared variants by their functional domain through associated constants:
`Image::TOKEN` (`Qwen4Exp`), `Image::ATTENTION` (`Qwen4Attn`),
`Image::ROUTED_EXPERTS` (`Qwen4Moe`), `Image::PREFILL` (`Qwen4Prefill`),
`Image::SCAN` (`Qwen4Scan`), `Image::BLOCK_PREFILL` (`GlmPrefill`),
`Image::SPARSE_MLA` (`GlmAttention`), `Image::BLOCK_ROUTED_EXPERTS`
(`GlmMoe`), `Image::DECODE` (`GlmDecode`) and `Image::DRAFTER`
(`Dflash2`). Each alias is the same value as its variant, so the driver still loads the same
cubin.

Only `Target::Sm121` has entries: every source under `kernels/` is named `*_sm121`.
`carried(Target::Sm100)` and `carried(Target::Sm120)` are empty and `missing`
names every part a checkpoint needs.

## Shapes must be the kernel's shapes

The constraints are the ones the kernels were compiled for, not the ones a checkpoint happens to
name:

* DeepSeek's b12x MXFP4 MoE body is compiled for 576 real columns, the TP4 rank-local width.
  `Part::Moe` and `Part::GroupedMoeGemm` declare `intermediate_equals(576)`; TP1
  and TP2 are an explicit refusal, not a resolution at 2304 or 1152.
* The shared GDN scan is compiled for the `(64, 64)`, `(128, 128)` and `(128, 256)` key/value
  templates; the Qwen device selects `q4p_gdn_scan_64_64`/`q4p_gdn_scan_128_256` for the two
  that are not the released body, so the catalogue accepts exactly those pairs. KDA is
  instantiated at all three but the GLM host selects only the released `(128, 128)` body, so
  `Part::KdaScan` claims only that pair. Anything else is a refusal, not a launch.
* The shared vision tower serves a 14x14 patch. `opt_vision_sm121.cu` also exports a 16-patch
  `opt_vision_patchify_16`, but the Qwen tower's orchestration and weight loading are still
  its own arm's (D84), so the catalogue does not claim a 16-patch tower and the Qwen3.8
  document that asks for one is an explicit refusal.
* The genuine Qwen FP8 grouped prefill bodies (`q4p_moe_gate_up_fp8`, `q4p_moe_swiglu_fp8`,
  `q4p_moe_down_fp8`) are per-symbol FP8 and prefill-only. `q4p_moe_swiglu_fp8` reads the
  gate/up tile's `float` rows, so its Activation entry also claims `F32`. `q4p_moe_gate_up`/
  `q4p_moe_down` are the NVFP4 grouped bodies and claim `NvFp4` only. The token FP8 experts
  (`q4e_expert_*_fp8`) serve both passes, because a short batch takes them too.
* The DeepSeek dense projection's b12x MXFP8 body is generic over the hidden width, so the
  `DenseGemm` entry is not pinned to 5120; 2560 resolves the same body.

## Adding an entry

1. Add the kernel source under `kernels/` and expose its image through
   `sparkflash_kernels::images` if it is a new image.
2. Add one `sm121(...)` entry per `(part, image)` pair, with the real exported symbol
   names, each symbol's own precision mask and pass, and the `ShapeConstraint` it was proven
   at. Add a `Part` variant if the operation needs a different shape rule.
3. Keep a DeepSeek entry canonical and first for any part DeepSeek implements.
4. Run `cargo test -p sparkflash-ops --lib parts` and
   `cargo clippy -p sparkflash-ops --all-targets -- -D warnings`.

The module tests require every part to have at least one sm121 entry, exactly one entry per
`(part, image)`, exactly one canonical entry per part, DeepSeek as canonical wherever it
appears, every symbol in its image's exported-name registry, no sm100/sm120 entries, and an empty
`missing` list for the DeepSeek V4.1, GLM-5.3 and Qwen3.8 configs on sm121.

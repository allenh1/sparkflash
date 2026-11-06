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

## Canonical implementations

`PartKernel::canonical` marks the implementation a target brings up first. There is exactly
one canonical entry per `(part, target)`, and it is listed first. **DeepSeek remains canonical
for every part it implements**: MLA and CSA2 attention, compression, the sparse indexer, MoE and
routed experts, hyper-connection, Engram, MTP, embedding, norm and the LM head. A Qwen or GLM
entry for the same part is an additional implementation, never the default.

A part DeepSeek implements therefore starts with an `Image::Deepseek` canonical entry; the
other model images follow. A part DeepSeek does not implement (grouped-query attention, Gated
DeltaNet, KDA, the dense MLP, the n-gram PLE, manifold hyper-connections) names its own
canonical image: grouped-query attention names `Image::Qwen4Attn`, Gated DeltaNet and KDA
name `Image::Qwen4Scan`, and manifold hyper-connections name `Image::GlmDecode`.

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
| `GlmDecode` | GLM-5.3 decode/token kernels: mHC mixers, router, indexer, LayerNorm, absorbed MLA and the tiled decode GEMV wrappers (BF16, block-scaled FP8 and routed-expert) | yes, when nvcc built the cubin |
| `GlmAttention` | GLM-5.3 sparse MLA attention and per-head projections | yes, when nvcc built the cubin |
| `Deepseek` | the shared PTX set the DeepSeek arm loads at run time | **no** |
| `Rocenante` | the one-shot collective | compiled at run time |

`Image::Deepseek` currently reports no embedded bytes: the DeepSeek arm loads those kernels
from the shared PTX set rather than from an `include_bytes!` image. It is still a real kernel
set and a canonical DeepSeek entry still names real symbols. Because of this,
`image_available` can be false for a part that `carried` reports, which is why the two
functions are separate.

Only `Target::Sm121` has entries: every source under `kernels/` is named `*_sm121`. A
target is not claimed for a part until its kernels are actually compiled for that target, so
`carried(Target::Sm100)` and `carried(Target::Sm120)` are empty and `missing` names every
part a checkpoint needs.

## Adding an entry

1. Add the kernel source under `kernels/` and expose its image through
   `sparkflash_kernels::images` if it is a new image.
2. Add one `sm121(...)` entry per `(part, image)` pair, with the real exported symbol
   names. A part may use several images; list each pair once.
3. Keep a DeepSeek entry canonical and first for any part DeepSeek implements.
4. Run `cargo test -p sparkflash-ops --lib parts` and
   `cargo clippy -p sparkflash-ops --all-targets -- -D warnings`.

The module tests require every part to have at least one sm121 entry with non-empty symbols,
exactly one canonical entry per part, DeepSeek as canonical wherever it appears, no sm100/sm120
entries, and an empty `missing` list for the DeepSeek V4.1, GLM-5.3 and Qwen3.8 configs on
sm121.

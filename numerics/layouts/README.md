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

# Expert layouts

Each file is the routed-expert layout a coverage dump is judged against (docs/NUMERICS.md §5),
schema `sparkflash-numerics-expert-layout/1`. `layers` is a list of contiguous spans:
`{"first": L, "count": N, "experts": E}` describes layers `L .. L+N` with `E` routed experts
each (`E = 0` for a dense layer). `draft_layers` records the checkpoint's MTP/draft layers; the
coverage command does not judge them yet.

The counts come from the released configuration named in each file's `source`:

- `qwen38.json`: `crates/sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json`,
  48 decoder layers x 512 routed experts.
- `glm53.json`: `models/configs/glm53_flash.hf.json`, 45 decoder
  layers; the first 3 are dense and the remaining 42 have 288 routed experts.
- `deepseek-v41.json`: `crates/sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json`,
  40 decoder layers x 384 routed experts.

Inkling-2 and Dory1.0 have no layout yet; a coverage command for them refuses with the model
named rather than guessing one.

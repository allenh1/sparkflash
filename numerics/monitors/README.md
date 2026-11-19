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

# Monitor dumps

A monitor dump is one probe's sampled values from one build: per MoE layer, the layer's hidden
sample and one sample per expert the probe routed to. A pair is a reference build's dump against a
candidate build's dump. `scripts/numerics.py monitors` compares them per layer and per expert
(docs/NUMERICS.md §5).

## The committed pair

The first real pair is here:

- `glm53-tp4-v1-reference.json` — GLM-5.3 at world 4 with `SPARKFLASH_DECODE_MMA=off`, whose
  decode runs the tiled row GEMV (version 1's arithmetic).
- `glm53-tp4-v2-candidate.json` — the shipped default, the FP8 few-row body (version 2).

Both ran the probe `decode-position-20` at position 20 with four rows, sampled the same 8 of 4096
hidden values, and hold the same 42 MoE layers.

```bash
python3 scripts/numerics.py monitors \
  --reference numerics/monitors/glm53-tp4-v1-reference.json \
  --candidate numerics/monitors/glm53-tp4-v2-candidate.json \
  --layer-bar 0.5 --expert-bar 0.5
```

The pair passes. Its achieved maxima are **3.158e-01** per layer and **3.511e-01** per shared
expert. The bars are a **regression guard, not a quality claim**: they sit above the measured
maxima so a later pair that moves further fails, and they say nothing about whether the candidate's
arithmetic is good enough to ship. `numerics/registry.toml` names the pair and these bars in
GLM-5.3's `[[monitors]]` row, and `scripts/numerics.py ci` judges every committed row. The
aggregate `[prerequisites] monitors` flag is false because Qwen3.8, Inkling-2 and Dory1.0 have no
row; a true flag requires a passing row for every captured logprobs model.

## What a routing divergence means

The router's top-k is not arithmetic. A slightly different hidden state can put a row's maximum on
a different expert, so the candidate may hold an expert the reference did not, and the reference
may hold one the candidate did not. That is a measured fact about the candidate — the candidate
routed that row to a different expert — and it is a real quality signal, not a malformed pair. The
judge intersects each layer's expert ids, computes the expert error only over the experts both
dumps hold, and counts and prints the rest. The exit code follows the bars alone.

In this pair, **7 of 42 layers** routed one row to a different expert: **14 experts are unshared**
(7 reference-only, 7 candidate-only) and **329 are shared**.

| layer | reference only | candidate only |
| --: | --: | --: |
| 16 | 199 | 158 |
| 20 | 181 | 14 |
| 23 | 87 | 255 |
| 32 | 205 | 56 |
| 40 | 255 | 44 |
| 41 | 92 | 287 |
| 42 | 64 | 119 |

The printed report carries a shared/unshared count beside each layer and names each differing
layer's ids; `--json` carries the same under `routing` (`layers_differing`, `experts_unshared`,
`reference_only`, `candidate_only`, `experts_shared`, `per_layer`) and the achieved maxima under
`maxima`.

Do not add a dump that was not produced by a serving run; a synthetic fixture belongs in the
self-tests, not here.

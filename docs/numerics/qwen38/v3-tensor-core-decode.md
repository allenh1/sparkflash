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

# Qwen3.8-Flash-Next, world 1, numerics version 3: the tensor-core decode returns

**Registry state: version 3, on version 1's own reference
(`references/qwen38-tp1-sm121a.json`, `37578a09…`).** Version 2 is retired with its
reference (`references/qwen38-tp1-sm121a-v2.json`, `9e122aa7…`) and the reason "breaks
resumed = fresh". World 2 and world 4 are restored at version 1 on their existing captures
(`d1d546ae…`, `007e7046…`); nothing was recaptured.

## Why version 2 could not stand

Version 1's reason for decoding on the tensor cores was invariance, not speed: every row count
from a one-row step to a 4,096-row chunk ran the tile's arithmetic, so a continued conversation
held the bits of a cold prefill of its transcript (`v1-tc-decode.md`,
`docs/NUMERICS.md` §1). Version 2 put the decode steps back on the token kernels for the 2.45%
the speed rule measured at one lane. Those kernels match the tile to a tolerance, not to its bits
(`device/prefill.rs`: "a chunk here matches the token kernels to a tolerance rather than bit for
bit"), so the invariance no longer held.

The test that pins it is
`device_parity::a_continued_conversation_is_the_bits_of_a_cold_prefill`. It now builds its plan
from the shipped profile (the chunk pinned to 512 rows so the prompts cross a boundary) instead
of constructing `PrefillPlan::new(CHUNK, true)`. At version 2's default, on a GB10 at
`sm_121a`, it fails on the 73-token prompt:

```text
73-token prompt, the next turn: 64 of 64 logits differ, the first at 0 (0.2048311 against 0.2048319)
```

Version 1's own result on the same test was bit-exact, and the test is why the tensor-core decode
was there.

## The speed measurement, and what it now decides

One build (`18a76075`, binary `102d216d`), the declared world-1 configuration, context
65,536, one process, fresh boots alternating `SPARKFLASH_DECODE_MMA`:

| boot | route | tok/s | per stream |
| :-- | :-- | --: | --: |
| on 1 | tensor-core decode | 35.82 | 37.25 |
| off 1 | token kernels | 36.81 | 38.28 |
| off 2 | token kernels | 36.77 | 38.25 |
| on 2 | tensor-core decode | 35.96 | 37.41 |

The token kernels are 2.45% ahead at the arm's default one lane. The rule now says it compares
only routes that both meet §1's invariances; this pair does not, so the number is a measurement of
the route version 2 served, not a reason to keep it. The tensor-core route stays the served one.

## What changed

- `crates/sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.toml`: `[decode]
  tensor_cores = true`, and the code default follows it (`unwrap_or(true)`).
- The arm's ledger row and the walk's declaration state the few-row twins of the tile again.
- Qwen leaves the shared `decode.tensor_cores = true` refusal; the refusal still holds for
  Inkling, GLM and Dory, whose token kernels are not the tile's twins.
- The registry moves world 1 to version 3 on version 1's card, retires version 2 with its card,
  and restores worlds 2 and 4 on their version-1 captures.

## What is not re-measured

Version 3 is version 1's arithmetic, so it is version 1's reference `37578a09…`, its anchor
`8c4ccd63…` and its floor (`v1-tp-floor.md`) unchanged. The version-2 capture
(`numerics/captures/qwen38/v2-token-kernels.json`) and its 982-position precision-tail floor
stay as the record of the route that was tried. Open: Qwen still has no `[[monitors]]` row.

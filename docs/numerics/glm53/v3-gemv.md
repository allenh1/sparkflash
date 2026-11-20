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

# GLM-5.3-Flash, numerics version 3: back to the row GEMV

**Registry state: version 3, on the reference version 1 declared**
(`numerics/references/glm53-tp4-sm121a.json`, `827ec5f6…`, captured 2026-11-05). Version 2 is
retired with "the FP8 few-row body is slower in both measured pairs; version 3 returns to the row
GEMV"; its reference file stays in history and the committed monitor pair still guards it.

Version 2 moved a decode step's block-scaled FP8 projections onto the shared tile's few-row body
(`glm_decode_fp8b128_rows`). It carried a real quality shift (+1.373e-01 mean NLL, 93.38% top-1)
and the owner approved it as the tensor-core decode change. Its report left the speed A/B
unmeasured. The written speed rule (docs/NUMERICS.md, *The speed rule*) now decides it: keep the
route only when, on alternated fresh-boot pairs, the served concurrency's aggregate tok/s is at
least +2% better and single-request tok/s is no more than 3% worse, both pairs agreeing. GLM's
served concurrency is four lanes, drafting as served (DFlash 2).

## What was measured

One build (`18a76075`, binary `102d216d`), the same checkpoint, tokenizer, converted weights,
context 1,048,576, lanes 4, DFlash 2, alternating `SPARKFLASH_DECODE_MMA` across four fresh
boots: ON, OFF, OFF, ON. Each boot: 32-token warm-up, then
`bench_serving.py --levels 1,4 --tokens 256 --rounds 3`; medians of the rounds.

| boot | route | L1 tok/s | L4 tok/s | accept L1/L4 |
| :-- | :-- | --: | --: | --: |
| on1 | FP8 few-row body | 42.06 | 56.22 | 0.59 / 0.70 |
| off1 | tiled row GEMV | 46.20 | 61.50 | 0.62 / 0.71 |
| off2 | tiled row GEMV | 46.35 | 61.42 | 0.62 / 0.71 |
| on2 | FP8 few-row body | 41.78 | 47.93 | 0.59 / 0.70 |

The ON/OFF ratios, where the rule needs four-lane at least 1.02 and single at least 0.97 in both
pairs:

| pair | single | four-lane |
| :-- | --: | --: |
| on1 vs off1 | 0.9104 | 0.9141 |
| off2 vs on2 | 0.9014 | 0.7804 |

The OFF boots agree to 0.3% (L1) and 0.1% (L4), so the reference side is stable; the ON side's
clean rounds agree to 0.7% (L1) and 0.25% (L4). The second ON boot's four-lane median was
depressed by an outside deploy loop (two rounds at 42.34 and 47.93 tok/s); its cleanest round,
56.08 tok/s, is still 0.912 of the OFF median, so the verdict does not depend on that load.

**The FP8 few-row body is slower on both axes in both pairs.** It misses the +2% four-lane bar by
about nine points and the -3% single-request guard by about six. The quality cost buys nothing.

## What changed

- `crates/sparkflash-model-glm5-next/src/device.rs`: the code default for `decode.tensor_cores`
  is off (`unwrap_or(false)`); the shipped `glm53_flash.toml` still leaves the field unset, so the
  served default is the tiled row GEMV.
- `crates/sparkflash-model-glm5-next/src/lib.rs`: `TOKEN_DECODE` states the row GEMV, and the
  serving profile check refuses `decode.tensor_cores = true` with that reason. The ledger row and
  the check are the same constant, as before.
- The FP8 few-row body is still in the catalogue and on the device path; a build that wants to
  measure it again flips that one constant (as this window did with the switch, before the flip).

Version 3's arithmetic is version 1's: the v2 report's separation capture showed
`SPARKFLASH_DECODE_MMA=off` reproduces version 1's reference exactly (0 of 997 positions differ,
top-1 100%), and this window's OFF boots are that route. The reference named here is therefore
version 1's own card, not a recapture.

The declared quality distance of version 2 (+1.373e-01 mean NLL, 93.38% top-1 against version 1's
anchor) is now history: the served arithmetic is the one the anchor was captured on.

## Reading

GLM is back on the arithmetic its anchor and version 1 describe, with the version 2 body recorded
as measured-and-held. The committed monitor pair (`glm53-tp4-v1-reference.json` against
`glm53-tp4-v2-candidate.json`) still passes at its bars and remains the evidence that the body is
a small, routed-expert-local perturbation; it is no longer the served default. No open speed item
remains for GLM.

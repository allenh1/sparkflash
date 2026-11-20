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

# Qwen3.8-Flash-Next, world 1, numerics version 2: the token kernels

**Registry state: retired.** Version 2 captured on 2026-11-19 on `18a76075` (binary
`102d216d`) with `SPARKFLASH_DECODE_MMA=off`, and served until the resumed = fresh test
refused it. It is retired with its own reference and the reason "breaks resumed = fresh"; version
3 returns world 1 to version 1's tensor-core decode
(`v3-tensor-core-decode.md`). The world-2 and world-4 configurations, retired here with their
tensor-core captures, are restored at version 1 on those same captures.

The speed rule (docs/NUMERICS.md, *The speed rule*) measured the token kernels ahead at the arm's
default one lane, but the rule compares only routes that both meet §1's invariances. This one does
not: the token kernels match the tile to a tolerance, not to its bits.

## What was measured

One build, the declared world-1 configuration (context 65,536, one process), fresh boots
alternating `SPARKFLASH_DECODE_MMA`, one 32-token warm-up then
`bench_serving.py --levels 1 --tokens 256 --rounds 3`:

| boot | route | tok/s | per stream |
| :-- | :-- | --: | --: |
| on 1 | tensor-core decode | 35.82 | 37.25 |
| off 1 | token kernels | 36.81 | 38.28 |
| off 2 | token kernels | 36.77 | 38.25 |
| on 2 | tensor-core decode | 35.96 | 37.41 |

The ratios are **0.9731** and **0.9780**. The single-request guard (no more than 3% worse) passes
in both pairs; the served-concurrency aggregate (the same number here, because the arm's default
lane count is one) is not **+2% better** in either - it is 2.45% worse on the means. The gap has
narrowed from the 2026-11-04 measurement's ~8%, but the rule asks for a win.

## What changed

- `crates/sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.toml`: `[decode]
  tensor_cores = false`, and the code default follows it (`unwrap_or(false)`) so a profile
  without the field also serves the token kernels.
- The arm's ledger row states the row GEMV and the serving profile check refuses
  `decode.tensor_cores = true` with a message citing this report.
- The tensor-core decode code stays; a later route adopts it only by passing the rule.

## The floor

Two free captures are bit-identical. The forced comparison against version 1's reference over 982
positions is a precision tail: mean NLL delta **+8.361e-04**, mean absolute **4.991e-03**, p99
5.759e-02, max **1.109e-01**, top-1 agreement **99.3890%**, mean KL 2.180e-04. The reference's own
mean NLL is **0.2228 a token** over 988 tokens.

The templated known answers all pass: `4`, `Paris`, `READY` (temperature 0, 128 tokens,
`finish_reason: stop`).

## Why it was retired

The test `device_parity::a_continued_conversation_is_the_bits_of_a_cold_prefill`, with its plan
built from the shipped profile, fails at this version's default: on the 73-token prompt all 64
logits differ from a cold prefill of the same transcript (the first 0.2048311 against 0.2048319).
Version 2 was 2.45% faster at one lane and not the same model on resume. Version 3 restores version
1's bits (`v3-tensor-core-decode.md`).

## Reading

Qwen's world-1 decode was returned to the token kernels its prefill path uses, 2.45% faster at one
lane, and then returned to the tile because the speed was bought with the resumed = fresh
invariance. Open: Qwen has coverage, not monitors.

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

**Registry state: version 2, captured on 2026-11-19 on `18a76075` (binary `102d216d`) with
`SPARKFLASH_DECODE_MMA=off`.** It replaces the world-1 version 1 reference, retired with "the
tensor-core decode is 2.45% behind the token kernels at one lane; version 2 returns to the token
kernels". The world-2 and world-4 configurations are retired with their tensor-core captures in
history, because the arm serves world 1 and the profile change is measured there.

The speed rule (docs/NUMERICS.md, *The speed rule*) judged the decode route on the arm's default
one lane. The tensor-core decode is not kept.

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

## Reading

Qwen's world-1 decode is back on the token kernels its prefill path uses, 2.45% faster at one lane,
with the tensor-core route measured and held for the next attempt. The world-2 and world-4
entries are retired with their captures; a re-added world starts at a version above 1 and takes the
same rule. Open: the
coverage/monitor rows the registry still names as missing (Qwen has coverage, not monitors).

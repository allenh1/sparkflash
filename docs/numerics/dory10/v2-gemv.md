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

# Dory1.0-4B, numerics version 2: back to the row GEMV

**Registry state: version 2, captured on 2026-11-19 on the pre-change build `84f07db7`
(binary `4de25a5e`).** It replaces version 1, retired with "the few-row body is 33% slower at one
lane; version 2 returns to the row GEMV"; version 1's reference and history stay.

Version 1 moved Dory's one-to-eight-row projections onto the wide tile's few-row body
(`q4p_dense_rows`). It was declared on determinism and a body-equals-tile device test; the speed
A/B was unmeasured. The written speed rule (docs/NUMERICS.md, *The speed rule*) now decides it, and
the body loses badly.

## What was measured

Two builds, fresh boots, one process on the head at world 1, context 8,192, no drafter, one
32-token warm-up then `bench_serving.py --levels 1 --tokens 256 --rounds 3`:

| boot | route | tok/s | per stream |
| :-- | :-- | --: | --: |
| body 1 | few-row body (`18a76075`, `102d216d`) | 17.77 | 17.95 |
| gemv 1 | row GEMV (`84f07db7`, `4de25a5e`) | 26.19 | 26.65 |
| gemv 2 | row GEMV | 26.15 | 26.60 |
| body 2 | few-row body | 17.55 | 17.73 |

The ratios are **0.679** and **0.671** against a rule that needs 1.02 at the served concurrency
and 0.97 at one request. The causes are visible in the logs: the GEMV build packs 289 dense
projections beside the BF16 rows (23.1% fewer bytes a pass), while the body build reads the BF16
rows. One-process Dory accepts **one lane only** - `--concurrency` and `--fuse-batch` are the
packed-model host driver's knobs and the recurrence group path needs at least two ranks - so the
16-lane half of the rule degenerates to the single-request number here; that is recorded rather
than worked around.

## What changed

The per-arm glue that sent one-to-eight-row steps to `q4p_dense_rows` is gone from
`crates/sparkflash-model-dory/src/device.rs`: `gemv_batch` and `gemv_group` always take the
packed or BF16 row GEMV. The typed operand (`Bf16Rows`), its admission (`dense_rows_fits`) and
the body launch (`dense_rows`) stay, marked test-only, so the next route adopts a checked operand
rather than a shape; the shared kernel binding stays as well.

## The floor

The reference is two free captures, bit-identical, and its forced comparison against version 1's
reference over 1,024 positions is a precision tail: mean NLL delta **-3.062e-05**, mean absolute
**4.962e-04**, p99 6.610e-03, max **1.006e-02**, top-1 agreement **100.0000%**, mean KL
2.586e-06. The reference's own mean NLL is **0.1616 a token** (1,024 tokens over the eight
prompts).

Known answers, through the chat template at 128 tokens: "the capital of France" answers
`Paris`; the arithmetic and `READY` probes returned empty `content` with
`finish_reason: length` because the model spent the whole budget inside `reasoning_content`.
That is recorded in the floor as two failed probes with the explanation, not retried and not waved
through; the probes' reasoning text is in the window report.

## Reading

Dory is back on the arithmetic its ring captures and version 1's *device tests* described, at 33%
more tokens a second, with the body's typed admission kept for a later route. The world-2
recurrence ring remains bit-identical to version 1's reference, so it needs no entry of its own.
Open: the ring's speed against a single process, and the coverage/monitor rows the registry still
names as missing.

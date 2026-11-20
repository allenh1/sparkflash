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

# Dory1.0-4B-BF16, numerics version 1: the decode rows on the wide tile's few-row body

**Registry state: version 1, captured on 2026-11-19 at world 1.** The entry points at
`numerics/references/dory10-tp1-sm121a.json` (`99715d0072eab1e1…`). It is the configuration's
first reference, so there is no earlier reference to move and no bump: version 1 is the
arithmetic this change lands, and the capture pins it. Two captures are bit-identical and a
`check` passed on two boots; the numbers are below.

**The world-4 entry was retired.** The released checkpoint carries two recurrent loops and
`DoryStage::of_rank` refuses a world that does not divide them, so the ring's legal widths are
1 and 2, not 4; the phase-0 registration at world 4 could only have been served by rewriting
`n_recurrent_loops`. The ring also serves under the TP=1 numerics key by design: its rank 0
calls `serve_backend` (world 1) because each rank runs the same kernels on the latent rows it
receives. Both halves of that claim are now measured. `crates/sparkflash-model-dory/tests/recurrence.rs`
holds the ring against the single process on a GPU (eight tests: the loopback ring, concurrent
lanes, the drafting passes, the end-of-turn id and the drafting rollback, each bit for bit), and
the served world-2 ring's capture of 2026-11-19 is **bit-identical to this reference**: 8 of 8
prompts, 1024 of 1024 logprobs, top-1 agreement 100%, and its own teacher-forced replay scores
exactly zero (`numerics/captures/dory10/world2-ring.json` and `-forced.json`). So the ring needs
no separate registry entry: its arithmetic is this one.

## What changed

Dory's projections are BF16 with f32 activations. A decode step, a verify and a few lanes (one to
`TOKEN_ROWS` = 8 rows) now take `q4p_dense_rows`, the shared wide tile's few-row BF16 body,
instead of `q4e_gemv_bf16` / `q4e_gemv4_bf16`. `dory_mma_wide` - the prompt tile - is
`shared_prefill_wide_body`, and the few-row body runs that tile's chain (the f32 activation
split into a bf16 head and the bf16 of its remainder, each 16-column K step's head MMA and then
its remainder, K ascending), so a decode row is a cold prompt row.

- Each projection is one `q4p_dense_rows` launch: the body's chain is per matrix and row, so
  a matrix's output is the bytes its lone launch gives it (the arm's old fused GEMV grouped
  four projections into one launch; the few-row body does not need to fuse to keep the bits).
- The body reads each matrix's BF16 rows. The arm's packed copies stay with the wider row
  GEMV: the catalogue's packed few-row symbol is a separate body, and the packed copy carries
  the same words, so this is a bytes choice and not an arithmetic one.
- A wider chunk keeps `dory_mma_wide` (nine rows up), and a shape the body does not cover
  (whole eight-column writes and a whole 32-column K run) keeps the BF16 row GEMV.
- No body was added: the catalogue's BF16 few-row body serves this precision.

## What was measured before the cluster

- **The decode row is the prompt tile's row.** On the sm_121a device, at every token count from
  1 to 8, `q4p_dense_rows` over the same BF16 weights and f32 activations is bit-equal, row by
  row, to `dory_mma_wide` (and to `glm_prefill_bf16_wide`, the same shared wide body). The
  test is
  `sparkflash-kernels/tests/token_bodies_device.rs::the_bf16_rows_body_is_the_wide_tiles_row`;
  it is the launch the arm's `gemv_batch`/`gemv_group` path makes.
- **The packed body is the BF16 rows.** The same file's packed test holds
  `q4p_dense_packed_rows` to the BF16 tile at 1, 2, 4 and 8 rows, and checks the copy unpacks to
  the BF16 rows word for word; Dory's packed copies are read through it.
- **The routing band.** A CPU test holds the body to one through eight rows and rejects the
  ninth, a shape without whole eight-column writes, a K run that is not whole 32-column blocks
  and an unaligned activation
  (`device::tests::the_decode_band_takes_the_wide_tiles_few_row_body_from_one_to_eight`).

## Measured on the cluster, 2026-11-19

Served as one process at world 1 on the head node from the branch built at `f1f86bf4` (release,
`5a092d5a`), the released snapshot untouched, `/readyz` in 3 s. The server reported its identity
as `world=1 profile=reference plan=tp arch=sm_121a`, status `unregistered` before the capture.

Two captures are bit-identical and `check --reference` passed on two separate boots:

```text
99715d0072eab1e1cef89dbfe990faa2de2b053320c61b25726855aa460e22e3  dory10-tp1-v1-a.json
99715d0072eab1e1cef89dbfe990faa2de2b053320c61b25726855aa460e22e3  dory10-tp1-v1-b.json
```

The lanes probe was partial - it was stopped by the window's time budget - but the numbers it
did produce were 16.6 tok/s alone and 16.6 tok/s aggregate for four at once, with all four lanes
bit-identical to alone.

There is **no served delta** to report, and that is the honest shape of a first reference. The
measurement of this change against the row-GEMV decode would need a build that still runs the old
rows, which no longer exists on this branch.

## What is still not measured

- **The change against the row-GEMV arithmetic.** The body is the tile's row (the device test
  above), but how far the served logits move from the GEMV's was never captured.
- **The speed A/B.** Measured on 2026-11-19: the few-row body is **33% slower** than the row
  GEMV, not flat. See *The speed rule* below.
- **The sliding window's effect on a continued turn.** The body is a per-row arithmetic and does
  not change the cache or the window, but a continued turn's tokens after a >8-row reply were not
  compared with a cold prompt.
- **The recurrence ring's speed.** Its bits are measured (the world-2 capture equals this
  reference and the loopback tests hold it to a single process), but tokens a second at world 2
  against world 1 were not compared.

## The speed rule: the body is 33% slower, and version 2 is owed

The written speed rule (`docs/NUMERICS.md`) was applied on the two builds the change separates:
the current one with the few-row body (`18a76075`, binary `102d216d`) against the pre-change
build with the row GEMV (`84f07db7`, binary `4de25a5e`), one process on the head at world 1,
context 8,192, fresh boot each, `bench_serving.py --levels 1 --tokens 256 --rounds 3`:

| boot | route | tok/s | per stream |
| :-- | :-- | --: | --: |
| body 1 | few-row body | 17.77 | 17.95 |
| gemv 1 | row GEMV | 26.19 | 26.65 |
| gemv 2 | row GEMV | 26.15 | 26.60 |
| body 2 | few-row body | 17.55 | 17.73 |

The ratios are 0.679 and 0.671, against a rule that needs 1.02; the single-request guard (0.97) is
missed by the same distance. The body reads the BF16 rows (the GEMV build logs 289 projections
packed beside them, 23.1% fewer bytes a pass; the body build logs no packing), so the deficit is the
extra bytes a pass. One-process Dory accepts **one lane** - `--concurrency` and `--fuse-batch`
are the packed-model knobs and the group path needs at least two ranks - so the 16-lane part of the
rule degenerates to the single-request number here; that is recorded, not worked around.

**Dory therefore returns to the row GEMV as numerics version 2.** The reference is captured and
stored as `numerics/captures/dory10/v2-gemv.json` (`4f3af7e0…`, eight prompts, 1,024 tokens,
built from `84f07db7`), two free captures bit-identical, and its floor's forced comparison
against version 1 (1,024 tokens) is a precision tail: mean NLL delta -3.062e-05, mean absolute
4.962e-04, max 1.006e-02, top-1 100.0000%, mean KL 2.586e-06. The reference's own mean NLL is
0.1616 a token. Known answers: "the capital of France" answers "Paris"; the arithmetic and READY
probes spent their 128-token budget inside `reasoning_content` and returned empty `content`, which
is recorded as observed rather than retried. The registry declaration (version 2, retiring version
1 with this reason) is the next change; the code still serves the body until it lands.

## Reading

The device test establishes that a decode row is the prompt tile's row, so the packed copy does
not change the arithmetic. The cluster capture establishes that the released checkpoint serves
deterministically at world 1 and pins the bits by a fingerprint. The ring's capture and its tests have
since landed (2026-11-19) and agree with this reference bit for bit; what remains is its speed
A/B.

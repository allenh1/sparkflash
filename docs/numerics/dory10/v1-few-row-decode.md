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
receives. That claim cites `tests/recurrence.rs`, which does not exist - the ring's bits against
a single process are **not tested** - so this reference is the world-1 configuration, and the
ring's world-2 capture and that test are the named follow-up.

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
- **The speed A/B.** Whether the one-row step is flat or slower and whether the several-row step
  is faster are both **unmeasured**. The plan expects flat-to-slightly-slower at one row and
  faster at several; that is not evidence.
- **The sliding window's effect on a continued turn.** The body is a per-row arithmetic and does
  not change the cache or the window, but a continued turn's tokens after a >8-row reply were not
  compared with a cold prompt.
- **The recurrence ring.** Its legal width with this checkpoint is 2; no world-2 capture exists
  and the claim that its bits equal one process's is untested.

## Reading

The device test establishes that a decode row is the prompt tile's row, so the packed copy does
not change the arithmetic. The cluster capture establishes that the released checkpoint serves
deterministically at world 1 and pins the bits by a fingerprint. What the arm still owes is the
ring: a world-2 capture, a test that its bits are a single process's, and the speed A/B.

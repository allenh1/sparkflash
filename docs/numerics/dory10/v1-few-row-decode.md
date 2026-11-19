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

**Registry state: version 1, uncaptured.** The registry entry names the world-4 sm_121a
configuration and leaves `reference` and `fingerprint` out: no checkpoint has been captured for
this arm, so there is no earlier reference to move and no fingerprint to invent. Version 1 is the
configuration's first reference, and it is the arithmetic this change lands, so the entry keeps
version 1 and gains this report instead of a bump.

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

## What was measured

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

The device run, on the sm_121a host:

```text
$ cargo test -p sparkflash-kernels --test token_bodies_device -- --ignored --nocapture --test-threads=1
running 3 tests
test the_bf16_rows_body_is_the_wide_tiles_row ... ok
test the_fp8_token_bodies_are_the_tiles_rows ... ok
test the_packed_token_body_is_the_bf16_tiles_row ... ok
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## What could not be measured

- **The served delta.** No cluster window and no released Dory checkpoint were available, so the
  change was not captured against a reference: the mean NLL delta, top-1 agreement and greedy
  text flips against the row-GEMV decode are **unmeasured**. The registry entry stays uncaptured
  until the world-4 recurrence ring takes its first reference.
- **The speed A/B.** Whether the one-row step is flat or slower and whether the several-row step
  is faster are both **unmeasured**. The plan expects flat-to-slightly-slower at one row and
  faster at several; that is not evidence.
- **The sliding window's effect on a continued turn.** The body is a per-row arithmetic and does
  not change the cache or the window, but a continued turn's tokens after a >8-row reply were not
  compared with a cold prompt here (no checkpoint).

To measure the served configuration, on the cluster with Dory served at world 4:

```bash
python3 scripts/numerics.py capture --url http://127.0.0.1:8001 --out numerics/references/dory10-tp4-sm121a.json
python3 scripts/numerics.py check --url http://127.0.0.1:8001 --reference numerics/references/dory10-tp4-sm121a.json
```

## Reading

The property this change establishes is the device test's: a decode row is the prompt tile's
row, so a continued turn resumes on the bits a cold prompt leaves, and the packed copy does not
change the arithmetic. What it does not establish, until the cluster capture, is how far the
served logits move and whether the step got faster.

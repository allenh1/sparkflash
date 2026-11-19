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

# Inkling-2-Small, numerics version 1: the decode rows on the tile bodies

**Registry state: version 1, captured on 2026-11-19.** The entry names the world-4 sm_121a
configuration and now points at `numerics/references/inkling2-tp4-sm121a.json`,
`387647bcc3268572…`. It is the configuration's first reference, so there is no earlier reference
to move and no bump: version 1 is the arithmetic this change lands, and the first capture pins
it. Two captures on the served group are bit-identical and the served gate's lanes check passed;
the numbers are below.

This is the first declared change to Inkling's arithmetic: its decode rows move from the row
GEMV/matvec bodies to the prompt tile's bodies.

## What changed

- **The head.** A decode head row (one to eight rows) now takes `q4p_dense_rows`, the shared
  wide tile's few-row BF16 body. The row is `glm_prefill_bf16_wide`'s - the tile Inkling's
  tensor walk runs - and the body reads the matrix's packed copy where the packed uploader made
  one. A wider head walk, or a shape the body does not cover, keeps the batched row GEMV.
- **The NVFP4 experts.** The one-to-eight-row steered band now takes
  `q4p_moe_rows_gate_up_capped` and `q4p_moe_rows_down`: the grouped tile's chain over the
  interleaved w13, the softcapped SwiGLU, the same split bf16 activation planes and the same
  per-expert outer scales. The rows are `q4p_moe_gate_up_capped`'s and `q4p_moe_down`'s.
- **The MXFP8 experts.** The same one-to-eight-row band takes the grouped MXFP8 tile body
  itself (`q4p_moe_gate_up_capped_mx8`, `q4p_moe_down_mx8`): the catalogue has no MXFP8
  few-row form, and the 128-row tile computes the item's at-most-eight rows exactly (the other
  rows are zero and skip their MMA). The bodies are catalogued on both passes now.
- **The row rules stay.** The other dense projections (attention, router, the dense MLP) keep
  `q4e_gemv_bf16`/`q4e_gemv4_bf16` at one to eight rows; the steered band from nine to sixty-four
  rows keeps the arm's row expert kernels; a band whose down projection gathers, a head-only
  activation (`SPARKFLASH_INKLING_PLANES=head`), a BF16 expert layer and a shape without whole
  32-column K runs (128 for NVFP4) keep the row walk.

## What was measured

- **The head is the wide tile's row.** On the sm_121a device, at every token count from 1 to 8,
  a `q4p_dense_rows` launch over the same weights, f32 activations and columns is bit-equal,
  row by row, to `glm_prefill_bf16_wide` - the prompt tile - and to `dory_mma_wide`, which
  is the same shared wide body. The test is
  `sparkflash-kernels/tests/token_bodies_device.rs::the_bf16_rows_body_is_the_wide_tiles_row`.
- **The NVFP4 expert rows are the grouped tile's rows.** On the same device, at 1, 2, 4 and 8
  rows, over a generated plan (one expert, two slots a row) and the same interleaved w13, outer
  scales, split activation planes and routing weights, the rows bodies' gate/up planes and down
  partials are bit-equal, word for word, to `q4p_moe_gate_up_capped` / `q4p_moe_down`. The test
  is `sparkflash-kernels/tests/expert_rows_device.rs::the_expert_rows_bodies_are_the_grouped_tiles_rows`.
- **The routing band.** A CPU test holds the band to one through eight rows and rejects the
  ninth, a BF16 expert layer, a head-only activation, a gathered band, a missing grouped table
  and a shape the bodies do not cover
  (`device::chunk::tests::the_decode_band_takes_the_expert_rows_body_from_one_to_eight_rows`).

The device run, on the sm_121a host:

```text
$ cargo test -p sparkflash-kernels --test token_bodies_device -- --ignored --nocapture --test-threads=1
running 3 tests
test the_bf16_rows_body_is_the_wide_tiles_row ... ok
test the_fp8_token_bodies_are_the_tiles_rows ... ok
test the_packed_token_body_is_the_bf16_tiles_row ... ok
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ cargo test -p sparkflash-kernels --test expert_rows_device -- --ignored --nocapture --test-threads=1
running 1 test
test the_expert_rows_bodies_are_the_grouped_tiles_rows ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Both compare bits. Neither is a speed measurement, and neither runs the released checkpoint.

## Measured on the cluster, 2026-11-19

Served at world 4 on the four DGX Sparks from the branch built at `f1f86bf4` (release,
`5a092d5a`), Inkling-2-Small over 1,048,576 tokens, launched by `scripts/inkling-serve-group.sh`
with `BIN` pointing at that binary and `LANES=4`. The group answered `/readyz` in 22 s. The
server reported its identity as `world=4 profile=reference plan=tp arch=sm_121a`, status
`unregistered` before the capture, registered at version 1.

Two captures are bit-identical, so the served configuration is deterministic:

```text
387647bcc3268572b2c4023978692fad7b150bcf63a0cfc12c7d5dfaaab23f16  inkling2-tp4-v1-a.json
387647bcc3268572b2c4023978692fad7b150bcf63a0cfc12c7d5dfaaab23f16  inkling2-tp4-v1-b.json
```

The served gate's lanes check passed on the same boot: four prompts together equal each alone,
twice, in 58 s.

There is **no delta** to report, and that is the honest shape of a first reference: the entry
had no reference to move from, so version 1 does not declare a change, it pins the arithmetic.
The measurement of this change against the row-GEMV decode would need a capture on a build that
still runs those bodies, which no longer exists on this branch.

## What is still not measured

- **The change against the row-GEMV arithmetic.** The bodies are the tiles' rows bit for bit
  (the device tests above), but how far the served logits move from the row GEMV's was never
  captured, and this first reference cannot supply it.
- **The speed A/B.** Whether the one-row step is flat or slower and whether the several-row step
  is faster are both **unmeasured**. The plan expects flat-to-slightly-slower at one row and
  faster at several, as on DeepSeek and Qwen; that expectation is not evidence. The MXFP8 rows
  take the 128-row tile at one to eight rows, which is the least likely of the three bodies to be
  faster at one row, and it is kept because it is the catalogue's MXFP8 body and its bits are the
  tile's by construction.
- **The released layers' MXFP8 scale codes.** The MXFP8 test is generated operands; the released
  checkpoint's per-32 ue8m0 codes have not been read on a device here.

The reference is checked by recapturing it on a freshly booted group:

```bash
python3 scripts/numerics.py check --url http://127.0.0.1:8000 \
    --reference numerics/references/inkling2-tp4-sm121a.json
```

The declaration check (`python3 scripts/numerics.py ci`) still reports the missing coverage and
monitor prerequisites, and Inkling-2 has no `[[anchor]]`, so this entry cannot bump until those
land; it is no longer one of the check's uncaptured ones, because it has a reference and a
fingerprint.

## Reading

The device tests establish that the head's row and the NVFP4 experts' rows are the prompt tiles'
rows, and that the MXFP8 rows are the tile itself. The cluster capture establishes the rest of
what a first reference can: on the released checkpoint the group serves deterministically, four
lanes together compute what each alone computes, and the bits are pinned by a fingerprint the
declaration check can hold. Whether the step got faster, and how far these rows sit from the row
GEMV's logits, remain open and are named above.

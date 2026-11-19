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

**On 2026-11-19 this reference was found to pin degenerate output.** The window's forced captures
showed the captured arithmetic decoding eight prompts to token soup, and the head's few-row body
was reading the row GEMV's packed layout through the tiled stage. The fix is commit `6dd347c9`;
the reference file and its registry entry are deliberately left as captured, because a reference
is replaced by a reviewed re-declaration, not by an edit. **Version 1 must be re-declared from a
clean build** (a new capture with the same forced-capture discipline), and **no deployment may be
proposed on the current v2-less state**: the only declared arithmetic is the degenerate one. The
full record is *The 2026-11-19 window* below.

This is the first declared change to Inkling's arithmetic: its decode rows move from the row
GEMV/matvec bodies to the prompt tile's bodies.

## What changed

- **The head.** A decode head row (one to eight rows) was moved to `q4p_dense_rows`, the
  shared wide tile's few-row BF16 body, whose row is `glm_prefill_bf16_wide`'s - the tile
  Inkling's tensor walk runs. As captured, the launch also handed the body the matrix's packed
  copy; that is the defect below (the body decodes only the *tiled* packed layout and Inkling
  packs in place in the *row walk's* layout). After `6dd347c9` a packed head keeps the batched
  row GEMV - the body would otherwise read the packed bytes as BF16 words - and only a
  checkpoint-form head, which has no packed copy, takes the body. A wider head walk, or a shape
  the body does not cover, keeps the GEMV either way.
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

## The 2026-11-19 window: the registered v1 pins degenerate output

A cluster window measured on one GPU, with the same launcher (`scripts/inkling-serve-group.sh`,
`LANES=4`, `PREFILL=tensor`, `CONTEXT=1048576`), the same checkpoint, and the reference's own
tokens forced through (`capture --forced-from`):

| build | generated | against `387647bc…` |
| :-- | :-- | :-- |
| the branch tip (`e60f00c5`) | eight prompts decoded to multilingual token soup | **bit-identical**: `numerics.py check` PASS |
| `7850cffc` (head still on the row GEMV) | a clean chat opener | mean NLL **+5.824**, top-1 **29.49%**, max **47.09** |

The divergence is structured, not a precision tail. The worst position of every prompt is position
0: there the reference puts essentially all mass on `200001` (`<|message_model|>`, logprob ~0),
while `7850cffc`'s argmax is content token `125148` at `-1.156` and `200001` is at `-48.25`.
Equal prompt ids, equal forced tokens, one decode step apart. The side that gives `200001`
probability ~1 is the reference, the captured tip; the pre-change build is the one that puts it at
`-48.25`.

### What was wrong

Commit `84f07db7` handed `q4p_dense_rows` the head's packed copy. The body's `Q4pRowsPacked`
stage decodes the **tiled** packed layout (`q4e_packed_tiled_a/b`: plane A at `256 j + 8 l` of a
step), while Inkling packs **in place in the row walk's layout**
(`PackedSet::new(form, Placement::InPlace)`, `Layout::Walk`, never `.tiled()`; its reader is
`q4e_gemv_packed`). `Shared::dense_rows` filtered the copy only through `admits` (columns, row
count, a 16-byte-aligned activation), which says nothing about layout, so every decode head row
rebuilt its chunks from the wrong offsets: plane A from another chunk, the plane-B nibble from
another chunk, the step's top from a different lane. The head's logits were garbage at the first
decode row and at every one after, and `200001` maximised them.

The expert band's change is not implicated by this measurement. The few-row NVFP4/MXFP8 bodies
take the same grouped table, plan, split activation planes, outer scales and routing-weight-one
convention as the `run_band_mma` launches, and the window's signature is the head's: the first
decode row's hidden state is the prompt's own last row, which the change does not touch. The
expert rows remain unproven on the released checkpoint.

### The fix

Commit `6dd347c9`: `Shared::dense_rows` routes by the packed copy's layout (`DenseRowsInput`):

- **no packed copy** (`SPARKFLASH_DENSE_WEIGHTS=checkpoint`) - the body reads the matrix's BF16
  rows, which is the wide tile's row;
- **a tiled copy** - the body reads the packed rows its stage decodes;
- **the walk layout** - the body refuses the projection and the head falls back to the row GEMV,
  which is `7850cffc`'s arithmetic and the build the window measured clean.

A CPU test pins the three-way routing:
`sparkflash-model-inkling2`'s
`device::tests::the_few_row_head_body_takes_only_the_tiled_packed_copy`. No kernel changed, so
`token_bodies_device::the_bf16_rows_body_is_the_wide_tiles_row` - the body-equals-tile test - is
unchanged and keeps passing; it cannot see a wrong *input*, which is exactly what this was.

### The consequence for v1

**`numerics/references/inkling2-tp4-sm121a.json` (`387647bc…`) pins the defect, not clean
arithmetic.** It is not edited and its registry entry is not moved: a reference is replaced by a
reviewed re-declaration. Inkling's v1 must be **re-declared** after the fix - a fresh capture from
a clean release build with the same forced-capture discipline (two captures bit-identical, the
served gate's lanes check, a server card that names the build), reviewed and registered as the
arm's new version. **No deployment may be proposed on the current v2-less state.**

### What the next GPU window runs

```sh
# 1. The bodies' device tests (unchanged: no .cu changed).
cargo test -p sparkflash-kernels --test token_bodies_device -- --ignored --nocapture --test-threads=1
cargo test -p sparkflash-kernels --test expert_rows_device -- --ignored --nocapture --test-threads=1

# 2. The fixed group on the released checkpoint, the capture's configuration.
BIN=<release build of 6dd347c9> LANES=4 PREFILL=tensor CONTEXT=1048576 \
    ./scripts/inkling-serve-group.sh launch

# 3. A greedy 32-token completion: the evidence is the text.
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"thinkingmachines/Inkling-2-Small","temperature":0,"max_tokens":32,
         "messages":[{"role":"user","content":"Hello, who are you?"}]}'

# 4. The forced comparison against the degenerate v1 reference, twice, bit-identical.
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 \
    --forced-from numerics/references/inkling2-tp4-sm121a.json \
    --out /tmp/inkling2-tp4-v2-a.json --binary <release build of 6dd347c9>
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 \
    --forced-from numerics/references/inkling2-tp4-sm121a.json \
    --out /tmp/inkling2-tp4-v2-b.json --binary <release build of 6dd347c9>
sha256sum /tmp/inkling2-tp4-v2-a.json /tmp/inkling2-tp4-v2-b.json
python3 scripts/numerics.py compare --anchor numerics/references/inkling2-tp4-sm121a.json \
    --candidate /tmp/inkling2-tp4-v2-a.json

# 5. The head's body against the row GEMV on the real checkpoint. With
#    SPARKFLASH_DENSE_WEIGHTS=checkpoint the head has no packed copy, so the few-row body runs on
#    its BF16 rows; with the default the head runs the walk-layout GEMV. The packed form's
#    round-trip is checked at load, so the two forced captures must be bit-identical.
SPARKFLASH_DENSE_WEIGHTS=checkpoint <the same launch> \
    && python3 scripts/numerics.py capture --url http://127.0.0.1:8000 \
        --forced-from numerics/references/inkling2-tp4-sm121a.json \
        --out /tmp/inkling2-tp4-checkpoint.json --binary <the same build>
```

Pass is: the device tests pass; the greedy completion begins with a plausible assistant opener
(not `200001` or a repeated special token); the two captures are byte-identical; the forced
comparison against the old reference reports a large **positive** NLL delta of the pre-change
build's order (+5.8 mean) - the forced tokens are the degenerate capture's own soup, and a clean
build assigns them low probability where the reference gave `200001` probability ~1; and the
checkpoint-form capture matches the packed-form capture bit for bit. Soup, a delta near zero, or a
body/GEMV mismatch is a failed fix.

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

There was **no delta** to report when the entry was captured, and that is the honest shape of a
first reference: the entry had no reference to move from, so version 1 did not declare a change,
it pinned the arithmetic. The 2026-11-19 window then measured the pre-change build against this
reference (+5.824 mean NLL, 29.49% top-1), but the reference is the degenerate capture, so that
delta measures the defect. Version 1 is being re-declared; see *The 2026-11-19 window* above.

## What is still not measured

- **The change against the row-GEMV arithmetic.** The bodies are the tiles' rows bit for bit
  (the device tests above), but this reference cannot measure the served distance: the 2026-11-19
  window did measure `7850cffc` against it (+5.824 mean NLL, 29.49% top-1), but the reference is
  the degenerate capture, so that number measures the defect, not the intended change. The clean
  delta arrives with the re-declaration.
- **The expert rows against the grouped tile on the released checkpoint.** The generated-operand
  device test says the rows bodies are the tiles' rows; the released checkpoint's NVFP4 and MXFP8
  operands have not been compared row against tile on a device. The next window's forced captures
  cover it end to end when the group is booted once with the decode band's rows and once with the
  chunk's grouped tile.
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

The device tests establish that the head's body and the NVFP4 experts' rows are the prompt tiles'
rows, and that the MXFP8 rows are the tile itself; the routing now keeps the packed head on the
row GEMV, so on a served (packed) checkpoint it is the head *GEMV* that runs, and only a
checkpoint-form head takes the body. The cluster capture establishes what a first reference can
about determinism: on the released checkpoint the group served deterministically and four lanes
together computed what each alone computed. It does not establish clean arithmetic, which is why
the reference is being re-declared. Whether the step got faster, and how far the expert rows sit
from the row GEMV's logits, remain open and are named above.

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

# Inkling-2-Small, numerics version 2: clean arithmetic after the head layout fix

**Registry state: version 2, captured on 2026-11-19 on commit `b62e2fcb` (binary
`23666b8d`, built with nvcc, embedding that commit).** It replaces version 1, which was retired
with "pinned degenerate output from the head layout defect (`6dd347c9`)". Version 1's reference
file and its registry history are untouched; a reference is replaced by a reviewed re-declaration,
never by an edit.

Version 1 pinned the arm's first arithmetic because two captures were bit-identical and the
served gate's lanes check passed. The bits were deterministic *and* wrong: the head's few-row
body read the row walk's packed layout through the tiled stage, so the first decode row's logits
were garbage and the model produced multilingual token soup. Commit `6dd347c9` routes the head by
the packed copy's layout (a tiled copy takes the body, the walk layout keeps the row GEMV), and
`158f5723` holds the few-row band's down grid to the output rows. Version 2 is the fixed build's
arithmetic, measured on the released checkpoint.

## What the floor records

The capture command now refuses to write a reference or an anchor without a correctness floor
(`docs/NUMERICS.md` §5): a forced comparison against the pre-change build or the host reference,
a known-answer decode of the fixed prompts, and the reference's own NLL on natural text. The
floor is in the card, outside the fingerprint, and the declaration check re-checks it.

Against `7850cffc`, the pre-change row-GEMV build (binary `32b7bd03`), teacher-forced on this
capture's own 970 greedy tokens:

| metric | candidate |
| :-- | --: |
| tokens scored | 970 |
| anchor NLL / token | 0.1806 |
| mean NLL delta | -5.002e-05 |
| mean absolute NLL delta | 1.691e-04 |
| p99 absolute NLL delta | 2.176e-03 |
| max absolute NLL delta | 7.307e-02 |
| top-1 agreement | 100.0000% |
| mean KL lower bound | 6.583e-06 |

With the head back on its GEMV, the remaining change against `7850cffc` is exactly the
one-to-eight-row expert band and the MXFP8 tile. The distance is a precision tail: no top-1
position changes, the mean absolute delta is 1.7e-04 nat, the largest single position is 7.3e-02
nat, none of the worst 20 positions is at position 0, and they spread over 7 of the 8 prompts
(`code` 6, `translation` 5, `raw-continuation` 4, `tool-json` 2, `summary`, `arithmetic`
and `logic` 1 each). The worst is `translation` position 68: 0.9156 nats on the clean build
against 0.8425 on the pre-change one, top-1 `149895` on both. This also closes the open item
"the expert rows against the grouped tile on the released checkpoint": 970 tokens forced through
both builds, with only that band differing, agree to 7e-02 nat.

The reference's own mean NLL on the suite's natural text is **0.1806 a token** (perplexity
1.198), against a 2.0 bar.

## Known answers

Greedy, temperature 0, 32 tokens:

- `The capital of France is` decodes `" Paris. It is a beautiful city with many famous
  landmarks. …"` — correct.
- `Hello, who are you?` decodes `"\n\nI'm Andy, a software engineer at Stripe. …"` — a coherent
  English opener, not a special token and not soup.
- `2+2=` decodes `"5\n\nA\n\nAnonymous\n\nGuest\nI have a 2+2=5 problem. …"`. This is
  recorded in the floor as a *failed* known answer with an explanation: the pre-change build
  decodes the identical text, and all 970 forced tokens agree, so it is the small model's own
  answer rather than a regression. The explanation is what lets the pin keep the number beside
  it instead of dropping the evidence.

## What was measured on the cluster

Served at world 4 on the four DGX Sparks, `BIN` the release build of `b62e2fcb`
(`23666b8d`, `inkling_sm121.cubin` 2,057,056 bytes), `LANES=4`, `PREFILL=tensor`,
`CONTEXT=1048576`. The group answered `/readyz` 26 s after launch and reported its card as build
`b62e2fcb48185220bb35a733ca614b475a4e31bd`.

Two free captures are byte-identical:

```text
8270fdc4712b306f199fc3762a1687260f6ec01f9727a8bb6de0e7bf18c99843  inkling2-v2-a.json
8270fdc4712b306f199fc3762a1687260f6ec01f9727a8bb6de0e7bf18c99843  inkling2-v2-b.json
```

The anchor (16 tokens a prompt, top-20) is also two-for-two identical
(`02578670…`), and the teacher-forced capture of the anchor's own 128 positions scores exactly
zero (mean, absolute and max delta 0.0; top-1 100%; KL 0), which is what makes the comparison
harness add nothing to a measured distance.

The reference card reports `status=registered, registered_version=1` with version 1's fingerprint,
because the binary was built while the registry still named version 1; the bits are clean and the
card's own fingerprint is `acd86d46…`. A rebuild from this registry reports version 2. This is the
same "a first capture reads unregistered" shape as any registry write that follows a capture.

The production server was restored from the release binary afterwards (`/readyz` 200, the same
model card and four ranks).

## Reading

The head is back on `q4e_gemv_packed` on the served (packed) checkpoint; the fix trades the
owner's tensor-core decode on this arm for correctness until the head has a tiled packed copy of
its own. The decode's *experts* still run the tile's few-row bodies, and the end-to-end forced
comparison above shows that change is a precision tail. What is still open:

- **The tensor-core head: measured, and the GEMV stays.** A tiled packed copy for the head (so
  `q4p_dense_rows` reads it) was built and served on `985dcbf6` (binary `8f1cdc99`). It is
  correct: its forced comparison against this reference over 970 positions is a precision tail
  (top-1 100.0000%, mean absolute NLL delta 5.742e-06, max 2.655e-03, 0 of the worst 20 at
  position 0, the worst 20 over 4 prompts), and the checkpoint-form capture is bit-identical to
  the default one, which proves the tiled copy exact. It is slower, though: two alternated
  fresh-boot pairs give 30.54 vs 34.50 and 32.57 vs 34.48 tokens/s, -6.6% to -8.5% for the tiled
  head, losing both pairs against a written +2% bar. The plan's regression clause keeps the GEMV
  head, so the tiled copy and its +355 MB arena were reverted (`4ee6eb45`); the typed packed
  layout and its routing contract stay.
- **Coverage and monitors.** The model has neither row yet; this bump carries the owner's
  approval for that reason.
- **The speed A/B.** Unmeasured here; the plan expects flat-to-slightly-slower at one row.
- **The prefill walk.** Still not named in the boot ledger (the open declaration gap from the
  version-1 window).

```bash
python3 scripts/numerics.py check --url http://127.0.0.1:8000 \
    --reference numerics/references/inkling2-tp4-sm121a-v2.json
```

# The per-position logits dump (O6 / task-19)

**Status.** The device chain's own output path for per-position logits. Functionally it is the
`task/logits-dump` `4f599b7` writer, now a first-class module with one documented schema and host
tests; nothing about the emitted bytes changed. No device run was made for O6 and none is claimed:
the writer and its schema are host-tested, and the device side of D40 is still measured in the
Lead's window.

## Where it lives

| | |
| :-- | :-- |
| producer | `crates/sparkflash-cli/src/logits_dump.rs` (`write_logits_dump`, the only writer) |
| command | `sparkflash device-chain --run-stage hybrid_sequence ... --dump-logits FILE` (D40 protocol step 2); every other stage rejects the flag (`device_chain.rs:598-601`) |
| consumer | `scripts/compare_native_vs_vllm.py` (`read_dump`, `read_logits`), exercised by `scripts/tests/test_compare_native_vs_vllm.py` |
| decision | D40 (`docs/DECISIONS.md`) freezes the artefact's role; this document freezes its fields |

The dump is written as soon as the compared rows exist and before the token-parity gate can
return, so a parity-failing run still leaves its logits. It changes no arithmetic and prints no
line: the stage's existing output is byte-identical with and without the flag.

## Schema (frozen)

`FILE` is JSONL: one object per compared position, in the order the chain compared them. `FILE.bin`
is the full-vocabulary sidecar.

| field | type | meaning |
| :-- | :-- | :-- |
| `position` | int | device position from the `hybrid_sequence` token table; a full 130 + 8 claim starts at `prompt_len - 1` = 129, a short run at 0 |
| `device_argmax` | int or null | the greedy token the device head selected; null when the row produced no token |
| `logits_argmax` | int | argmax of the dumped full-vocabulary logits; ties break to the lowest id |
| `reference_id` | int or null | the reference token id the token table compared at this position |
| `hybrid_id` | int or null | the chain's token id at this position |
| `fixture_id` | int or null | the fixture's recorded next token, on a full-claim comparison only |
| `exact_token_equal` | bool | whether `reference_id` equals `hybrid_id` |
| `vocab` | int | logits in this position's block (129,280 for the released checkpoint) |
| `logits_offset` | int | byte offset of this position's block in `FILE.bin` |
| `logits_bin` | string | the sidecar path, `FILE` + `.bin` |
| `top20` | array | up to 20 `[token_id, logit]` pairs, descending by logit and then ascending by id |

Fields are additive. The frozen consumer requires only `position`, `vocab` and `logits_offset`
(`CONSUMED_FIELDS` in the module), so an older dump that predates a parity-only field still joins.

**Sidecar.** `FILE.bin` holds consecutive little-endian f32 blocks, `vocab` values per dumped
position, in the same position order, with no padding. Each object's `logits_offset` is the sum of
the previous blocks' `vocab * 4`, and the file's length is that sum for every dumped position.

**Positions.** D40 joins the two sides on this integer: oracle `prompt_logprobs[p + 1]` is the
distribution after device position `p`. A prefill-only run (`--decode-steps 0`) therefore dumps the
prompt positions, which is what a D41 flooring run consumes; the trailing position with no oracle
counterpart is harmless because the comparison joins on `position`.

## Compatibility with the existing dumps

| dump | state |
| :-- | :-- |
| `crates/sparkflash-cli/tests/fixtures/logits-dump-demo-scale.jsonl` (9 positions 129..137, vocab 129,280, offsets tile a 4,654,080-byte sidecar) | the existing D40 protocol demo dump, sha256 `459bd9471f30629846f4859e124be2bbf2283369512a70c0df460fb788624a85`, kept verbatim as the compatibility fixture. Every key it carries is a subset of the table above; it does not carry `hybrid_id`, `fixture_id` or `exact_token_equal`, so a reader must not require them |
| `window-2026-10-18/runs70/logits-130x8.jsonl` (sha256 `f548d9c3...`) and `.bin` (`43b31438...`) | the 2026-10-18 window dumps, on the evidence node, not in the repository. They were not re-read for O6: named, not asserted |

**Writer equivalence.** The function body is `4f599b7`'s verbatim. The only differences are the
`ComparedRow` / `HybridTokenOutput` type aliases in the signature and `sidecar_path(path)` replacing
three inline bin-path lines; a line diff of the two function bodies shows nothing else.

## Host tests (no device)

```sh
cargo test -p sparkflash-cli logits_dump
cargo test -p sparkflash-cli --features cuda logits_dump
python3 -m unittest discover -s scripts/tests -v
```

- `the_dump_writes_one_jsonl_object_and_one_le_f32_block_per_position` - the moved `4f599b7` test:
  two positions, offsets 0 and 16, the sidecar's little-endian values, `fixture_id` on a full claim.
- `every_object_carries_exactly_the_frozen_fields` - the object's keys equal `FIELDS` in order, so
  a schema change is a test change and this document cannot drift.
- `the_existing_d40_demo_dump_matches_the_same_schema` - parses the existing demo dump above and
  checks the frozen consumer's fields, that every key it carries is documented here, that its
  offsets tile its sidecar, and that it names the sidecar the way `sidecar_path` does.
- `scripts/tests/test_compare_native_vs_vllm.py` reads the same fixture through the frozen consumer
  (`read_dump`), so the existing artefact and the writer are checked against one reader, not two.

## O6 record (task-19)

Before O6 the main branch already carried `4f599b7`'s writer, but it lived inside `device_chain.rs`
behind `cfg(feature = cuda)`, the plan still pointed at the branch as the place to land it, and the
only test could not compile without the device feature. O6 makes it the chain's own output path:

- the writer moved to `crates/sparkflash-cli/src/logits_dump.rs`, compiled in every build, with
  `FIELDS`, `CONSUMED_FIELDS` and `sidecar_path` as the one definition of the schema;
  `device_chain.rs` now only calls it from `hybrid_sequence` (the rest of the stage is unchanged);
- `grep -rn 'fn write_logits_dump' crates/` returns exactly one definition, so the audit path and the
  chain cannot drift into two dumps;
- the tests are host tests (no GPU, no driver): they run with and without `--features cuda`;
- the schema is this document; the plan, the window runbook and the stage-level audit now point at
  it instead of at the branch, and the existing D40 demo dump is a checked-in compatibility fixture
  read by both the Rust test and the frozen comparison script.

Verified for this record: `cargo test -p sparkflash-cli logits_dump` (3/3, both feature sets),
`python3 -m unittest discover -s scripts/tests` (48 tests, OK), and the writer-body diff against
`4f599b7` shows only the type aliases and the `sidecar_path` extraction above. No device was run;
the window measurement stays named and unclaimed.

**Unmeasured, named.** No device run: the writer has not been exercised on a real checkpoint in a
window, and the b70/runs70 dumps were not re-read. Those measurements belong to the Lead's window
and are not asserted here.

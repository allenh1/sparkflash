# Phase 2 sequence harness (milestone 2)

Status: **2A contract/format landed on main** as 13d880d (squash of !56) with the real-checkpoint
element-limit follow-up 51cd0c5 (squash of !58); the node golden fixture now exists
([phase2-golden-fixture.md](phase2-golden-fixture.md)); 2B is a CPU-pinned skeleton in draft MR !63
and its device comparison has not run; 2C has not started. This note records the accepted design,
the 2A contract, and the exact boundaries so no later claim is read more broadly than it was
measured; section 7 records the current Phase 2 status. **Phase 2 exit criterion 1 remains OPEN.**

Sources: `docs/build/roadmap-to-serving.md` section 5 (5.3 prefill/decode, 5.4 audit, 5.5 exit
criteria), `docs/DECISIONS.md` D7 (CED prefill bypass), `docs/engine/01-forward-order.md`
sections 1 and 3.9.

## 1. What milestone 2 runs

One sequence, one store, one prompt of **130 tokens** (it crosses the 128-token sliding window),
then **8 greedy decode steps** on the device path, compared against the host reference engine.

Honest labels, fixed now:

* the engine reference does **not** implement CED. The harness supplies the encoder/decoder
  scheduling; the comparison is tokens and logits **on the compared set only**, never a claim
  that the two paths are equal for every position;
* the compared set is exactly **{last prefill position, 8 decode positions}**. Adding any older
  position fails the comparison and names D7 (see below);
* MTP, batch>1 and multiple sequences are out of scope; one store = one sequence.

## 2. CED schedule as data (2A, `sparkflash-kv::schedule`)

`ced_schedule(prompt_tokens)` returns the cold-prompt plan plus the 40-row layer table and the
kernel-handle publication order, so the device path never re-derives CED from prose.

For `ced_schedule(130)`:

* encoder layers `0..=20` run over all 130 prompt tokens;
* decoder layers `21..=39` run over the last 128 tokens;
* `plan_request(130, 0, 0, false)` returns exactly
  `PrefixMatchPlan { total_prompt_tokens: 130, global_kv_cached_tokens: 0, uma_hit_tokens: 0,
  nvme_restored_tokens: 0, swa_tail_hit: false, encoder_swa_replay_tokens: 0,
  uncached_suffix_tokens: 130, decoder_swa_replay_tokens: 128 }`.
  Naming caveat: on a cold prompt the last field is the **CED decoder extent**, not a cache
  replay. The pin test asserts the equality with `min(prompt, 128)`, so a future change to
  `plan_request` cannot silently move the CED boundary.

Per-layer modes (engine `LayerPlan::from_config(v41_flash)` and `sparkflash_kv::csa2_mode`
agree; a test pins both): 0-1 SWA-only; 2/8/14/20 Full (own main KV + index-K + Top-K); 24/28/32/36
Reindex (main KV from layer 20, own indexer, own Top-K); every other layer Reuse. Publication
order: main KV, index-K and Top-K at each Full layer; the HSI pool right after layer 20's Top-K
(before its first consumer, layer 24); Top-K republished at 24/28/32/36. Window rings are
per-layer from construction and are not publications.

## 3. Fixtures (2A, `sparkflash-engine::sequence_fixture`)

Two fixtures, two jobs, never interchangeable:

| fixture | source | where | role |
|:--|:--|:--|:--|
| CI fixture `tests/fixtures/sequence-130-tiny.json` | deterministic finite-FP32 source, tiny config | dev host, committed | schema, compared set, comparator plumbing; **not** evidence about the model |
| node golden fixture `sequence-130-released.json` | real checkpoint via `ReferenceWeights` | node, later window | proof-of-reference for 2B/2C; tokens first, last-position logits second |

Schema v1 records: schema version, generator, source kind/seed/checkpoint, full `ModelConfig`,
prompt token ids, decode step count, window, compared positions, per-step kind/position/next
token/logits SHA-256 (full logits optional; CI records all, the node fixture records the last
prefill position), reserved `kv_rows` (empty in 2A, populated for the 2C audit), and provenance.
Provenance is deliberately asymmetric: the node golden fixture records the shell-runnable command,
host, generator binary SHA-256 and notes, while the CI fixture records only the canonical command,
because it must regenerate byte-identically on any host and binds no evidence to one machine or one
binary. Each `--note TEXT` on the generator command is appended verbatim to the fixture's notes, so
the node run records the fix commit, the rustc/zig versions and the squash expectation there without
changing the generated numbers or the compared set (the note does change the provenance bytes, which
is why a CI fixture refuses it: the CI provenance records the canonical command, and that command has no
`--note` flag).

Two element limits, deliberately named apart. For a real checkpoint the generator gives the engine
`linalg::CHECKPOINT_TENSOR_LIMIT` (`1 << 30` elements), the one constant every real-checkpoint
entry point takes from `linalg` - `sequence_fixture`, `one_token` and `boundary_dump` - so those
three entry points cannot drift. The released `embed.weight` and
`head.weight` are 661,913,600 elements each (and many attention tensors are 41,943,040), all
above `linalg::REFERENCE_TENSOR_LIMIT` (32 Mi elements). That constant remains the small-shape
guard the weight-free finite source keeps, and by design it refuses real checkpoint tensors: the
first node attempt against merged main 13d880d failed with "embed.weight holds 661913600 elements,
above the 33554432-element limit of the reference pass" before this fix. The fixture command is
otherwise unchanged, and the CI fixture bytes are unchanged by the fix.

Commands:

```text
# CI fixture (weight-free, CUDA-free; the committed file is exactly this output)
cargo run -p sparkflash-engine --example sequence_fixture -- \
    --config tiny --prompt-tokens 130 --decode-steps 8 --seed 7 --logits all \
    --out crates/sparkflash-engine/tests/fixtures/sequence-130-tiny.json

# node golden fixture (real checkpoint; run on the node, not the dev host)
cargo run --release -p sparkflash-engine --example sequence_fixture -- \
    --config released --checkpoint /path/to/snapshot --prompt-tokens 130 \
    --decode-steps 8 --seed 7 --logits prefill \
    --note "fix commit <sha>: real-checkpoint element limit; lands on main as a squash" \
    --out /home/user/phase2-evidence/sequence-130-released.json
```

The committed CI fixture: 29,508 B, sha256 `b9e209239be17e4edcd594156131b56d149af247c53f9c827b8e14cbecf2a094`,
regenerated byte-identically by its recorded command. Prompt token ids are synthetic ids inside
the vocabulary (recorded in the fixture); no tokenizer is involved.

## 4. Comparison contract

`compare_steps` is the primary bar: it checks set equality in both directions (a missing
compared position fails; an extra older position fails), then next-token equality, then the logits
SHA-256. `compare_logits` is the second bar: absolute max difference against the Phase-1 bar,
only where the fixture recorded logits. A widening of the compared set fails
`UnexpectedPosition` and its message names D7 and the offending position.

The logits bar is numerical and fixed: **2e-2 absolute**, the provisional Phase-1 ceiling
(`roadmap-to-serving.md`), exposed as `sparkflash_engine::sequence_fixture::LOGITS_ABS_BAR`.
The tests use that constant, not ad-hoc values; the only 0.0 comparison is a host fixture
regenerating itself exactly. 2B/2C may report the measured delta, but may not loosen the bar
without a Lead decision.

`exclusion_boundary(prompt_tokens, window)` fixes the positions older than the window
(`130 - 128 = 2` for the released model). Positions between the boundary and the last prefill
position are inside the window but still not compared: the engine's prefill returns logits for its
last position only, and the device CED decoder runs layers 21-39 on the last window, so those
positions have no comparable decoder output.

## 5. 2A test list (all green locally)

`sparkflash-kv` schedule module (8 tests): exact `plan_request(130,0,0,false)` pin; fresh
planner agreement; decoder extent clamp for 1/64/128/130/4096; window-crossing boundary; 40-row
mode/source table plus the store's `source_layer_for` agreement; exact publication order
(17 events, HSI pool before layer 24); empty-prompt refusal.

`sparkflash-engine` fixture module (15 tests): compared set exactly 9 positions; exclusion
boundary for 130/128 and 130/8; a matching tiny fixture passes tokens and logits; widening with
an older position fails naming D7 (positions 0 and 128); missing position fails; wrong
token fails while unchanged logits still pass; logits bar absolute and second (using
`LOGITS_ABS_BAR`, not a literal); missing device
logits fail only the second bar; schema-version and widened-set validation; deterministic prompt;
finite source refused for the released configuration; the released plan's `embed.weight` is
661,913,600 elements, above `REFERENCE_TENSOR_LIMIT` and inside `CHECKPOINT_TENSOR_LIMIT`
(`head.weight` ties it); the generator selects the checkpoint limit only for a real checkpoint and
keeps `1 << 22` for the finite source; empty prompt / zero decode steps refused.

`tests/sequence_fixture.rs` (7 tests): committed CI fixture regenerates byte-identically;
fixture contract (schema 1, 130 prompt, 9 steps, 8 decode, positions 129-137, kinds, digests,
finite logits, empty `kv_rows`); the committed fixture passes its own comparison; widening fails
naming D7 (positions 0, 64, 122 and 128, each rejection naming the position); released boundary (2 excluded positions, decoder 128, encoder 130); engine
`LayerPlan` agrees with the kv schedule on all 40 backbone layers and MTP 40-42 stays Mtp; the
two fixture weight kinds are pinned in the serialised schema.

## 6. 2B/2C notes fixed in advance

* 2B is a **new example** `crates/sparkflash-kernels/examples/m8_sequence.rs`; `m7_layer0.rs` stays
  untouched. `FlashMlaParams`/`engine_window_row` move to a public kernels module with the m7
  author's acknowledgement, and a test pins that both call sites use the shared builder.
* 2B prefill 130 tokens then 8 decodes, consuming the task-7 store; `window_rows` must be
  contiguous oldest-first. The device window slot is not one 528-B blob: it is two arrays per
  layer, 128 slots x 512 B FP8 codes and 128 slots x 16 B E8M0 scales. 528 B is only the
  host/audit raw slot size. The ring rotation copy into `window_rows` (512 B + 16 B per slot,
  per layer, per step) is measured and reported, not assumed cheap.
* 2B may split into 2B-i (prefill last position) and 2B-ii (8 decodes), but no partial
  exit-criterion claim is allowed.
* 2C runs one logged device run and only then claims roadmap 5.5.1-5.5.5: clean audit after
  prefill and after decodes; five corruption cases (Full main KV, Full index-K, index-K through a
  Reindex layer, source main KV through a Reuse layer, SWA-only window slot) each fail naming
  (kind, layer, position); per-token bytes measured from the live store per layer mode; the
  un-masked-score control breaks 5.5.4.

## 7. Status after the golden fixture (2026-10-18)

* **2A contract/format landed**: !56 (main 13d880d) produced the schema and the comparator; the
  real-checkpoint element-limit follow-up is !58 (main 51cd0c5). Sections 2-5 describe the contract
  as merged.
* **The golden fixture exists**: a real-checkpoint 130-token prefill + 8 greedy decodes, 2,536,194 B,
  sha256 `e6df8290...`; its provenance and measurement caveats are in
  [phase2-golden-fixture.md](phase2-golden-fixture.md). It is reference truth, not device evidence.
* **2B skeleton is CPU-pinned and in draft MR !63**: the `m8_sequence` harness compares token-first
  then logits at 2e-2, and refuses the device path in that draft. It was opened at bad3c84 against
  a26be2e; a rebase onto d3dcc3f is still owed.
* **2B device comparison has NOT run.** No token, logits, rotation-copy or device timing number for
  this milestone exists.
* **CSA-union format 2 landed** as !62 -> main d3dcc3f (pipeline 2763 green): the previous
  window-only device attention structurally excluded the compressed candidates the engine reference
  attends at position 129, so `window_kv_format=2` attends the window rows then the selected
  compressed ids, causal, with the sink first in the denominator. The 19-param ABI and formats 0/1
  are unchanged and the CPU twin (`mla_ref::attention_with_sink_union`) is bit-for-bit against the
  engine reference, but **format 2 has not executed on a device**; its semantics are D24.
* **No exit criterion is claimed.** Phase 2 criterion 1 (`roadmap-to-serving.md` section 5.5.1:
  prefill over 128 tokens and 8 decode steps, every decoded token reproduced token-for-token
  against the reference) remains OPEN. The precise remaining gap is the Lead-sequenced 2B device
  run: wire the format-2 dispatch and the measured `window_rows` rotation into `m8_sequence`, run
  prefill 130 + 8 decodes on the node against the golden fixture, and compare tokens first then
  logits at 2e-2. Criteria 2-5 are likewise unclaimed.

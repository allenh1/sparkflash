# Phase 2 sequence harness (milestone 2)

Status: **2A contract/format landed on main** as 13d880d (squash of !56) with the real-checkpoint
element-limit follow-up 51cd0c5 (squash of !58); the node golden fixture exists
([phase2-golden-fixture.md](phase2-golden-fixture.md)); 2B's device half (measured window rotation +
format-2/format-1 attention dispatch, no forward chain) is wired in
`crates/sparkflash-kernels/examples/m8_sequence.rs`, merged as !63 (1ceeb54); a first window found a
contaminated rotation clock and a decoder-row over-count, both fixed before merge, and the corrected
path then ran on the node in window 2026-10-18T18:32Z: 11,764 `copy_dtod` calls moving 198,832,128 B
in 0.025624 s of copy-only time (7,400.000 MiB/s), 360 dispatches and 320/320 identical repeats. The
2B device **comparison** (tokens then logits against the fixture) has **not** run: the forward chain is
not wired, no device token exists, and this is plumbing evidence only; 2C has not started. This note
records the accepted design, the 2A contract, and the exact boundaries so no later claim is read more
broadly than it was measured; section 7 records the current Phase 2 status. **Phase 2 exit criterion 1
remains OPEN.**

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

`compare_steps` is the primary bar: it checks set equality in both directions (a missing compared position fails; an extra older position fails), then next-token equality, then the logits SHA-256. **[Update 2026-10-19 (D39): next-token equality is retired as an acceptance criterion - generation is not deterministic; this comparator remains the fixture-internal regression check.]** `compare_logits` is the second bar: absolute max difference against the Phase-1 bar,
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

* 2B is a **new example** `crates/sparkflash-kernels/examples/m8_sequence.rs`; the `m7_layer0`
  harness stays untouched (now `examples/m7_layer0/`, split by 954d7af). `FlashMlaParams`/`engine_window_row` move to a public kernels module with the m7
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

## 7. Status after the corrected 2B device plumbing run (2026-10-18T18:32Z)

* **2A contract/format landed**: !56 (main 13d880d) produced the schema and the comparator; the
  real-checkpoint element-limit follow-up is !58 (main 51cd0c5). Sections 2-5 describe the contract
  as merged.
* **The golden fixture exists and is byte-pinned**: a real-checkpoint 130-token prefill + 8 greedy
  decodes, 2,536,194 B, sha256
  `e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308`; its provenance and
  measurement caveats are in [phase2-golden-fixture.md](phase2-golden-fixture.md). It is reference
  truth, not device evidence. The dev-host copy used for the local dry-run is
  `/home/user/phase2-evidence/devhost-copy-f72738b/sequence-130-released.json` (evidence, not the
  repo); the node path is `/home/user/phase2-evidence/sequence-130-released.json`.
  `m8_sequence`'s loader now pins that byte size and hash, so a different fixture with the same
  schema and token 491 is refused before a comparison.
* **CSA-union format 2 landed** as !62 -> main d3dcc3f (pipeline 2763 green): `window_kv_format=2`
  attends the window rows then the selected compressed ids, causal, with the sink first in the
  denominator. The 19-param ABI and formats 0/1 are unchanged; the CPU twin
  (`mla_ref::attention_with_sink_union`) is checked against the engine reference. Format 2 has since
  executed on devices: the isolated task-14 check and the 2026-10-18 plumbing window below. Its
  semantics are D24.
* **2B iteration 1 wires the device half in `crates/sparkflash-kernels/examples/m8_sequence.rs`
  (merged as !63, 1ceeb54).** It does NOT run the model forward and claims no exit criterion. Round 2
  of adversarial review found two blockers and two majors in the first commit (8c91fe5); the round-2
  fix commit (a6044c2) corrects them:
  * **rotation rows are anchored at each layer's first executed position.** Encoder layers start at
    0 and decoder layers at the CED boundary 2, so a decoder view at position `p` holds
    `min(p - 1, 128)` rows, not `min(p + 1, 128)`. The first commit used the latter and over-counted
    253 slots per decoder layer (4,807 slots, 2,538,096 B), which would also have rotated slots the
    layer never wrote if the forward chain reused it. `cuda_harness::ring_view_position` now takes
    the layer's first position, with lib tests for the decoder cases, and the plan, the device copy
    closure and the dry-run all call that one helper.
  * **corrected rotation accounting**: one rotation per token-layer execution, copying only the rows
    that execution's oldest-first view holds: **5,482 executions (prefill 5,162 + decode 320),
    376,576 slots, 198,832,128 B (codes 192,806,912 B + scales 6,025,216 B)**; one full ring
    67,584 B, all forty rings once 2,703,360 B, at most 21,928 `copy_dtod` calls and exactly 11,764
    expected under the segment geometry (5,082 one-run views x 2 calls + 400 wrapped views x 4); the
    measured run issued exactly those 11,764 calls.
    An independent closed-form test pins the per-layer split (encoder 200,256 slots, decoder
    176,320 slots); the earlier min(position + 1) total (381,383 slots) and the skeleton's
    forty-ring-volume x executions total are both named and pinned as wrong.
  * **rotation timing is copy-only.** `measure_rotation` sums the durations the closure attributes to
    its `rotate_window_ring` calls alone and reports the loop wall separately; the device closure
    times only those calls, and a test sleeps 60 ms between copies and asserts the sleep appears in
    the wall but not in the copy time. A dispatch or read-back can no longer be charged to the copy
    rate. The first 2026-10-18 window's rotation seconds and MiB/s came from the contaminated
    first-commit clock, so those numbers are superseded and **deliberately not quoted anywhere**;
    the corrected path's copy-only measurement is recorded in section 7 below.
  * the per-layer attention-kind table and the dispatch plan for the compared positions: 360 calls =
    18 window-only (format 1, layers 0-1) + 342 union (format 2, 38 layers); at position 129 the
    union calls carry 65 selected ids on the 18 ratio-2 layers and 130 on the 20 ratio-1 layers; the
    per-position **selected counts (window rows are not included in these numbers)** are 3770, 3790,
    3828, 3848, 3886, 3906, 3944, 3964, 4002.
  * `--dry-run` (host only, no CUDA): builds the schedule/rotation/dispatch plans, bounds-checks every
    union call through the shared `Format2AttentionCall::params()` builder (which now carries real
    indexer buffers instead of nulls), builds the expanded compressed views, and exercises the
    token-then-logits comparison plumbing against the fixture with three negative controls (wrong
    token, out-of-bar logits, position outside the compared set are all refused). It writes the same
    report shape the device run writes. The loader pins the fixture hash first.
  * `--prefill-only` is implemented: it narrows the compared set to the prefill position 129 for both
    the dispatch plan and the comparison, and the report labels the narrowed set (it used to be a
    parsed-but-ignored flag).
  * `--device`: measures the rotation sweep with real device-to-device copies on the corrected row
    arithmetic, reads every compared position's oldest-first view back and compares it byte-for-byte
    with the host slot order, then runs the dispatches on harness-owned deterministic inputs,
    comparing each prefill-position dispatch against the CPU twin (bar 2e-2) and repeating each
    decode dispatch to prove byte-identical output. It reads no weight and produces **no token and no
    logits delta**; the report states that the 9-position comparison was NOT attempted. The dispatch
    line counts every launch (first launches plus determinism repeats), not just records.
* **Device window 2026-10-18T18:32Z: the corrected 2B plumbing run (measured).** This is the
  run of record for 2B's device half. Binary built from merged main `1ceeb54` (squash of !63),
  aarch64 sha256 `8645b62104324d22fec1f64f222acc2052f745fa5b8e717d49c27be3e9b0f534`; fixture
  2,536,194 B `e6df8290...`; flash_mla cubin `flash_mla_csa2_sm121.cubin` 60,792 B `5bfee4c7...`;
  checkpoint snapshot `1bae0e10...`; run on `node0` in a Lead-sequenced vLLM-down window. The
  argv and every sha are recorded in the report itself. Artifacts:
  `/home/user/phase2-evidence/window-2026-10-18/runs2/m8-sequence-report.txt` (65,518 B, sha256
  `3fc6683a5d6c6a3bb1d25ad1eb03f77f6910b84de418231b38413eea4fb1aa6f`),
  `runs2/m8-corrected-device.log`, window log `window-b2-20261018T183240Z.log`.
  * **rotation, measured**: 5,482 token-layer executions (prefill 5,162 / decode 320); the plan's
    376,576 slots x 528 B = 198,832,128 B (codes 192,806,912 B + scales 6,025,216 B). The run issued
    **11,764 `copy_dtod` calls and moved 198,832,128 B in 0.025624 s of copy time**, i.e.
    **7,400.000 MiB/s copy-only**. Copy time is the sum of the `rotate_window_ring` call durations
    only; the loop wall 2.588988 s also covers view verification and the attention dispatch. One full
    ring is 67,584 B and the per-execution bound is 67,584 B. 360 oldest-first views were read back
    and compared byte-for-byte with the host slot order (24,330,240 B verified), zero mismatches.
  * **dispatch, measured**: 360 calls (342 union format 2 + 18 window-only format 1) over 9 positions x
    40 layers; grid (64 heads, 1, 1), block (256, 1, 1); 0 non-finite outputs; 40 device-vs-twin
    comparisons at position 129 all within the 2.0e-2 bar (worst max_abs 9.718239e-4 at layer 0, head
    5, channel 48; 32,766/32,768 exact bf16); 320/320 repeated-launch determinism checks
    byte-identical; dispatch wall 0.175593 s over the 360 first-launch records (+320 untimed repeat
    launches, 680 launches in all) = 487.758 us per first-launch record.
  * **buffers**: 9,926,144 B across 176 device buffers (m8's attention/KV plumbing set - a lower bound;
    no weights and no head are allocated by this run).
  * **what did NOT happen**: the model forward is not wired, so no device token was produced at any
    compared position 129..137 and no logits delta exists; the token-then-logits comparison was NOT
    attempted. This is a plumbing run, not a model forward, and it closes no exit criterion.
  * **vLLM handling in the window**: vLLM was stopped on all four nodes at 2026-10-18T18:32:40Z, the
    run completed, and vLLM was restarted at 18:34:00Z; it was healthy at 18:43:01Z after 550 s
    (`GET /v1/models` 200) with restart exit 0.
* **Performance: the rotation copy is measured, and it is small (task-10 baseline).** 11,764
  copies for 376,576 slots in 0.025624 s is **2.178 us per copy call and 4.674 us per token-layer
  execution, 7,400.000 MiB/s copy-only**. The same window's m7 stage-2 attention launch at 130 tokens
  was p50 0.2492 ms, so the whole 130-prefill + 8-decode rotation sweep costs about 103 such attention
  launches and one copy call is about 114x cheaper than one attention launch; the dispatch wall in
  the same run was 0.175593 s, so copy time is about 6.9x smaller than the dispatch+read-back wall.
  This answers the question task-10 was filed for: per-call driver overhead is real, but the sweep is
  a small share of the plumbing run. Batching each contiguous run into one pitched `cudaMemcpy2D` per
  view or a device-side copy kernel driven by the `RingSegment` table (with `host_ring_view` kept as
  the oracle) remains a Phase 3 optimization with this measurement as its baseline. The FP4 stage-2
  run in the same window also reproduced the merged-chain equality: 163838/163840 exact, max abs
  delta 2.4414063e-4, max ULP 1 (`runs/c-stage2-fp4-130.log`).
* **Host seams still open** (each is labelled in every report): the HSI pool population and the
  top-512 candidate merge are harness-supplied (the harness passes the reachable compressed rows
  directly); the expanded compressed view is built on the host; the query, window rings, compressed
  store, sink and indexer buffers are deterministic harness-owned content, not model state; and the
  per-layer forward chain is absent. The kernel stage-1 indexer buffers are zeroed host buffers, so
  the dispatch checks the ABI and layout, not the HSI selection. Format-2 dispatches are built and
  bounds-checked through the shared `Format2AttentionCall` builder, and format-1 window-only
  dispatches go through `cuda_harness::flash_mla_params` directly because that builder is format-2
  only; both take the 19-word order from the same constructor.
* **The format-2 builder defect is fixed.** Commit 0d715bd added
  `crates/sparkflash-kernels/src/sequence_chain.rs` in the !63 branch (the earlier note's claim that
  the module was outside that branch's write scope was wrong); `Format2AttentionCall` now carries the four
  indexer `Deviceptr` fields, passes them to their ABI words (8, 9, 11, 12; the sink stays at 10) and
  refuses nulls, so `params()`/`launch()` can issue a real format-2 launch. Tests in both crates pin
  the pointer words and the null refusal.
* **Example tests now run in CI.** `cargo test --workspace` builds examples but does not run their
  `#[cfg(test)]` blocks, so the 32 m8 invariants now run in a gate; `.gitlab-ci.yml` carries a
  `test-examples` job running `cargo test -p sparkflash-kernels --examples` (no GPU, no checkpoint).
* **Task-15 layer-0 smoke is NOT 2B evidence.** The 4-position layer-0 device smoke recorded a
  last-position residual within the bar and six tight row-0 bars slightly exceeded, and its 130-token
  run was cut off by the window. It is a single-layer plumbing check, not the 40-layer CED
  prefill/decode comparison this milestone needs.
* **2B device plumbing has run; the 2B device comparison has NOT.** The corrected run above measured
  the rotation and the dispatch and produced no device token at any compared position, so no token or
  logits delta exists for this milestone. The remaining gap is the forward chain (embedding ->
  40 layers -> lm head -> greedy selection at each compared position); with it wired, the same command
  can compare tokens first and logits at 2e-2. Phase 2 exit criterion 1 stays OPEN.
* **Exact device command** (built with `scripts/build-aarch64.sh --example=m8_sequence` from merged
  main `1ceeb54` and staged on the node next to the window cubins; `--out` is stamped per window):

```text
# The command that produced the run of record (verbatim from the report's argv line):
$ /home/user/phase2-evidence/window-2026-10-18/m8_sequence_1ceeb54_aarch64 --device \
  --checkpoint /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --index /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170/model.safetensors.index.json \
  --golden /home/user/phase2-evidence/sequence-130-released.json \
  --cubin-dir /home/user/phase2-evidence/window-2026-10-18/cubins \
  --out /home/user/phase2-evidence/window-2026-10-18/runs2/m8-sequence-report.txt \
  --provenance node
```

  Build with `scripts/build-aarch64.sh --example=m8_sequence`, stage the binary on the node next to
  the cubins, and stamp `--out` per window; the report records its own binary sha256 and full argv,
  so a report can always be matched back to the binary that wrote it.

  The report states rotation and dispatch separately and lists every per-step token as NOT PRODUCED
  until the forward chain is wired.
* **No exit criterion is claimed.** Phase 2 criterion 1 (`roadmap-to-serving.md` section 5.5.1:
  prefill over 128 tokens and 8 decode steps, every decoded token reproduced token-for-token
  against the reference) remains OPEN. The precise remaining gap is the forward chain: embedding ->
  40 layers -> lm head -> greedy selection at each compared position, consuming the rotated window
  ring and the format-2 dispatch 2B wires (merged). Criteria 2-5 are likewise unclaimed.

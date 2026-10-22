# Acceptance-gate rehearsal: the failure modes that must not cost a window

**Status: written 2026-10-19 (task-48 / board task-46).** Host-only: no device, no vLLM instance, no
network, no GPU. The window is expensive and unrepeatable, so this file and the tests beside it make the
machinery demonstrate its refusals instead of describing them.

## What is being rehearsed

The acceptance chain has two halves, both runnable offline:

    synthetic capture --scripts/d41_floors.py--> D41 gate bars (sparkflash-d41-bars/1)
    dump + .bin sidecar + oracle --scripts/compare_native_vs_vllm.py--> verdict

The rehearsal (`scripts/tests/test_gate_rehearsal.py`) drives the **real scripts as subprocesses**, in
temporary directories, exactly as `docs/build/window-runbook.md` does. Its valid bars are one real
`d41_floors.py` run over the synthetic capture builders in
`scripts/tests/test_d41_floors.py` (the builders are imported, not copied; the D41 test file is not
edited). Its synthetic dump/oracle pair uses 1,000 vocabulary and the pinned positions 129..137, with
`--expect-vocab 1000`; the released vocabulary (129,280) appears only in the wrong-vocabulary case.

Run it with:

```sh
python3 scripts/tests/test_gate_rehearsal.py -v
python3 -m unittest discover -s scripts/tests -v
```

## Covered failure modes

Every row is asserted to produce a non-zero exit and no evidence file, except the flagged run, whose
evidence file is written and must not say PASS or UNBARED.

| failure mode | how the rehearsal produces it | expected result | test |
| :-- | :-- | :-- | :-- |
| wrong vocabulary | dump rows carry vocab 1,000 while `--expect-vocab 129280` is declared | compare exits 2, names the vocab mismatch, writes no JSON/Markdown | `test_a_wrong_vocabulary_dump_is_refused` |
| non-tiling offsets | one row's `logits_offset` is shifted by 4 bytes | compare exits 2, names the tiling violation, writes nothing | `test_non_tiling_offsets_are_refused` |
| sidecar basename mismatch | a row's `logits_bin` names another file, and a second variant passes `--bin` with a foreign basename | compare exits 2, writes nothing | `test_a_row_naming_another_dumps_sidecar_is_refused`, `test_a_sidecar_with_another_dumps_name_is_refused` |
| missing `reference_id` | every row's `reference_id` is removed; metrics would otherwise pass the bars | compare exits **1** with verdict `NOT_LIKE_FOR_LIKE`, `prefix_mismatches == prefix_unknown == 9`, and an evidence file that says so (never PASS/UNBARED) | `test_a_missing_reference_id_is_not_like_for_like` |
| positions list not equal to 129..137 | dump 130..138 against the pinned list (129 missing); oracle 129..136 against the pinned list (137 missing); a duplicated position in the list | compare exits 2, names the missing/duplicate position, writes nothing | `test_a_positions_list_the_dump_cannot_satisfy_is_refused`, `test_a_positions_list_the_oracle_cannot_satisfy_is_refused`, `test_a_duplicate_position_list_is_refused` |
| bars file with an unknown name | the valid D41 bars file plus a `token_equality` bar | compare exits 2, names the unknown bar, writes nothing | `test_an_unknown_bar_name_is_refused` |
| demo-scale synthetic bars | the semantic content of `/home/user/phase2-evidence/demo-scale/bars.json` (plain `{"bars": ...}`, no schema/status/bar_class) | compare exits 2, "not a D41 gate bars file", writes nothing | `test_the_demo_scale_bars_file_is_refused` |
| capture with a missing class | the standard three-block capture with the `idle_distractor` block removed, `d41_floors.py --require-classes repeat,idle_distractor,busy_distractor` | floors exits 2, names `idle_distractor`, writes no JSON, Markdown or bars | `test_a_capture_missing_a_required_class_is_refused` |
| floors run emitting `requires_review` | the outlier construction of `test_d41_floors.py::test_review_status_exits_three_and_writes_outputs` | floors exits **3** and writes a bars file with `status: "requires_review"`; feeding that file to the comparison exits 2, names `requires_review`, writes nothing | `test_a_requires_review_floors_run_cannot_gate` |

The positive control (`test_a_correct_synthetic_run_passes`) is the other half of the proof: the same
fixtures and the same bars file produce verdict PASS with evidence, so the rehearsal is not a suite that
fails everything. It also exercises the two scripts against each other: the bars the comparison consumes
are exactly the bytes `d41_floors.py` wrote.

## What the consumer now refuses (the gate contract)

These checks live in `scripts/compare_native_vs_vllm.py` and are exercised by this rehearsal. The
consumer's `evaluate_bars` remains generic for unit tests; the gate-shape checks are in `main()`.

- **`--bars` is a gate input, not a hint.** It must be an object with
  `schema == "sparkflash-d41-bars/1"`, `status == "ok"`, `bar_class == "idle_distractor"` and a non-empty
  `bars` dict. A demo-scale file, a hand-copied provisional file and a `requires_review` D41 file are all
  refused (exit 2, nothing written) instead of being applied.
- **`--expect-vocab`** declares the checkpoint's vocabulary. Acceptance runs pass `129280`; a dump row
  with another vocab is refused before any metric is computed. It has no default, so synthetic runs
  without the flag keep their historical behaviour.
- **`--positions`** must name positions that exist in **both** the dump and the oracle. A requested
  position that is missing is refused; the list is no longer silently intersected, and duplicates or
  non-integers are refused. Without `--positions` the legacy intersection behaviour remains (used by the
  synthetic unit tests, not by the gate).
- **A prefix that cannot be proven fails the run.** `prefix_match` is tri-state; a row with `UNKNOWN`
  (no usable `reference_id`, or no oracle `target_token`) or `MISMATCH` counts in `prefix_mismatches`,
  and any such row forces verdict `NOT_LIKE_FOR_LIKE` and exit 1. The evidence file is written with the
  counts and the negative verdict; it is not suppressed, because it is the record of why the run cannot
  be used.

The runbook command gains `--expect-vocab 129280`, the pinned `--positions 129,130,...,137` (already
there) and no new bars flag: strictness is the default. `docs/build/window-runbook.md` §1/§3/§4 now
carry the same contract (audit-dev's runbook commit on this branch); this file did not edit the
runbook itself, to keep the docs change in one owner.

## What cannot be rehearsed offline

Named rather than claimed. These still need the window or depend on state this host does not have:

- **The device dump itself.** The rehearsal writes the dump bytes by hand; `write_logits_dump` and the
  chain's own buffer sizing/offsets are unit-tested in Rust, but "the real chain writes 9 rows at
  129..137 with vocab 129,280 and a 4,654,080-byte sidecar" is a window fact (docs/build/logits-dump.md,
  O6). A rehearsal cannot prove a real run produced it.
- **The real oracle.** `oracle_native.py`'s request, `prompt_logprobs` handling, the served-root check
  against a live `/v1/models` (D38) and the tokenizer/suite pinning cannot run without the released
  server. The rehearsal's oracle is a synthetic JSON with the same schema; it proves the consumer's
  handling, not the oracle's production.
- **The real D41 capture and floors values.** The idle proof needs live `/metrics`, and the measured
  floors depend on the actual oracle's self-consistency. The rehearsal proves the *shape and refusal
  paths* of the bars, not that the Lead's eventual 129,280/top-k/KL values are correct. D41 bars are
  floor-derived from a real capture; a synthetic capture's numbers are not gates.
- **Numeric acceptance.** D39 makes acceptance distributional (top-1, top-k, KL/JS, |delta|) against the
  released oracle; that comparison cannot be made without either side's real output. The rehearsal
  proves the comparison *fails closed* on malformed inputs, never that the chain is numerically close.
- **The staging/manifest layer.** Binary and cubin sha256s, the 19-cubin set, the aarch64 preflight and
  the `window-artifacts-*.txt` manifest are window artifacts; a missing/wrong cubin or a binary built at
  the wrong tip is detected by the runbook preflight, not by this rehearsal.
- **Operational failure modes.** GPU OOM, a killed run, a partial dump (the sidecar length check catches
  truncation, but not a crash mid-write), vLLM restart failure, client traffic during the window and
  ssh/launch failures are Lead-owned window operations (docs/build/window-runbook.md).
- **Selection/discrete stages.** The indexer top-k / reachability questions are task-12 (N9) and are not
  comparison inputs.
- **A discrete selection flip.** The released oracle records no `topk_idxs` (D39), so a selection flip
  has no offline reference; it needs the pre/post long-context measurement named on task-12.

## Records

- Rehearsal: `scripts/tests/test_gate_rehearsal.py` - 13 tests, host-only, ~4.5 s.
- Consumer changes: `scripts/compare_native_vs_vllm.py` (`parse_positions`, `check_gate_bars`,
  `--expect-vocab`, strict `--positions`, `NOT_LIKE_FOR_LIKE`; strict gate-shape check in `main()`).
- Base: rebased onto batch-9 main `0d9bf57`; the original base was
  `origin/task28-window-seam-fixes` (`e785850` = main `134b444` + audit-dev's nine commits), whose
  content batch 9 squash-merged (`0d9bf57` differs from it only by the D42/serving docs). The rebase
  replayed the branch's four commits with no content conflicts, because the three files the batch-9
  squash could have rewritten are byte-identical between `e785850` and `0d9bf57`.
  audit-dev's follow-up on this branch (`f433011` unit tests for the new gate contract, `c04a584`
  runbook §1/§3/§4) closes the two expected discoveries: the consumer unit tests now use a
  gate-shaped bars fixture and assert the `NOT_LIKE_FOR_LIKE` exit, and the runbook command carries
  `--expect-vocab 129280`.
- No window, no device, no instance was used. No arithmetic or kernel change is claimed.

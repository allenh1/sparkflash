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

# Coverage dumps and the growing suite

No real dump is committed yet, so `[prerequisites] coverage` stays false. Do not add a dump that
was not produced by a serving run; a synthetic fixture belongs in the self-tests, not here.

## The bar, and the judge

`numerics/registry.toml` declares the agreed bar:

```toml
[prerequisites]
coverage_slots = 0.93
```

It is the fraction of (layer, routed-expert) slots a committed dump must reach, not a count, and
it is deliberately below 1.0. Real serving reached 92-93% of Qwen3.8's 48x512 slots after 240-330
inputs and every layer still carried gaps, so "every routed expert of every layer" was an
unstated bar a real run cannot meet. 0.93 is the owner's agreed value; the bar stays in the
registry so no reader has to infer it.

```bash
python3 scripts/numerics.py coverage --dump numerics/coverage/<model>-<world>.json --min-inputs 8
python3 scripts/numerics.py coverage-gaps --dump "$W/qwen38-coverage.json" --json "$W/gaps.json"
```

`coverage` passes when the dump reaches at least `coverage_slots` and exits non-zero below it.
Either way it prints the achieved fraction, the declared bar and the missing expert ids per layer,
so a reader can see how much headroom is left. It refuses a dump whose model and layout disagree,
and a registry whose bar is missing or outside (0, 1]. `coverage-gaps` is the same inspection
without the gate: it prints the missing count per layer and writes every missing expert to JSON;
it does not decide a prerequisite. The 48x512 Qwen3.8 layout has 24,576 routed slots; the two
cluster runs left about 1,720-1,966 of them unreached after 240-330 inputs.

`[prerequisites] coverage` turns true only in a reviewed edit that also sets
`coverage_dump = "numerics/coverage/<dump>.json"` and commits that dump. `scripts/numerics.py ci`
then judges the named dump at `coverage_slots` and fails a true flag whose dump is missing, names
another model, or is below the bar. A false flag makes no claim and names no dump.

## The sweep

[`scripts/numerics.py coverage-sweep`](../../scripts/numerics.py) sends a deterministic corpus to a
server that is writing the dump, reads the dump before and after each prompt, and keeps a prompt
only if its forward reached a slot that no earlier kept prompt reached. The measurement is the
selection; the generator only proposes candidates.

The checked-in corpus is [`corpus.json`](corpus.json) (character classes, code, numerals, non-Latin
scripts, long documents, repetition). `--generate N` adds a deterministic generated corpus from the
same families, seeded by `--seed`.

The rules:

- **Deterministic.** The corpus order is the file order; `--order shuffled` sorts by
  `sha256("<seed>:<name>")`. The generator is a pure function of `--seed` and its index. A rerun
  with the same server and dump makes the same decisions.
- **Measured.** `kept` means the prompt's hit delta contained a slot outside the selected union.
  A prompt that reaches nothing new is `discarded`; an empty delta is reported separately.
- **No duplicate inputs.** A prompt whose content-addressed input name is already in the dump is
  skipped without a request. Every prompt a sweep sends is written to the seen manifest
  (`DUMP.seen.json` by default), and a later sweep skips it by content digest. A legacy dump whose
  inputs are `input-0`, `input-1`, ... carries no content, so the sweep refuses until you pass
  `--input-suite FILE` (the suite that produced those inputs) or an explicit `--seen` manifest.
- **Model and layout are checked.** The dump must name the layout's model, every layer must be
  present with exactly the layout's routed-expert count, and the live server must name the same
  model. A mismatch is a refusal, not a warning.
- **Marginal coverage is reported.** Every `--batch` sent prompts produce a row: sent, kept,
  discarded, new slots, missing before/after and coverage before/after. The final JSON is stable
  and can be compared across windows.

`--out` is the next batch: a `sparkflash-numerics-prompts/1` suite of just the kept prompts, in
selection order. It reproduces the sweep's forwards with greedy decode and the same `--max-tokens`
(default 1); feed it to `capture --max-tokens 1`, or keep it as the reproducible subset that added
the measured slots. The base dump's inputs are not in it: reproduce the whole dump by appending to
that dump, or prepend the `--input-suite`. The dump itself remains the evidence and includes every
prompt the sweep sent.

## Estimate for Qwen3.8

24,576 routed slots. The two real runs reached 92-93%, leaving about 1,720-1,966 of them (the
judge's rounded gap is 1,800-2,000). Taken as a bracket (240 inputs at 92%, 330 at 93%), the
difference is 246 slots over 90 inputs, about 2.7 new slots per input at the tail, so closing the
remaining gap needs roughly 650-750 *contributing* prompts. A greedy sweep discards non-contributors, so budget 1,500-2,500 measured candidates for
650-800 kept. This is an estimate from the two runs' marginal rate, not a measured coverage claim;
the only real numbers are 92-93% after 240-330 inputs.

## The window command

Run it on an idle Qwen3.8 world-1 server that is already writing the dump. The starting dump is a
copy of the partial one; the sweep appends to it.

```bash
W=/home/allenh1/cluster-tests/$(date +%Y%m%dT%H%M)-coverage-sweep
mkdir -p "$W"
cp "$QWEN_COVERAGE_DUMP" "$W/qwen38-coverage.json"

# Only for a dump whose inputs are legacy input-0/input-1 names.
INPUT_SUITE_ARGS=()          # e.g. INPUT_SUITE_ARGS=(--input-suite "$QWEN_OLD_SUITE")

SPARKFLASH_MODEL=qwen3.8-flash-next-nvfp4 \
SPARKFLASH_CHECKPOINT="$QWEN_CHECKPOINT" \
SPARKFLASH_TOKENIZER="$QWEN_TOKENIZER" \
SPARKFLASH_CONTEXT=65536 \
SPARKFLASH_HTTP_BIND=127.0.0.1:8002 \
SPARKFLASH_NUMERICS_COVERAGE="$W/qwen38-coverage.json" \
SPARKFLASH_NUMERICS_MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4 \
  scripts/serve-single-node.sh launch
# Wait for /readyz.
python3 scripts/numerics.py coverage-sweep \
  --url http://127.0.0.1:8002 \
  --dump "$W/qwen38-coverage.json" \
  --layout numerics/layouts/qwen38.json \
  --corpus numerics/coverage/corpus.json \
  --generate 2000 --seed 20261119 --order shuffled \
  --batch 64 --max-candidates 256 \
  --out "$W/next-batch.json" --json "$W/sweep.json" \
  "${INPUT_SUITE_ARGS[@]}"
scripts/serve-single-node.sh stop

python3 scripts/numerics.py coverage-gaps --dump "$W/qwen38-coverage.json" --json "$W/gaps.json"
python3 scripts/numerics.py coverage --dump "$W/qwen38-coverage.json" \
  --layout numerics/layouts/qwen38.json --min-inputs 8
```

The seen manifest belongs to the dump and makes each window continue where the last stopped; delete
it when you start a fresh dump. Rerun the same command until `--max-candidates` exhausts the
corpus, then raise `--generate` or change `--seed` for more candidates. A rerun is deterministic
given the same dump, server and seed.

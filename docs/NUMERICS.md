# Numerics

**The rule:** nothing about a served configuration's arithmetic changes unless someone declared
it. A declared change must reproduce an independently written reference of the new arithmetic
and stay within a measured distance of a frozen reference. Everything else is a bug, and the
bit gate catches it the way it always has.

The bit gate exists to prove the implementation is correct and that model quality never degrades
silently. It was never there to block speed. This policy keeps that guarantee for every
accidental change exactly as it was, and lets a deliberate change land only with evidence at
least as strong as the bit gate. The decision is [D67](DECISIONS.md); the issue that argued it is
#7.

## 1. What stays exactly as it was

1. **Bit-exactness is the default gate for every change.** A change that declares nothing must
   reproduce the current references bit for bit. A bit change in such a change is a bug.
2. **The invariances are hard gates in every configuration:** run-to-run determinism, batch and
   concurrency invariance, speculation never changes the output, chunk-width invariance, and a
   resumed request matches a fresh one. They cost almost nothing (the batch-invariant tensor-core
   decode made eight-stream steps faster) and they are the guarantees users see.
3. **Frozen references are never edited quietly.** An anchor (§4) is never edited at all.

## 2. Declared changes

A deliberate change to the arithmetic is a declared event. There are two kinds:

| Kind | Examples | What it changes | Quality risk |
| :-- | :-- | :-- | :-- |
| Reassociation | split-K, a different TP split, expert-parallel sums, autotuned tiles | the order of FP32 sums only | none systematic, but greedy text can flip at a near-tie |
| Precision | narrow projections on tensor cores with BF16 activations, FP8 or FP4 activations | the values themselves | real, and it grows with generation length |

A declaration is all of:

- **A bumped numerics version** (§5) for each affected configuration, by exactly one.
- **An independent reference of the new arithmetic**, written from the checkpoint's published
  recipe, not derived from the kernel. Quantized operands (FP4/FP8 codes and their scales) must
  match it bit for bit: quantization is elementwise and deterministic, and this is where a scale
  applied twice or a wrong scale group lives. Where the summation order is ours, the accumulated
  output must match a reference that sums in the same order, bit for bit; where the tensor core
  fixes the order inside an instruction, it must stay within the FP32 bound against f64 math,
  about `K x 6e-8 x sum|a_i b_i|`. A pure reassociation has no new operands, so only the second
  rule applies to it.
- **A quality report** against the frozen anchor (§4), committed under `docs/numerics/<model>/`
  as a new file and named by the registry entry.
- **Regenerated references** in the same change: the bit fixtures and the registry reference.

Defaults:

- A change that reduces precision is **opt-in only**, as a named numerics profile, with its
  measured deltas published.
- The default arithmetic changes only by an explicit maintainer decision recorded in
  [DECISIONS.md](DECISIONS.md), never automatically because a bar passed. One bump may record that
  decision as its `approved` block in the registry (§5).
- An unknown `--numerics` value is refused. The one profile today is `reference`.
- DeepSeek production does not change arithmetic until it has been measured under this policy.

## 3. What catches each kind of mistake

| Mistake | What catches it |
| :-- | :-- |
| A refactor changes the math by accident (bad index, missed tile, race) | the bit gate: changed bits without a declaration fail |
| Output depends on batch, concurrency or speculation | the invariance tests |
| Nondeterminism | the run-to-run test |
| A declared change implemented wrong | bit-for-bit comparison of the quantized operands against the independent reference |
| A wrong partition in a reordered sum | bits against a reference in the new order, or the f64 bound |
| A correct change that hurts quality more than expected | the teacher-forced comparison against the anchor; it ships only by decision |
| Slow drift from many small changes | every version is measured against the same anchor, never its predecessor |
| A path no test reaches | the coverage inputs (§4) |

After a declared change, the next refactor is bit-gated against the new version's references.

The one residual risk is a kernel and an independent reference that misread the spec the same way.
The bit check then passes and only the quality comparison stands in the way. Independent
authorship, an external cross-check where another engine runs the same arithmetic, the known
error shape of FP4/FP8 rounding (a misread looks like a bias or a spike in one expert, which the
monitors flag) and the rarity of these changes are the mitigations.

## 4. The frozen quality reference

- **Anchor.** One frozen reference per model, captured at high precision (today's arithmetic at
  the smallest world size, or BF16 activations). Every numerics version of the model is measured
  against it. Anchors are `[[anchor]]` entries in the registry; the declaration check refuses any
  edit to one, its removal, or a change to its file. The anchor requirement is **per model**: a
  configuration may bump only when its own model has an anchor and a committed capture
  teacher-forced from that anchor. A model whose evidence is `text` has no token ids to force and
  is exempt from the anchor requirement, but it also cannot bump: there is no comparison that
  would justify the change. That is DeepSeek's position, and it is why the aggregate
  `[prerequisites] anchor` flag can be false without blocking a model that has its own anchor.
- **Teacher-forced comparison.** The anchor's own generated tokens are fed through the candidate
  (`forced_tokens`, §6), and every position's distribution is compared: NLL delta, top-1
  agreement and a KL lower bound. Both arms see the same tokens, so a near-tie changes one
  position's score instead of derailing the rest of the text, and both arms are deterministic,
  so the result is too.
- **Per-layer and per-expert error monitors.** Relative error against the reference at every
  layer and every shared routed expert, so a bug in one expert shows as a spike when the
  end-to-end KL looks fine; a row that reached a different expert is reported beside the error,
  because the router's choice is itself a measured fact.
- **Coverage inputs.** Prompts that reach at least the registry's `coverage_slots` of the
  routed (layer, expert) slots, chunk boundaries, 32k+ contexts, several batch shapes and
  speculation depths. The suite is grown and measured by `coverage-sweep` (§5): the generator
  proposes candidates, the dump's hit delta selects them.
- **Bars.** The TP=1 to TP=4 divergence already shipped is the reassociation floor: a change at
  or below it cannot be told apart from what is served today. A precision change above it is a
  real quality change, and its report says so in numbers.
- **Baselines.** The trades already shipped (GLM-5.3's E4M3 attention window, Qwen3.8's 16-bit
  activations, a BF16 head and remainder, against the checkpoint's W4A4 recipe) each get a
  measured entry against the anchor.

The KL the tool reports is a lower bound on the true KL. A server returns the top-k logprobs, not
the whole row, so the comparison partitions the vocabulary into the tokens both arms list plus
one bucket for everything else. A coarser partition never has more KL than the full vocabulary.

## 5. Mechanics

### The registry

[`numerics/registry.toml`](../numerics/registry.toml) holds one `[[config]]` per served
configuration, keyed by model, world size, numerics profile, parallel plan (`tp`) and CUDA
architecture (or `host`):

| Field | Meaning |
| :-- | :-- |
| `version` | starts at 1; each declared change bumps it by one |
| `evidence` | `logprobs` (every token's id, logprob and top-k) or `text` (greedy text, for a path that serves neither logprobs nor ids) |
| `reference` | the capture under `numerics/references/` |
| `fingerprint` | sha256 of the capture's canonical bits |
| `report` | the quality report under `docs/numerics/`: required from version 2; a version-1 entry may name the baseline report that measured it |
| `captured`, `note` | when, and what a reader should know |

`reference` and `fingerprint` are omitted together for a configuration that is registered but
not yet captured: the binary reports it `unregistered`, and `check` refuses with "no reference
captured yet" rather than comparing. A first capture fills both in; from then on a changed capture
is a declared change.

An entry may also carry the owner's approval of one declared change, as a nested table:

```toml
approved = { decision = "tensor-core decode is an approved arithmetic change", date = "2026-11-18", report = "docs/numerics/qwen38/v2-approval.md" }
```

The approval replaces the per-model coverage and monitors prerequisites (§5). The declared change,
the entry's own `report` and the per-model anchor rule still apply. It is part of the record, like
a version: the declaration check refuses its removal, and refuses an empty `decision` or `date`, a
`report` that does not exist, and an approved bump that changes nothing.

`[[retired]]` keeps a removed configuration's last version, so removing and re-adding one cannot
reset it. A reference's `server` card is what the capturing binary reported, and it is outside the
fingerprint: the registry can name a capture's fingerprint only after the capture exists, so a
first capture reads `unregistered`, and a recapture names the version and fingerprint it replaced.
The registry entry is the record; `check` and `ci` accept the card's shape and never compare the
card against the fingerprint, though `ci` does require a new capture's card to name its build.
The card names the build the capture came from. `binary_sha256` is the sha256 of the server
binary named by `capture --binary` (or `SPARKFLASH_BIN`). `commit` is the served card's
`build.id`: the commit the binary was compiled from, with `-dirty` appended when the source tree
was dirty at build time, so a dirty build is recorded as dirty rather than silently named exact.
`capture --repo` (default: the checkout the tool runs from) is consulted only when the served card
names no commit - including a card that reports `unknown` - and `server.build_source` says which
of the two the commit came from (`card`, `repo`, or `null` when neither named one). `capture`
warns on stderr when the served card names no build commit, and `build_note` gives the one-line
reason a value is null; a value is never invented. The card stays outside the fingerprint, so
naming a build changes no bits.

`server.build_source` is also the build record's version note. A reference that predates it has no
such key and is grandfathered: the declaration check judges only a reference or `[[anchor]]` that
this change adds, and a new one whose card carries `build_source` but no commit (or says
`unknown`) is refused. The capture command writes the key unconditionally, so a new capture is
always judged; an old reference is accepted without it. `text` evidence is weaker than
`logprobs`: an arithmetic change that flips no greedy token is invisible to it. DeepSeek's TP=4
device path is the only `text` configuration.

### The correctness floor

A first reference or a re-declaration must show that the arithmetic it pins is right, not merely
repeatable: two identical captures of token soup are still soup. Inkling's version 1 was pinned on
"two captures identical and the lanes gate passed" and froze degenerate output, so a pin now carries
a floor taken outside the tool: `capture --reference` and `capture --anchor` refuse to write
without one.

The card's `floor` object sits outside the fingerprint, like the server card, and holds:

- `against`: what the capture was measured against - the pre-change build, or the arm's host
  reference where one exists.
- `forced`: the teacher-forced comparison's `positions`, `mean_nll_delta`,
  `mean_abs_delta`, `max_abs_delta` and `top1_agreement`.
- `known_answers`: the fixed prompt set's greedy decodes, each with its prompt, expected text,
  the text the model produced, and whether it passed; at least one names an expected answer.
  The probe goes through the chat template and says what it wants ("Answer with the number only:
  what is 2 + 2?"), not a bare continuation ("2+2="), which asks the model to continue web text:
  version 2's floor recorded "5 ... I have a 2+2=5 problem" for the bare form, which is a
  corpus completion, not an arithmetic answer. A failed known answer is fixed by changing the
  probe or the model, never waved through because two builds agree.
- `nll_per_token`: the reference's own mean NLL on its natural-text prompts.
- `explanation`: null, or the reason a bar below is deliberately exceeded.

Unless a model's `[[precision]]` row tightens or loosens them, the bars are: top-1 agreement at
least `0.90`, mean absolute NLL delta at most `1.0` nat, max absolute NLL delta at most `20`
nats, and the reference's own NLL at most `2.0` nats/token. A distance far beyond the arm's
precision tail - Inkling's 47-nat swings and 29.49% top-1 were a layout defect, not precision - is
refused unless the card records an explanation beside the numbers, and a failed known answer is
judged the same way. The declaration check applies the rule to every reference and `[[anchor]]`
this change adds; a card without `build_source` predates both rules and is grandfathered.

An anchor is frozen, but a replacement may be recorded with
`replaces = { reference, fingerprint, reason }` naming the card and fingerprint it supersedes -
for an anchor that pinned a defect rather than the arithmetic. The old card stays in history,
unchanged, and the new card carries its own build and floor.

### The speed rule

Tensor cores are expected to lose a little at one row and win at several, so a one-row-only A/B
tests the wrong thing. One rule judges every arm, written here before any run:

> A tensor-core decode route is kept when, on alternated fresh-boot pairs, the served
> concurrency's aggregate tok/s is at least +2% better **and** single-request tok/s is no more
> than 3% worse, both pairs agreeing. Otherwise the arm keeps its row GEMV for that route,
> recorded with the numbers.

Served concurrency is the arm's default lane count (Inkling 4, GLM 4, Dory 16, Qwen its
default), with drafting as served. The measurement itself:

- a fresh boot for every sample, with the arm's launcher, checkpoint, launcher flags and
  drafting as served; the routes alternate across boots (A then B, B then A), at least two pairs;
- at each boot, after two warm-up requests, **single-request** decode: three greedy requests of
  256 tokens, tokens over wall time, median;
- **served concurrency**: one request a lane at the arm's default lane count, same shape,
  aggregate completion tokens over the wall time of the slowest lane, repeated; median;
- the aggregate decides; the single-request number is the guard; both pairs must agree. Record
  every number and the verdict (kept or held back) in the arm's numerics report, so a route held
  back says which bar it missed.

The rule applies to every route from now on, not only decode. A route that was already in
service when the rule was written is measured once at the next window and recorded; if it fails,
it becomes a declared version whose reference is the row-GEMV arithmetic.

### The served identity

The binary embeds the registry. At boot every serve path prints one line:

```text
sparkflash numerics: <model> world=4 profile=reference plan=tp arch=sm_121a status=registered registered_version=1 fingerprint=<16 hex> evidence=logprobs overrides=none
```

and `/v1/models` carries the same as `numerics` beside `build`. `build.id` is the commit
`crates/sparkflash-api/build.rs` embeds at compile time (with `-dirty` when the tree was dirty, or
`unknown` when the build ran without a git HEAD), `build.version` stays the package version, and
the serve boot line names the same commit. When the served entry records an
approval, the line appends `approved=<date>` and the card's `numerics` carries the `approved`
object (`decision`, `date`, `report`). The status is one of:

- `registered`: the registry has an entry and no arithmetic switch is set; `version` is the
  entry's.
- `unregistered`: no reference exists for this configuration, so nothing vouches for its bits;
  `version` is null.
- `overridden`: a switch that can change the arithmetic is set (`overrides` names it), so the
  registered reference no longer describes this process; `version` is null and
  `registered_version` says what the default would serve.

The arithmetic switches are the profile overrides whose field is in
`sparkflash_model::numerics::ARITHMETIC_FIELDS`, plus the arm switches not yet in the profile
table. An arm switch set to the value its arm takes when unset (production DeepSeek's
`SPARK_COMM_REDUCE=nccl`) selects the same arithmetic and keeps the claim; any other value
withdraws it. A test requires every profile override to be classified as arithmetic or as covered
by an invariance gate, so a new one cannot land unclassified.

### The declaration check

`scripts/numerics.py ci` runs in the `scripts-tests` CI job against the merge request's base (or
the push's previous tip, or the merge base with `main`). It fails:

- a reference whose bits changed without a version bump, unless the new capture only extends the
  old one (more prompts, or a generation that ran out of budget continued past it; a capture of
  the same length keeps every bit, its finish reason included);
- a version bump whose bits did not change (a reference moved to another path with the same
  fingerprint is a rename, which needs no bump);
- a bump by more than one, or a version that went down;
- a bump while a global prerequisite below (`declaration_check`, `coverage`, `monitors`) is false
  at the base or at the head and the entry carries no usable `approved` record, or while the
  bumping configuration's own model has no `[[anchor]]` or no committed capture teacher-forced
  from it (an approval does not replace the per-model rule);
- an `approved` record that is not a table, whose `decision` or `date` is empty, whose `date` is
  not `YYYY-MM-DD`, or whose `report` is not under `docs/numerics/`;
- an approved bump whose report does not exist, is not new or changed in this change, or that
  declares no change;
- an approval removed from an entry that carried one, or added without the bump it records;
- a bump on `text` evidence, which has no token ids to teacher-force;
- a bump without a new report;
- any change to an anchor, its removal, or a change to its file, unless the head anchor's
  `replaces` record names the base anchor's reference and fingerprint and a reason;
- a reference this change adds whose server card records a build (`server.build_source`) but
  names no commit or says `unknown`, or a `[[anchor]]` this change adds whose reference does the
  same. A reference without `build_source` predates the build record and is grandfathered;
- a reference or `[[anchor]]` this change adds that carries `build_source` but no correctness
  floor, or whose floor misses a bar without an explanation (see *The correctness floor* above);
- a registry the serving binary's own parser refuses: `ci` runs
  `cargo run -p sparkflash-model --example validate_registry` on `numerics/registry.toml`, so
  the Python gate and the binary share one parser. The version 2 declaration's `[[retired]]`
  reference passed the Python gate and aborted the next release at boot; this closes that gap;
- a fingerprint that does not match its file, or a reference file no entry names;
- a removed configuration without a `[[retired]]` entry;
- an edited or deleted file under `numerics/captures/`, which holds the evidence behind reports;
- a changed setting in an arm profile (`crates/sparkflash-model-*/config/*.toml`) that
  `sparkflash_model::numerics::NEUTRAL_FIELDS` does not list, at the base and at the head, unless
  every reference of that model is recaptured in the same change. The rules above then judge the
  recaptured bits. The profile names its model (`model`, held to its weights' descriptor by a
  `sparkflash-model` test), and a profile that names none fails. A profile moved to another path
  with every setting kept is a rename and needs nothing. Only Qwen3.8's profile is compiled into
  a binary today; GLM-5.3 and DeepSeek take their defaults from code, and their profiles are gated
  so that one an arm starts reading already is;
- a field that joins `NEUTRAL_FIELDS` without a new report under `docs/numerics/` that names it
  and the invariance gate that holds it. The check trusts that list to waive a recapture, so the
  list is judged in the same diff.

`scripts/numerics.py self-test` shows the declaration refusals on synthetic registries; the
unit tests under `scripts/tests/test_numerics.py` cover the per-model anchor and both dump judges.

The bits themselves are checked against a running server by `scripts/numerics.py check`, during a
window on the cluster, as the existing gates are. CI has no GPU or weights; it checks that what is
committed is consistent and declared. It sees the declared surfaces and every input it trusts (the
registry, the references, the profiles and the neutral list); an arithmetic change in code, a
kernel or a default set in Rust, is the bit gate's to catch.

### Prerequisites

No version may be bumped, which is to say no arithmetic change may land, until the global
`declaration_check` is true **and** the bumping configuration's own model has all three of its
per-model prerequisites — a frozen anchor with a teacher-forced comparison, its own `[[coverage]]`
row that passes, and its own `[[monitors]]` row that passes — **or** the entry carries an
`approved` record, the owner's decision that one declared change is expected, which replaces the
coverage and monitors prerequisites but nothing else.

`anchor`, `coverage` and `monitors` are **per model**, shaped like `[[anchor]]`: the registry
holds one row a model, and `ci` resolves the bumping entry's model to its rows. A model with no
row has that prerequisite missing, so it cannot lean on another model's evidence. The
`[prerequisites]` flags are the **aggregate report** over every model whose captured evidence is
`logprobs`: `ci` prints which models each flag covers, and a true flag that claims more than the
registry holds is a finding (a false flag with an uncovered model is consistent — the per-model
lookup is what refuses that model).

| Prerequisite | State |
| :-- | :-- |
| `anchor`: the frozen anchor and the teacher-forced comparison | per model. Qwen3.8, GLM-5.3, Inkling-2 and Dory1.0 each have a `[[anchor]]` and a committed capture teacher-forced from it; DeepSeek is exempt (its device path serves no logprobs, so a text reference has no token ids and cannot teacher-force), and a text-evidence change is refused rather than excused. The aggregate flag is **true**; `scripts/numerics.py ci` prints the covered, exempt and uncovered models, and refuses a true flag whose coverage is incomplete. |
| `declaration_check`: the CI check above | **done**, and global: it gates every bump. |
| `coverage`: inputs that reach at least `coverage_slots` of the routed (layer, expert) slots, with hit counts | per model. **Qwen3.8 world 1 has its row** — the swept dump reaches **24,494 of 24,576** slots (99.67%) over 416 inputs, judged at the declared **0.93** with `min_inputs = 8`. **GLM-5.3, Inkling-2 and Dory1.0 have no row**, so an unapproved bump on any of them is refused with the model named; the aggregate flag is therefore **false**. |
| `monitors`: per-layer and per-shared-expert relative error against the reference, with any routing divergence reported | per model. **GLM-5.3 has its row** — the committed pair is judged at layer bar **0.5** and expert bar **0.5**, and passes with maxima **3.158e-01** per layer and **3.511e-01** per shared expert; 7 of 42 layers routed one row to a different expert, leaving 14 experts unshared (7 reference-only, 7 candidate-only) and 329 shared. **Qwen3.8, Inkling-2 and Dory1.0 have no row**, so the aggregate flag is **false**. |

**No model has both rows today.** Qwen3.8 has coverage and no pair; GLM-5.3 has a pair and no
dump; Inkling-2 and Dory1.0 have neither. Every *unapproved* bump is therefore refused, naming the
model and the missing item; an approved bump still lands, because the approval replaces exactly
these two prerequisites. Nothing here grows a model's coverage or judges a new pair: that is the
coverage and monitor windows' work, and it adds the model's row in the same reviewed change.

The coverage evidence is one row a model:

```toml
[[coverage]]
model = "nvidia/Qwen3.8-Flash-Next-NVFP4"
dump = "numerics/coverage/qwen38-world1.json"
min_inputs = 8
```

`dump` is a JSON dump under `numerics/coverage/` and `min_inputs` is the coverage suite's declared
input floor (a positive integer; the dump must name at least that many distinct inputs). The bar is
the shared `coverage_slots`, which is the fraction of (layer, routed-expert) slots a committed dump
must reach, in (0, 1]; it is **0.93**, not 1.0: real serving reached 92-93% of Qwen3.8's 48x512
slots with 240-330 inputs and every layer still carried gaps, so requiring every slot would have
made the prerequisite unattainable. The row's model must have a committed layout under
`numerics/layouts/`, and the dump must name that model or it is refused.

The monitor evidence is also one row a model:

```toml
[[monitors]]
model = "zai-org/GLM-5.3-Flash"
reference = "numerics/monitors/glm53-tp4-v1-reference.json"
candidate = "numerics/monitors/glm53-tp4-v2-candidate.json"
layer_bar = 0.5
expert_bar = 0.5
```

Both paths must be JSON dumps under `numerics/monitors/`, and the bars finite and positive. The
bars are a regression guard, not a quality claim: they sit just above the measured maxima so a
later change that doubles the error fails.

A row is validated when the registry is parsed: its model is present and non-empty, its paths are
under the right directory and end in `.json`, its bars are finite and positive, its
`min_inputs` is positive, and there is **one row a model** (a second row for the same model is
refused). A malformed row fails the parse with the model named. A committed row is evidence and is
judged by `ci` whether or not the aggregate flag is true; the flag only adds the aggregate claim.

`ci` resolves the bumping entry's model to its `[[coverage]]` and `[[monitors]]` rows in
`declaration_findings`. An unapproved bump whose model has no row, whose row is malformed (the
parse refuses it first), or whose row fails its bar is refused, and the finding names the model and
the reason:

```text
<configuration>: version N declares an arithmetic change, but '<model>' has no passing per-model
evidence: the model '<model>' has no [[coverage]] row; ... (docs/NUMERICS.md §5); add the model's
[[coverage]] and [[monitors]] rows with their committed evidence, or record the owner's approval
```

An `approved` record accepts that one bump when its `decision` is a non-empty one line, its
`date` is the day the decision was taken, and its `report` exists and is new or changed in the
same diff. It replaces the coverage and monitors prerequisites (per model, whatever rows exist);
the declared change (a changed fingerprint), the entry's own `report` and the model's anchor and
teacher-forced comparison still apply, so a model with no anchor cannot bump under an approval.
The record stays with the entry afterwards and its removal is refused. This is how a deliberate
arithmetic change whose output shift the owner has accepted lands while a model's coverage or
monitor evidence is missing. A defect is still a defect: the approval says a shift is expected,
not that any output is acceptable.

Otherwise the gate is exactly the bit gate. The per-model rules are enforced by
`scripts/numerics.py ci`: a bump whose model has no `[[anchor]]`, no `[[coverage]]` row or no
`[[monitors]]` row, whose anchor has no committed teacher-forced capture, or whose evidence is
`text` is refused with the model named, while a model that has its own evidence is not blocked
by another model's missing row.

#### The coverage dump

A coverage run writes one JSON document for one served configuration. It holds the inputs that
ran and, for every decoder layer, one hit count per routed expert. A dense layer is still a
layer: it appears with an empty hit list. The format is versioned as
`sparkflash-numerics-coverage/1`:

```json
{
  "schema": "sparkflash-numerics-coverage/1",
  "model": "zai-org/GLM-5.3-Flash",
  "inputs": [{"name": "retrieval", "tokens": 1024}],
  "layers": [
    {"layer": 0, "hits": []},
    {"layer": 3, "hits": [12, 4, 1]}
  ],
  "captured": "2026-11-19"
}
```

The model's expert layout is committed under `numerics/layouts/` as
`sparkflash-numerics-expert-layout/1`: contiguous spans of decoder layers, each with the number
of routed experts in them (`0` for a dense layer), with the released configuration it came from.
The checkpoint's MTP/draft layers are listed under `draft_layers` but are not judged by this dump
yet; they run in the drafter's pass.

```bash
python3 scripts/numerics.py coverage --dump numerics/coverage/glm53-tp4.json --min-inputs 8
```

The command refuses a dump whose schema or model is wrong, whose model has no committed layout,
whose layer list omits, repeats or invents a layer, whose hit list does not have exactly one
non-negative integer per routed expert, whose dense layer carries hits, or that names fewer
distinct inputs than the required `--min-inputs` declaration of the coverage suite. It also
refuses a registry whose `coverage_slots` is missing or outside (0, 1]. It prints the achieved
fraction of routed (layer, expert) slots, the declared bar, and the missing expert ids per layer;
it passes at or above the bar and exits non-zero below it. A dump at 0.93 with every layer
carrying gaps is a pass: the gaps are printed and counted, not hidden.

#### Growing the coverage suite

The judge's gaps are a planning input, not just a verdict. `coverage-gaps` prints them without the
gate, and writes every missing expert to JSON:

```bash
python3 scripts/numerics.py coverage-gaps --dump "$W/qwen38-coverage.json" --json "$W/gaps.json"
```

`coverage-sweep` turns a corpus into the next batch and measures every prompt:

- It reads the dump before and after each prompt, so the slots a prompt actually reached decide
  whether it is kept. A prompt is kept only when its delta contains a slot no earlier kept prompt
  reached; the rest are discarded. Per-batch rows report sent, kept, discarded, new slots, and
  missing/coverage before and after. The final JSON is deterministic given the dump, server, seed
  and corpus.
- The order is the corpus order, or a seeded `sha256("<seed>:<name>")` shuffle. The generator is a
  pure function of the seed; it proposes candidates, and the measured hit delta selects among them.
- It does not repeat an input. A prompt whose content-addressed input name (`input-<16 hex>`, FNV-1a
  over the token ids) is already in the dump is skipped without a request, and every sent prompt is
  recorded in `DUMP.seen.json` (or `--seen FILE`). A legacy dump with `input-0` ... names carries
  no content, so it needs `--input-suite FILE` (the suite that produced those inputs) or an
  explicit seen manifest.
- It refuses a layout whose model is not the dump's, a dump whose model is not the layout's, a
  server that names another model, a layer whose hit list does not match the layout, and a corpus
  whose schema or names are wrong.
- `--out` is the kept prompts as a `sparkflash-numerics-prompts/1` suite; the dump remains the
  evidence and contains every prompt the sweep sent. The suite reproduces the sweep's forwards with
  greedy decode and the same `--max-tokens` (default 1); the base dump's inputs are not in it, so
  reproduce the whole dump by appending to that dump or by prepending the `--input-suite`.

The corpus is [`numerics/coverage/corpus.json`](../numerics/coverage/corpus.json); `--generate N`
adds a deterministic generated corpus from character classes, code, numerals, non-Latin scripts,
long documents and repetition.

**Estimate.** 48 x 512 = 24,576 routed slots. The two runs reached 92-93%, leaving about
1,720-1,966 of them (the judge's rounded gap is 1,800-2,000). Taken as a bracket (240 inputs at
92%, 330 at 93%), the difference is 246 slots over 90 inputs, about 2.7 new slots per input at the
tail, so the remaining gap needs roughly 650-750 *contributing* prompts. A greedy sweep discards non-contributors; budget 1,500-2,500 measured
candidates for 650-800 kept. That is an estimate from the two runs' marginal rate, not a measured
coverage claim.

The window command, on an idle world-1 server already writing the dump:

```bash
W=/home/allenh1/cluster-tests/$(date +%Y%m%dT%H%M)-coverage-sweep
mkdir -p "$W"
cp "$QWEN_COVERAGE_DUMP" "$W/qwen38-coverage.json"

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
  --input-suite "$QWEN_OLD_SUITE"
scripts/serve-single-node.sh stop

python3 scripts/numerics.py coverage-gaps --dump "$W/qwen38-coverage.json" --json "$W/gaps.json"
python3 scripts/numerics.py coverage --dump "$W/qwen38-coverage.json" \
  --layout numerics/layouts/qwen38.json --min-inputs 8
```

Drop `--input-suite` when the starting dump's inputs are already content-addressed. The seen
manifest belongs to the dump and makes each window continue where the last stopped; delete it when
starting a fresh dump. Rerun the same command until `--max-candidates` exhausts the corpus, then
raise `--generate` or change `--seed`.

#### The monitor dump

A monitor run samples the reference forward and the candidate forward on the same probe. Each
dump records, for every MoE decoder layer the probe reached and every expert it routed to, the
values at a fixed set of hidden indices, so `monitors` can compute a relative L2 error without
shipping whole activations. A dense layer has no routed expert and is not recorded. The format is
versioned as `sparkflash-numerics-monitors/1`:

```json
{
  "schema": "sparkflash-numerics-monitors/1",
  "model": "zai-org/GLM-5.3-Flash",
  "probe": {"name": "decode-position-3", "prompt_ids": [42], "position": 3, "rows": 1},
  "sample": {"width": 4096, "indices": [0, 1, 2, 3]},
  "layers": [
    {"layer": 3, "values": [0.1, 0.2, 0.3, 0.4],
     "experts": [{"expert": 0, "values": [0.1, 0.2, 0.3, 0.4]}]}
  ],
  "captured": "2026-11-19"
}
```

```bash
python3 scripts/numerics.py monitors \
    --reference numerics/monitors/glm53-tp4-v1-reference.json \
    --candidate numerics/monitors/glm53-tp4-v2-candidate.json \
    --layer-bar 0.5 --expert-bar 0.5
```

The error is `||candidate - reference|| / ||reference||` on the sampled values. The command refuses
two dumps that name different models, probes or sampled indices, that do not hold the same layers,
that hold a non-finite or duplicate value, or whose reference sample is all zeros on a layer or a
shared expert (the relative error is undefined); it refuses a bar that is not finite and positive;
and it exits non-zero when any layer exceeds `--layer-bar` or any shared expert exceeds
`--expert-bar`, printing every failure and writing the per-layer, per-expert and routing result
when asked.

A routing difference is **reported, not refused**. A slightly different hidden state can change
the router's top-k choice, so a candidate may send a row to an expert the reference did not. The
command intersects each layer's expert ids and judges the expert bar over the experts both dumps
hold; it counts, per layer and in total, the experts only one dump holds (`reference_only` and
`candidate_only`), prints each differing layer's ids, and carries the same in `routing`
(`layers_differing`, `experts_unshared`, `experts_shared`, `per_layer`). The layer bar is
judged unchanged, and the exit code follows the bars alone. A divergence is itself a quality
signal — the candidate's router made a different choice on that row — and it is reported beside
the error, never hidden in a pass.

The committed GLM-5.3 pair is judged at layer bar **0.5** and expert bar **0.5**. It passes with
maxima **3.158e-01** per layer and **3.511e-01** per shared expert; 7 of 42 layers routed one row
to a different expert, leaving 14 experts unshared (7 reference-only, 7 candidate-only) and 329
shared. The bars are a **regression guard, not a quality claim**: they sit above the measured
maxima so a future pair that moves further is caught, and they say nothing about whether the
candidate is good enough to ship. The registry's `[[monitors]]` row for GLM-5.3 names this pair
and these bars; the aggregate `[prerequisites] monitors` flag is false because Qwen3.8, Inkling-2
and Dory1.0 have no row, so an unapproved bump on any of them is refused.

The Rust writer is `sparkflash_model::numerics_monitors::Recorder`. It is off unless
`SPARKFLASH_NUMERICS_MONITORS` names an output path; `SPARKFLASH_NUMERICS_SAMPLE` sets the number
of evenly spaced hidden indices (default 8) and `SPARKFLASH_NUMERICS_MODEL` overrides the model
id. The document declares the exact indices it sampled, and the judge refuses two dumps whose
indices differ. The recorder merges a document across forwards only when model, probe and sample
all match and the values agree; a different probe or sample is refused.

A monitor run is **one probe per process**, and the probe is the **first decode step**: the first
forward that reaches a MoE layer with a row count at most the arm's few-row band, which is eight
rows on both wired arms (the tiled GEMV's batch, and the few-row tile bodies' widest forward). A
wider forward is a prefill chunk, and a prefill must not consume the probe. The document says what
the probe is: `probe.rows` is the forward's row count (one for a plain decode step), `probe.position`
is the absolute position of its first row (the prompt's length for the first decode step), and
`probe.name` is `decode-position-<position>`. A prefill probe therefore cannot be written: the
writer (`sparkflash_model::numerics_monitors::Recorder::new`) refuses a probe wider than the band,
and each call site checks the row count before it opens the recorder. A later decode step is a
different probe, which the recorder reports as `AlreadyExists`; the serving wrapper logs it and
stops recording that forward rather than failing the request. Use a fresh path per run and send the
request after `/readyz`.

This rule is why the probe can see a decode-only change. Before it the probe was the process's
first forward, which on a real request is the prompt's prefill; a prefill runs the bulk GEMM on
both builds, so a pair taken then was blind to a difference confined to the decode body. The first
GLM-5.3 pair — one build with `SPARKFLASH_DECODE_MMA=off`, one with the shipped default — came out
byte-identical and the judge passed at 0.000e+00 without measuring the body. It was retaken; the
committed pair probes the first decode step and measures the FP8 few-row body, as the numbers
above record.

A real reference/candidate pair needs two builds. The echo of one process is not a candidate. The
first real pair is committed under `numerics/monitors/` — GLM-5.3's reference build
(`SPARKFLASH_DECODE_MMA=off`, the tiled row GEMV) against the shipped default (the FP8 few-row
body) — and `numerics/registry.toml` names it in GLM-5.3's `[[monitors]]` row. The aggregate
`[prerequisites] monitors` flag stays false while any captured logprobs model has no row. On the
cluster, with the wired Qwen3.8 host
reference:

```bash
W=/home/allenh1/cluster-tests/$(date +%Y%m%dT%H%M)-monitors
mkdir -p "$W"
cat > "$W/one.json" <<'JSON'
{"schema": "sparkflash-numerics-prompts/1",
 "prompts": [{"name": "probe", "prompt": "<the fixed probe prompt>"}]}
JSON

# Reference build, into reference.json.
SPARKFLASH_MODEL=qwen3.8-flash-next-nvfp4 \
SPARKFLASH_CHECKPOINT="$QWEN_CHECKPOINT" \
SPARKFLASH_TOKENIZER="$QWEN_TOKENIZER" \
SPARKFLASH_CONTEXT=65536 \
SPARKFLASH_HTTP_BIND=127.0.0.1:8002 \
SPARKFLASH_BIN="$REFERENCE_BIN" \
SPARKFLASH_NUMERICS_MONITORS="$W/qwen38-monitors-reference.json" \
SPARKFLASH_NUMERICS_SAMPLE=8 \
SPARKFLASH_NUMERICS_MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4 \
  scripts/serve-single-node.sh launch
python3 scripts/numerics.py capture --url http://127.0.0.1:8002 \
  --suite "$W/one.json" --out "$W/reference-drive.json" --max-tokens 1
scripts/serve-single-node.sh stop

# Candidate build, into candidate.json. Same checkpoint, tokenizer, context and probe.
SPARKFLASH_MODEL=qwen3.8-flash-next-nvfp4 \
SPARKFLASH_CHECKPOINT="$QWEN_CHECKPOINT" \
SPARKFLASH_TOKENIZER="$QWEN_TOKENIZER" \
SPARKFLASH_CONTEXT=65536 \
SPARKFLASH_HTTP_BIND=127.0.0.1:8002 \
SPARKFLASH_BIN="$CANDIDATE_BIN" \
SPARKFLASH_NUMERICS_MONITORS="$W/qwen38-monitors-candidate.json" \
SPARKFLASH_NUMERICS_SAMPLE=8 \
SPARKFLASH_NUMERICS_MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4 \
  scripts/serve-single-node.sh launch
python3 scripts/numerics.py capture --url http://127.0.0.1:8002 \
  --suite "$W/one.json" --out "$W/candidate-drive.json" --max-tokens 1
scripts/serve-single-node.sh stop

# The run's declared bars; this page does not invent them.
python3 scripts/numerics.py monitors \
  --reference "$W/qwen38-monitors-reference.json" \
  --candidate "$W/qwen38-monitors-candidate.json" \
  --layer-bar <declared> --expert-bar <declared> \
  --json "$W/qwen38-monitors-result.json"
```

The first **decode step** is the probe: the request's prompt is a prefill, wider than the band
and so not the probe, and the forward that generates its first token is. The reference and
candidate must run the same prompt with the same tokenizer and context so the probe's ids and
position match; a different chunking is refused by the judge. Check the document's `probe` before
judging it — `rows` is 1 and `position` is the prompt's length — and use a fresh path and send the
request after `/readyz`, because a boot warmup can be the first decode step.

#### Emission sites

The dump is written by the serving side, never reconstructed from a report. The Rust writer is
`sparkflash_model::numerics_coverage::Recorder`; a recorder is created with the model's per-layer
expert counts, and each router decision calls `Recorder::hit(layer, expert)`. It is opt-in:
`SPARKFLASH_NUMERICS_COVERAGE` names the output path and is unset by default, and
`SPARKFLASH_NUMERICS_MODEL` overrides the model id for a tiny fixture. With the switch unset the
caller holds `None` and records nothing. The recorder merges the document already at the path, so
a serving process accumulates a suite across forwards. Each forward opens a content-addressed
input, `input-<16 hex>` (FNV-1a 64 over the token ids' little-endian bytes); a repeated input adds
no second entry, and `coverage-sweep` computes the same name from `/tokenize`'s ids to skip an
input the dump already holds. A document for another model or expert layout is replaced by the new
run, not merged.

| Model | Coverage emission site | Monitor emission site |
| :-- | :-- | :-- |
| Qwen3.8-Flash-Next | `crates/sparkflash-model-qwen4-exp/src/forward.rs`, `moe_forward` (the host reference forward; **wired**, test `the_reference_forward_can_record_routing_for_a_coverage_dump`), and the world-1 device arm's `crates/sparkflash-model-qwen4-exp/src/device/blocks.rs`, `DeviceSequence::mlp_branch` after the router launch (**wired**, `DeviceSequence::record_routing`) | the same host reference `moe_forward` (**wired**, test `the_reference_forward_can_record_a_monitor_dump`), and the world-1 device arm's `DeviceSequence::record_probe` after `moe_combine` (**wired**, sampling `branch` and each slot of `expert_accumulator`); the TP arm is named |
| GLM-5.3-Flash | the device arm's `crates/sparkflash-model-glm5-next/src/device.rs`, `DeviceRank::combine_experts` after its combine kernel (**wired**, `DeviceRank::record_routing` reads the router's `expert_indices` back into `crates/sparkflash-model-glm5-next/src/device/numerics.rs`, `record_routing`; the MTP draft passes `None` and records nothing); the host reference's `crates/sparkflash-model-glm5-next/src/forward.rs`, `Moe::apply` — **named** | the device arm's `crates/sparkflash-model-glm5-next/src/device.rs`, `combine_experts` after its combine kernel (**wired**, `DeviceRank::record_layer` and `crates/sparkflash-model-glm5-next/src/device/numerics.rs`, sampling `partial` and each slot of `expert_out` on rank 0); the host reference's `Moe::apply` — **named** |
| DeepSeek-V4.1-Flash | `crates/sparkflash-deepseek-forward/src/moe_device.rs`, `MoeDeviceBackendImpl::ffn` (the selected ids are already copied to the host at `moe_device.rs:643`) — **named** | `crates/sparkflash-deepseek-forward/src/moe_device.rs`, `MoeDeviceBackendImpl::ffn`'s routed combine — **named** |

The Qwen host reference is wired for **both** dumps and is covered by the two CPU tests named
above; the cross-language test `the_dump_round_trips_through_the_python_monitor_command` writes a
monitor document and judges it with `scripts/numerics.py monitors`. The Qwen3.8 world-1 device
arm — the path the cluster window exercised — is wired for both as well:

- `DeviceSequence::forward_batch` opens the recorders once a forward
  (`crates/sparkflash-model-qwen4-exp/src/device/numerics.rs`, `Switches::begin`), so coverage
  opens one named input a forward. The monitor recorder takes that forward as the probe only when
  its row count is at most the few-row band (`DeviceSequence::begin_numerics` passes the batch's
  row count); a prefill forward opens no monitor recorder and leaves the probe to the first decode
  step after it.
- `DeviceSequence::record_routing` copies the router's `[batch][top-k]` ids back after
  `mlp_branch`'s router launch and counts them.
- `DeviceSequence::record_probe` samples the first row's `moe_combine` branch and each routed
  expert's `expert_accumulator` slot, divided by its routing weight — the expert kernels write
  `weight * body` there. The MTP layer is a draft pass and is not recorded.

GLM's device arm is wired for both dumps:

- `DeviceRank::begin_numerics` opens the recorders once a step from the step's first run
  (`crates/sparkflash-model-glm5-next/src/device/numerics.rs`, `Switches::begin`). Coverage opens
  one content-addressed input on every forward, the prompt's prefill included, and on every rank:
  the router's chosen ids are the same view of the layer on every rank and the coverage format has
  no rank field. The monitor recorder takes only a decode step, and `DeviceRank` passes its rank
  so only rank 0 opens one.
- `DeviceRank::combine_experts` counts each routed layer's chosen ids after its combine kernel:
  `record_routing` copies the router's `[rows][top-k]` ids back from `expert_indices` and hands
  them to `numerics::record_routing`, one `hit` a selected expert. `combine_experts` takes the
  decoder layer as an `Option<usize>`: the token and bulk paths pass the layer's index, and the
  MTP layer's draft pass passes `None`, so the draft's routing is not recorded. The parameter
  changes no arithmetic.
- The monitor half is unchanged: `record_layer` copies back the probe row's routed ids and
  weights, the layer's `partial` row and each slot's `expert_out` row, and
  `numerics::record_probe` divides each expert row by its routing weight and samples at the
  document's indices. The dump is rank 0's slot of the tensor-parallel expert sum — the shared and
  routed experts' columns are split, so `partial` and `expert_out` hold this rank's partial
  products — and the rank gate keeps one writer a launch. A reference build and a candidate run
  the same split, so the judge compares like with like. A read-back or write failure is logged
  once and serving continues.

Both switches read **once** when a session is made: `SPARKFLASH_NUMERICS_COVERAGE` and
`SPARKFLASH_NUMERICS_MONITORS` name output paths, `SPARKFLASH_NUMERICS_MODEL` overrides the dump's
model id, and `SPARKFLASH_NUMERICS_SAMPLE` sets the monitor sample's evenly spaced hidden-index
count (default 8). With them unset a forward pays one `Option` check and a layer one more: no
environment read, no formatting, no allocation. A recorder that cannot open or write logs once and
serving continues; a request never fails for a diagnostic. The rules — the document shapes, the
input naming and accumulation, and the one-probe rule — are unit-tested with a synthetic table in
each wired device arm's `device/numerics.rs`
(`crates/sparkflash-model-qwen4-exp/src/device/numerics.rs` and
`crates/sparkflash-model-glm5-next/src/device/numerics.rs`), so they need no GPU. The GLM
coverage recorder is seeded from `mlp_layer_types`: a dense decoder layer has no routed experts,
so it is listed with an empty hit list, as `numerics/layouts/glm53.json` declares it.

The other arms are **named, not wired**. DeepSeek's `ffn` does have its ids on the host, but the
forward crate calls the backend one layer at a time with no per-forward boundary, so a recorder
there would open one input per layer and a layer list the judge refuses. The Qwen TP arm's
`mlp_branch` leaves only this rank's share of the expert sum in `branch`, so monitor dumps are the
world-1 arm's; a sharded arm logs once and records none. The device-batch path
(`DeviceSequence::forward_rows`) is not wired. Wiring DeepSeek is a device window's work, and the
first real dumps are still ahead.

The exact window command for the Qwen3.8 world-1 dumps (both switches on in one process, the
first decode step is the probe, then the two judges):

```bash
W=/home/allenh1/cluster-tests/$(date +%Y%m%dT%H%M)-numerics
mkdir -p "$W"
SPARKFLASH_MODEL=qwen3.8-flash-next-nvfp4 \
SPARKFLASH_CHECKPOINT="$QWEN_CHECKPOINT" \
SPARKFLASH_TOKENIZER="$QWEN_TOKENIZER" \
SPARKFLASH_CONTEXT=65536 \
SPARKFLASH_HTTP_BIND=127.0.0.1:8002 \
SPARKFLASH_NUMERICS_COVERAGE="$W/qwen38-coverage.json" \
SPARKFLASH_NUMERICS_MONITORS="$W/qwen38-monitors.json" \
SPARKFLASH_NUMERICS_MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4 \
SPARKFLASH_NUMERICS_SAMPLE=8 \
  scripts/serve-single-node.sh launch
# After /readyz: three passes of the reference suite's eight prompts, one token each. Every
# request's prefill is at least one forward, and each forward opens one named input.
for pass in 1 2 3; do
  python3 scripts/numerics.py capture --url http://127.0.0.1:8002 \
    --suite numerics/prompts/reference.json --out "$W/drive-$pass.json" --max-tokens 1
done
scripts/serve-single-node.sh stop

python3 scripts/numerics.py coverage --dump "$W/qwen38-coverage.json" --min-inputs 8
# The monitors judge needs a reference build's dump against this candidate; the pair block above
# runs both builds. This window's file is the candidate, and the bars are the run's to declare.
python3 scripts/numerics.py monitors --reference "$W/qwen38-monitors-reference.json" \
  --candidate "$W/qwen38-monitors.json" --layer-bar <declared> --expert-bar <declared>
```

The GLM-5.3-Flash arm serves only at world 4, so its pair is **one build run twice**: the tip with
`SPARKFLASH_DECODE_MMA=off` (the tiled row GEMV, which is version 1's arithmetic) and the tip with
the shipped default (the tile's few-row body, version 2). Only rank 0 records — the dump is rank
0's slot of the tensor-parallel expert sum — so run with rank 0 local, use a fresh path for each
run, and send the request after `/readyz`:

```bash
W=/home/allenh1/cluster-tests/$(date +%Y%m%dT%H%M)-glm53-monitors
mkdir -p "$W"
cat > "$W/one.json" <<'JSON'
{"schema": "sparkflash-numerics-prompts/1",
 "prompts": [{"name": "probe", "prompt": "<the fixed probe prompt>"}]}
JSON

# Reference build: the tip with SPARKFLASH_DECODE_MMA=off, the tiled row GEMV (version 1's
# arithmetic).
SPARK_NODES="$GLM_NODES" SPARK_LOCAL_RANKS=0 \
SPARKFLASH_MODEL=zai-org/GLM-5.3-Flash \
SPARKFLASH_CHECKPOINT="$GLM_CHECKPOINT" \
SPARKFLASH_TOKENIZER="$GLM_TOKENIZER" \
SPARKFLASH_CONVERTED_DIR="$GLM_CONVERTED_DIR" \
SPARKFLASH_CONTEXT=131072 \
SPARKFLASH_HTTP_BIND=127.0.0.1:8002 \
SPARKFLASH_DECODE_MMA=off \
SPARKFLASH_BIN="$GLM_TIP_BIN" \
SPARKFLASH_NUMERICS_MONITORS="$W/glm53-monitors-reference.json" \
SPARKFLASH_NUMERICS_SAMPLE=8 \
SPARKFLASH_NUMERICS_MODEL=zai-org/GLM-5.3-Flash \
  scripts/serve-four-node.sh launch
python3 scripts/numerics.py capture --url http://127.0.0.1:8002 \
  --suite "$W/one.json" --out "$W/reference-drive.json" --max-tokens 1
scripts/serve-four-node.sh stop

# Candidate build: the same binary with the shipped default, the tile's few-row body. Same nodes,
# checkpoint, tokenizer, context and probe.
SPARK_NODES="$GLM_NODES" SPARK_LOCAL_RANKS=0 \
SPARKFLASH_MODEL=zai-org/GLM-5.3-Flash \
SPARKFLASH_CHECKPOINT="$GLM_CHECKPOINT" \
SPARKFLASH_TOKENIZER="$GLM_TOKENIZER" \
SPARKFLASH_CONVERTED_DIR="$GLM_CONVERTED_DIR" \
SPARKFLASH_CONTEXT=131072 \
SPARKFLASH_HTTP_BIND=127.0.0.1:8002 \
SPARKFLASH_BIN="$GLM_TIP_BIN" \
SPARKFLASH_NUMERICS_MONITORS="$W/glm53-monitors-candidate.json" \
SPARKFLASH_NUMERICS_SAMPLE=8 \
SPARKFLASH_NUMERICS_MODEL=zai-org/GLM-5.3-Flash \
  scripts/serve-four-node.sh launch
python3 scripts/numerics.py capture --url http://127.0.0.1:8002 \
  --suite "$W/one.json" --out "$W/candidate-drive.json" --max-tokens 1
scripts/serve-four-node.sh stop

# The window's bars; this page does not invent them.
python3 scripts/numerics.py monitors \
  --reference "$W/glm53-monitors-reference.json" \
  --candidate "$W/glm53-monitors-candidate.json" \
  --layer-bar <declared> --expert-bar <declared> \
  --json "$W/glm53-monitors-result.json"
```

**The probe is now the first decode step.** The pair above is a retake: its probe is the forward
that generates the first token after the prompt's prefill, so the two runs measure the row the
decode body writes. A pair taken before this rule recorded the prompt's prefill instead, whose bulk
GEMM runs on both builds — that pair's dumps came out byte-identical and the judge passed at
0.000e+00 without measuring the body, so it is blind to the difference it was meant to measure and
must not be used. Before judging a dump, check that `probe.rows` is 1 and `probe.position` is the
probe prompt's length. A boot warmup that generates a token first can still become the probe, so a
fresh path and a post-`/readyz` request are what name the forward the two runs share.

**A GLM-5.3-Flash coverage sweep.** The GLM arm serves only at world 4, so a coverage window is one
run of the tip, not a pair. `SPARK_LOCAL_RANKS=0` keeps rank 0 on the launching node, and rank 0's
file is the suite the operator collects: every rank records its own routing view, and the router's
chosen ids are the same on all of them, but the monitor dump's rank-sharing caveat does not apply
to coverage and the format has no rank field. Use a fresh window directory and send the sweep after
`/readyz`; a run against the committed layout and the declared `--min-inputs` of eight is:

```bash
W=/home/allenh1/cluster-tests/$(date +%Y%m%dT%H%M)-glm53-coverage
mkdir -p "$W"

SPARK_NODES="$GLM_NODES" SPARK_LOCAL_RANKS=0 \
SPARKFLASH_MODEL=zai-org/GLM-5.3-Flash \
SPARKFLASH_CHECKPOINT="$GLM_CHECKPOINT" \
SPARKFLASH_TOKENIZER="$GLM_TOKENIZER" \
SPARKFLASH_CONVERTED_DIR="$GLM_CONVERTED_DIR" \
SPARKFLASH_CONTEXT=131072 \
SPARKFLASH_HTTP_BIND=127.0.0.1:8002 \
SPARKFLASH_NUMERICS_COVERAGE="$W/glm53-coverage.json" \
SPARKFLASH_NUMERICS_MODEL=zai-org/GLM-5.3-Flash \
  scripts/serve-four-node.sh launch
# Wait for /readyz.
python3 scripts/numerics.py coverage-sweep \
  --url http://127.0.0.1:8002 \
  --dump "$W/glm53-coverage.json" \
  --layout numerics/layouts/glm53.json \
  --corpus numerics/coverage/corpus.json \
  --generate 2000 --seed 20261119 --order shuffled \
  --batch 64 --max-candidates 256 \
  --out "$W/next-batch.json" --json "$W/sweep.json"
scripts/serve-four-node.sh stop

python3 scripts/numerics.py coverage-gaps --dump "$W/glm53-coverage.json" --json "$W/gaps.json"
python3 scripts/numerics.py coverage --dump "$W/glm53-coverage.json" \
  --layout numerics/layouts/glm53.json --min-inputs 8
```

Rerun the same command until `--max-candidates` exhausts the corpus, then raise `--generate` or
change `--seed`, as for Qwen3.8.

### Across architectures and versions

sm_100 and sm_12x cannot be bit-compared, so each architecture keeps its own entry and reference.
Across architectures the anchor comparison and the error bounds apply. A greedy-text comparison is
valid within one numerics version; across versions the teacher-forced comparison replaces it.

## 6. Running it

```bash
# Capture a reference from a running server (prints the registry entry to paste). The served
# card's build.id supplies server.commit; --binary and --repo name the binary and the fallback
# tree, and neither changes the fingerprint.
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 \
  --binary target/release/sparkflash --repo . \
  --out numerics/references/qwen38-tp4-sm121a.json

# A reference or an anchor needs its correctness floor: the forced comparison, the known-answer
# decode and the reference's own NLL, recorded in a JSON file (see *The correctness floor*).
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 \
  --reference --floor /tmp/v2-floor.json --binary target/release/sparkflash --repo . \
  --out numerics/references/qwen38-tp4-sm121a-v2.json

# The DeepSeek device path serves only greedy text.
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 --evidence text --out numerics/references/deepseek-v41-tp4-sm121a.json

# Check a server against its registered reference, bit for bit.
python3 scripts/numerics.py check --url http://127.0.0.1:8000

# Teacher-force a candidate on an anchor's tokens, then score it.
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 --forced-from numerics/references/qwen38-anchor.json --top-logprobs 20 --out numerics/captures/qwen38/tp4-v1.json
python3 scripts/numerics.py compare --anchor numerics/references/qwen38-anchor.json --candidate numerics/captures/qwen38/tp4-v1.json --markdown docs/numerics/qwen38/v1-tp-floor.md

# Judge the coverage dump a run wrote against the committed layout and the registry's bar.
python3 scripts/numerics.py coverage --dump numerics/coverage/glm53-tp4.json --min-inputs 8

# Report what a partial coverage dump still misses, with every expert in JSON.
python3 scripts/numerics.py coverage-gaps --dump "$W/qwen38-coverage.json" --json "$W/gaps.json"

# Measure a corpus against a live dump and write the next batch of prompts.
python3 scripts/numerics.py coverage-sweep --url http://127.0.0.1:8002 \
  --dump "$W/qwen38-coverage.json" --layout numerics/layouts/qwen38.json \
  --corpus numerics/coverage/corpus.json --generate 2000 --seed 20261119 \
  --out "$W/next-batch.json" --json "$W/sweep.json"

# Compare the committed GLM-5.3 monitor pair at its declared bars, over the experts both routed
# to; a routing divergence is reported beside the errors.
python3 scripts/numerics.py monitors \
  --reference numerics/monitors/glm53-tp4-v1-reference.json \
  --candidate numerics/monitors/glm53-tp4-v2-candidate.json \
  --layer-bar 0.5 --expert-bar 0.5 --json "$W/glm53-tp4-result.json"
```

The prompt suite is [`numerics/prompts/reference.json`](../numerics/prompts/reference.json). A
capture tokenizes it once through the server's `/tokenize` and stores the ids, so a check replays
ids and measures the arithmetic, never the tokenizer.

Two request extensions on `/v1/completions` make this work, and both are refused rather than
ignored where they cannot be honoured:

- **`return_tokens_as_token_ids: true`** (vLLM's): every reported token and `top_logprobs` key is
  the string `token_id:N`, so a capture keys by id and the same capture runs against vLLM. Chat
  completions accept it too.
- **`forced_tokens: [ids]`** (SparkFlash's): the server feeds these tokens in place of its own
  choice and scores each one under the row the model produced for it. It needs `logprobs`, one
  prompt, `n = 1`, greedy neutral sampling, no stop, and `max_tokens` absent or equal to its
  length; a sequence longer than the output ceiling or the context is refused, never cut. Only a
  driver that declares `forced_decoding` on the model card accepts it: the generic host driver and
  the Qwen3.8 and GLM-5.3 TP drivers do, while the DeepSeek device path and the reference engine
  refuse it.

## 7. Where it stands

Measured 2026-11-03 on sm_121a from the official checkpoints. Every served configuration below is
registered at version 1.

| Configuration | Evidence | Anchor | Distance from the anchor |
| :-- | :-- | :-- | :-- |
| Qwen3.8-Flash-Next, world 1 | logprobs | world 1, token-kernel decode | mean KL 2.8e-04, top-1 99.39% ([report](numerics/qwen38/v1-tc-decode.md)) |
| Qwen3.8-Flash-Next, world 2 | logprobs | world 1, token-kernel decode | mean KL 1.1e-04, top-1 99.70% ([report](numerics/qwen38/v1-tc-decode.md)) |
| Qwen3.8-Flash-Next, world 4 | logprobs | world 1, token-kernel decode | mean KL 2.4e-04, top-1 99.60% ([report](numerics/qwen38/v1-tc-decode.md)) |
| GLM-5.3-Flash, world 4 | logprobs | world 4, f32 latent cache | mean KL 1.7e-02, top-1 96.70% ([report](numerics/glm53/v1-e4m3-window.md)) |
| DeepSeek-V4.1-Flash, world 4 | text | none | not measurable: the path serves no logprobs |

What the round established:

- **Forced decoding is exact.** On the host driver (Qwen3.8 world 1) and on the TP driver (GLM-5.3
  world 4), a server teacher-forced on its own anchor scores exactly zero on every metric. The
  harness adds nothing to a measured distance.
- **Run-to-run determinism holds** for every configuration above: each recapture matched its
  reference bit for bit. DeepSeek's holds only from a freshly booted server (below).
- **Debug and release builds agree.** Qwen3.8 world 1 and DeepSeek each reproduce their reference
  bit for bit from a debug and a release build of the same source.
- **The context does not confound GLM-5.3's comparison.** The anchor runs at `--context 131072`;
  the shipped default scores it bit for bit there and at its default 1M window.
- **Qwen3.8's reassociation floor is small and has no direction.** Measured with the token-kernel
  decode the anchor keeps, TP=2 sits at mean KL 1.3e-04 and TP=4 at 2.3e-04 from it, and every
  top-1 flip sits where the anchor's top two tokens were within 0.02 nats
  ([report](numerics/qwen38/v1-tp-floor.md)).
- **Qwen3.8 decodes on the tensor cores (D68), just above that floor.** Version 1 runs decode
  steps and verifies through the tensor-core image its prefill already used, in few-row forms
  that give each row the wide tiles' bits. Against the anchor, world 1 sits 24% above the TP=4
  floor's KL with 6 top-1 flips in 988, every one within 0.072 nats; worlds 2 and 4 sit on their
  floors. `SPARKFLASH_DECODE_MMA=off` serves the anchor's arithmetic, reported as `overridden`.
- **GLM-5.3's shipped E4M3 window is a real quality change.** It sits about 75 times above
  Qwen3.8's floor, and some of its flips are confident. Nearly all of the distance is the E4M3
  cache, not the bulk FP8 prefill. The report says what the measurement cannot settle.
- **DeepSeek's device path breaks the resume invariance (§1).** On a warm server, 7 of the suite's
  8 prompts produce different greedy text than on the suite's first pass after boot. Two passes
  on a warm server agree with each other, and so do two first passes after separate boots. With
  `SPARKFLASH_PREFIX_CACHE=0`, every pass agrees. So a request resumed from the prefix a reused
  lane holds can produce different greedy text than the same request prefilled whole. Greedy text
  cannot tell whether that is reassociation or a defect; logprobs on this path, or a device-side
  comparison of the resumed and whole prefills, can. Until it is fixed, DeepSeek's reference is the
  suite's first pass on a freshly booted server, so its check needs one too: the operator boots the
  server and runs `numerics.py check` before any other request, since the tool only takes `--url`.
- **Qwen3.8's continued turns are the bits of cold ones (D68).** The host driver's prefix reuse
  (D66) continues a chat turn from the sequence a worker kept and prefills only the new suffix.
  While Qwen3.8 decoded on its token kernels, the continued turn's history had been computed in
  different arithmetic from a cold prefill of the same transcript. On the suite, continued turns
  sat at mean KL 6.4e-04 from cold ones and changed the greedy text of 2 of 8 prompts (3 of 8
  without speculation). With the tensor-core decode, 8 of 8 continued turns are bit-exact, with
  and without speculation, and `a_continued_conversation_is_the_bits_of_a_cold_prefill` holds the
  invariance on the device. The host driver also serves GLM-5.3 on a single node, which no
  registered configuration covers; whether its bulk prefill and token path agree bit for bit was
  not measured here.
- **A capture must ask for greedy.** The API samples at its default temperature of 1.0 when a
  request names none, so `numerics.py` sends `temperature: 0` explicitly.

### The anchor question, answered

The old table said the anchor prerequisite was satisfied by "Qwen3.8 and GLM-5.3 anchored and
compared", while the flag was false and named DeepSeek as the reason. That reading made one arm's
inability to serve logprobs a global veto. The intended reading is the one §4 states: one anchor
per model, and a configuration is measured against its own model's anchor. The tool now enforces
that. Qwen3.8 and GLM-5.3 have an anchor and a committed capture teacher-forced from it, so they
are covered. DeepSeek is **exempt from the anchor requirement and from bumping**: its device path
serves no logprobs, a text reference has no token ids to teacher-force, and §2 freezes its
arithmetic until it has been measured under this policy. If a logprobs path for DeepSeek lands,
the exemption ends with it, and an anchor capture would be the first requirement.

All four logprobs models now have an anchor and a committed teacher-forced capture, so the
aggregate `anchor` flag is **true**; a true flag with an uncovered model still fails `ci`. The
coverage and monitor evidence is per model (§5). Qwen3.8 has a `[[coverage]]` row — its swept
dump reaches 24,494 of 24,576 slots (99.67%) at the declared 0.93 — and GLM-5.3 has a
`[[monitors]]` row: the committed reference/candidate pair is judged at layer bar 0.5 and expert
bar 0.5, with 7 of 42 layers routing one row to a different expert (14 experts unshared, 329
shared) and maxima 3.158e-01 per layer and 3.511e-01 per shared expert. Inkling-2 and Dory1.0 have
neither row, and GLM-5.3 has no coverage dump while Qwen3.8 has no monitor pair, so both aggregate
flags are **false** and every unapproved bump is refused with the model named. An approved bump
needs neither row. No flag was flipped on synthetic data, and no model's coverage was grown or new
pair judged to make a flag true. The coverage suite is now
growable: `coverage-gaps` names the missing experts, `coverage-sweep` measures a deterministic
corpus and writes the kept prompts as the next batch, and the corpus and generator live under
`numerics/coverage/`. The commands that wrote the coverage dump and the monitor pair are in §5;
DeepSeek's device site and the GLM host reference are still named there; the GLM device arm now
counts coverage, but no GLM coverage dump has been run yet. The sweep's estimate for Qwen3.8's remaining
gap is 650-750
contributing prompts (1,500-2,500 measured candidates); that is an estimate from the two runs'
marginal rate, not a measured coverage claim.

## How this reconciles the earlier decisions

- **D39** (native arithmetic, numeric acceptance against the reference forward) is unchanged: it
  defines DeepSeek's reference arithmetic.
- **D51** let performance kernels reassociate within a per-kernel tolerance. For a served
  configuration that is now a declared change; the per-kernel tolerance becomes part of the
  evidence (§2's f64 bound), not a license.
- **D54**'s numerical gate stays the speed row's correctness check against the frozen CPU twin,
  with its bars. Its "changed arithmetic is not licensed" clause now reads: licensed only through
  this policy.

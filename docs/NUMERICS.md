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
  [DECISIONS.md](DECISIONS.md), never automatically because a bar passed.
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
  edit to one, its removal, or a change to its file.
- **Teacher-forced comparison.** The anchor's own generated tokens are fed through the candidate
  (`forced_tokens`, §6), and every position's distribution is compared: NLL delta, top-1
  agreement and a KL lower bound. Both arms see the same tokens, so a near-tie changes one
  position's score instead of derailing the rest of the text, and both arms are deterministic,
  so the result is too.
- **Per-layer and per-expert error monitors.** Relative error against the reference at every
  layer and expert, so a bug in one expert shows as a spike when the end-to-end KL looks fine.
- **Coverage inputs.** Prompts that reach every routed expert in every layer, chunk boundaries,
  32k+ contexts, several batch shapes and speculation depths.
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

`[[retired]]` keeps a removed configuration's last version, so removing and re-adding one cannot
reset it. A reference's `server` card is what the capturing binary reported, and it is outside the
fingerprint: the registry can name a capture's fingerprint only after the capture exists, so a
first capture reads `unregistered`, and a recapture names the version and fingerprint it replaced.
The registry entry is the record; neither `check` nor `ci` reads the card. `text` evidence is weaker than `logprobs`: an arithmetic change that flips no greedy
token is invisible to it. DeepSeek's TP=4 device path is the only `text` configuration.

### The served identity

The binary embeds the registry. At boot every serve path prints one line:

```text
sparkflash numerics: <model> world=4 profile=reference plan=tp arch=sm_121a status=registered registered_version=1 fingerprint=<16 hex> evidence=logprobs overrides=none
```

and `/v1/models` carries the same as `numerics` beside `build`. The status is one of:

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
- a bump while any prerequisite below is false, at the base or at the head;
- a bump without a new report;
- any change to an anchor, its removal, or a change to its file;
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

`scripts/numerics.py self-test` shows each refusal on synthetic registries.

The bits themselves are checked against a running server by `scripts/numerics.py check`, during a
window on the cluster, as the existing gates are. CI has no GPU or weights; it checks that what is
committed is consistent and declared. It sees the declared surfaces and every input it trusts (the
registry, the references, the profiles and the neutral list); an arithmetic change in code, a
kernel or a default set in Rust, is the bit gate's to catch.

### Prerequisites

No version may be bumped, which is to say no arithmetic change may land, until all four are true
in the registry's `[prerequisites]`:

| Prerequisite | State |
| :-- | :-- |
| `anchor`: the anchors and the teacher-forced comparison | Qwen3.8 and GLM-5.3 anchored and compared; DeepSeek has no anchor, because its device path serves no logprobs and cannot teacher-force |
| `declaration_check`: the CI check above | **done** |
| `coverage`: inputs that reach every routed expert of every layer, with hit counts | not started |
| `monitors`: per-layer and per-expert relative error against the reference | not started |

Until then the gate is exactly the bit gate.

### Across architectures and versions

sm_100 and sm_12x cannot be bit-compared, so each architecture keeps its own entry and reference.
Across architectures the anchor comparison and the error bounds apply. A greedy-text comparison is
valid within one numerics version; across versions the teacher-forced comparison replaces it.

## 6. Running it

```bash
# Capture a reference from a running server (prints the registry entry to paste).
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 --out numerics/references/qwen38-tp4-sm121a.json

# The DeepSeek device path serves only greedy text.
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 --evidence text --out numerics/references/deepseek-v41-tp4-sm121a.json

# Check a server against its registered reference, bit for bit.
python3 scripts/numerics.py check --url http://127.0.0.1:8000

# Teacher-force a candidate on an anchor's tokens, then score it.
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 --forced-from numerics/references/qwen38-anchor.json --top-logprobs 20 --out numerics/captures/qwen38/tp4-v1.json
python3 scripts/numerics.py compare --anchor numerics/references/qwen38-anchor.json --candidate numerics/captures/qwen38/tp4-v1.json --markdown docs/numerics/qwen38/v1-tp-floor.md
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

## How this reconciles the earlier decisions

- **D39** (native arithmetic, numeric acceptance against the reference forward) is unchanged: it
  defines DeepSeek's reference arithmetic.
- **D51** let performance kernels reassociate within a per-kernel tolerance. For a served
  configuration that is now a declared change; the per-kernel tolerance becomes part of the
  evidence (§2's f64 bound), not a license.
- **D54**'s numerical gate stays the speed row's correctness check against the frozen CPU twin,
  with its bars. Its "changed arithmetic is not licensed" clause now reads: licensed only through
  this policy.

# GLM-5.3-Flash, numerics version 1: the E4M3 attention window

Version 1 is the shipped default at world 4 on sm_121a, registered in
[`numerics/registry.toml`](../../../numerics/registry.toml). It makes two precision trades:

- the attention window's latent cache is stored as E4M3;
- prompts prefill in bulk chunks whose GEMMs take FP8 activations.

This report measures both trades against an anchor that makes neither. It is the baseline
[NUMERICS.md](../../NUMERICS.md) §4 asks for: a trade already shipped, measured in numbers.

## The anchor

GLM has no world-1 device arm, so the anchor cannot be the smallest world. Instead it is world 4
with the latent cache in f32: `SPARKFLASH_GLM_KV_CACHE=f32` at `--context 131072`. An f32 cache
cannot take the bulk chunk, so the anchor's prompts run the token path, which removes the
second trade too. The anchor is `references/glm53-anchor.json`: greedy generation over the eight
prompts of the reference suite, up to 128 tokens each, 970 in all, top-20 logprobs.

Two checks make the anchor trustworthy:

- **Self-check:** the f32 server forced on its own anchor (`captures/glm53/f32-self-v1.json`)
  scores exactly zero on every metric. A forced capture reproduces free-running generation bit
  for bit on the TP driver.
- **Determinism:** `numerics.py check` recaptured the version-1 reference and matched it bit for
  bit.

The anchor runs at `--context 131072`, and the shipped default below at its default context, the
1M window. The shipped default forced on the anchor at `--context 131072` produces the same
fingerprint as at the default context (`b75febde3418734e`), so the context does not confound the
comparison.

## The shipped default

Anchor: `zai-org/GLM-5.3-Flash world=4 profile=reference plan=tp arch=sm_121a` (`6de4f3ab892cfe9e`)
Candidate: `zai-org/GLM-5.3-Flash world=4 profile=reference plan=tp arch=sm_121a` (`b75febde3418734e`), teacher-forced on the anchor's tokens, top-20 logprobs

| metric | candidate |
| :-- | --: |
| tokens scored | 970 |
| anchor NLL / token | 0.1220 |
| mean NLL delta | +2.749e-02 |
| mean absolute NLL delta | 4.631e-02 |
| p99 absolute NLL delta | 7.051e-01 |
| max absolute NLL delta | 4.832e+00 |
| top-1 agreement | 96.7010% |
| mean KL lower bound | 1.738e-02 |
| max KL lower bound | 2.364e+00 |

## The E4M3 cache alone

The same server with `SPARKFLASH_GLM_PREFILL=0` (`captures/glm53/e4m3-token-prefill-v1.json`)
keeps the E4M3 cache and prefills on the token path, as the anchor does. Its distance from the
anchor is therefore the E4M3 cache's alone.

Anchor: `zai-org/GLM-5.3-Flash world=4 profile=reference plan=tp arch=sm_121a` (`6de4f3ab892cfe9e`)
Candidate: `zai-org/GLM-5.3-Flash world=4 profile=reference plan=tp arch=sm_121a` (`d63cb5ad79d7d9b5`), teacher-forced on the anchor's tokens, top-20 logprobs

| metric | candidate |
| :-- | --: |
| tokens scored | 970 |
| anchor NLL / token | 0.1220 |
| mean NLL delta | +2.417e-02 |
| mean absolute NLL delta | 4.277e-02 |
| p99 absolute NLL delta | 7.052e-01 |
| max absolute NLL delta | 6.235e+00 |
| top-1 agreement | 96.8041% |
| mean KL lower bound | 1.621e-02 |
| max KL lower bound | 3.391e+00 |

The KL column is a lower bound. It partitions the vocabulary into the tokens both arms list in
their top-20 plus one bucket for the rest.

## Reading

- **The E4M3 cache is almost all of it.** Alone it accounts for a mean KL of 1.62e-02 and a top-1
  agreement of 96.80%. The bulk FP8 prefill moves those to 1.74e-02 and 96.70%. The two do not
  add exactly, because each changes the other's inputs, but the prefill's share is small.
- **This is a real quality change, not reassociation.** GLM has no reassociation floor of its
  own, because it has no world-1 arm to measure one against. The closest reference is Qwen3.8's
  TP=4 floor ([v1-tp-floor.md](../qwen38/v1-tp-floor.md)): a mean KL of 2.3e-04 and 1 top-1 flip
  in 988 tokens. GLM's shipped default is about 75 times that KL, with 32 flips in 970 tokens.
  The models differ, so the ratio is an order of magnitude, not a bar, but nothing about a
  reordered sum produces this.
- **Some flips are confident.** Qwen's TP flips all sit where the anchor's top two tokens were
  within 0.02 nats. Of GLM's 32 flips, 6 are where the anchor led by 0.1 nats or less, 18 by 0.1
  to 0.5, 2 by 0.5 to 1, and 6 by more than 1 nat. At four of those six, E4M3 picks neither the
  anchor's token nor its runner-up.
- **One stretch dominates the worst numbers.** Five of the six confident flips, and 7 of the 32
  flips, fall in positions 78 to 91 of the raw continuation, which is the 16-token prompt with
  no chat template. There the anchor's own text loses coherence. Without that prompt, the
  shipped default scores a mean KL of 6.5e-03, top-1 97.86% and mean absolute NLL delta 2.5e-02
  over 842 tokens. That is still about 26 times Qwen's floor.
- **It lowers confidence in the anchor's tokens.** The mean signed NLL delta is +2.7e-02, which
  is 22% of the anchor's 0.122 nats per token. Qwen's TP=4 is +7.4e-04. Teacher forcing is
  scored on the anchor's own choices, so any perturbation tends to raise this number. Its size,
  not its sign, is the finding.

The E4M3 window shipped before this policy existed. This report does not reverse that choice. It
records what the trade costs, so the next change to GLM's arithmetic is measured against the same
anchor and the same numbers. What this measurement cannot settle is whether the confident flips are
what E4M3 does to this model or a defect in the E4M3 path. A per-layer error monitor (prerequisite
`monitors`) or an independent reference of the E4M3 cache would. Until one exists, the distance
above is the shipped baseline, not a bar that licenses more.

## Reproducing

From a checkout, with GLM served at world 4 on the cluster:

```bash
# The anchor and its self-check: the f32 latent cache.
#   serve with SPARKFLASH_GLM_KV_CACHE=f32 and --context 131072
python3 scripts/numerics.py capture --url http://127.0.0.1:8001 --top-logprobs 20 --out numerics/references/glm53-anchor.json
python3 scripts/numerics.py capture --url http://127.0.0.1:8001 --forced-from numerics/references/glm53-anchor.json --top-logprobs 20 --out numerics/captures/glm53/f32-self-v1.json

# The shipped default: the version-1 reference, the determinism check and the forced capture.
python3 scripts/numerics.py capture --url http://127.0.0.1:8001 --out numerics/references/glm53-tp4-sm121a.json
python3 scripts/numerics.py check --url http://127.0.0.1:8001 --reference numerics/references/glm53-tp4-sm121a.json
python3 scripts/numerics.py capture --url http://127.0.0.1:8001 --forced-from numerics/references/glm53-anchor.json --top-logprobs 20 --out numerics/captures/glm53/e4m3-v1.json

# The E4M3 cache alone: serve with SPARKFLASH_GLM_PREFILL=0.
python3 scripts/numerics.py capture --url http://127.0.0.1:8001 --forced-from numerics/references/glm53-anchor.json --top-logprobs 20 --out numerics/captures/glm53/e4m3-token-prefill-v1.json

# The tables above.
python3 scripts/numerics.py compare --anchor numerics/references/glm53-anchor.json --candidate numerics/captures/glm53/e4m3-v1.json
```

Captured 2026-11-03 from the official checkpoint (`zai-org/GLM-5.3-Flash`, snapshot `03eb5366`).

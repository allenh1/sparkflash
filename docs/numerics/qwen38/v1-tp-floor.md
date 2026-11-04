# Qwen3.8-Flash-Next, numerics version 1: the tensor-parallel floor

Version 1 is today's arithmetic, registered in [`numerics/registry.toml`](../../../numerics/registry.toml)
for world sizes 1, 2 and 4 on sm_121a. This report measures how far TP=2 and TP=4 sit from the
world-1 anchor. It is the reassociation floor of [NUMERICS.md](../../NUMERICS.md) §4.

## What differs, and what does not

All three worlds load the same checkpoint (NVFP4 weights, f32 activations, BF16 KV cache) and
run the same kernels. A larger world splits each projection across ranks and adds the partial
sums in an NCCL all-reduce, so TP=2 and TP=4 change only the order of FP32 additions. Nothing is
quantized differently. Any distance below is what reassociation alone does to this model.

## Method

- **Anchor:** `references/qwen38-anchor.json`. This is the world-1 configuration's greedy
  generation over the eight prompts of the reference suite: up to 128 tokens each, 988 in all,
  top-20 logprobs.
- **Candidates:** each candidate is teacher-forced on those 988 tokens (`forced_tokens`) at top-20,
  so every position is scored under the same context in both arms.
- **Self-check:** the world-1 server forced on its own anchor
  (`captures/qwen38/tp1-self-v1.json`) scores exactly zero on every metric. That proves a forced
  capture reproduces free-running generation bit for bit on the host driver, so a nonzero
  distance below is the arithmetic and not the harness.
- **Determinism:** `numerics.py check` recaptured the world-1 reference and matched it bit for bit.

## TP=2

Anchor: `nvidia/Qwen3.8-Flash-Next-NVFP4 world=1 profile=reference plan=tp arch=sm_121a` (`8c4ccd634bae03a0`)
Candidate: `nvidia/Qwen3.8-Flash-Next-NVFP4 world=2 profile=reference plan=tp arch=sm_121a` (`50847d160e339a8f`), teacher-forced on the anchor's tokens, top-20 logprobs

| metric | candidate |
| :-- | --: |
| tokens scored | 988 |
| anchor NLL / token | 0.2228 |
| mean NLL delta | +4.837e-05 |
| mean absolute NLL delta | 3.093e-03 |
| p99 absolute NLL delta | 5.893e-02 |
| max absolute NLL delta | 1.083e-01 |
| top-1 agreement | 99.7976% |
| mean KL lower bound | 1.283e-04 |
| max KL lower bound | 2.132e-02 |

## TP=4

Anchor: `nvidia/Qwen3.8-Flash-Next-NVFP4 world=1 profile=reference plan=tp arch=sm_121a` (`8c4ccd634bae03a0`)
Candidate: `nvidia/Qwen3.8-Flash-Next-NVFP4 world=4 profile=reference plan=tp arch=sm_121a` (`db24239081d46805`), teacher-forced on the anchor's tokens, top-20 logprobs

| metric | candidate |
| :-- | --: |
| tokens scored | 988 |
| anchor NLL / token | 0.2228 |
| mean NLL delta | +7.360e-04 |
| mean absolute NLL delta | 4.246e-03 |
| p99 absolute NLL delta | 6.499e-02 |
| max absolute NLL delta | 2.114e-01 |
| top-1 agreement | 99.8988% |
| mean KL lower bound | 2.263e-04 |
| max KL lower bound | 2.181e-02 |

The KL column is a lower bound. It partitions the vocabulary into the tokens both arms list in
their top-20 plus one bucket for the rest.

## Reading

- **Top-1 flips are near-ties.** TP=2 changes the top-1 token at 2 of 988 positions and TP=4 at
  1. At each flip, the anchor's top two tokens were within 0.003 to 0.020 nats of each other.
- **No direction.** The mean signed NLL delta is +4.8e-05 (TP=2) and +7.4e-04 (TP=4), against
  an anchor NLL of 0.223 per token. That is two to three orders of magnitude below the mean
  absolute delta, so the reordered sums do not make the model systematically less sure of the
  anchor's tokens.
- **No outlier prompt.** Without the raw continuation, the distances barely move: KL 1.3e-04 and
  2.5e-04, top-1 99.77% and 99.88%.

These numbers are the bar for Qwen3.8. A future change at or below them cannot be told apart
from serving at another world size today. A precision change above them is a real quality
change, and its report must say so in numbers.

## Reproducing

From a checkout, with each configuration served in turn on the cluster
(`--context 65536`; TP=2 and TP=4 use the lead node and the first one or three workers):

```bash
# World 1: the reference, the anchor, the self-check and the determinism check.
python3 scripts/numerics.py capture --url http://127.0.0.1:8002 --out numerics/references/qwen38-tp1-sm121a.json
python3 scripts/numerics.py capture --url http://127.0.0.1:8002 --top-logprobs 20 --out numerics/references/qwen38-anchor.json
python3 scripts/numerics.py capture --url http://127.0.0.1:8002 --forced-from numerics/references/qwen38-anchor.json --top-logprobs 20 --out numerics/captures/qwen38/tp1-self-v1.json
python3 scripts/numerics.py check --url http://127.0.0.1:8002 --reference numerics/references/qwen38-tp1-sm121a.json

# World 2 and 4 (W=2 or 4): the reference and the forced capture.
python3 scripts/numerics.py capture --url http://<lead node>:8003 --out numerics/references/qwen38-tpW-sm121a.json
python3 scripts/numerics.py capture --url http://<lead node>:8003 --forced-from numerics/references/qwen38-anchor.json --top-logprobs 20 --out numerics/captures/qwen38/tpW-v1.json

# The tables above.
python3 scripts/numerics.py compare --anchor numerics/references/qwen38-anchor.json --candidate numerics/captures/qwen38/tpW-v1.json
```

Captured 2026-11-03 from the official checkpoint
(`nvidia/Qwen3.8-Flash-Next-NVFP4`, snapshot `fc694b54`).

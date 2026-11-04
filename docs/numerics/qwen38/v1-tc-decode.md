# Qwen3.8-Flash-Next, numerics version 1: decode on the tensor cores

Version 1 of every Qwen3.8 configuration (worlds 1, 2 and 4 on sm_121a) decodes with the
tensor-core image's arithmetic. This report measures that arithmetic against the frozen anchor and
against the tensor-parallel floor. The decision is [D68](../../DECISIONS.md).

## What changed, and why

The tensor-core prefill image computes each matrix multiply with every activation split into a
BF16 head and a BF16 remainder (16 significant bits), the weights' own BF16 or FP4 values, and an
FP32 accumulator, its K dimension split by the chunk's width. Decode steps and speculative verifies,
1 to 63 rows, used to run the token kernels instead: the same weights against the f32 activation,
summed in FMA chains in their own order. So a token the server decoded and the same token re-read
in a cold prefill went through different arithmetic.

That broke the resume invariance ([NUMERICS.md](../../NUMERICS.md) §1). With prefix reuse on
(D66), a continued chat turn decodes on from the caches its earlier turns wrote. The same
conversation sent to a server that does not hold it is prefilled cold, through the tensor-core
image. Turning the cache off would not have fixed it, because a long cold transcript also crosses
from tensor-core rows into token-kernel rows at its last partial chunk.

How much it mattered was measured at world 1 on the suite's eight prompts. Each prompt's greedy
answer (turn 1) was followed by a short user turn. That transcript was sent twice: to the warm
server that had just decoded turn 1, which prefilled only the new turn, and cold, after an
unrelated prompt had taken the sequence, teacher-forced on the warm reply's 64 tokens.

| decode | speculation | turns bit-exact | greedy text changed | mean KL lower bound | top-1 agreement |
| :-- | :-- | --: | --: | --: | --: |
| token kernels | DFlash | 1 of 8 | 2 of 8 | 6.4e-04 | 99.18% |
| token kernels | none | 0 of 8 | 3 of 8 | 8.1e-04 | 98.77% |
| tensor cores (version 1) | DFlash | 8 of 8 | 0 of 8 | 0 | 100% |
| tensor cores (version 1) | none | 8 of 8 | 0 of 8 | 0 | 100% |

Before this change, a continued turn sat about five times the TP=2 floor away from the same turn
served cold. A chat client sends exactly that request on every turn after its first.

Version 1 now runs every row count, from a one-row decode step to a 4,096-row chunk, through the
tensor-core image's arithmetic. A step below 64 rows takes few-row forms of the image's kernels.
They use the same operand split, the same K split (each matrix split as a full 512-row chunk
would split it) and the same summation order, and each row comes out with the bits the wide tiles
give it. This is DeepSeek's rule: its tensor-core decode was made batch-invariant the same way
and got faster. `SPARKFLASH_DECODE_MMA=off` puts decode back on the token kernels: the anchor's
arithmetic, reported on `/v1/models` as `overridden`.

## The bits

- **Row count:** at every row count from 1 to 63, each few-row kernel gives every row the bits
  the wide tile gives it. That covers the dense projection, the token GEMM, the fused split
  reduction and epilogue, and a launch that groups up to four matrices of one input
  (`tests/few_rows.rs`).
- **Continued turns:** a continued conversation's logits, and its caches at every layer, are the
  bits of a cold prefill of its whole transcript
  (`device_parity::a_continued_conversation_is_the_bits_of_a_cold_prefill`).
- **Batches and verifies:** a row decodes to the same bits alone, beside other rows and inside a
  speculative verify (`tests/batch_exactness.rs`, `tests/verify_rows.rs`).
- **The independent reference:** the device loop stays within the tensor-core image's bound
  against the f32 host oracle, prefill and decode both (`device_parity`). That bound is ten times
  the measured 1e-5 relative.
- **Determinism:** `numerics.py check` recaptured the world-1 reference and matched it bit for bit.

## Distance from the anchor

The anchor is `references/qwen38-anchor.json`, frozen at the token-kernel decode (it was captured
before this change, and anchors are never edited). Each world was teacher-forced on the anchor's
988 tokens at top-20 (`captures/qwen38/tpW-tc-decode-v1.json`). The TP floor columns repeat
[v1-tp-floor.md](v1-tp-floor.md): the same worlds with the token-kernel decode.

| metric | world 1 | world 2 | world 4 | TP floor, world 2 | TP floor, world 4 |
| :-- | --: | --: | --: | --: | --: |
| tokens scored | 988 | 988 | 988 | 988 | 988 |
| anchor NLL / token | 0.2228 | 0.2228 | 0.2228 | 0.2228 | 0.2228 |
| mean NLL delta | +6.992e-04 | -1.317e-04 | +6.459e-04 | +4.837e-05 | +7.360e-04 |
| mean absolute NLL delta | 4.996e-03 | 2.676e-03 | 4.508e-03 | 3.093e-03 | 4.246e-03 |
| p99 absolute NLL delta | 5.977e-02 | 4.379e-02 | 6.239e-02 | 5.893e-02 | 6.499e-02 |
| max absolute NLL delta | 2.112e-01 | 1.090e-01 | 2.126e-01 | 1.083e-01 | 2.114e-01 |
| top-1 agreement | 99.3927% | 99.6964% | 99.5951% | 99.7976% | 99.8988% |
| mean KL lower bound | 2.805e-04 | 1.077e-04 | 2.360e-04 | 1.283e-04 | 2.263e-04 |
| max KL lower bound | 2.189e-02 | 1.617e-02 | 2.193e-02 | 2.132e-02 | 2.181e-02 |

The KL column is a lower bound. It partitions the vocabulary into the tokens both arms list in
their top-20 plus one bucket for the rest.

## Reading

- **World 1 sits just above the floor.** Its mean KL (2.8e-04) is 24% above the TP=4 floor's
  (2.3e-04), and it changes the top-1 token at 6 of 988 positions where the floor changes 1 or 2.
  Its mean absolute NLL delta (5.0e-03) is 18% above the TP=4 floor's. By NUMERICS.md §4 this is
  a real precision change, slightly larger than serving at another world size.
- **Worlds 2 and 4 sit on the floor.** World 2's distance is below its own floor (KL 1.1e-04
  against 1.3e-04); world 4's is 4% above its floor's KL.
- **Every flip is a near-tie.** At the six world-1 flips, the anchor's top token led the token
  the candidate chose by 0.003 to 0.072 nats (code, tool-json, retrieval, summary, and two in the
  raw continuation). Five of the six flip at world 2 or 4 under the same arithmetic too, and two
  (retrieval, summary) flip in the token-kernel TP floor as well. The margin decides them, not a
  systematic shift.
- **No direction.** The mean signed NLL delta at world 1 (+7.0e-04) is the TP=4 floor's
  (+7.4e-04), and world 2's is negative. Against an anchor NLL of 0.223 per token, the 16-bit
  activation split does not make the model systematically less sure of the anchor's tokens.
- **The raw continuation carries the excess.** Without it, world 1's KL is 2.3e-04 against the
  TP=4 floor's 2.5e-04, and its top-1 agreement is 99.53%.

What this cannot settle: the suite is 988 tokens from eight prompts, so a flip count of 6 against
1 or 2 is a small-number difference. Coverage inputs and per-layer monitors (§5's prerequisites)
are not built, so a change confined to experts these prompts do not reach would not show here.
What it does settle: version 1 decodes exactly as it prefills. A continued turn is the bits of
the cold one, which the token-kernel decode could not give at any distance.

## Reproducing

From a checkout, with each configuration served in turn on the cluster (`--context 65536`;
TP=2 and TP=4 use the lead node and the first one or three workers):

```bash
# World 1: the reference, the determinism check and the forced capture.
python3 scripts/numerics.py capture --url http://127.0.0.1:8002 --out numerics/references/qwen38-tp1-sm121a.json
python3 scripts/numerics.py check --url http://127.0.0.1:8002 --reference numerics/references/qwen38-tp1-sm121a.json
python3 scripts/numerics.py capture --url http://127.0.0.1:8002 --forced-from numerics/references/qwen38-anchor.json --top-logprobs 20 --out numerics/captures/qwen38/tp1-tc-decode-v1.json

# World 2 and 4 (W=2 or 4): the reference and the forced capture.
python3 scripts/numerics.py capture --url http://<lead node>:8003 --out numerics/references/qwen38-tpW-sm121a.json
python3 scripts/numerics.py capture --url http://<lead node>:8003 --forced-from numerics/references/qwen38-anchor.json --top-logprobs 20 --out numerics/captures/qwen38/tpW-tc-decode-v1.json

# The table above.
python3 scripts/numerics.py compare --anchor numerics/references/qwen38-anchor.json --candidate numerics/captures/qwen38/tpW-tc-decode-v1.json
```

Captured 2026-11-03 from the official checkpoint
(`nvidia/Qwen3.8-Flash-Next-NVFP4`, snapshot `fc694b54`).

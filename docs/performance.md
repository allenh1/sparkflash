# Performance

Measured on four DGX Sparks (GB10, 128 GB each) at tensor parallelism 4, one network cable per
node with NCCL on both of its RoCE devices (about 200 Gb/s; [serving.md](serving.md#network)
explains the two), default GPU clocks, serving `deepseek-ai/DeepSeek-V4.1-Flash` from the
`sparkflash convert` per-rank files. All numbers are client-side, over HTTP streaming, on
2026-10-27 unless marked otherwise.

## Decode

Greedy, 256-token completions, prompts of about 800 tokens (60 filler records plus a question),
all streams starting together, each wave run on an otherwise idle server. *Per request* is each
stream's own rate from its first to its last token, which is what a user watching one stream
sees; *aggregate* is all tokens over the wall time of the run, prefill included.

| Workload | 1 stream | 4 streams | 8 streams | 16 streams |
| :-- | --: | --: | --: | --: |
| Code (per request) | **125.5 tok/s** | **51.0 tok/s** | **30.5 tok/s** | **18.8 tok/s** |
| Code (aggregate) | | 155 tok/s | 180 tok/s | 200 tok/s |
| Prose (per request) | **73.9 tok/s** | 39.5 tok/s | | |
| Prose (aggregate) | | 114 tok/s | | |

- DSpark speculation verifies about **5.3 tokens per step** on code and **2.7** on prose at one
  stream (`k = 5` drafts, depth chosen per step). The step itself is ~44 ms for code at full
  depth and ~37 ms for prose.
- Up to 32 decoding sequences share one speculative step of up to 64 verify rows, drafted in
  passes of sixteen; the rows go to the likeliest drafts, by expected tokens per modelled
  millisecond. Sixteen code streams still verify 4.1 tokens per step, at a ~191 ms step.
- Time to first token is ~0.2-0.5 s at one stream, ~1.4 s median at four code streams, ~2.4 s at
  eight and ~3.8 s at sixteen, since every prompt arrives at once and prefills alongside the
  others' decode.
- At 32 code streams: 10.9 tok/s per request and 185 tok/s aggregate, with a 12 s median time to
  first token while the 32 prompts prefill.

## Against the MiaAI-Lab SGLang build

The fastest published V4.1-Flash setup on four Sparks we know of is the
[MiaAI-Lab SGLang build](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/main/docs/tp4.md).
Its numbers come from sparkDash 1.8.8's DecodeBench, which we replicated request for request:
the same prompts, 256 tokens at temperature 0 with thinking off, a discarded first sweep, and
the single-stream prose and code figures as the median of five runs. Every wave ran on an
otherwise idle server.

Per-stream tok/s, **SparkFlash** / SGLang:

| Workload | 1 stream | 2 | 4 | 8 | 16 |
| :-- | --: | --: | --: | --: | --: |
| Prose | **86.2** / 87.7 | **58.1** / 61.9 | **41.7** / 41.4 | **28.5** / 30.8 | **18.3** / 22.2 |
| Code | **131.8** / 124.8 | **92.2** / 88.4 | **60.8** / 63.2 | **38.6** / 41.4 | **23.1** / 29.3 |
| Structured | **151.7** / 152.4 | **114.6** / 103.3 | **88.3** / 70.4 | **63.7** / 44.2 | **39.3** / 44.4 |
| JSON | **126.9** / 118.9 | **94.2** / 89.8 | **65.7** / 76.2 | **43.8** / 60.2 | **28.1** / 42.7 |

Aggregate tok/s, **SparkFlash** / SGLang:

| Workload | 2 streams | 4 | 8 | 16 |
| :-- | --: | --: | --: | --: |
| Prose | **114** / 120 | **157** / 164 | **205** / 238 | **258** / 343 |
| Code | **175** / 175 | **224** / 247 | **270** / 310 | **304** / 438 |
| Structured | **219** / 178 | **224** / 240 | **271** / 296 | **343** / 572 |
| JSON | **182** / 174 | **246** / 302 | **316** / 472 | **365** / 660 |

Tokens per verify step here, by stream count (1 / 2 / 4 / 8 / 16): prose 3.2 / 2.6 / 2.6 / 2.4 /
2.3, code 5.7 / 5.5 / 5.2 / 4.9 / 4.2.

- SparkFlash does not implement `ignore_eos` or `min_tokens`, which sparkDash sends, so most
  structured-output streams stop at their natural end before 256 tokens (2 of 2 at two streams,
  14 of 16 at sixteen). Their per-stream rates stand; their aggregates count fewer tokens.
- **One stream:** level. SparkFlash is ahead on code (131.8 against 124.8 tok/s) and JSON, level
  on structured output, and 2% behind on prose. It verifies the same tokens per step SGLang
  reports (~3 on prose, ~6 on code), at a step of ~37 ms (prose) and ~43 ms (code). Both now
  run their collectives over both RoCE devices of one cable (SGLang through a one-shot RDMA
  kernel, SparkFlash through NCCL).
- **Two to eight streams:** close. At two streams SparkFlash is ahead on code, structured output
  and JSON. At four and eight, code and prose are level to 8% behind and structured output is
  ahead, but JSON trails by 14% at four and 27% at eight. SparkFlash verifies up to 64 rows a step
  and keeps code at ~5 tokens a step through eight streams. Its decode GEMVs run on CUDA cores so
  that greedy output does not depend on the batch, so each extra row still costs compute, and the
  step grows from ~43 ms at one code stream to ~125 ms at eight. SGLang runs its routed experts
  on tensor cores.
- **Sixteen streams:** the cliff is gone. SparkFlash speculates for up to 32 sequences, so
  sixteen streams keep 2.3 (prose) to 4.2 (code) tokens a step and 18-39 tok/s each, up from
  ~12. SGLang is still ahead here: 1.3x per stream on code, 1.5x on JSON, and 1.3-1.8x in
  aggregate. Tensor-core decode kernels whose per-row reduction order does not depend on the batch
  would keep batch invariance while making extra rows cheap.

## Long context

Cold (nothing cached, the server's first prompts after a restart), greedy. The 209K prompt is
a needle in a haystack (a fact at half depth in filler records); the others are shipping-log
filler with a question:

| Prompt | Time to first token | Prefill rate |
| --: | --: | --: |
| 209,631 tokens | **44.4 s** (one run) | ~4,700 tok/s |
| 67,570 tokens | **11.8 s** (10.8-14.2 s over three servers) | ~5,700 tok/s |
| 32,613 tokens | **5.1 s** (median of six 32K prompts over three servers: 4.9-5.2 s, two outliers at 7.1 and 8.4 s) | ~6,400 tok/s |

- The 200K needle is found.
- Decode stays usable at depth: **56.0 tok/s** at a 208K-token context (2.1 tokens a step, at a
  37 ms step).
- A prompt that extends a previous turn reuses that turn's KV: only the new tokens are prefilled.
- Running streams keep decoding while a long prompt prefills (slower, but never stalled).

## Boot

**25 s** from the server starting to `/v1/models` answering (25.3 and 25.4 s on the two
restarts for this page, by the server's own boot timer), reading 77.7 GB per node from NVMe
(`O_DIRECT`, ~3.1 GB/s) and loading pre-built cubins. Timed from running the launch script, which
starts the four ranks over ssh, the same restarts took 29.5 s. There is no compilation, graph
capture or Python import at boot.

## Correctness

- With `SPARKFLASH_PREFILL_EXACT=1`, eight and sixteen concurrent greedy requests produce output
  **byte-identical** to the same prompts run one at a time, and the one-at-a-time output matches
  the previous build's (checked before every deployment).
- Greedy decode is the same at any concurrency: decode kernels keep each row's arithmetic order.
- Seeded sampling emits the same tokens whether or not speculation is on.
- Engram n-grams are hashed over the tokenizer's compressed ids, as the reference does, and
  checked against the reference implementation's hash columns.

## History

Single-stream decode on the same four Sparks, from the first end-to-end device run:

| Date | Commit | Milestone | Single stream |
| :-- | :-- | :-- | --: |
| 2026-10-20 | `d195b70` | first coded-weight decode harness | 1.06 tok/s |
| 2026-10-21 | `4417297` | host path, async device work | 1.7 tok/s |
| 2026-10-24 | `f361c35` | whole decoder layer on the device, NCCL | 6.9 tok/s |
| 2026-10-24 | `a7ea50b` | DSpark speculative decode on the device | 76.9 tok/s code |
| 2026-10-25 | `9b0c858` | `sparkflash serve` on the device path, 1M context | 87 tok/s |
| 2026-10-26 | `19c8054` | batched verify, adaptive depth, tick-ahead, bulk prefill | 112.8 tok/s code |
| 2026-10-26 | `771ed84` | Engram hashed over compressed ids (as the reference) | 117.9 tok/s code |
| 2026-10-27 | !188 | 64-row verify steps for up to 32 sequences, tensor-core drafter head, faster bulk prefill | 125.5 tok/s code |

The Engram fix raised code acceptance from 5.0 to 5.3 tokens per step on the same prompt; prose
went from 2.7 to 2.6.

200K cold time to first token went 119 s -> 100 -> 87 -> 70 -> 63.9 s over four prefill changes
(FP4 index scorer, paired chunks, sequence-parallel seams, merged expert pass). The bulk attention
at two blocks an SM then took 32K prompts from 9.2-9.3 s to 8.2-8.3 s, and a 209K-token prompt
to 57.3 s. Running only layers 0-20 on a prompt's rows before its last 2,541 then took cold 32K
prompts from 8.2 s to 5.1 s, 64K from 18.1 s to 11.8 s, and the 209K needle to 44.4 s (72-87 s
with the skip off on the same day, slower than the 57.3 s measured before).

## Reproducing

- Decode: stream `/v1/chat/completions` with `temperature: 0`, `max_tokens: 256`, and time the
  content deltas per request; run N such requests at once for the concurrent rows.
- Long context: a cold prompt (vary a salt so no prefix is cached), `max_tokens: 48`, time to the
  first content delta.
- Measure on an otherwise idle cluster, and discard the first runs after a restart.

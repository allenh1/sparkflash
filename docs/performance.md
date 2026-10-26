# Performance

Measured on four DGX Sparks (GB10, 128 GB each) at tensor parallelism 4, NCCL over one 200 Gb/s
RoCE link per node, default GPU clocks, serving `deepseek-ai/DeepSeek-V4.1-Flash` from the
`sparkflash convert` per-rank files. All numbers are client-side, over HTTP streaming, on
2026-10-26 unless marked otherwise.

## Decode

Greedy, 256-token completions, prompts of about 2K tokens (a block of filler records plus a
question), each run with a fresh system prompt so nothing is reused from the KV cache. *Per
request* is each stream's own rate from its first to its last token, which is what a user
watching one stream sees; *aggregate* is all tokens over the wall time of the run.

| Workload | 1 stream | 4 streams | 8 streams |
| :-- | --: | --: | --: |
| Code (per request) | **117.9 tok/s** | **44.5 tok/s** | **25.4 tok/s** |
| Code (aggregate) | | 146 tok/s | 157 tok/s |
| Prose (per request) | **69.8 tok/s** | 31.9 tok/s | |
| Prose (aggregate) | | 103 tok/s | |

- DSpark speculation verifies about **5.3 tokens per step** on code and **2.6** on prose at one
  stream (`k = 5` drafts, depth chosen per step). The step itself is ~46 ms for code at full
  depth and ~36 ms for prose.
- Four code streams vary from run to run (41.5 and 47.5 tok/s per request in two runs); eight
  are steady (24.8-25.9 tok/s over three runs).
- Time to first token for these prompts is ~0.1 s at one stream, ~0.4 s median at four and
  ~0.65 s at eight. Right after a restart the first prompts are slower (up to 0.5 s alone and
  2.3 s at eight) while their Engram table rows are first read from NVMe.
- Up to eight decoding sequences share one speculative step. Past eight, every sequence gets one
  plain (unspeculated) token per step, which is why sixteen streams fall off a cliff (below).

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
| Prose | **81.7** / 87.7 | **51.9** / 61.9 | **32.4** / 41.4 | **20.1** / 30.8 | **11.8** / 22.2 |
| Code | **126.6** / 124.8 | **82.7** / 88.4 | **47.5** / 63.2 | **23.6** / 41.4 | **11.4** / 29.3 |
| Structured | **144.3** / 152.4 | **99.5** / 103.3 | **62.3** / 70.4 | **34.3** / 44.2 | **17.2** / 44.4 |
| JSON | **120.3** / 118.9 | **80.2** / 89.8 | **48.0** / 76.2 | **25.5** / 60.2 | **12.9** / 42.7 |

Aggregate tok/s, **SparkFlash** / SGLang:

| Workload | 2 streams | 4 | 8 | 16 |
| :-- | --: | --: | --: | --: |
| Prose | **101** / 120 | **119** / 164 | **147** / 238 | **177** / 343 |
| Code | **159** / 175 | **156** / 247 | **158** / 310 | **171** / 438 |
| Structured | **191** / 178 | **179** / 240 | **185** / 296 | **180** / 572 |
| JSON | **156** / 174 | **181** / 302 | **177** / 472 | **193** / 660 |

Tokens per verify step here, by stream count (1 / 2 / 4 / 8 / 16): prose 3.2 / 2.8 / 2.9 / 2.0 /
1.0, code 5.7 / 5.5 / 4.3 / 2.3 / 1.0.

- SparkFlash does not implement `ignore_eos` or `min_tokens`, which sparkDash sends, so most
  structured-output streams stop at their natural end before 256 tokens (2 of 2 at two streams,
  14 of 16 at sixteen). Their per-stream rates stand; their aggregates count fewer tokens.
- **One stream:** level. SparkFlash is ahead on code and JSON and behind on prose and structured
  output. It verifies the same tokens per step SGLang reports (~3 on prose, ~6 on code), so the
  difference is step time: ~39 ms (prose) and ~45 ms (code) here against ~33 and ~39 ms there.
  Its collectives use a one-shot RDMA kernel on both network ports (~2.6 ms of all-reduce per
  step); SparkFlash uses NCCL on one port.
- **Two to eight streams:** SGLang pulls ahead, 1.3x per stream at four code streams and 1.8x at
  eight. SparkFlash verifies at most 16 rows per step across all sequences, and its decode GEMVs
  run on CUDA cores so that greedy output does not depend on the batch. Each extra row costs
  compute, so the step grows (45 ms at one code stream, 90 at four, 98 at eight) while each
  sequence's share of the 16 rows shrinks to about one draft at eight. The SGLang build runs its
  routed experts on tensor cores and keeps full drafts for up to 16 sequences. Tensor-core decode
  kernels whose per-row reduction order does not depend on the batch would keep batch invariance
  while making extra rows cheap.
- **Sixteen streams:** SparkFlash's batched speculative step serves at most eight sequences;
  with more decoding, every sequence gets one plain token per step. Aggregate throughput holds
  (~170-190 tok/s) but each stream drops to ~12 tok/s, against ~22-44 for SGLang. Speculating
  in groups of eight would remove the cliff.

## Long context

Cold (nothing cached), 48 greedy tokens, needle-in-a-haystack prompts, measured at `19c8054`
(before the Engram fix, which changes which Engram rows a prompt reads but not how many):

| Prompt | Time to first token | Prefill rate |
| --: | --: | --: |
| 203,324 tokens | **63.9 s** (65.1 s replicate) | ~3,200 tok/s |
| 31,945 tokens | **9.4 s** | ~3,400 tok/s |

- The 200K needle is found. Output is identical with and without the merged expert pass (one of
  the bulk prefill's optimisations, on by default), which takes 200K from 67.7 s to 63.9 s.
- Decode stays usable at depth: **38.4 tok/s** at a 208K-token context (measured 2026-10-25 at
  `65e957c`, before the later decode improvements).
- A prompt that extends a previous turn reuses that turn's KV: only the new tokens are prefilled.
- Running streams keep decoding while a long prompt prefills (slower, but never stalled).

## Boot

**25 s** from launching the four processes to `/v1/models` answering (24.5-25.4 s on twelve of
the last fifteen restarts, 29-31 s on the other three), reading 77.7 GB per node from NVMe (`O_DIRECT`, ~3.1 GB/s) and loading
pre-built cubins. There is no compilation, graph capture or Python import at boot.

## Correctness

- With `SPARKFLASH_PREFILL_EXACT=1`, eight concurrent greedy requests produce output
  **byte-identical** to the stored single-stream reference (checked before every deployment).
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

The Engram fix raised code acceptance from 5.0 to 5.3 tokens per step on the same prompt; prose
went from 2.7 to 2.6.

200K cold time to first token went 119 s -> 100 -> 87 -> 70 -> 63.9 s over the last four
prefill changes (FP4 index scorer, paired chunks, sequence-parallel seams, merged expert pass).

## Reproducing

- Decode: stream `/v1/chat/completions` with `temperature: 0`, `max_tokens: 256`, and time the
  content deltas per request; run N such requests at once for the concurrent rows.
- Long context: a cold prompt (vary a salt so no prefix is cached), `max_tokens: 48`, time to the
  first content delta.
- Measure on an otherwise idle cluster, and discard the first runs after a restart.

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
- Time to first token is ~0.2-0.5 s at one stream, ~1.2 s median at four code streams, ~2.0 s at
  eight and ~3.5 s at sixteen, since every prompt arrives at once and prefills alongside the
  others' decode, two or three short prompts a tick (~1.4, ~2.4 and ~4.5 s one a tick).
- At 32 code streams: 10.9 tok/s per request and 185 tok/s aggregate, with a 12 s median time to
  first token while the 32 prompts prefill.

### Sampling changes the rate

The tables above and below are greedy, which is speculation's best case: predictable text makes
for long acceptance runs. The same server, one stream, 256-token completions, five runs each,
measured 2026-10-30:

| Prompt | Sampling | Tokens per step | Median |
| :-- | :-- | --: | --: |
| Code | greedy, thinking off | 5.8 | **133 tok/s** |
| Prose | greedy, thinking off | 3.1 | **86 tok/s** |
| Code | temperature 1.0, top_p 0.95, thinking on | 2.5-3.4 | **80 tok/s** |
| Chat, "how a CPU cache hierarchy works" | temperature 1.0, top_p 0.95, thinking on | 1.9-2.3 | **59 tok/s** |

The step stays at ~40 ms in every row; what moves is how many tokens a step verifies. Sampling
shortens acceptance runs and removes the tail of near-certain drafts, so 133 tok/s is a ceiling on
the most predictable workload, and about 80 tok/s is what interactive sampled code sees. Two
identical greedy prose runs measured 87 and 52 tok/s on the same accept length (3.11) because the
step itself stretched, so single-stream figures are a band of roughly ±20%, not a constant.

## Against the MiaAI-Lab SGLang build

The fastest published SGLang V4.1-Flash setup on four Sparks we know of is the
[MiaAI-Lab SGLang build](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/main/docs/tp4.md)
(building on [`b12x`](https://github.com/local-inference-lab/b12x)'s `RoCEnante` one-shot RoCE
collectives, [`dgx-spark-networking`](https://github.com/christopherowen/dgx-spark-networking), and
[`knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4`](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4);
see also [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) and
[`blaesus/dsgx10`](https://github.com/blaesus/dsgx10)).
Its numbers come from [sparkDash 1.8.8's
DecodeBench](https://github.com/MiaAI-Lab/sparkDash), which we replicated request for request:
the prompts and their per-stream prefixes
([`src/shared/llmPrompts.js`](https://github.com/MiaAI-Lab/sparkDash/blob/main/src/shared/llmPrompts.js)),
the decode-rate definition `(completion_tokens - 1) / (last - first token)`
([`server/collectors/DecodeBench.js`](https://github.com/MiaAI-Lab/sparkDash/blob/main/server/collectors/DecodeBench.js)),
256 tokens at temperature 0 with thinking off, a discarded first sweep, and the single-stream
prose and code figures as the median of five runs. Every wave ran on an otherwise idle server.
Both tables below are a 2026-10-27 capture; the sweep client lives outside this tree, so a
re-run has to reproduce the protocol from those two files.

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
  and keeps code at ~5 tokens a step through eight streams. The step grows from ~43 ms at one
  code stream to ~125 ms at eight, mostly because more rows pick more distinct routed experts,
  each a separate read (see below). SGLang runs its routed experts on tensor cores.
- **Sixteen streams:** the cliff is gone. SparkFlash speculates for up to 32 sequences, so
  sixteen streams keep 2.3 (prose) to 4.2 (code) tokens a step and 18-39 tok/s each, up from
  ~12. SGLang is still ahead here: 1.3x per stream on code, 1.5x on JSON, and 1.3-1.8x in
  aggregate.

**Re-run 2026-11-11** on `implement-work-item-20` (the WS1-WS4 batch), one fresh epoch on port
8177, the same prompt set and protocol through
[`scripts/probes/sparkdash_decode.py`](../scripts/probes/sparkdash_decode.py) - the client the
paragraph above could not name. One stream (median of four waves): prose **81.6**, code **133.6**,
structured **159.9**, JSON **131.2** tok/s, against the capture's 86.2 / 131.8 / 151.7 / 126.9, so
code, structured output and JSON are level to 5 % ahead and prose is 5 % behind. The waves above
one stream, per-stream / aggregate: prose 64.1 / 116, 44.4 / 158, 30.8 / 212, 18.7 / 257; code
90.9 / 158, 56.4 / 193, 37.6 / 240, 22.6 / 257; structured 117.4 / 209, 82.1 / 284, 53.5 / 353,
31.9 / 379; JSON 95.4 / 166, 64.1 / 217, 43.3 / 277, 27.6 / 312. Tokens a verify step match the
capture (prose 2.9 / 2.6 / 2.6 / 2.4, code 5.5 / 5.4 / 5.2 / 5.2, JSON 5.1 / 5.1 / 5.0 / 5.0, and
structured pinned at 6.0 through eight streams). The code and JSON *aggregates* read 10-15 % below
the capture while their per-stream rates are level, which puts the difference in the prefill a cold
wave pays rather than in decode: a cold 16-stream wave spends 2-4 s of its 13-16 s span before its
first token.

### What an extra verify row costs

Rank 0's kernel time per step, from an Nsight Systems capture of code streams (PDL off, so the
steps run longer than above; 2026-11-01):

| Streams (rows a step) | Step | Routed experts | Dense projections | Collectives | Attention |
| :-- | --: | --: | --: | --: | --: |
| 1 (5.6) | 45 ms | 19.9 ms | 11.1 ms | 4.3 ms | 2.7 ms |
| 4 (20.9) | 97 ms | 56.2 ms | 15.8 ms | 8.2 ms | 5.5 ms |
| 8 (38.7) | 164 ms | 80.7 ms | 23.1 ms | 37.4 ms | 8.4 ms |

- **Routed experts** are half (eight streams) to two thirds (four) of what the extra rows add,
  and they are memory-bound. Each distinct expert a layer's rows pick is a 4.7 MB read per rank.
  Windows of recorded routes pick 22 distinct experts a layer at one stream's six rows, 56 at
  four streams and 82 at eight. Routes from different conversations pick more.
  - Reading each of those experts once, the routed kernels run at 218 GB/s.
  - With the real routes, where several rows share an expert, the kernels come within 2% (one
    stream), 4% (four) and 11% (eight) of that floor.
  - So the per-row FMA work is mostly hidden under the reads.
- **Dense projections** (the attention projections and the shared expert) add ~5 µs a row a layer
  on the CUDA cores: 213 µs a layer at one row, 445 µs at 48.
- **Collectives** at eight streams are mostly transfer once decode is steady. In a capture of all
  four ranks at once (2026-11-01), each rank waits 2-7 ms of a ~135 ms step for its peers. The
  rank that arrives last changes from call to call, mostly on the routed-expert kernels, whose
  time differs by rank, and no one node is slow. Rank 0's 28 ms in calls longer than 200 µs in the
  capture above came mostly from its first steps.
- **The Engram gather** is the largest host stall at eight streams. On text the server has not
  seen, every rank's GPU idles 3.7-4.6 ms a step before layer 1's Engram write while its table rows
  are read from NVMe. Once those rows are in the reader's cache the wait is ~0.1 ms, so a benchmark
  that repeats its prompts hides it.

Tensor cores need not make greedy output depend on the batch: an output keeps its bits if the
tile shape, the split of K and the order of each output's reduction do not depend on the row
count. The bulk prefill's GEMMs already have that property: a row gets the same bits alone as
among 48.

Decode runs the projections with FP8 or bf16 activations that way, as block-scaled MMA in
`kernels/opt_gemv_sm121.cu` (`SPARKFLASH_DECODE_MMA=off` keeps them on the CUDA cores). A block
keeps the GEMV's split of K over its eight warps, chains each warp's MMAs in ascending K from zero
and adds the warps' sums in warp order, so a row's bits do not depend on how many rows share the
launch or where the row sits in it. Projections with f32 activations, such as the shared expert's,
stay on the GEMV and the routed experts on their read-bound kernels. The five attention
projections of a layer, weights streamed from memory on one Spark:

| Rows | 1 | 6 | 24 | 32 | 48 | 64 |
| :-- | --: | --: | --: | --: | --: | --: |
| CUDA cores | 167 µs | 177 µs | 214 µs | 243 µs | 335 µs | 442 µs |
| Tensor cores | 176 µs | 179 µs | 188 µs | 181 µs | 199 µs | 211 µs |

Every row of every launch from 1 to 64 rows matches the row launched alone, and the error against
a double-precision reference is the GEMV's; `crates/sparkflash-kernels/tests/decode_mma_device.rs`
checks both on a device. On the four Sparks (code streams, CUDA and tensor cores alternated twice;
2026-11-01) the step goes from 43.2 to 43.1 ms at one stream and from 131.1 to 124.6 ms at eight
(5% less). Each request decoded the same alone as among eight on both, but five of eight
completions differ between them a few tokens in, because a tensor core sums a block's 32 products
in its own order. Tokens accepted a step move with the text, so per-stream throughput rose 1.5% at one
stream and 2% at eight. GPUs with several times this memory bandwidth read the routed experts in a
fraction of the time, so there the per-row FMA work dominates and tensor cores matter more.
There the small projections also leave most SMs idle: wkv runs on 16 blocks and wq_a on 40.
On one Spark, 32 blocks already keep memory busy, so splitting wkv's K over more blocks would save
2.6 µs a layer.

### A single stream's step folds a third of its slots, and reads at the same rate as any other

The step-shape model above prices rows at concurrency. A single stream is a different shape: three
rows (an anchor and its drafts) over a union of experts far smaller than the slot count, because
consecutive positions of one sequence route alike. The MoE id dump refused exactly this shape (it
demanded 32 rows), so the union came from four measured c1 steps first:

| step | slots | distinct experts | folded |
| :-- | --: | --: | --: |
| three rows | 18 | 12 | 33 % |
| three rows | 18 | 12 | 33 % |
| three rows | 18 | 9 | 50 % |
| four rows | 24 | 14 | 42 % |

Four steps are a sample, and a favourable one. The engine counts this itself, since only the plan
kernel knows the group count (`SPARKFLASH_MOE_PLAN_STATS=1`): over a c1 plus c8 run it reports
**14.52** distinct experts a call in the up-to-32-row band, 25.29 at 64 rows, 53.64 at 128, 69.12
above - and over a c8 plus c16 run, 13.02 / 31.32 / 53.01 / 80.42.

Those counts are what a rate must be divided by - not the routed-row count, which is
`tokens x top_k` and is what the kernel's grid is sized at. Divided correctly, the dedup kernels run
at **204 GB/s at c1** (47,626 expert groups, 140 GB of gate/up in 0.69 s) and **207 GB/s at c8/c16**
(818,853 groups, 2,416 GB in 11.69 s; the down projection 199 GB/s). That is 75 % of the part's
273 GB/s peak, the same figure the band table above records, and it does **not** improve or worsen
with the shape.

The correction worth keeping: an earlier version of this section compared a c1 rate computed from the
dump's twelve-expert sample against the *33..64-row band's* 231 GB/s and read a 1.4x gap out of it.
Blended to blended there is no gap - the samples differed, not the kernel. A rate here is only
comparable when both sides are divided by the plan's group count over the same calls.

**Occupancy was tested and is not the lever.** The gate/up kernel compiles to 255 registers (the ptxas
cap) at 128 threads - 32,640 of a GB10 SM's 65,536, so two blocks a multiprocessor and eight warps of
the forty-eight it can hold. `__launch_bounds__(128, 3)` fits the body in 168 registers (280 B of
spills) for twelve warps, `__launch_bounds__(128, 4)` in 128 (5,072 B of spills) for sixteen. Cold c1,
three samples an arm: **72.6 / 73.5 / 71.6 tok/s a stream** for 2 / 3 / 4 blocks, decode kernel time
36.54 against 36.81 ms, greedy text byte-identical (`a629cb2d93922881`). Flat inside the spread, so
the kernel is not waiting on warps it could have; the change was reverted rather than carried.

**The depth policy is at its optimum, and a corrected row price does not move it.** The c1 row
price looks wrong on paper: `STEP_COST` charges 4 ms a verified row, and a row really costs about
5.2 ms at a single stream (six experts x 4.42 MB x forty layers at 204 GB/s). Both directions were
run on the cluster rather than argued. Forcing the floor up (`SPARKFLASH_DSPARK_MIN_DEPTH=5`, verify
every draft) is 13 % **slower** - 62.9 against 72.6 tok/s a stream - so the policy is not drafting
too shallow. Pricing the row at the measured 5.2 ms (`SPARKFLASH_SPEC_ROW_MS=5.2`) is exactly
neutral: over six distinct prompts, paired prompt by prompt, **+0.0 %** (mean 71.40 either way,
paired sd 2.8, t = 0.00 on 5 df). Greedy text is byte-identical on both arms
(`a629cb2d93922881`), as a policy change must be.

**A measurement warning this cost a round to learn.** Prompts differ by 4-5 % at c1 - larger than
most effects worth chasing - and `gate_bench.py` used to seed each sample at `1000 + sample` with no
way to move the base, so every invocation at the default sample count re-sent prompt 1000. Three
separate runs then looked like three samples and were one prompt measured three times; that is how
a 5.2 ms row price appeared to be worth +3.9 % before it measured +0.0 %. The base is now
`GATE_SEED`, and a comparison is paired prompt by prompt across arms.

The one c1 cost that is measured as waiting rather than working is the exchange, and it is
measured as latency rather than bandwidth. Four drained runs, each over a different workload, report
what a round costs:

| run | row bytes a round | arrival wait | peers arrive at |
| :-- | --: | --: | :-- |
| c1 | 38,244 | 25.1 us | 0 / 11.9 / 15.0 / 19.3 us |
| c1 | 38,244 | 23.7 us | 0 / 14.5 / 13.6 / 17.2 us |
| c1 + c8 | 94,031 | 27.4 us | 0 / 17.7 / 18.3 / 20.0 us |
| c8 + c16 | 235,851 | 57.1 us | 0 / 37.8 / 39.0 / 45.7 us |

A round carries six times the payload between the first and last row, and its wait grows by 32 us:
the marginal rate is 198 KB / 32 us = **6.2 GB/s**, and the fixed part is what a small round pays
regardless. At c1, 38 KB costs about 14 us to deliver, which is roughly 2 us of transfer at that
marginal rate and **12 us of doorbell, wake-up and completion latency**. So c1's exchange is not
waiting for a peer to finish computing - the wait barely moves between c1 and c1+c8 (25 to 27 us)
even as the group's load changes; it is paying a per-round latency, 93 rounds a step,
**about 2.3 ms of a 36.9 ms step**.

Two paired A/Bs close the two ways out of that wait, both on the same six cold prompts with the
same seeds, so each is compared prompt by prompt against the shipped arm (mean 71.40 tok/s a
stream):

| arm | mean | delta | paired sd |
| :-- | --: | --: | --: |
| the NCCL ring the exchange replaced (`SPARKFLASH_EXCHANGE=off`) | 71.47 | +0.1 % | 0.12 |
| the single seam kernel instead of partials + tail (`SPARKFLASH_SEAM_SPLIT=0`) | 70.58 | **-1.1 %** | 0.12 |

The transport does not matter at this size - a 4-rank collective round costs what it costs - so the
wait is not the exchange's implementation. And the split seam is the right call even at three rows,
by 1.1 % with a paired sd of 0.12, so the two-launch shape is not the waste it looks like. Both arms
produced byte-identical text (`a629cb2d93922881`).

Recovering it is a protocol change, not a knob: the knobs that exist
(`SPARKFLASH_ONESHOT`, `SPARKFLASH_TWO_SHOT_ROWS`) select the NCCL ring path the direct-RDMA
exchange replaced, and nothing exposes "how many rounds may be in flight". Pipelining rounds is the
change, and it is the one class of edit this codebase has already been burned by: the exchange is a
global round queue that every rank must enter in the same order with the same row sizes, and a
per-round shape that differs between ranks fails at the next collective (D108's round-3039 failure,
and this goal's own round-477 boot that a half-staged binary caused). It is worth about 6 % of a c1
step and it is the last measured lever in this direction.

### The shared step's cost is two bands and a per-sequence term

`share_rows` does not charge a draft a fixed price: it divides the step's expected tokens by
`multi_step_ms(seqs, drafts, model)`, so the model's ratios decide which sequences get verify rows
and how many. Until this measurement that model was a guess - 27 ms + 4 ms a row, plus 2.3 ms a
sequence past the first, taken from a c1 six-row step and a c16 64-row step that happened to fit one
line (D105).

The engine prints one line a tick under `SPARKFLASH_BATCH_TRACE=1`, and a pure decode tick is
exactly one shared step: its rows are the anchors plus the verified drafts, its milliseconds the
step's wall. Fitting `scripts/probes/fit_step_bands.py` over 611 such ticks, from cold c4, c8 and
c16 waves (a fresh nonce and a fresh prose slice a request, greedy, thinking off, 192-token
completions), solves both bands and one shared per-sequence term in a single least-squares:

| band | model | ticks |
| :-- | :-- | --: |
| up to 16 rows | 25.5 ms + 2.68 ms a row | 315 |
| past 16 rows | 41.8 ms + 1.71 ms a row | 296 |
| a sequence past the first | 1.59 ms | 611 |

RMS 4.1 ms. The bands meet at the bend within 0.8 ms (68.4 modelled at 16 rows from below, 69.2
from above), so the bend is the data's and not the fit's: the marginal row falls past 16 rows, from
2.68 ms to 1.71 ms, because a wider step pipelines the row work a narrow one pays at its seams.
Against that, `STEP_COST`'s 4 ms a row is 1.5x the narrow marginal and 2.3x the wide one, and 2.3 ms
a sequence is 45 % high.

`SPARKFLASH_SPEC_BATCH_COST` takes `a0,a1,b0,b1[,seq_ms]`; a four-number value keeps the 2.3 ms
per-sequence term, the rows it prices are the step's own (its anchors plus the verified drafts,
the count the trace and the fit use), and an unset value is the pre-bend flat model exactly - 27 ms
for a c=1 step's anchor, 4 ms a verified row, 2.3 ms a sequence past the first - so the switch
cannot move a step it was not asked to move. The flat band is stated over the step's rows, which is
`23 + 4 x rows` for that same curve. A value that is not four or five non-negative finite numbers
is refused when the profile is applied rather than silently taking the flat model. The model is a
group knob: the depth it picks is the row layout every rank sends, so a rank whose model differs
desynchronises the direct-RDMA exchange (D110).

**A/B, one fresh epoch an arm, the same cold corpus, tracing on for both so the comparison is like
for like.** Per-stream decode rate: flat 38.4 against fitted 38.7 at c4, 24.9 against 24.8 at c8,
15.6 against 15.4 at c16; the aggregate, which includes each wave's cold prefill, moves 150 / 154
and 187 / 180. Inside the 4-5 % spread, so the rate is conserved - but the decision the model makes
is not:

| c | drafts a sequence | mean step |
| --: | :-- | --: |
| 4 | 1.70 -> 1.93 | 58.8 -> 61.4 ms |
| 8 | 1.40 -> 2.12 | 84.9 -> 96.4 ms |
| 16 | 1.22 -> 1.83 | 124.3 -> 150.1 ms |

The fitted model verifies half a draft a sequence more at c8 and c16, and the step grows by exactly
what the new row price says it should, so the tokens a millisecond do not move. That is the
self-cancelling-ratio lesson `optimization-strategy.md` already records for `DFLASH2_STEP_COST`: the
share compares a draft's chained probability against `rate` times a row's cost, so a model that is
wrong on both axes in the same direction lands near the right threshold anyway. Here the flat 4 ms a
row and 2.3 ms a sequence are both high by 1.4-1.5x in the mix the waves actually run, and the
plateau is wide enough that the depth mix never mattered to the rate. The deepest step seen is 48
rows at c16 (16 anchors plus two drafts each), well under `R_MAX` = 64, so it is the pricing that
stops the share, not the budget.

Text is byte-identical on every arm (sha256 `a629cb2d93922881`, 128 tokens), as a cost model must
be. What the measurement buys is a price the engine can defend for the shapes nobody has run yet -
c2-c4 at full depth, a wider `R_MAX`, and item C's lanes, which price their admission against the
same model.

### The routed MoE's grouping is built once a stage call

`opt_moe_gate_up_dedup_sm121` used to derive the expert grouping from the router's ids in **every
one of its blocks**, before a weight byte moved, and the launch sizes its grid at the row count, so
blocks below the distinct-expert count paid it only to leave. `opt_moe_plan_sm121` (one block, 128
threads) builds it once and the dedup blocks copy it; `SPARKFLASH_MOE_PLAN_INCTA=1` puts the old
in-block build back as the A/B arm (issue #20 WS1a, [DECISIONS.md](DECISIONS.md) D102).

Four Sparks, greedy, thinking off, 256-token completions, RDMA exchange, one pinned code prompt, each
cell a warmed server; kernels from one `nsys` capture of rank 0 over the same c16 cell:

| c16, code | decode tok/s a stream | aggregate tok/s | tokens a step | gate/up | down | plan |
| :-- | --: | --: | --: | --: | --: | --: |
| in-block build (`SPARKFLASH_MOE_PLAN_INCTA=1`, and the previous build) | 26.0 | 349 | 3.89 | **801 µs** | 407 µs | – |
| built once (the default) | 26.3 | 352 | 3.89 | **783 µs** | 404 µs | 8.7 µs |

- The in-block build costs **18.6 µs a gate/up launch**; hoisting it pays **8.7 µs once**, so the
  stage is 9.9 µs cheaper, about **0.4 ms of a 40 ms step** over 40 layers. That is inside the ±1.5%
  run-to-run spread of the end-to-end harness, which is why the wall-clock A/B only points the right
  way (26.3 against 26.0 tok/s a stream) while `nsys` resolves the mechanism.
- At one stream nothing moves (127.4 against 127.7 tok/s): two rows make a two-entry plan, and the
  copy is 8.7 µs of a 31 ms step. The router kernel is unchanged (18.1 µs a layer a launch).
- Every cell's text is **byte-identical** across the hoisted arm, the in-block arm and the previous
  build, at one and sixteen streams, on both prompts.

### The expert read is at 194-231 GB/s, not 112: divide by the experts, not the rows

The survey put the routed expert read at "at most 112 GB/s at c16 — 41% of this part" and named the
one-byte weight stream as the suspect (§6.0.5). Both halves of that were wrong, and the number it
asked for instead — the plan's own count — is now an instrument: `SPARKFLASH_MOE_PLAN_STATS=1`
makes `opt_moe_plan_sm121` add the grouping it just built into a host-owned device pair, and the
server prints one line a row-count band at drain.

The reason a single mean is not enough: the routed-row count a dedup launch is sized at is
`tokens x top_k`, **not** the number of experts it reads. At c16 a forward carries 64 tokens, so
`gridY = 384` — the number the survey's table reads as "rows" — while the plan groups **81.5**
distinct experts out of them. And the cells do not all run at that size, so the bands are reported
separately rather than averaged into one rate no launch runs.

One c16 code cell, 4,840 plan launches (40 layers x 121 forwards); `nsys` means over the same
bands of `opt_moe_gate_up_dedup_sm121` and `opt_moe_down_dedup_sm121` from the rank-0 capture:

| band | calls | mean experts a call | gate/up | down |
| :-- | --: | --: | --: | --: |
| `rows <= 32` | 1,200 | 12.43 | 204 GB/s (75%) | 151 GB/s (55%) |
| `33..64` | 280 | 23.63 | **231 GB/s (85%)** | 196 GB/s (72%) |
| `65..128` | 360 | 40.76 | 196 GB/s (72%) | 188 GB/s (69%) |
| `> 128` (62% of calls) | 3,000 | 81.50 | 194 GB/s (71%) | 197 GB/s (72%) |

- **The issue's own acceptance bar for WS1(b) is already met.** It asks for `>= 180 GB/s` at c16,
  and the dominant band reads gate/up at 194 GB/s (down at 197). At most 1.4x is left on this read
  before it hits the 273 GB/s the survey uses as this part's peak, and the pipe would have to reach
  95% of peak to take it.
- **The survey's mechanism described a different kernel.** The one-byte stream it indicted
  (`w_row[i0 / 2u]`, a nibble at a time) is `opt_moe_gate_up_body`, the per-row path; the dedup
  kernel has read 16-byte `ld.global.nc` chunks a lane since it was written. Its 41% was an
  artifact of the denominator: bytes are `count` expert matrices, and `count` is 3.2x the 25.5 it
  assumed, which is very nearly the factor by which the rate moves.
- **So the c16 cost is bytes moved, not bandwidth wasted.** A call in the dominant band moves
  255 MB gate/up and 128 MB down **a layer**, so a 40-layer forward moves 15.3 GB a rank, in about
  79 ms of a roughly 147 ms step. That is because 64 tokens touch 81.5 of 384 experts. Cutting it
  means fewer experts a forward, which is a batching and row-pricing question (issue #20 WS4), not a
  kernel-pipeline one. **WS4's next datum is the per-row expert ids of one c16 step**, which the rows
  state already holds (`workspace.moe_ids`, `[rows][top_k]` u32 global ids from the router,
  [moe_routed_device.rs](../crates/sparkflash-kernels/src/moe_routed_device.rs:673)), so it needs a
  gated host copy rather than new kernel work. With that matrix two things can be counted and neither
  is a guess: the union a step pays for, and each row's *marginal* experts - the quantity a price
  would go on, since a row sharing seven of its eight slots costs one matrix and a row bringing
  twenty new ones costs twenty.

  `SPARKFLASH_MOE_IDS_DUMP=20` on a c16 step (7,538-token prompts, 16 streams) gives that matrix,
  and the answer is not the one the collapse suggests:

  | rows kept (a prefix, the only shape a depth policy can pick) | experts | a row |
  | --: | --: | --: |
  | 10 | 29 | 2.90 |
  | 20 | 48 | 2.40 |
  | 30 | 56 | 1.87 |
  | 40 (the whole step, 240 slots) | **62** (3.87x collapse) | **1.55** |

  **The direction this points is deeper, not shallower** - and it corrects an earlier reading of
  this same step. Marginal cost decays hard (the first rows bring six and four new experts, the last
  ten bring six between them, and 28 of the 62 experts are touched by exactly one row), so a prefix
  gets *cheaper per row as it grows*: 2.90 experts a row at ten rows against 1.55 at forty. Keeping
  the 32 rows with the smallest marginals would cover 33 experts, but no policy can pick those -
  the ids arrive from the router *after* the rows are chosen - so that figure is not headroom and
  is withdrawn. What a price has to do is charge the marginal row what it costs rather than the
  average, which over-prices exactly the deep rows a step should prefer;
  [ws4_row_pricing.py](../scripts/probes/ws4_row_pricing.py) is the analysis, and it prints this
  table from a capture, with the by-depth and by-lane groupings.

  **The by-depth capture, which is the one a price needs.** A merged verify step of 32 rows over 9
  lanes at 16 streams, 192 slots, gives 69 experts - a 2.78x collapse, 2.16 a row - and split by
  draft depth:

  | depth | rows | marginal experts | a row |
  | --: | --: | --: | --: |
  | 0 (the anchor) | 9 | 26 | 2.89 |
  | 1 | 9 | 19 | 2.11 |
  | 2 | 6 | 14 | 2.33 |
  | 3 | 4 | 4 | 1.00 |
  | 4 | 2 | 4 | 2.00 |
  | 5 | 2 | 2 | 1.00 |

  A draft's marginal cost roughly halves after the anchor and settles near one expert a row from
  depth 3 - the rows a flat 4 ms over-prices most. Per lane it is wilder (0.50 to 4.00 experts a
  row), which says the number of lanes and the shallow drafts are what the step pays for.

  Fitting both recorded shapes gives the two bands D105 asks for, with the wall 22.8 ms fixed +
  0.71 ms a row + 0.97 ms an expert, and experts a step as `29.5 x rows^0.24` for merged multi-lane
  rows (69 at 32, 81.5 at 64) against `5.9 x rows^0.34` for one sequence's drafts (11 at six, 22.2
  at 48). It reproduces both anchors - 22.8 + 6 x 0.71 + 11 x 0.97 = 38.0 and 22.8 + 64 x 0.71 +
  81.5 x 0.97 = 147 - and charges the deep rows what they cost. Wiring it behind a switch and
  A/B-ing it is the last step, and it needs both shapes measured in one session.

  **The A/B says the price is inert, and that is WS4's answer.** The same switch run at both
  values - 4 ms and the measured 0.75 - moves nothing: prose at c1 reads 85.2 tok/s against 85.6,
  code 139.5 against 139.8, c16 is 29.7 either arm, and the **tokens a step is identical to two
  decimals in every cell** (3.00 prose, 5.67 code, 2.53 at c16), which is the direct evidence that
  the policy chose the same depths. The depth is set by the draft chain's own decay against a floor,
  not by what a row costs: at 0.95 acceptance the chain pays to the last draft either way, and at
  0.66 it stops either way. So a price on rows cannot buy deeper verification, and what is left of
  WS4 is the step's *shape* - how many lanes merge into one verify step - which is a scheduler
  change outside WS4 as the issue writes it (D106).

  **The constant that comparison would replace is `spec.rs`'s `STEP_COST`: 27 ms plus 4 ms a
  verified row, documented as "each row reads its own six experts a layer".** Two recorded points
  say that per-row term is too high where it matters most. A c1 step is 38 ms at about six rows and
  a c16 step is 147 ms at 64, which fits 26.7 ms fixed plus **1.88 ms a row** - the fixed part
  agreeing with the 27 to within noise, the marginal off by 2.1x. The expert accounting closes it:
  at c16 a row costs 1.55 experts (62 over 40) and an expert-matrix costs about 0.97 ms (79 ms over
  81.5 experts), so 1.5 of the measured 1.88 ms is the expert read and the rest is the seam, head
  and collectives. At c1 the one-stream cell groups 22.2 experts over 48 rows - 0.46 a row - so the
  model's 4 ms cannot be expert cost there either, which says the per-row term is shape-dependent
  and one constant cannot serve both. That is what a price has to fix, and it is the reason the
  first thing to change is not the selection but the cost model it selects against.
- The one-stream cell groups **16.1** experts a call in its `rows <= 32` band (1,120 calls) and
  22.2 in `33..64` (1,600 calls), against the c16 bands' 12.4 and 81.5. Its rate is provisional —
  the per-call times above are from a c16 capture, and a one-stream launch is a different shape —
  but on those times it reads at 264 and 217 GB/s, and the issue's c1 bar is `>= 90 GB/s`.
- The instrument is text-neutral: the c16 cell hashes `4b2edf6e…`, the same as the recorded
  hoisted-arm cell, and it costs two atomics a plan launch when armed and one uniform branch when
  not.

### The prefill indexer re-packed the whole key history every chunk

Issue #20 WS2a. `bulk_index` packed **every** compressed key from position zero into MMA fragment
order on every chunk and every index-source layer; the ticket's estimate is ~53 MB a layer a chunk at
209K. The pack kernel's `frag = 1` address is `(key >> 3) * 512 + (key & 7) * 64 + c * 4`, a function
of the absolute key index **provided the first key of a launch is a multiple of eight** (the layout
groups eight keys into 512 bytes), so appending the positions a chunk completed writes the same bytes
as re-packing all of them. The append is the default; `SPARKFLASH_INDEX_PACK_APPEND=0` selects the
full re-pack (the A/B arm) and `SPARKFLASH_INDEX_PACK_LOG=1` prints the blocks each call asks for.

One process, prefills of 6,741 / 26,943 / 53,871 tokens, blocks summed over the layers that share a
chunk boundary (`blocks_repack` is what the old call asked for):

| chunk p0 | n | layers | blocks now | blocks re-packed | |
| --: | --: | --: | --: | --: | --: |
| 0 | 1024 | 8 | 10,240 | 26,624 | 2.6x |
| 6,738 | 4,096 | 4 | 40,936 | 108,340 | 2.6x |
| 19,026 | 4,096 | 4 | 40,960 | 231,220 | 5.6x |
| 31,034 | 4,096 | 4 | 40,960 | 351,300 | 8.6x |
| 47,418 | 4,096 | 8 | 40,960 | 1,339,364 | 32.7x |
| 51,514 | 2,357 | 8 | 23,564 | 1,400,640 | 59.4x |

**The gate the issue calls the real one is met: the packed blocks a chunk asks for are constant in
`p0`** (40,960 across six chunk boundaries while the re-pack grows 13x), and the ratio grows with the
history exactly as the mechanism predicts. Roles that share an owner inside one chunk now pack nothing
after the first, which is the "owner 3 is packed five times in one chunk" case.

- **At this point it was not the default, and it must not have been: it changed the text.** (The
  scope bugs below were found later; the final measurement makes it the default.) On real prose (`book_32k`, the
  Project Gutenberg corpus the ledger draws from) the two arms differ on every warm cell — 7,538
  tokens: append `6d090718…` against re-pack `418f6a1e…`; 30,246 tokens: `f9d50149…` against
  `5385ddcb…` — while each arm reproduces itself request after request. So the difference is the code's,
  not the harness's.
- **The first version of that comparison was worthless, and finding out why took three launches.** The
  first A/B used synthetic prompts (the same eleven lines repeated hundreds of times) and "found" a
  difference; a control arm contradicted it: two launches of the *same* binary on the same prompt
  returned `d2fd3cd4…` and `eae262e2…`. The protocol that does hold is the one below — real prose, the
  first request discarded as the warm-up, and the next two required to agree before any arm is
  compared. Cold and warm are different inputs: 30,246 tokens cost 4,519 ms cold and 969 ms warm, and
  their texts differ (cold 30,246/4.519 = 6,692 tok/s, which is the published ~6,400 at 32K, so the
  harness reproduces the doc's own baseline).
- **One real bug it caught: the watermark outlived its sequence.** The watermark is on the forward and
  a lane serves many requests, so after the boot warm-up the next request found positions already
  packed, appended nothing, and scored the warm-up's keys. `index_pack_have` now refuses a watermark
  for a chunk at position zero or one whose key count is below the store's, and there is a test for
  it. That fix is necessary and **not sufficient**: the arms still differ, so a second, unexplained
  divergence remains.
- The eight-key alignment is a necessity, not a tidy-up: at 53,871 tokens the last boundary is
  `p0 = 51,514`, which is 2 mod 8, and the first build appended from there and addressed the fragment
  layout wrongly. `index_pack_start` rounds down to the super-block, re-packing up to seven positions
  it already packed, which is idempotent.
- **The second optimisation read as a content bug, and was a scope bug.** The layer-to-owner map is
  `[0,1,2,3,3,3,3,3]` — five of a chunk's eight index layers are owner 3, so five consecutive calls
  run the key pack over the same boundary. Skipping four of the five
  (`SPARKFLASH_INDEX_PACK_DEDUPE`) broke the run's *own* determinism: two consecutive warm requests
  on one prompt returned `418f6a1e…` then `a6461b79…`, and at 30,246 tokens three requests gave
  three texts. The checksum probe (`SPARKFLASH_INDEX_PACK_VERIFY=1`) then showed what the
  divergence was not: the five calls pack identical bytes. What broke was the repeat memory
  outliving its chunk — the boundary `(p0, n_max)` is a property of the prompt, so a second request
  over the same prompt matched the first's entry and skipped packs it had never run, reading the
  previous request's keys. Scoping the memory to one chunk pass (`pack_scope_is_new`) fixed it; the
  dedupe still ships off only because its fixed arm has no device A/B yet.
- **The append's own divergence had the same shape, one level up.** The watermark and the repeat
  memory also outlived the *sequence*. The store is one chunk state's scratch, and a chunk state is
  shared by every sequence of a serving epoch, so a watermark that is a bare position says nothing
  about whose keys are under it: a shorter prompt's pack at the boundary a longer prompt had reached
  would be mistaken for that prompt's own, and the longer prompt's next chunk would append onto the
  shorter one's head. The record now carries the sequence (`BulkState::set_index_seq`), the
  continuity check still requires `p0` to be the pack's token end, and placement clears both
  (`BulkState::reset_index`, including a lane re-placed after a preemption).

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

Issue #20 WS2 asked what the indexer's key packing costs a prefill. `bulk_index` packs the whole
compressed-key history into MMA fragment order on every chunk *and* every index-source layer -
eight of them per pass - so the work is the history times the layers. Two independent reductions
are measured here. The store is keyed by kv-source owner and five of the eight layers map to owner
3, so those five pack identical bytes (measured: layers 20, 24, 28, 32, 36 read the same head and
tail at one boundary) and four of the five packs are redundant; and because the pack kernel's
fragment address is a function of the absolute key index, a chunk need only pack the positions it
completed.

Three arms, one binary, three identical requests per arm, `SPARKFLASH_INDEX_PACK_LOG=1`:

| Prompt | Every pack | Owner-3 repeat skipped | Append only | Both |
| :-- | --: | --: | --: | --: |
| 7,538 tokens | 705,512 (11.3 MB) | 176,544 (2.8 MB) | 227,464 (3.6 MB) | — |
| 30,246 tokens | 3,523,016 (56.4 MB) | 1,449,904 (23.2 MB) | 877,704 (14.0 MB) | — |
| 60,488 tokens | 5,900,112 (94.4 MB) | — | 1,737,424 (27.8 MB) | 1,721,040 (27.5 MB) |

Every arm answers the text at every length measured: three identical requests give one sha, and the
append's is the every-pack arm's own (`d12c925e` at 7,538, `4fdfb215` at 30,246, `f87f6de6` at
60,488). The append is therefore the **default**, `SPARKFLASH_INDEX_PACK_APPEND=0` restores the
re-pack as the A/B arm, and the saving is 3.1x / 4.0x / 3.4x. The skip ships off
(`SPARKFLASH_INDEX_PACK_DEDUPE=1`). The append is *between* chunks; the four owner-3 repeats are
*inside* one, so the append leaves them in place — turning the skip on is worth 22% of the pack
traffic at 7,538 tokens and under 1% at 60,488, where the history the first layer packs dominates.

**The byte saving does not move time to first token at these lengths**, and saying so is the point of
measuring it: a cold 60,488-token prompt took 9,199 / 9,406 / 8,886 ms with every pack, the append,
and both, which is a 3% spread in the noise of one run each. The packed keys are 94 MB of a prefill
that also streams every expert weight and reads the whole KV history, so this workstream buys
traffic and bus time, not latency; its case would be a long enough prompt, or a decode step, where
that share is larger.

It took three fixes to get there, each found by the next prompt length, and the first two made the
text move: the packed scratch needed a slot per kv-source owner (one buffer for four owners left
the head as the previous owner's keys), and the watermark and the repeat memory had to live on the
chunk state rather than on the layer that three chunk states share (a shared watermark skipped
packs the second state still needed - that arm looked *better* at 30,246 tokens, 6.9x, precisely
because it was not doing the work).

Three switches remain because the prefill path may be wrong in ways a 60K prompt does not reach:
`SPARKFLASH_INDEX_PACK_APPEND=0`, `SPARKFLASH_INDEX_PACK_DEDUPE=1`, and
`SPARKFLASH_INDEX_PACK_VERIFY=1` for the checksum probe that found all three defects.

### Sonar, one boot of this build

[Sonar](https://github.com/christopherowen/sonar) measures a steady state rather than a ramp: a closed
loop with a warm-up, then windows, `temperature: 0`, 256 tokens a request, over its own 256-prompt
chat set. One boot of the branch, 2026-11-11, `sonar suite --suite smoke` plus a 16-stream loop
(60 s warm-up, 300 s measured):

| cell | rate | interval | TTFT p50 | TPOT p50 |
| :-- | --: | --: | --: | --: |
| 1 stream | 65.3 tok/s | ±4.2% | 231 ms | 14.9 ms |
| 4 streams | 127.3 tok/s | ±2.4% | 291 ms | 30.6 ms |
| 16 streams | 196.3 tok/s | ±1.4% | 439 ms | 79.8 ms |
| cold prefill, 8K | 5,220 tok/s | ±1.4% | | |

**The one-stream figure is not a slow engine, and the number that settles it is acceptance.**
Sonar's prompts are short chat questions, and this model thinks before it answers, so they verify
**2.46 tokens a step** (256 tokens, 152 accepted, identical across three runs) where a code prompt
verifies 4.50 to 5.8. The step is about 37 ms either way, so 256 tokens is 104 steps, which is 3.85 s
and 66 tok/s - the measured 65.3. The 125-133 tok/s this page quotes elsewhere is the same engine on
code, whose texts accept nearly twice as deep. Comparing a figure across corpora without the tokens
a step is comparing different workloads, not different builds.

This is one boot and no A/A floor, so it is a measurement and not a verdict: Sonar's `table` needs
candidate, reference and candidate again, with the floor recorded from repeated runs of one build.

## Reproducing

- Decode: stream `/v1/chat/completions` with `temperature: 0`, `max_tokens: 256`, and time the
  content deltas per request; run N such requests at once for the concurrent rows.
- Long context: a cold prompt (vary a salt so no prefix is cached), `max_tokens: 48`, time to the
  first content delta.
- Measure on an otherwise idle cluster, and discard the first runs after a restart.

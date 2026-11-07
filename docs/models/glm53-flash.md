# GLM-5.3-Flash (FP8, TP=4)

zai-org/GLM-5.3-Flash is a 321B-parameter (18B active) hybrid-attention MoE. The pack is
`crates/sparkflash-model-glm5-next`; this page covers the serving frontend that turns the rank
into an OpenAI-shaped endpoint.

## The model

| Property | Value |
| :-- | :-- |
| Layers | 45 (34 Kimi Delta Attention, 11 DeepSeek Sparse Attention), 3 dense MLPs then 42 MoE |
| Experts | 288 routed, 8 a token, one shared expert; block-scaled FP8 |
| Hidden / expert width | 4,096 / 2,048 |
| Attention | NoPE MLA, 64 heads, q-LoRA 1,536, kv-LoRA 512, indexer top-k 2,048 |
| Hyper-connections | `hc_mult = 4`, 20 Sinkhorn iterations |
| Vocabulary / context | 154,880 / 1,048,576 positions |
| Release | BF16 642.6 GB; FP8 328.3 GB (82 GB a rank over four 128 GB nodes) |

The architecture, tensor plan, host reference and device rank are documented in the crate root
([crate](../../crates/sparkflash-model-glm5-next/)); the device arm runs the Qwen4-Exp pack's
cubin.

## The serving frontend

`sparkflash serve --model zai-org/GLM-5.3-Flash` dispatches to the pack's arm. The arm is the
TP=1 device backend behind the generic `sparkflash-serve` host driver:

The web console configures the four-rank release by selecting **Model** `zai-org/GLM-5.3-Flash`
under Advanced settings. It then downloads the checkpoint, converts one rank per Spark, and starts
`scripts/serve-four-node.sh` with `--model glm5.3-flash-fp8`; GLM needs no external cubin
directory because its kernels are compiled into the binary.

* `Glm5NextDeviceBackend` (`src/serve.rs`) owns the device rank behind a mutex and opens one
  `DeviceSequence` a request. A rank's forward takes `&mut self` because a step writes the rank's
  scratch, so sessions serialize their steps; that is also the tensor-parallel requirement, where
  every rank runs the same row set.
* `Glm5NextSession` implements `ModelSession`. `prefill` runs the prompt as one greedy step per
  `MAX_ROWS` (eight) rows, reads the last row's logits with `DeviceRank::last_logits`, and samples
  on the host; `decode` runs one row the same way. The host sampler applies temperature, top-k,
  top-p, the three penalties and a seed. `speculation` is the zero default: the MTP and DFlash 2
  drafters are wired in the `glm-tp4` harness, and the MTP layer in the four-rank serve arm
  (below), not in the session yet.
* `GlmCodec` (`src/codec.rs`) renders the released `chat_template.jinja` over a Hugging Face
  `tokenizer.json`: the `[gMASK]<sop>` prefix, the default `Reasoning Effort: Max` system turn,
  the tool catalogue and its `<tool_call>` XML, assistant `reasoning_content`, `clear_thinking`,
  the merged `<|observation|>` tool turn, and the `<|assistant|><think>` generation prompt. The
  text-only subset is pinned to the template's output byte for byte in the module's tests.
  Divergences: a tool signature's and a tool call's argument keys serialize in sorted order where
  the template writes the caller's order, a run of tool results keeps the caller's order instead
  of re-sorting by call id, and an image/audio part is refused because the towers are not loaded.
* The descriptor (id, family, weights selector, context, vocabulary and capabilities) lives in
  `sparkflash-model`'s registry, so `/v1/models` and the API card report the pack rather than a
  V4.1 constant.

### Running the tiny fixture

The pack writes a synthetic checkpoint for any configuration, which is how the frontend is
exercised without 328 GB of weights:

```bash
cargo build -p sparkflash-cli
sparkflash serve --model zai-org/GLM-5.3-Flash --synthetic --bind 127.0.0.1:8000
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"zai-org/GLM-5.3-Flash","messages":[{"role":"user","content":"hello"}],"max_tokens":8}'
```

The fixture runs with a byte codec: it proves the dispatch, driver, backend, session and router,
not the released tokenizer or the model's quality. The gate is
`crates/sparkflash-model-glm5-next/tests/serve_tiny.rs`, which skips on a build without the cubin.

### The released model at TP=4

The released FP8 checkpoint is 82 GB a rank over four nodes, so it is served by the four-rank
serve-owned control channel (the `tp4-serve` feature). Build and launch one process a rank:

```bash
# A CUDA 13 toolkit must be on PATH so the Qwen4-Exp cubin is embedded at build time.
cargo build --release -p sparkflash-cli --features tp4-serve

# Once per node, for that node's rank: one ~78 GB file in the kernels' upload layout.
sparkflash convert --checkpoint "$CKPT" --output-dir "$CONV" \
    --tp-world-size 4 --tp-rank "$RANK"

SPARKFLASH_GLM_CONVERTED_DIR="$CONV" \
sparkflash serve --model zai-org/GLM-5.3-Flash --checkpoint "$CKPT" \
    --tokenizer "$CKPT/tokenizer.json" --tp4-rank "$RANK" --tp4-world-size 4 \
    --tp4-peers "$PEER0,$PEER1,$PEER2,$PEER3" --bind 0.0.0.0:8000
```

Rank 0 owns HTTP, the tokenizer and the control hub; ranks 1-3 load the same GLM rank and run the
same token ids, and rank 0 samples once a step, so every rank commits the same token. A greedy
request without logprobs whose penalties change nothing takes the step's argmax as the device
picked it, and the logits row stays on the device; any other request reads the row and samples it
on the host. Such a greedy request also decodes speculatively: the MTP layer drafts
`SPARKFLASH_GLM_DRAFTS` tokens after each token (2 by default; 0 drafts none), one verify step runs
the token and its drafts, and the drafts greedy would have picked are kept with the token after
them, so the tokens are the one-token steps' (`DeviceRank::verify_batch_greedy`, held to plain
greedy by `tests/device_forward.rs`). Rank 0 decides, and the Request frame carries the decision,
so every rank runs the same drafts and verifies. A count past 7 refuses at startup, as does a rank
without the MTP layer unless the count is 0.

The engine collective is the TCP seam `glm-tp4` already opens, partials on the f32 wire. By
default the seam gets the run's NCCL communicator as its device leg, as `glm-tp4` does on the f32
wire: each sublayer's partials are exchanged on the devices and summed in rank order by the same
kernel, so the tokens are the TCP path's. That is `SPARK_COMM_REDUCE`'s default on the device arm
for every model, DeepSeek's included. The NCCL transport variables in [serving.md](../serving.md)
apply. `SPARK_COMM_REDUCE=tcp` keeps the partials on the host seam, which needs no NCCL; any other
value refuses. A step of 16 rows or more (a prefill chunk) sums two-shot instead, with the
reduce-scatter and gather the Qwen3.8 arm shares (`reduce_two_shot`), on the device scatter
DeepSeek's two-shot uses: each rank sums its quarter of the
elements in the same rank order and the sums are gathered, so the bits are the one-shot's at half
the bytes a rank (`SPARKFLASH_TWO_SHOT_ROWS`, default 16, 0 for never).

### Speculation from a served request

#### Served, with the DFlash 2 block drafter

The four-rank serve arm drafts with a DFlash 2 block drafter when the profile names one
(`SPARKFLASH_GLM_DFLASH2`, the typed `speculation.drafter` field): every rank loads it, a
request's window is a block-drafting one, and the depth is the shared speculator's
([D72](../DECISIONS.md)) — adaptive, because the block drafter says how sure it is of each draft,
weighed against `DFLASH2_STEP_COST`. This is the first served measurement of it (2026-11-05, four
Sparks, converted ranks, one stream, greedy, 24-token prompt, 200 generated tokens, one run each):

| Drafter | Decode | Time to first token | A step | Tokens a step |
| :-- | --: | --: | --: | --: |
| MTP layer, 2 drafts | 51.3 tok/s | 498 ms | ~78 ms | 2.6 |
| **DFlash 2 block, adaptive depth** | **80.7 tok/s** (1M window) / **82.1** (64k) | 1,378 / 602 ms | 89 / 87 ms | **6.9** |

The request's own counters (28 verify steps for 200 tokens, `dspark_speculative`) say what the
drafter is worth: **196 drafts proposed, 13 gated by the depth policy, 177 accepted (90.3 %),
7.32 tokens a verify step**. The drafter is not the problem and a deeper chain would not help —
the block proposes seven drafts, so eight tokens a step is the ceiling and 86 ms a step puts that
at 93 tok/s.

#### The gap to 100 tok/s

The target is a **step budget, not a drafter budget**: at 7.3 tokens a step, 100 tok/s needs
**73 ms** where the arm spends 86-89; at the eight-token ceiling it needs 80.

`SPARKFLASH_GLM_PROF=1` (the typed `diagnostics.profile` field) puts marks around the step's
stages, and the steady state of a 200-token request at eight verified rows says where the 86 ms
goes (134.6 ms profiled a step — a mark drains the stream, so each stage is an upper bound):

| stage | profiled | share | scaled to 86 ms |
| :-- | --: | --: | --: |
| routed experts (`moe`) | 83.0 ms | 61.7 % | ~53 ms |
| mHC and the layer's all-reduce (`mhc_in`) | 28.3 ms | 21.0 % | ~18 ms |
| KDA attention (`attn_kda`) | 13.0 ms | 9.7 % | ~12 ms |
| sparse attention (`attn_dsa`) | 5.6 ms | 4.1 % | |
| dense layers (`mlp`) | 2.4 ms | 1.8 % | |
| head and fold (`tail`) | 0.7 ms | 0.5 % | |

The experts cost what they cost because the decode path reads each *selected pair*: a step of eight
rows over eight experts a row is 64 expert reads a layer, where a prefill chunk's grouped path
reads each distinct expert once. That is the lever for the last 20 %: halving the expert reads
takes the step to about 60 ms, which at 7.3 tokens a step is past 100 tok/s.

The gain is entirely in tokens a step: a verify of seven rows costs 89 ms against the two-row
MTP step's 78 ms, and 6.9 accepted tokens turn that into 1.6x the rate. A chat request with a
thinking block measures 59 tok/s, because reasoning accepts fewer drafts.

Two defects this measured, both fixed:

- **A model knob that only rank 0 inherited split the group.** `serve-four-node.sh` forwarded
  `SPARKFLASH_{PREFILL,BATCH,DECODE,ENGRAM}_*` and now forwards `SPARKFLASH_{GLM,QWEN4,SPEC,DRAFTS}*`
  too. Before that, rank 0 ran the block drafter and ranks 1-3 the MTP layer, and the group's
  first collective blocked with no error: 503 after two minutes with `steps=0`.
- **A block-drafting window cannot be reset in place.** `DeviceRank::reset` refuses one (its
  drafter context is keyed to the rows the request captured), so a request now replaces the
  window: the old one is dropped first, because two 18.2 GiB windows do not fit beside the
  94 GB rank at the 1M default.

Measured on the four Sparks (converted ranks, the 1M window, greedy, one request at a time):

| Collective | Load a rank | Decode a request | 2,204-token prompt: first token, then a step |
| :-- | --: | --: | --: |
| TCP f32 seam | 12.7-13.7 s | 16.9 tok/s (59.2 ms a step) | 45.1 s, 73.0 ms |
| NCCL device leg | 14.8-15.7 s | **27.0 tok/s (36.7 ms)** | **30.3 s, 50.9 ms** |

The four 256-token answers were the same text on both. A step is the same at a 65,536-position
window and with the f32 cache (36.6-36.8 ms). At 2,204 positions and a 65,536 window the f32
cache is the faster one: 46 ms a step and a 26.9 s prefill, against E4M3's 51 ms and 28.7 s.

With the window allocated at load (below), the device pick and the latent attention's loads issued
ahead (below), the NCCL leg decodes at 29.4 tok/s (34.0 ms a step) without drafting, and the
2,204-token prompt takes 27.3 s to its first token, then 48.5 ms a step; the texts are the ones
above. MTP speculation, one request at a time, same texts as without it:

| Drafts | Code prompts | Prose prompts | Prose, 2,204 positions |
| --: | --: | --: | --: |
| 0 | 29.3 tok/s | 29.5 tok/s | 20.6 tok/s |
| 1 | | 42.1 tok/s (1.83 a step) | 27.2 tok/s |
| **2 (default)** | **49.0 tok/s (2.64 a step)** | **45.3 tok/s (2.36 a step)** | **33.2 tok/s** |
| 3 | 50.0 tok/s (3.17 a step) | 43.1 tok/s (2.63 a step) | 25.3 tok/s |

A sampled request and a logprobs request decode a token a step, with the same text as without
drafting. The drafting window's MTP caches leave the prompt's first-token time as it was.

### Bulk prefill

A prompt of [MIN_PREFILL_ROWS](#the-released-model-at-tp4) (16) rows or more runs in chunks through
the tensor cores
(`kernels/glm_prefill_sm121.cu` for the dense GEMMs and `kernels/glm_moe_sm121.cu` for the
routed experts, embedded in the binary). The release quantises an FP8
projection's input to E4M3 codes with an f32 scale per 128 columns, as the decode GEMV does, so
by default the bulk GEMMs take it the same way: each chunk row is quantised once
(`glm_quantise_rows`) and an `m16n8k32` E4M3 x E4M3 MMA runs against the block-scaled weight, both
scales folded into a per-block partial sum after every 128 K columns. A BF16 weight (the KDA
projections) takes its f32 activation as a bf16 head and remainder, two `m16n8k16` chains a
stage. The output tile is 128x128; the dense GEMMs visit their tiles in groups of eight output
columns, so the blocks running at once share their weight tiles in L2 (a 6,144 x 4,096 x 4,096
FP8 GEMM went from 8.49 to 2.15 ms, 96 TFLOP/s). A tile lands in shared memory row-major with a
swizzle, so a weight row's 64 bytes stay together: split over two 32-column halves, as the first
version stored them, the expert stream held at about 126 GB/s against 205 GB/s for a plain copy.

The routed experts are grouped by expert into items of up to 128 sorted slots: each item gathers
its rows, runs the same MMA body over the expert's gate/up slice once, SwiGLUs, runs the down
slice, and scatters the weighted rows, so an expert's 6.3 MB rank slice is read once for the chunk
instead of once per row. The router and the indexer's projections stay on the token GEMV
(`SPARKFLASH_GLM_PREFILL_EXACT=0` puts them on the tensor cores too) so expert choice and selected
tokens are the decode path's. The KDA layers' short convolution and recurrence run over the whole
chunk in one launch each, with the Qwen3.8 arm's prefill kernels (`q4p_gdn_conv`, `q4p_kda_scan`),
instead of a row at a time (the scan splits each head's value columns over blocks until the heads
fill the 48 SMs, `gdn_scan_shape`; a column's recurrence is the same in any split, so the bits
are too); the mHC's narrow projections (24 or fewer outputs) have a kernel of
their own that reads the chunk's rows once. A chunk's sublayer partials are summed with the
two-shot all-reduce past 16 rows ([TP=4](#the-released-model-at-tp4)), half the bytes a rank
of the one-shot exchange.

Two things had to change for a chunk as wide as 4,096 rows. The per-layer KDA buffers are one
allocation shared by every layer (a layer's launches are stream-ordered, so no two layers hold one
buffer at once), and a chunk's KDA convolution windows are a one-float placeholder, because a chunk
never verifies and only a verify keeps windows for a commit to restore. At 4,096 rows that is
**9.6 GiB of device scratch** instead of ~55 GiB (the rank prints it at load: `GLM bulk prefill:
4096 rows a chunk, 9.6 GiB of chunk scratch`), plus 2.4 GiB of page-locked host memory the ranks
read the LM head's logits through, which is what lets the chunk fit beside the 78 GB of weights and
the 1M window. Most of the 9.6 GiB is the indexer's worst-case score rows and the LM head's
full-chunk logits, so a narrower chunk saves less than its row count suggests and a wider one costs
more. The MTP layer's stash keeps a chunk's last eight rows, the most a draft reads; it had copied
all 4,096 into a buffer that holds eight.

Measured on the four Sparks, one cold random-word prompt, greedy, per-request prefill rate (the
prompt's tokens over time to first token). The first block is the first version, at ~6.1k tokens:

| Chunk | Prefill | Time to first token at ~6.1k |
| --: | --: | --: |
| 8 rows (token path, `SPARKFLASH_GLM_PREFILL=0`) | 82.3 tok/s | 75.2 s |
| 512 | 205 tok/s | 30.1 s |
| 2,048 | 296 tok/s | 21.0 s |
| 4,096 | 314-316 tok/s | 19.4-19.7 s |

and what the changes above made of the 4,096-row default, one run a step. A run's rate moves by
about 10% between runs, so a small step is within the noise; the last step's prefill share is
mostly that, since the split attention is the decode path's:

| Build | 2.2k | 6.5k | 17.3k | 34.8k |
| :-- | --: | --: | --: | --: |
| bf16 split activations, before tonight's changes | 682 | 574 | 665 | 601 |
| E4M3 activations, narrow mHC kernel | 733 | 758 | 714 | 656 |
| two-shot chunk reduce | 701 | 821 | 774 | 699 |
| row-major weight tiles | 851 | 910 | 824 | 762 |
| KDA recurrence over the chunk | 893 | 974 | 888 | 797 |
| **KDA convolution over the chunk, split decode attention ([Context](#context)): the default** | **1,013** | **1,114** | **1,040** | **915** |

vLLM serving the same checkpoint on the same four Sparks prefills one request at 647 tok/s.

A fresh server's first cold 6.1k-token prompt measures 610 tok/s to first token on the default
build (2026-11-03); the rates above are from a server already serving.
- 8,192 rows does not serve: the ranks exit while loading, so 4,096 is the cap. 4,096 is also the
  default; `SPARKFLASH_GLM_PREFILL_CHUNK` picks another width. Below 16 rows a prompt runs on the
  token path, and a width past 4,096 is clamped to it rather than refused.
- The chunk's scratch is allocated only for a configuration the chunk can take: the f32 latent
  cache (whose absorbed MLA stages run at most eight rows) and a shape the bulk MLA does not serve
  load without it, and their prompts prefill on the token path. A DFlash 2 block-drafting
  request takes the chunk too: the chunk captures the drafter's target rows for the prompt's last
  `window_left` positions (2,047, the only ones its first block reads) and commits them to the
  drafter's context a step's rows at a time, as Qwen's chunks do. `SPARKFLASH_GLM_PREFILL_DRAFTING=0`
  (`prefill.drafting_prompt`) keeps such a prompt on the token path instead.
- A chunk's output is **not bit-identical** to the token path's: the MMAs sum each output's 128-wide
  K stage in their own order, so a chunk edge and the token path disagree in the last bits and a
  greedy completion can part company a few tokens in (4 of 4 code prompts did, at 6.1k tokens).
  `SPARKFLASH_GLM_PREFILL=0` keeps the token path's bits, as
  [SPARKFLASH_PREFILL_EXACT](../serving.md) does for the other packs. Within one launch a row's
  bits do not depend on which other rows share it, and the two-shot reduce, the row-major tiles,
  the chunk-wide KDA kernels and the split attention each left the greedy texts as they were.
- `SPARKFLASH_GLM_PREFILL_ACTIVATION` picks how the bulk GEMMs take their activations: `fp8` (the
  default for this release), `split` (a bf16 head and remainder, the first version's arithmetic
  and the default for a release that does not quantise activations) or `bf16` (the head alone).
  `fp8` on a release that does not quantise activations, and any other value, refuse at startup.
  The E4M3 GEMM takes at most 16,384 inner columns; a wider projection is refused with a message
  naming `split` rather than skipped.

### Startup: the converted rank file

The checkpoint path re-reads and re-lays-out every tensor. On the four Sparks that took **321-353 s
a rank** and about six minutes to the HTTP bind. The converted rank file is one allocation the arm
uploads in place, read with the same parallel direct-I/O path the V4.1 ranks use:
`SPARKFLASH_GLM_CONVERTED_DIR` selects it, and a directory that does not hold this rank's file is
a refusal rather than a silent fallback to the slow path.

Measured on the four Sparks (121 GiB usable a node, TCP f32 collective, 524,288-position context):

| Path | Load a rank | Server ready after launch |
| :-- | --: | --: |
| Checkpoint | 321-353 s | ~6 min |
| Converted rank file | **13.0-14.0 s** | **~24 s** |

Every node needs its own `rank-R-of-4.bin`; the file is node-local, not shared. Each rank prints
`GLM rank R/4 loaded in N s from converted <path>` at startup.

### Context

The sparse layers cache the compressed MLA latent, not an expansion of it. The served cache is
the release's **E4M3 activation**: 512 codes and four f32 block scales a position a layer, 528 B,
plus the indexer's 128-wide key and gate (f32) and its pooled key. That is 1.7 KiB a position a
layer, **18.2 KiB a position a rank** over the eleven sparse layers:

| Window | E4M3 latent + indexer, a rank | f32 latent + indexer |
| --: | --: | --: |
| 65,536 | 1.1 GiB | 2.2 GiB |
| 524,288 | 9.1 GiB | 17.2 GiB |
| 1,048,576 | **18.2 GiB** | 34.4 GiB |

The default is the model's full **1,048,576** in E4M3: 18.2 GiB beside the 82 GB rank, leaving
about 18 GiB for the kernels' scratch and the node. `LatentCache::Auto` picks per rank: the E4M3
latent when the release's dynamic activation applies and kv_b is block-scaled, which it is.
The f32 column is the expansion's exact operand and is not a serving configuration at 1M (it
leaves 1-2 GiB); `SPARKFLASH_GLM_KV_CACHE=f32` selects it on the serve arm, whose default is
`fp8`, and `f32` wants `--context 524288` or less. It is also the one cache the bulk prefill
cannot take: its absorbed MLA stages run at most eight rows, so a prompt of 16 rows or more with
the f32 cache falls back to the token path.

Attention runs in latent space. The query is projected into the latent once a row
(`q4e_glm_mla_absorb`), the softmax is over the latent (`q4e_glm_mla_attend`), and the weighted
latent sum is projected by kv_b's value rows: MLA absorption (arXiv:2405.04434) rather than the
expanded per-head K/V an earlier version of the arm cached. A rank's kv_b rows are dequantised to
f32 once at load (184 MB a rank) because absorption folds the block scales into the weight where
the FP8 GEMV folds them into the activation. Each head's value rows are stored transposed, so the
value phase reads them coalesced.

Both kernels run one block a head and row (16 blocks a rank at one row), so they are bound by load
latency, not bandwidth, and they issue eight loads ahead of the first multiply; the multiplies stay
in the same order, so the bits do too. That matters under PDL: before it, the next kernel's early
weight loads and the side-stream prefetch slowed the two kernels several-fold, and PDL gained
nothing on this arm (35.2 ms a harness step either way). With the loads ahead PDL gains 2.4 ms:
35.3 to 32.3 ms a step, the same last logits.

That one kernel still runs a block a head and row, so a decode step's attention uses 16 of the
48 SMs and each block gathers its 2,048 selected latents one after another: past a few thousand
positions it was the step's largest cost after the experts. The decode path now runs it in four
launches (`q4e_glm_mla_scores`, a warp a selected position, reading its latent once for every
head; `q4e_glm_mla_softmax`; `q4e_glm_mla_mix`, a lane a latent column; `q4e_glm_mla_project`,
every row of a run over one read of kv_b), each sum in the order the one kernel keeps and each
reduction its tree, so the outputs are its bits (`tests/glm_kernels.rs` holds the four to the one
kernel on both cache formats). The gathers are issued before any is decoded, and the scores
kernel scores four positions a staged query read. At 13k positions the attention went from
18.5 ms to about 2 ms a step, and the step from 64.2 to 48.3 ms, with the same greedy texts.
`q4e_glm_mla_attend` remains the reference the split is tested against.

Keeping kv_b in BF16, widened exactly in the kernels, gives the same bits and halves the weight
read: 0.3 ms a harness step, and the same 34 ms in serve. But at 2,204 positions it stepped 51.6 ms
against 48.1-48.6 for f32, two servers each way, so it was not kept. The two kernels' position
loops compile to the same instructions, so the loss comes from what runs beside them, not their
code.

The E4M3 cache is a serving trade, not the reference's arithmetic: the release's `kv_b_proj` is
BF16, so the expansion multiplied an f32 latent, while the delivered window multiplies an E4M3
one. On the tiny fixture - random weights, whose MoE routing flips on small perturbations - the
device/host seam moves by about 1e-2 relative with occasional top-k flips; that is why the parity
tests run the f32 cache. The live model's probe requests are correct at 1M (below).

Each rank allocates its window once, at load, and every request puts it back at position 0
(`DeviceRank::reset`). As in a fresh window, only the convolution history and the recurrent memory
are cleared; the caches are read below the position only. A window that does not fit beside the
weights is refused at startup. Allocating the window for every request had cost each one about
1.4 s before its first token at the 1M default: over four short prompts the first-token median went
from 2.0 s to 0.42 s and the slowest from 2.1 s to 0.84 s, with the same texts. The first request
after a start still ran its 40-token prompt in 0.81 s, against 0.34-0.42 s for the ones after it.

## Limits

* The four-rank control protocol carries temperature and seed only; top-k, top-p and the
  penalties are the session defaults on that path (the TP=1 path carries all of them).
* Prompt logprobs are a typed refusal on the four-rank path; the rank's step keeps one logits row.
* A client that aborts a request ends that request for the workers with an End frame; the group
  stays up. An idle gap longer than the control read timeout is also an idle wait, not a dead
  peer, so a quiet server keeps its workers.
* The arm streams the model's `<tool_call>` XML as text; it does not parse it into the OpenAI
  `tool_calls` field the way the Qwen arm's dialect layer does. The prompt side (the catalogue and
  the XML instructions) is rendered, so a client that reads the XML can use it, but a client that
  expects structured calls cannot yet. The `<think>` block is split into
  `reasoning_content` by [`GlmResponseEngine`](../../crates/sparkflash-api/src/glm.rs).
* One TP=1 process cannot hold the released checkpoint; the arm refuses that configuration with
  the 328.3 GB reason rather than attempting the upload.
* The 1M default needs the E4M3 latent cache; the f32 cache needs 34.4 GiB a rank there and does
  not fit beside the weights, so it wants `--context 524288` or less; see [Context](#context).
* The vision tower is planned in the checkpoint and not loaded, so the descriptor declares no
  vision channel.
* Fused batching and drafting in the session are not implemented; the four-rank serve arm drafts
  with the MTP layer, one request at a time. `--fuse-batch` and `--kv-budget-gib` are therefore
  refused at TP=1, and a group refuses those two and `--concurrency` above 1, rather than ignore
  them.

## Evidence

* Prompt encoding: `codec::tests` compares the rendered text with the released template's output
  for user, system, effort, `clear_thinking`, tool-catalogue and tool-call cases, and asserts the
  special ids the tokenizer must carry.
* Backend contract: `serve::tests` pins the descriptor and that a backend and a session are
  `Send`/`Sync` where the serving layer requires it.
* Router integration: `tests/serve_tiny.rs` drives the real router over a synthetic checkpoint on
  a host with the cubin. Its `tiny` fixture predates the mixer's 256-wide stream bound and is
  refused before the router is built, so the device gate is `tests/device_forward.rs` above and
  the serving path is exercised live on the four Sparks.
* The four-rank adapter was run on four GB10 Sparks: converted ranks load in 13.0-14.0 s a rank,
  the server binds about 24 s after launch, and a chat request answered correctly with the
  reasoning channel split. A 2 s abort of a 256-token request left every rank alive and the next
  request served. The control and parse paths are also covered by the `serve::tp4_serve_tests`
  unit tests, including a shortened-timeout idle gap.
* The absorbed attention is held to the expansion's numbers by `tests/device_forward.rs` on an
  `sm_121a` node: all eleven cases pass, including the release's activation scheme (median seam
  exact, none past the 1e-5 relative-L2 bar), batched rows bit-identical to one-token steps, the
  verify/commit and speculative paths bit-identical to plain greedy, the converted rank file
  bit-identical to the checkpoint, and a reset window bit-identical to a fresh one. World 1, 2 and
  4 all agree with the host world.
* The full **1,048,576-position** default was exercised on the four Sparks with the E4M3 cache:
  the server advertised `max_model_len 1048576`, a chat request allocated the 18.2 GiB window and
  answered "The capital of France is Paris." (103 GiB used, 18 GiB available), and a 5,976-token
  prompt whose first sentence named a secret number answered `4271`: the indexer selected outside
  the last 2,048 tokens, which is the DSA path at a window the f32 cache cannot serve. Decode
  measured 15-16 tok/s and prefill 51-56 tok/s on that node.

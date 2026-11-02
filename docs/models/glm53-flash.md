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
value refuses.

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

The default is the model's full **1,048,576**: 18.2 GiB beside the 82 GB rank, leaving about 18 GiB
for the kernels' scratch and the node. The f32 column is the expansion's exact operand; it is what
`DeviceRank::load` uses by default (and what the parity tests run), but at 1M it leaves 1-2 GiB and
is not a serving configuration. `SPARKFLASH_GLM_KV_CACHE=f32` selects it on the serve arm; the
default there is `fp8`, and `f32` wants `--context 524288` or less.

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
  with the MTP layer, one request at a time.

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

# Serving

`sparkflash serve` runs one process per DGX Spark. Rank 0 serves HTTP; ranks 1-3 only run the
model. [scripts/serve-four-node.sh](../scripts/serve-four-node.sh) builds and launches the four
command lines over ssh; this page describes what it sets up and what the server offers.

The lead node can do all of this from a browser: `cargo run --bin sparkflash -- --web` starts
[the console](web.md), which adopts and deploys the same binary, downloads the model once and
copies it over, works out the network settings, converts the weights and launches. The manual path
below remains the reference for what those buttons run.

## Prerequisites

On every node, at the same paths:

| What | How |
| :-- | :-- |
| The checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash` from Hugging Face (its `tokenizer.json` is the tokenizer) |
| The binary | `cargo build --release -p sparkflash-cli --features cuda,tp4-serve` (CUDA 13, `sm_121a`) |
| The kernels | `scripts/build-cubins.sh <dir>`: one `.cubin` per `kernels/*.cu` |
| NCCL | a `libnccl.so.2` built for aarch64 + CUDA 13 (`SPARKFLASH_NCCL_LIB`) |
| Converted weights | `sparkflash convert` (below): one ~78 GB file per rank, for a ~25 s boot |
| Networking | the four nodes reach each other over RoCE (ConnectX-7) and over ssh from the launch host |

### Network

A Spark's ConnectX-7 reaches the GB10 over two PCIe Gen5 x4 links of about 100 Gb/s each, so each
cabled port appears as two RDMA devices, each with its own netdev and address. For a cable in the
second port they are `rocep1s0f1` (`enp1s0f1np1`) and `roceP2p1s0f1` (`enP2p1s0f1np1`). One device
carries about half of the port's 200 Gb/s; list both in `NCCL_IB_HCA` to use all of it. A cable in
the other port adds no bandwidth, because both ports share the same two links.

With `NCCL_IB_GID_INDEX=3` (RoCE v2 over IPv4), NCCL takes each peer's MAC address from the node's
ARP cache once, when it connects, and never checks it again. A stale entry, for example one left by
a device that used to hold a peer's address, makes every connection to that address fail about
30 s after launch with `IBV_WC_RETRY_EXC_ERR`, while ping and ssh still work because ordinary
traffic re-resolves it. Before launching, `ip neigh show <address>` on every node should give the
MAC in that peer's `/sys/class/net/<netdev>/address`; pinging the address from each netdev
refreshes a stale entry. If a node's two netdevs share a subnet, each must answer ARP only for its
own address (`arp_ignore=1`), or peers can cache the other half's MAC. Index 1 (RoCE v2 over the
IPv6 link-local address) derives the MAC from the address and does not use the ARP cache.

### Converting the weights

Run once per node, for that node's rank (or `--ranks 0,1,2,3` on one machine and copy the files):

```bash
sparkflash convert --checkpoint "$CKPT" --output-dir /data/sparkflash-conv \
    --tp-world-size 4 --tp-rank "$RANK" --no-pad-moe-640 --no-verify
```

The file holds that rank's weights already in the kernels' layout, so boot is one large parallel
read. Without converted files the server reads and re-lays-out the checkpoint shards at every
boot, which is much slower.

## Launching

```bash
export SPARK_NODES="node0 node1 node2 node3"            # ssh names, rank order
export SPARK_PEERS="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4" # RoCE addresses, rank order
export SPARKFLASH_CHECKPOINT=/data/DeepSeek-V4.1-Flash
export SPARKFLASH_TOKENIZER=$SPARKFLASH_CHECKPOINT/tokenizer.json
export SPARKFLASH_CUBINS=/opt/sparkflash/cubins
export SPARKFLASH_NCCL_LIB=/opt/nccl/lib
export SPARKFLASH_CONVERTED_DIR=/data/sparkflash-conv
export NCCL_IB_HCA=<both RoCE devices of the cabled port> NCCL_SOCKET_IFNAME=<one of their netdevs>
export NCCL_IB_GID_INDEX=3
scripts/serve-four-node.sh print    # show the four commands
scripts/serve-four-node.sh launch   # start them
scripts/serve-four-node.sh status
scripts/serve-four-node.sh stop     # SIGTERM: rank 0 drains in-flight requests, then all exit
```

Rank 0 answers on `0.0.0.0:8000` (`SPARKFLASH_HTTP_BIND`) about 25 s after launch. `/readyz`
turns 200 when it accepts requests. If any rank dies, the others exit rather than hang.

## API

| Route | Notes |
| :-- | :-- |
| `POST /v1/chat/completions` | Streaming and buffered; tools and tool calls (DeepSeek V4.1 DSML, or Qwen3.8 `<tool_call>` XML, parsed into OpenAI `tool_calls`); `reasoning_effort` / thinking with the reasoning returned separately (`reasoning_content` for Qwen); images; `n`, `seed`, `stop`, `max_tokens`, `temperature`, `top_p`, `top_k`; `stream_options.include_usage` |
| `POST /v1/completions` | Text or token-id prompts, streaming and buffered; a prompt carrying an image marker is refused (this route has no image part to fill it) |
| `POST /v1/responses` | OpenAI Responses API: string, message, content-part (`input_text`, `input_image`), function-call and function-output input; instructions; reasoning items; Responses tools and tool calls as `function_call` items; buffered and SSE (`response.created` through `response.completed`) |
| `POST /v1/messages` | Anthropic Messages API: system blocks, text/thinking/tool-use/tool-result/image blocks, a `system` role inside `messages` (folded into the system prompt), `thinking` `enabled`/`disabled`/`adaptive` with `output_config.effort`, Anthropic tools and tool calls as `tool_use` blocks; buffered and SSE (`message_start` through `message_stop`) |
| `GET /v1/models`, `/v1/models/{id}` | The served model is `deepseek-ai/DeepSeek-V4.1-Flash` |
| `POST /tokenize`, `/detokenize` | vLLM-compatible |
| `GET /health`, `/readyz` | Liveness (503, with the reason, once the device loop has failed or stalled), and readiness (also 503 while draining) |
| `GET /metrics` | Prometheus: requests running / waiting / rejected, prompt and generated tokens, time to first token, DSpark accepted/proposed counters and last-request decode/prefill/acceptance gauges |
| `GET /v1/cluster/stats` | JSON: per-node state, KV cache, speculation and request timing statistics |

Authentication is off by default; `--api-key KEY` (or `SPARKFLASH_SERVE_API_KEY`) requires a
bearer token on every route except health, readiness and metrics.

Requests accept the served id, its last path segment, a trailing context suffix (`model[1m]`, the
Anthropic spelling Claude Code forwards), or any id listed in `SPARKFLASH_MODEL_ALIASES` (comma
or space separated). The alias list is what lets a client configured for another provider
(Claude Code sends its `ANTHROPIC_DEFAULT_*_MODEL` value) talk to this server unchanged.

### The model card names the weights that answered

`GET /v1/models` reports the checkpoint `sparkflash serve` loaded, not merely the id a client
asked for. `root` is the `--checkpoint` path when one was declared (and the model id otherwise),
`checkpoint` carries the same value as its own member, and `build` carries the package version
plus `SPARKFLASH_BUILD_ID` when an operator set one. A finetune served under the released model's
id is therefore distinguishable from the released weights (DECISIONS.md D38, D61).

The `capabilities` block names the modes the server actually enforces, so an unsupported mode is
visible before a request rather than as a typed 400 mid-run. On the device path, where the head can
mask its logits against a schema, `response_format_modes` is `["text", "json_object", "json_schema"]`
and `tool_choice_modes` is `["auto", "none", "required", "named"]` with `tool_choice_forced: true`;
a process without the tokenizer's vocabulary table advertises `["text"]`, `["auto", "none"]` and
`false` instead, because that is all it can guarantee (DECISIONS.md D64).

Speculative-decode counters are exported twice from one increment: the `sparkflash_draft_tokens_*`
spelling, and vLLM's (`vllm:spec_decode_num_accepted_tokens_total`,
`vllm:spec_decode_num_draft_tokens_total`, `vllm:spec_decode_num_drafts_total`), which is what
`tool-eval-bench --spec-bench` and vLLM dashboards read. `sparkflash_verify_steps_total` is the
step denominator behind tokens-per-verify-step (D61).

### Not supported yet

- **Responses and Anthropic shapes the chat template cannot represent are refused typed**, not
  silently dropped: hosted/built-in/custom Responses tools, `item_reference`, `previous_response_id`
  (this server keeps no response store), JSON output formats, Anthropic document/search blocks,
  and a tool choice that forces a call (OpenAI `required`, Anthropic `any`, or a named
  function/tool) on an engine that cannot force one. Images are read on all three conversation routes (chat message `image_url`/
  `input_image` parts, Responses `input_image` parts, Anthropic `image` blocks); Completions takes
  text only and refuses an image by name, whether it arrives as an `<image>...</image>` tag, a
  placeholder spelling or the image token in pre-tokenized ids, or as a carrier key in the body. Both adapters render the same V4.1 chat template and tokenize it through the
  checkpoint tokenizer, so with that tokenizer attached the prompt ids are the checkpoint's real
  ids.
- `logprobs` / `top_logprobs`, and repetition, presence and frequency penalties, on the device path
  (`--tp4-device`): refused with an error rather than ignored. The single-node reference engine
  serves both, with vLLM's definitions: the repetition penalty covers the prompt and the generated
  tokens, the presence and frequency penalties only the generated tokens. The TP=4 host loop
  (without `--tp4-device`) decodes greedily only; it serves logprobs but ignores the penalties.
- `response_format` `json_object` / `json_schema` on an engine without constrained decoding:
  refused with a typed 400. The device path enforces them (see above) and compiles the caller's
  schema before it serves, so a keyword the byte automaton cannot enforce is still a 400 rather
  than a stream that fails midway. The keyword subset is listed in
  `crates/sparkflash-api/src/constrained.rs`. A sequence that reaches a point where no vocabulary
  token continues the schema ends with an error of its own; the requests decoding beside it carry
  on (D65).
- `response_format` together with a `tool_choice` that forces a call: refused with a typed 400,
  because the call's arguments are not the response_format document. A named `tool_choice` whose
  function name holds a quote, an angle bracket or a line break is refused too: the forced call
  spells the name into the prompt, where those characters would end it early.
- `ignore_eos` and `min_tokens`: ignored, like other unknown request fields; generation stops at
  the end-of-sequence token.
- Anthropic thinking blocks carry an empty `signature`, and `stop_sequence` is always `null`:
  the generation seam does not expose Anthropic's encrypted thinking signature or which stop
  sequence matched, so the adapter does not invent either. `stop_sequences` still stop generation,
  but the finish reason cannot distinguish a matched sequence from natural end-of-turn, so both
  report `stop_reason: "end_turn"` with `stop_sequence: null`. An internal loop abort is also
  reported as `end_turn`.

## Behaviour

- **Concurrency.** Up to 32 sequences run at once (`SPARKFLASH_LANES`); up to 64 more wait in a
  queue, after which requests are refused (503). Up to 32 decoding sequences share one speculative
  step of up to 64 verify rows, drafted in passes of sixteen.
- **Prefix reuse.** A request that extends an earlier request's prompt (a multi-turn chat or an
  agent loop) reuses its KV. Concurrent requests with a common prefix prefill it once.
- **Long prompts.** Prompts are prefilled in chunks between decode steps, so running streams
  continue (more slowly) while a 200K-token prompt is read.
- **Context.** Up to 1,048,576 tokens per sequence (`--max-model-len` lowers it).
- **Sampling.** Greedy below `temperature` 1e-5. Seeded requests (`seed`) are reproducible, and
  speculation does not change what they produce.
- **Failures.** A device error ends every request on the device and queued for it with an error,
  and all four ranks exit, so a process supervisor can restart the group. When the loop makes no
  progress for `SPARKFLASH_STALL_SECS` (120 s) while it has work, `/health` and `/readyz` answer
  503 with the reason and new requests are refused; the server log records when the stall began
  and ended.

## Tuning knobs

Defaults are what the numbers in [performance.md](performance.md) were measured with.

| Variable | Effect |
| :-- | :-- |
| `SPARKFLASH_PREFIX_REUSE_MIN_ROWS=N` | The fewest shared rows a match on a lane holding a long prefix must have to be reused (default 256). Below it the lane is kept for a later turn that shares more of the prefix, so repeated short prompts on a full server re-prefill whole. `0` reuses every match, which is the pre-`!197` behaviour; a value that is not a row count refuses at startup |
| `SPARKFLASH_PREFILL_EXACT=1` | Prefill every prompt on the exact one-row path: slower long prompts, output bit-identical to a single-stream reference |
| `SPARKFLASH_PREFILL_DECODER_SKIP=0` | Run every layer over every prompt row (slower long prompts; for comparison). By default a prompt's rows before its last 2,541 run only layers 0-20, which hold every cache the later layers read |
| `SPARKFLASH_PREFILL_MOE_MERGE=0` | Run each in-flight prefill chunk's experts separately (slower; for comparison) |
| `SPARKFLASH_WARMUP=0` | Skip the warm-up pass before HTTP binds (the first request then pays the one-time setup) |
| `SPARKFLASH_MOE_DEDUP=0` | Run the routed experts once per routed row instead of once per distinct expert (slower at several streams; for comparison). Both paths produce the same bits, so the output does not move |
| `SPARKFLASH_MOE_PLAN_INCTA=1` | Rebuild the routed MoE's expert grouping in every dedup block instead of once a stage call (slower at c16; the A/B arm of [issue #20](https://git.allen-software.com/allenh1/deepseek-v4.1-flash-spark/-/work_items/20) WS1a, DECISIONS.md D102). The plan is the same bytes either way, so the output does not move. `SPARKFLASH_MOE_PLAN_STATS=1` prints the plan's per-band counts and bytes, but only on the hoisted path (the plan kernel that accumulates them is not launched with `INCTA=1`) |
| `SPARKFLASH_INDEX_PACK_APPEND=0` | Re-pack the whole packed-key history every prefill chunk instead of appending only the groups the chunk completed (the A/B arm; performance.md's WS2 section). The append is the default and reproduces the re-pack's text, but the saving is bus time, not time to first token at the measured lengths |
| `SPARKFLASH_INDEX_PACK_DEDUPE=1` | Skip the repeated owner-3 pack inside one chunk pass (opt-in). The append removes the *earlier chunks'* re-pack; inside one chunk the five owner-3 layers all run at the chunk's `p0`, so each would re-pack the whole history without this - the four skips are 22% of the pack traffic at 7,538 tokens and under 1% at 60,488. `SPARKFLASH_INDEX_PACK_LOG=1` prints the blocks each call asks for, and `SPARKFLASH_INDEX_PACK_VERIFY=1` checksums the store after each pack |
| `SPARKFLASH_SPEC_BATCH_COST=a0,a1,b0,b1[,seq_ms]` | Price a shared step with the fitted two-band model (D110): `a0 + a1*rows` up to 16 rows, `b0 + b1*rows` past it, plus `seq_ms` a sequence, over the step's rows (every anchor plus the verified drafts). Four numbers keep the engine's per-sequence term; unset is the old flat model, and a value that is not four or five non-negative numbers refuses at startup |
| `SPARKFLASH_SPEC_ROW_MS=N` | Price a verified row at `N` ms in the depth policy for the WS4 A/B (`speculation.row_ms`; unset keeps the engine's 4 ms; anything but a non-negative number refuses at startup). The measured A/B at c1 was neutral (DECISIONS.md D106, D113), so this is an instrument, not a default |
| `SPARKFLASH_MOE_IDS_DUMP=<layer>` | Copy one step's routed expert ids at `<layer>` to the host and print them (a layer number, e.g. `20`; the first 48 steps that reach it, on a prefill or a decode step). The datum a row price needs, and the one the plan-stat instrument cannot report (WS4; performance.md) |
| `SPARKFLASH_DECODE_MMA=off` | Run the dense decode projections on the CUDA cores instead of the tensor cores (`on` by default): for DeepSeek-V4.1-Flash 5% slower at eight streams, with the greedy output from before the tensor-core decode. Read by Qwen3.8-Flash-Next too, where `off` puts decode steps and verifies back on its token kernels: faster at one stream (20.1 against 18.5 tok/s without a drafter, 29.9 against 28.0 with DFlash 2), but a continued chat turn is then no longer bit for bit a cold prefill of its transcript ([qwen38](models/qwen38-flash-next-nvfp4.md#decode-on-the-tensor-cores)). A request decodes the same alone as in a batch either way ([performance](performance.md#what-an-extra-verify-row-costs)) |
| `SPARKFLASH_GLM_PREFILL=0` | Run GLM-5.3-Flash prompts on the token path (one GEMV a few rows) instead of the bulk prefill: 82 instead of 1,114 tok/s at a 6.5k-token prompt, with the bit-identical greedy output ([glm53-flash](models/glm53-flash.md#bulk-prefill)) |
| `SPARKFLASH_GLM_PREFILL_CHUNK=N` | The GLM bulk prefill's chunk width in rows (default 4,096). Wider chunks read each weight fewer times per prompt; 4,096 is the widest that fits beside the weights, its 9.6 GiB of chunk scratch and the 1M window. A prompt below 16 rows runs the token path whatever this says, and a width past 4,096 is clamped to it |
| `SPARKFLASH_GLM_PREFILL_EXACT=0` | Put GLM's router and indexer projections on the tensor cores too, instead of the token GEMV (`on` by default): the same 316 tok/s at a 6.1k-token prompt when it was measured, without the guarantee that expert choice and selected tokens are the decode path's |
| `SPARKFLASH_GLM_PREFILL_ACTIVATION=fp8\|split\|bf16` | How GLM's bulk prefill GEMMs take their activations: `fp8` (the default for the release, which quantises them) as the release's E4M3 codes into an E4M3 x E4M3 MMA; `split` as a bf16 head and remainder (the first version's arithmetic); `bf16` the head alone. Any other value, and `fp8` on a release that does not quantise activations, refuses at startup |
| `SPARKFLASH_TWO_SHOT_ROWS=N` | The fewest rows a tensor-parallel step sums with the two-shot all-reduce (reduce-scatter, then gather) instead of the one-shot exchange (default 16; 0 never). Read by the DeepSeek, GLM-5.3-Flash and Qwen3.8 device arms; a value that is not a row count refuses |
| `SPARKFLASH_SEQUENCE_PARALLEL=0`, `SPARKFLASH_SEQUENCE_OVERLAP=0` | Qwen3.8 at TP=2/TP=4: run a prefill chunk's hyper-connections and PLE reads replicated on every rank instead of on each rank's own rows, or run a sequence-parallel chunk on one stream instead of as two halves whose collectives overlap the other half's compute (both `1` by default, bit-identical either way). GLM-5.3-Flash at TP=4 reads `SPARKFLASH_SEQUENCE_PARALLEL` for its bulk prefill chunks' hyper-connections the same way, on one stream: overlapped halves measured slower there ([glm53-flash](models/glm53-flash.md#bulk-prefill)). Set the same on every rank; a value other than 0 or 1 refuses |
| `SPARKFLASH_ENGRAM_DIRECT=off` | Read the Engram tables through the page cache instead of with `O_DIRECT` (`on` by default). Through the cache, the ~200 GB of random lookups keep each node's memory full and the kernel reclaiming while it decodes. A table that cannot be read with `O_DIRECT` is refused at startup unless this is `off` |
| `SPARKFLASH_ENGRAM_SHARD=off` | Read every Engram column of a prompt chunk on every rank instead of a quarter of them each, exchanged on the device (`on` by default). Either way the arena holds the same bytes and the output is the same; sharded, each node issues a quarter of the chunk's random reads. A decode step does not shard: see the note below |
| `SPARKFLASH_ENGRAM_DEDUPE=0` | Read a gather's repeated table entries once and copy them to every row that looks them up (`on` by default). A verify step over eight lanes repeats a share of its 2/3/4-grams, and a row's own columns can hash to one bucket; each distinct entry is read once. Bit-identical either way |
| `NCCL_*`, `SPARKFLASH_SERVE_API_KEY`, `SPARKFLASH_PREFILL_*`, `SPARKFLASH_BATCH_*`, `SPARKFLASH_DECODE_*`, `SPARKFLASH_ENGRAM_*` | Passed to every rank by the launcher |
**Measured 2026-11-11: the stage is live; the counter is not.** `sparkflash_engram_lookups_total`
reads 0 here because nothing on the device path increments it: the counters belong to the host
cache (`sparkflash-engram/src/cache.rs` calls `EngramLayerTelemetry::record_lookups`), while this
path's reader (`tp4_forward/spec.rs`'s `EngramReader`) reads the tensor rows itself and holds no
handle on that telemetry, so `/metrics` publishes the initialised snapshot. Three observations
stand against reading those zeros as an idle stage:

- rank 0 of the `20261111T181950Z-final` run held two `O_DIRECT` descriptors on
  `model-00047`/`model-00048`, the checkpoint shards that carry
  `layers.{1,14}.engram.embed.{weight,scale}` (`/proc/<pid>/fd` flags `O_DIRECT|O_CLOEXEC`);
- its `read_bytes` moved 5.7 MB serving one cold 7,538-token request and 2.7 MB on the repeat,
  against 0 in an idle window - the order the sharded plan predicts (6 of the 24 columns a rank,
  ~12 MB of row reads for 7,538 rows), not zero;
- `pack_engram` is not what this path reads. `EngramReader` pages the two tensors out of the
  checkpoint's `SafeTensorsSource` in place, so `--engram-dir` and `pack-engram` belong to the
  host reference path, and the ~200 GB of tables were never missing from this deployment.

**The shard switch, measured (2026-11-11, issue #20 WS5).** Fresh epoch per arm, greedy, thinking
off, cold prompts (a random 24-char nonce ahead of each), 8-token completions, two samples an arm:

| Arm | 8k prefill | 32k prefill | 64k prefill |
| :-- | --: | --: | --: |
| `SPARKFLASH_ENGRAM_SHARD` on (default) | 5,124 / 5,048 tok/s | **6,708 / 6,703** | 6,889 / 6,884 |
| `SPARKFLASH_ENGRAM_SHARD=off` | 5,109 / 5,048 | 6,445 / 6,415 | 6,821 / 6,964 |

Sharding is worth 4.2 % at 32k (both samples) and nothing measurable at 8k or 64k, so it stays on
by default. The output is the same text either way: sha256 `a6cfb97f191ee5a5` over 47 tokens on
all three arms. A first sample on a freshly killed epoch read low (4,469 / 5,950 / 6,322) and the
next two on the same epoch did not, so one wave of one boot is not an arm.

**The decode side (issue #20 WS5 items 1-3), measured 2026-11-11.** Two of the three levers
were tried; one is in and one is out on evidence.

- **the dedupe is in** (`SPARKFLASH_ENGRAM_DEDUPE`, on by default). A read whose table bytes an
  earlier read of the same slot already lands becomes a copy, made once that slot's reads have
  landed (`EngramGather::settle`) and before its upload. The key is (slot, table bytes), so two
  rows - or two columns of one row - that hash to the same bucket cost one read. **Measured at c8**
  on cold prompts (eight streams, fresh nonce and corpus slice a request, greedy, thinking off,
  one fresh epoch an arm, 128-token completions, port 8189/8187): on 134.7 tok/s mean over eight
  samples (median 134.5), off 132.4 mean over seven (median 133.0). That is +1.1 to +1.7 %, inside
  a 4-5 % sample spread, so it is a direction and not yet a number - the reader's entry cache
  already answers a repeated entry, which is why the mechanism's saving barely reaches the step.
  Text is identical either way (sha256 `efb0f3d38a387f1d` over 49 tokens).
- **the shard is out.** `spec_multi_step`/`plain_step` passed the exchange above 16 rows a step,
  each rank reading a quarter of every row's columns into a run-scope share buffer on
  `EngramDevice`. The eight-stream burst killed the device loop within a second: rank 1 reported
  `round 3039 did not complete: rank 0 published a row length that is not the one this round sent`
  while rank 0 was at `all_reduce_moe` and rank 1 at `all_reduce`. The ranks were one collective
  out of phase, so a step's share exchange was not symmetric - the direct-RDMA exchange is a
  global round queue and needs the same size and the same *order* on every rank, which a
  per-step, per-rank-shaped slot does not guarantee. With `SPARKFLASH_ENGRAM_SHARD=off` the same
  workload is clean (the eight samples above). Landing it needs a fixed `R_MAX`-shaped exchange
  (and a permute that takes the per-rank stride) plus a rank-invariant shard decision; that is a
  separate change, not a constant.
- **the early anchor submit is not in.** `spec_step` already reads a lone sequence's anchor under
  its draft (`read_ahead`), but `spec_multi_step` cannot: its arena rows are step rows and an
  anchor's row is the running sum of the drafted depths. Landing it means either a second pinned
  buffer with a per-row upload or a row table, and the anchors are only `1/g` of a step's rows.

**Still open: the acceptance number.** Read wait at c8 on cold text (`<= 1.0 ms/step`, from
3.7-4.6) is not met by the dedupe alone; the shard was the lever that would have met it, and it
needs the fixed-shape exchange above. The c8 aggregate gate (+5 % on cold prompts) reads +1-2 %
today. The scripted arms are `SPARKFLASH_ENGRAM_DEDUPE=0` (this one) and, once it exists, the
fixed-shape shard.

**The 1M-context ceiling is lane memory, not the index (2026-11-11).** A 1M-token prompt is
advertised (max_context_tokens = 1_048_576) and reaches the engine (32 MB request bodies are
accepted), and nothing in the block selection is 16-bit: selected is uint32_t at four bytes an
entry, and block_size/num_blocks are 32-bit. pico-dgx-ds4's 17-bit block list is a
shared-memory packing of their blist, a constraint our path does not have. What bounds a
session is the lane budget: ~1.9 GB for 250K tokens (lanes.rs), so ~7.6 GB for 1M against the
24 GiB per-rank pool (SPARKFLASH_LANE_MEMORY_MB) - one 1M session fits, sixteen do not, and a
sequence that cannot grow waits for memory rather than evicting another lane. D109. The bulk
prefill's packed-key store is part of that budget too: one scratch slot per kv-source owner
(four owners) holds `max_positions * index_dim / 2` code bytes plus its scales, which four
owners multiplies by four - the price of the owner-slot fix that made the append correct.

**Benchmarks need a fresh epoch.** The `final` instance, up for two hours and sixty-odd requests
(one of 210k tokens), read 2,945 / 2,547 / 2,163 tok/s on the same three prompts - 1.7x, 2.6x and
3.2x slower than a fresh boot of the same binary, which restored it. Whatever grows over a long
epoch (lane state, allocator, cache residency) is worth more than every switch in this table, so
take a configuration measurement on a fresh boot and a fresh boot only.

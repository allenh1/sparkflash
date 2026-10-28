<!--
Copyright (c) 2026 Hunter L. Allen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Serving

`sparkflash serve` runs one process per DGX Spark. Rank 0 serves HTTP; ranks 1-3 only run the
model. [scripts/serve-four-node.sh](../scripts/serve-four-node.sh) builds and launches the four
command lines over ssh; this page describes what it sets up and what the server offers.

The lead node can do all of this from a browser: `scripts/setup.sh` builds the server and starts
[the console](web.md), which installs it on the other nodes, downloads the model once and copies
it over, works out the network settings, converts the weights and launches. The manual path below
remains the reference for what those buttons run.

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
| `POST /v1/chat/completions` | Streaming and buffered; tools and tool calls (DeepSeek V4.1 format, parsed into OpenAI `tool_calls`); `reasoning_effort` / thinking with the reasoning returned separately; images; `n`, `seed`, `stop`, `max_tokens`, `temperature`, `top_p`, `top_k`; `stream_options.include_usage` |
| `POST /v1/completions` | Text or token-id prompts, streaming and buffered |
| `POST /v1/responses` | OpenAI Responses API. **Not implemented yet:** refused with 501 (see below) |
| `POST /v1/messages` | Anthropic Messages API. **Not implemented yet:** refused with 501 (see below) |
| `GET /v1/models`, `/v1/models/{id}` | The served model is `deepseek-ai/DeepSeek-V4.1-Flash` |
| `POST /tokenize`, `/detokenize` | vLLM-compatible |
| `GET /health`, `/readyz` | Liveness (503, with the reason, once the device loop has failed or stalled), and readiness (also 503 while draining) |
| `GET /metrics` | Prometheus: requests running / waiting / rejected, prompt and generated tokens, time to first token, DSpark accepted/proposed counters and last-request decode/prefill/acceptance gauges |
| `GET /v1/cluster/stats` | JSON: per-node state, KV cache, speculation and request timing statistics |

Authentication is off by default; `--api-key KEY` (or `SPARKFLASH_SERVE_API_KEY`) requires a
bearer token on every route except health, readiness and metrics.

### Not supported yet

- **`/v1/responses` and `/v1/messages` still need implementing.** They do not tokenize their
  requests yet: their prompt is a row of placeholder token ids sized from the text, so the model
  would answer something unrelated to the request. Until they are implemented, both answer 501
  with an error that points to `/v1/chat/completions`. Implementing them means converting
  their requests (system prompt, messages, tool calls and tool results, reasoning) to the chat
  template and tokenizing them like `/v1/chat/completions`, returning tool calls as
  `tool_use` / `function_call` items, and adding streaming.
- `logprobs` / `top_logprobs`, and repetition, presence and frequency penalties: refused with an
  error rather than ignored.
- `response_format` `json_object` / `json_schema` (structured output needs constrained decoding,
  which the server does not have): refused.
- `ignore_eos` and `min_tokens`: ignored, like other unknown request fields; generation stops at
  the end-of-sequence token.

## Behaviour

- **Concurrency.** Up to 32 sequences run at once (`SPARKFLASH_LANES`); up to 64 more wait in a
  queue, after which requests are refused (503). Up to eight decoding sequences take one
  speculative step together; with more than eight decoding, every sequence gets one token per
  step and no speculation, so per-stream speed drops sharply past eight.
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
| `SPARKFLASH_PREFILL_EXACT=1` | Prefill every prompt on the exact one-row path: slower long prompts, output bit-identical to a single-stream reference |
| `SPARKFLASH_PREFILL_DECODER_SKIP=0` | Run every layer over every prompt row (slower long prompts; for comparison). By default a prompt's rows before its last 2,541 run only layers 0-20, which hold every cache the later layers read |
| `SPARKFLASH_PREFILL_MOE_MERGE=0` | Run each in-flight prefill chunk's experts separately (slower; for comparison) |
| `SPARKFLASH_WARMUP=0` | Skip the warm-up pass before HTTP binds (the first request then pays the one-time setup) |
| `NCCL_*` | Passed to every rank by the launcher |

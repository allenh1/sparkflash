# Serving

`sparkflash serve` runs one process per DGX Spark. Rank 0 serves HTTP; ranks 1-3 only run the
model. [scripts/serve-four-node.sh](../scripts/serve-four-node.sh) builds and launches the four
command lines over ssh; this page describes what it sets up and what the server offers.

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
export NCCL_IB_HCA=<your RoCE device> NCCL_SOCKET_IFNAME=<its netdev> NCCL_IB_GID_INDEX=3
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
| `POST /v1/responses` | OpenAI Responses API, buffered |
| `POST /v1/messages` | Anthropic Messages API, buffered |
| `GET /v1/models`, `/v1/models/{id}` | The served model is `deepseek-ai/DeepSeek-V4.1-Flash` |
| `POST /tokenize`, `/detokenize` | vLLM-compatible |
| `GET /health`, `/readyz` | Liveness (503, with the reason, once the device loop has failed or stalled), and readiness (also 503 while draining) |
| `GET /metrics` | Prometheus: requests running / waiting / rejected, prompt and generated tokens, time to first token |
| `GET /v1/cluster/stats` | JSON: per-node state, KV cache, speculation and request timing statistics |

Authentication is off by default; `--api-key KEY` (or `SPARKFLASH_SERVE_API_KEY`) requires a
bearer token on every route except health, readiness and metrics.

### Not supported yet

- `logprobs` / `top_logprobs`, and repetition, presence and frequency penalties: refused with an
  error rather than ignored.
- `response_format` `json_object` / `json_schema` (structured output needs constrained decoding,
  which the server does not have): refused.
- Streaming on `/v1/responses` and `/v1/messages`.
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
| `SPARKFLASH_PREFILL_MOE_MERGE=0` | Run each in-flight prefill chunk's experts separately (slower; for comparison) |
| `SPARKFLASH_WARMUP=0` | Skip the warm-up pass before HTTP binds (the first request then pays the one-time setup) |
| `NCCL_*` | Passed to every rank by the launcher |

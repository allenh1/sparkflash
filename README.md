# SparkFlash

**DeepSeek-V4.1-Flash on four NVIDIA DGX Sparks: 125 tok/s single-stream, 25 s boot, 1M context.**

SparkFlash is an inference server written from scratch in Rust and CUDA for exactly one job:
serving [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) on a
four-node DGX Spark (GB10) cluster. Every kernel is written for `sm_121a`, the weights are stored
in the layout those kernels read, and the whole serving path, from HTTP to the last collective, is
one native binary per node. It speaks the OpenAI Chat Completions API, so existing clients and
coding agents that use it work unchanged.

| Measured on 4x DGX Spark | |
| :-- | --: |
| Single stream, code | **125.5 tok/s** |
| Single stream, prose | **73.9 tok/s** |
| 4 concurrent streams, code | **51 tok/s each** |
| 8 concurrent agents, code | **30.5 tok/s each** (180 tok/s total) |
| 16 concurrent agents, code | **18.8 tok/s each** (200 tok/s total) |
| 200K-token prompt, cold | **57 s** to first token |
| Context per sequence | **1,048,576 tokens** |
| Server start to serving | **25 s** |

Details and methodology: [docs/performance.md](docs/performance.md).

## Quick start

You need four DGX Sparks on DGX OS (CUDA 13), linked by their ConnectX-7 ports (RoCE), with
passwordless ssh from the node you launch on, and about 560 GB free on each node's NVMe (476 GB
of checkpoint, 78 GB of converted weights).

**1. On every node:** clone the repository at the same path, and download the checkpoint to the
same path.

```bash
git clone <this repository> ~/sparkflash && cd ~/sparkflash
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /data/DeepSeek-V4.1-Flash
```

**2. On every node:** build the server and its kernels, and fetch NCCL (no root needed; it lands
in `target/deps/nccl`). Install Rust with [rustup](https://rustup.rs) first; the toolchain is pinned
by `rust-toolchain.toml`.

```bash
cargo build --release -p sparkflash-cli --features cuda,tp4-serve
scripts/build-cubins.sh target/cubins
scripts/fetch-nccl.sh
```

**3. On every node:** convert that node's quarter of the weights once (rank 0 on the first node,
1 on the second, and so on). This takes about five minutes and makes every later boot a ~25 s
read.

```bash
target/release/sparkflash convert --checkpoint /data/DeepSeek-V4.1-Flash \
    --output-dir /data/sparkflash-conv --tp-world-size 4 --tp-rank 0 --no-pad-moe-640 --no-verify
```

**4. From the first node:** launch all four ranks.

```bash
export SPARK_NODES="spark0 spark1 spark2 spark3"            # ssh names, rank order
export SPARK_PEERS="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4"   # RoCE addresses, rank order
export SPARKFLASH_CHECKPOINT=/data/DeepSeek-V4.1-Flash
export SPARKFLASH_TOKENIZER=/data/DeepSeek-V4.1-Flash/tokenizer.json
export SPARKFLASH_CONVERTED_DIR=/data/sparkflash-conv
export SPARKFLASH_CUBINS=$PWD/target/cubins
export SPARKFLASH_NCCL_LIB=$PWD/target/deps/nccl
export NCCL_IB_HCA=<both RoCE devices of the cabled port, from rdma link> NCCL_SOCKET_IFNAME=<one of their netdevs>
export NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=5 NCCL_DMABUF_ENABLE=1
scripts/serve-four-node.sh launch
```

**5. Use it.** Rank 0 serves on port 8000.

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "messages": [{"role": "user", "content": "Write a Python LRU cache."}],
  "stream": true
}'
```

`scripts/serve-four-node.sh stop` drains in-flight requests and shuts the group down. More on
flags, the API surface and tuning: [docs/serving.md](docs/serving.md).

## Or let the lead node do the setup

`sparkflash --web` turns the manual bring-up above into a dark control plane on the lead node. It
reads `sparkflash.toml`, finds the other Sparks over mDNS, checks passwordless SSH (and prints the
`ssh-copy-id` command when it fails), runs the one-time per-rank conversion, launches/stops the
four ranks, and graphs decode speed, prefill, DSpark acceptance, cache hit rate, tokens by day and
per-rank state. History is written to disk only while the cluster is idle, never during inference.

```bash
cp sparkflash.toml.example sparkflash.toml     # edit the four nodes and paths
target/release/sparkflash --web --config sparkflash.toml
# control plane on http://127.0.0.1:8790
```

The model (weights) and engine (how they run) are separate selectors, with V4.1-Flash the only
choice in each today. The full config, routes, discovery and telemetry design are in
[docs/web.md](docs/web.md).

## Why DGX Spark owners should care

A DGX Spark is not a small datacenter GPU. It has 128 GB of memory shared by the CPU and GPU, about
250 GB/s of bandwidth to it, and a 200 Gb/s NIC. General-purpose servers were designed for HBM
GPUs on NVLink and carry assumptions that do not hold here. SparkFlash is built around the
Spark's constraints instead:

- **Decode is bandwidth-bound, so every byte read counts.** Each decode step reads each weight
  once for all running sequences. The MoE kernels deduplicate experts across rows, and the next
  layer's weights are prefetched into L2 while the current layer finishes.
- **Speculation on the device, sized per step.** V4.1-Flash's DSpark head drafts five tokens;
  SparkFlash decides how many to verify each step from the drafts' confidence and a cost model of
  the hardware. Code runs at five to six tokens per step.
- **Tensor parallelism laid out for four nodes.** Every expert is split four ways, so no node
  waits on another's experts. Small exchanges use a one-shot device exchange instead of NCCL's
  ring.
- **Long prompts on tensor cores.** Prefill runs 4096-token chunks as FP8/FP4 tensor-core GEMMs,
  three chunks in flight so their collectives overlap compute, with one pass over the expert
  weights for all three.
- **Restarts are cheap.** Weights are converted once into each rank's final memory layout, so a
  boot is a parallel 78 GB read plus loading pre-built kernels: 25 s, with no JIT, no graph
  capture and no Python start-up.

The design is described in [docs/design.md](docs/design.md); the model itself in
[docs/architecture/](docs/architecture/).

## Compared with vLLM and SGLang

vLLM and SGLang are excellent general-purpose servers, and both run V4.1-Flash on Sparks through
community recipes. SparkFlash makes the opposite trade: one model and one cluster shape, in
exchange for start-up time, determinism and a small, self-contained serving stack.

| | SparkFlash | vLLM, community recipe | SGLang, MiaAI-Lab build |
| :-- | :-- | :-- | :-- |
| Single stream, code | **131.8 tok/s** | 76.0 tok/s | 124.8 tok/s |
| Single stream, prose | 86.2 tok/s | 62.4 tok/s | **87.7 tok/s** |
| 4 streams, code, each | 60.8 tok/s | not stated | **63.2 tok/s** |
| 8 streams, code, total | 270 tok/s | 264 tok/s | **310 tok/s** |
| 16 streams, code, total | 304 tok/s | not stated | **438 tok/s** |
| Prefill, ~32K-token prompt | 3,950 tok/s | **5,700 tok/s** | 4,800 tok/s |
| Start-up to serving | **25 s** | 2-7 minutes | 160-170 s |
| Greedy output at any concurrency | **identical** (decode always; prefill in exact mode) | can vary with the batch | identical run to run |
| Network, per node | one cable, both RoCE devices | not stated | one cable, both RoCE devices |
| In the serving path | one Rust binary per node | Python, PyTorch, JIT kernels | Python, PyTorch, JIT kernels |

The vLLM figures are self-reported by community members for four Sparks (TileLang-based recipe,
October 2026) in the NVIDIA forum's
[DeepSeek 4.1 Flash recipe thread](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438).
The SGLang figures are from the
[MiaAI-Lab TP4 notes](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/main/docs/tp4.md)
(September 2026; prefill on real text). SparkFlash's decode figures here were measured with the
same benchmark, sparkDash 1.8.8's DecodeBench prompts and protocol, so those two columns compare
like for like; the full grid is in [docs/performance.md](docs/performance.md). The vLLM column
used other prompts, and single-stream speculative decoding is very prompt-dependent, so treat it
as indicative.

What this means in practice:

- **Start-up and recovery:** a 25 s restart, no Python environment and no kernel compile cache
  make upgrades, crashes and experiments cheap.
- **Reproducibility:** greedy decode does not depend on what else is running, and with
  `SPARKFLASH_PREFILL_EXACT=1` neither does prefill, so an agent's behaviour can be replayed
  exactly and regressions bisected.
- **Network:** both use one cable per node and both of its RoCE devices, so the differences
  below are not the network.
- **Raw speed:** one stream is level with the MiaAI-Lab SGLang build (ahead on code and JSON,
  2% behind on prose). Up to eight streams it is close: 4% behind per stream at four code
  streams and 7% at eight. SparkFlash now speculates for up to 32 sequences, so sixteen code
  streams get 23 tok/s each against SGLang's 29, where they used to get ~12. SGLang still has
  more aggregate throughput at sixteen streams (1.3-1.8x), and both Python stacks prefill long
  prompts faster. Closing these gaps is the next work; see
  [docs/performance.md](docs/performance.md).

### API

One server answers OpenAI Chat Completions and Completions, with V4.1's tool calls parsed into
standard `tool_calls`, its reasoning returned separately, images, `n`, `seed` and `stop`, plus
vLLM-compatible `/tokenize`, `/detokenize`, `/metrics`, `/health` and `/readyz`.

Not supported yet: the OpenAI Responses and Anthropic Messages routes, which answer 501 until they
are implemented ([docs/serving.md](docs/serving.md#not-supported-yet) lists what that takes);
`logprobs`, repetition, presence and frequency penalties, and `response_format` JSON modes (these
return an error rather than being ignored); and `ignore_eos` / `min_tokens`, which are ignored
(generation stops at the end-of-sequence token).

## Building and testing without a cluster

Most of the workspace (API, tokenizer and chat template, scheduler, KV planning, Engram hashing,
reference kernels) is testable on any Linux machine, with no GPU and no weights:

```bash
cargo test --workspace
python3 -m unittest discover -s scripts/tests
```

The device path builds with `--features cuda,tp4-serve` and needs CUDA 13 and `sm_121a` to run.

## Repository layout

```
crates/
  sparkflash-cli/        the `sparkflash` binary: serve, convert, models, the TP=4 device runtime
  sparkflash-api/        OpenAI / Anthropic HTTP + SSE server, tool and reasoning parsing
  sparkflash-model/      model descriptors, registry, engine selectors and PromptCodec contract
  sparkflash-model-deepseek-v41/  the V4.1 model pack: config, plans, host forward, encoding, loader
  sparkflash-weights/    generic storage formats, safetensors container, weight sources, host arithmetic
  sparkflash-kernels/    sm_121a CUDA kernels' Rust bindings and CPU reference kernels
  sparkflash-device/     device arena, converted-weight loader, device sessions
  sparkflash-comm/       NCCL and one-shot RDMA collectives
  sparkflash-engine/     compatibility facade re-exporting the V4.1 host pack
  sparkflash-engram/     Engram n-gram hashing and table storage
  sparkflash-kv/         KV cache planning
  sparkflash-scheduler/  batching and speculation scheduling
  sparkflash-serve/      model-agnostic host driver and request lifecycle
  sparkflash-vision/     image decoding and resizing for the vision path
  sparkflash-trace/      CPU and GPU trace points
  sparkflash-web/        `sparkflash --web`: TOML control plane, mDNS/SSH discovery, jobs, graphs
kernels/                 CUDA sources (compiled to cubins by scripts/build-cubins.sh)
scripts/                 launch, build and verification tooling
docs/                    design, performance, serving and model architecture
reference/               upstream model card, tokenizer and encoding reference
```

## Status

SparkFlash serves one model, DeepSeek-V4.1-Flash, on one topology: four DGX Sparks at tensor
parallelism 4. Other models, node counts and GPUs are not supported yet. The model identity and
selector boundary now lives in `sparkflash-model`, and the V4.1 host model — plans, reference
forward, checkpoint loader, sequence fixture and prompt encoding — lives in
`sparkflash-model-deepseek-v41`; `sparkflash-engine` is a compatibility facade. The generic
storage container, weight sources and host arithmetic are in `sparkflash-weights`, the HTTP
layer holds `Arc<dyn PromptCodec>`, and a toy dense model in `sparkflash-model`'s tests keeps
the contracts from assuming V4.1 geometry. Shared telemetry and the vision front end are model
contracts too, so `sparkflash-api`'s normal dependency tree no longer contains the V4.1 pack,
Engram, KV or the vision crate. `sparkflash-model::backend` names the host execution seam, and
`sparkflash-serve` now provides a generic `HostDriver` over it; its toy test serves a chat
completion through the real router without any V4.1 type. The V4.1 reference driver still lives
in the CLI until the remaining Phase 3 move. The path to a second model is
[docs/multi-model-plan.md](docs/multi-model-plan.md).

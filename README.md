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

### What you need

- **Four DGX Sparks** running DGX OS, on the same network as the computer you set them up from,
  with the same user account name on each (the account made at first boot is fine).
- **One QSFP cable per Spark** from its ConnectX-7 to a 200 GbE switch. Either port works; a
  second cable adds nothing (see [the fast network](#the-fast-network)).
- **About 600 GB free on each Spark**: the model is 510 GB, and each Spark's prepared quarter of
  it is another 78 GB.
- **Internet on the first Spark**, which downloads the model once and copies it to the others.

Nothing needs installing by hand. DGX OS already ships the CUDA toolkit, the compilers, `curl`,
`rsync`, `python3` and `ssh`. The one missing piece, Rust, is installed by the setup script into
your home directory, without an admin password. The other three Sparks need nothing at all: the
first one installs everything on them.

### 1. Run the setup script on one Spark

Pick any Spark to be the lead, open a terminal on it, and run:

```bash
git clone <repository URL> ~/sparkflash
```

```bash
~/sparkflash/scripts/setup.sh
```

The script checks the Spark, installs Rust, and builds SparkFlash (several minutes the first time).
It then lists the computers it finds on your network and asks which three are your other Sparks.
For each one it asks you to type `yes` to trust it and your password there, once, so this Spark
can reach them from then on. Last, it starts the SparkFlash console in the background and prints
its address, for example:

```text
Open the SparkFlash console in a browser on any computer on this network:

    http://node0.local:8790
```

Running the script again is safe: it skips whatever is already done.

### 2. Follow the Setup page

Open that address in a browser. The **Setup** page walks through seven steps, one button each,
and checks every Spark after each one, so it always shows what is done and what is next.

| Step | What happens |
| :-- | :-- |
| 1. Find your Sparks | Pick your other three Sparks from the ones on the network. If this Spark can't sign in to one yet, the page shows the one command that fixes it. |
| 2. Connect the fast network | Checks the cables and works out the dual-rail NCCL settings from what each Spark reports. If a port still needs an address or a larger MTU, it shows the commands to paste on that Spark. |
| 3. Install on the other Sparks | Copies the server, its GPU kernels and NCCL from this Spark. |
| 4. Download the model | Fetches [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (510 GB) onto this Spark, with progress. An interrupted download resumes where it stopped. |
| 5. Copy the model to the other Sparks | Copies it over the fast network to all three at once. |
| 6. Prepare the weights | Each Spark converts its quarter of the model once, in parallel (about five minutes). This is what makes every later start take 25 s. |
| 7. Start the server | Starts all four Sparks and shows the API address. |

Steps that don't depend on each other are open at the same time, so the software installs while
the model downloads. Everything runs on the Sparks: you can close the browser and come back to
the same address to check progress.

The DeepSeek checkpoint needs no Hugging Face account. For a model that does, save your token in
`~/.cache/huggingface/token` on the lead Spark before step 4.

### 3. Use it

The lead Spark serves the OpenAI Chat Completions API on port 8000; step 7 shows the exact
address. Point any OpenAI client or coding agent at it, or try:

```bash
curl -s http://node0.local:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "messages": [{"role": "user", "content": "Write a Python LRU cache."}],
  "stream": true
}'
```

Start and stop the server from the console's **Overview** page, which also graphs decode speed,
prefill, draft acceptance, cache hits and tokens served. The console keeps running in the
background; `~/sparkflash/scripts/setup.sh --stop` stops it, and running the script again starts
it. More on the console: [docs/web.md](docs/web.md). More on the API and tuning:
[docs/serving.md](docs/serving.md).

### The fast network

The four Sparks run one model together, so every token crosses the fast network several times.
A Spark's ConnectX-7 reaches the GB10 over two PCIe Gen5 x4 links of about 100 Gb/s each, so one
cabled port shows up as **two** RoCE devices, each with its own network interface. NCCL gets the
port's full 200 Gb/s only when it is told to use both (dual-rail). A cable in the second port adds
no bandwidth, because both ports share the same two links.

Step 2 reads every Spark's RoCE devices and works out the settings for you:

| Setting | What step 2 sets it to |
| :-- | :-- |
| `NCCL_IB_HCA` | both RoCE devices of the cabled port, for example `rocep1s0f1,roceP2p1s0f1` |
| `NCCL_SOCKET_IFNAME` | the interface each Spark routes the fast subnet through, for example `enP2p1s0f1np1` |
| `NCCL_IB_GID_INDEX` | the GID slot holding the ports' RoCE v2 IPv4 addresses (3 on DGX OS) |
| each Spark's peer address | that Spark's address on the same interface |

It also checks that every port links at 200 Gb/s, uses MTU 9000, and, when a Spark's two
interfaces share a subnet, that each answers ARP only for its own address (`arp_ignore=1`,
`arp_announce=2`). The console never changes a Spark's network itself. When something needs
fixing, it shows a short script for that Spark, to paste into a terminal there, which writes
`/etc/netplan/60-sparkflash-fabric.yaml` (and `/etc/sysctl.d/60-sparkflash-fabric.conf` when ARP
needs it) and applies it:

```bash
sudo tee /etc/netplan/60-sparkflash-fabric.yaml > /dev/null <<'EOF'
network:
  version: 2
  ethernets:
    enP2p1s0f1np1:
      mtu: 9000
      addresses: [192.0.2.21/24]
    enp1s0f1np1:
      mtu: 9000
      addresses: [192.0.2.11/24]
EOF
sudo chmod 600 /etc/netplan/60-sparkflash-fabric.yaml
sudo tee /etc/sysctl.d/60-sparkflash-fabric.conf > /dev/null <<'EOF'
net.ipv4.conf.all.arp_ignore = 1
net.ipv4.conf.all.arp_announce = 2
EOF
sudo sysctl -p /etc/sysctl.d/60-sparkflash-fabric.conf
sudo netplan apply
```

New addresses join the subnet the other Sparks' fast ports already use, or `10.200.0.0/24` when
none has one yet. After pasting, press **Check again**, then **Use these settings**, which saves
the NCCL values and peer addresses into `sparkflash.toml`.

### Manual setup

The console runs the same commands you can run by hand, which is useful for scripting or for a
cluster managed some other way. On every node, at the same paths:

```bash
git clone <repository URL> ~/sparkflash && cd ~/sparkflash
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /data/DeepSeek-V4.1-Flash
cargo build --release -p sparkflash-cli --features cuda,tp4-serve   # Rust from https://rustup.rs
scripts/build-cubins.sh target/cubins                               # nvcc is in /usr/local/cuda/bin
scripts/fetch-nccl.sh                                               # NCCL into target/deps/nccl
target/release/sparkflash convert --checkpoint /data/DeepSeek-V4.1-Flash \
    --output-dir /data/sparkflash-conv --tp-world-size 4 --tp-rank 0 --no-pad-moe-640 --no-verify
```

Use `--tp-rank 0` on the first node, 1 on the second, and so on. Then, from the first node, with
passwordless SSH to the others:

```bash
export SPARK_NODES="node0 node1 node2 node3"                      # ssh names, rank order
export SPARK_PEERS="192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14"  # fast-network addresses, rank order
export SPARKFLASH_CHECKPOINT=/data/DeepSeek-V4.1-Flash
export SPARKFLASH_TOKENIZER=/data/DeepSeek-V4.1-Flash/tokenizer.json
export SPARKFLASH_CONVERTED_DIR=/data/sparkflash-conv
export SPARKFLASH_CUBINS=$PWD/target/cubins
export SPARKFLASH_NCCL_LIB=$PWD/target/deps/nccl
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_GID_INDEX=3
scripts/serve-four-node.sh launch
```

`scripts/serve-four-node.sh stop` drains in-flight requests and shuts the group down. The
details are in [docs/serving.md](docs/serving.md).

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
  weights for all three. Layers 21-39 read only caches that layer 20 owns, so they run on a
  prompt's last 2,541 rows alone.
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
| Prefill, ~32K-token prompt | 6,400 tok/s | **6,800 tok/s** | 4,800 tok/s |
| Start-up to serving | **25 s** | 2-7 minutes | 160-170 s |
| Greedy output at any concurrency | **identical** (decode always; prefill in exact mode) | can vary with the batch | identical run to run |
| Network, per node | one cable, both RoCE devices | not stated | one cable, both RoCE devices |
| In the serving path | one Rust binary per node | Python, PyTorch, JIT kernels | Python, PyTorch, JIT kernels |

The vLLM figures are self-reported by community members for four Sparks (TileLang-based recipe,
October 2026; the prefill figure is the recipe's October 25 build on source text) in the NVIDIA
forum's
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
  more aggregate throughput at sixteen streams (1.3-1.8x). Long-prompt prefill is now ahead of
  SGLang and about 6% behind the latest vLLM recipe. Closing these gaps is the next work; see
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
  sparkflash-web/        `sparkflash --web`: guided setup, mDNS/SSH discovery, node checks, jobs, graphs
kernels/                 CUDA sources (compiled to cubins by scripts/build-cubins.sh)
scripts/                 launch, build and verification tooling
docs/                    design, performance, serving and model architecture
reference/               upstream model card, tokenizer and encoding reference
```

## Contributing

Contributions are welcome under the Apache-2.0 license. Start with
[CONTRIBUTING.md](CONTRIBUTING.md): it states the commit sign-off rule, the CI gates, and the
evidence a new model must bring (tested configuration, clean and measured kernels, benchmark
protocol, weights revision, paper or reference implementation, licensing, and supported
platforms).

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

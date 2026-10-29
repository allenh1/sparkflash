# SparkFlash

**DeepSeek-V4.1-Flash on four NVIDIA DGX Sparks: 132 tok/s single-stream, 25 s boot, 1M context.**

> **Setting up with an AI agent?** Send your agent to [docs/agent-setup.md](docs/agent-setup.md) to set up and run this repo.

SparkFlash is a native Rust and CUDA inference server purpose-built for serving
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) on a four-node DGX
Spark (GB10) cluster, building on architectural concepts from
[`b12x`](https://github.com/local-inference-lab/b12x),
[NVIDIA Flare (`flare-local-inference`)](https://github.com/NVIDIA/flare-local-inference),
[`dgx-spark-networking`](https://github.com/christopherowen/dgx-spark-networking), and the DGX
Spark community (see [Acknowledgments & Prior Art](#acknowledgments--prior-art) and
[NOTICE](NOTICE)). Every kernel is targeted to `sm_121a`, the weights are stored in the layout
those kernels read, and the whole serving path, from HTTP to the last collective, is one native
binary per node. It speaks the OpenAI Chat Completions API, so existing clients and coding agents
that use it work unchanged.

## Benchmarks

Four DGX Sparks at tensor parallelism 4, one network cable per node, default clocks, greedy,
client-side over HTTP.

| Benchmark | Result |
| :-- | --: |
| Single stream, code | **131.8 tok/s** |
| Single stream, prose | **86.2 tok/s** |
| 4 streams, code, each | **60.8 tok/s** |
| 8 streams, code, total | **270 tok/s** |
| 16 streams, code, total | **304 tok/s** |
| Prefill, ~32K-token prompt | **6,400 tok/s** |
| Start-up to serving | **25 s** |
| Greedy output at any concurrency | decode identical; prefill with `SPARKFLASH_PREFILL_EXACT=1` |
| Network, per node | one cable, both RoCE devices |
| In the serving path | one Rust binary per node |

Decode: sparkDash 1.8.8 DecodeBench prompts, 256 tokens, temperature 0, idle server. Full results:
[docs/performance.md](docs/performance.md).


## Quick start

### What you need

- **Four DGX Sparks** running DGX OS, on the same network as the computer you set them up from,
  with the same user account name on each (the account made at first boot is fine).
- **One QSFP cable per Spark** from its ConnectX-7 to a 200 GbE switch. Either port works (see [the fast network](#the-fast-network)).
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
git clone https://github.com/allenh1/sparkflash ~/sparkflash
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

| Step                                  | What happens                                                                                                                                                                              |
|:--------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Find your Sparks                   | Pick your other three Sparks from the ones on the network. If this Spark can't sign in to one yet, the page shows the one command that fixes it.                                          |
| 2. Connect the fast network           | Checks the cables and works out the dual-rail NCCL settings from what each Spark reports. If a port still needs an address or a larger MTU, it shows the commands to paste on that Spark. |
| 3. Install on the other Sparks        | Copies the server, its GPU kernels and NCCL from this Spark.                                                                                                                              |
| 4. Download the model                 | Fetches [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (510 GB) onto this Spark, with progress. An interrupted download resumes where it stopped.          |
| 5. Copy the model to the other Sparks | Copies it over the fast network to all three at once.                                                                                                                                     |
| 6. Prepare the weights                | Each Spark converts its quarter of the model once, in parallel (about five minutes). This is what makes every later start take 25 s.                                                      |
| 7. Start the server                   | Starts all four Sparks and shows the API address.                                                                                                                                         |

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
git clone https://github.com/allenh1/sparkflash ~/sparkflash && cd ~/sparkflash
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
  following the online confidence + step-cost approach pioneered in
  [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference), SparkFlash decides how many
  drafts to verify each step from the drafts' calibrated confidence and a cost model of the
  hardware. Code runs at five to six tokens per step.
- **Tensor parallelism laid out for four nodes.** Every expert is split four ways, so no node
  waits on another's experts. Small exchanges and the one-shot RDMA collective path build on the
  `RoCEnante` dual-rail architecture from [`b12x`](https://github.com/local-inference-lab/b12x) and
  [`dgx-spark-networking`](https://github.com/christopherowen/dgx-spark-networking).
- **Long prompts on tensor cores.** Prefill runs 4096-token chunks as FP8/FP4 tensor-core GEMMs,
  three chunks in flight so their collectives overlap compute, with one pass over the expert
  weights for all three. Layers 21-39 read only caches that layer 20 owns, so they run on a
  prompt's last 2,541 rows alone.
- **Restarts are cheap.** Inspired by [`b12x`](https://github.com/local-inference-lab/b12x) and
  [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference), weights are converted once into
  each rank's final memory layout, so a boot is a parallel 78 GB `O_DIRECT` read plus loading
  pre-built kernels: 25 s, with no JIT, no graph capture and no Python start-up.

The design is described in [docs/design.md](docs/design.md); the model itself in
[docs/architecture/](docs/architecture/).


### API

One server answers OpenAI Chat Completions (`/v1/chat/completions`), Completions (`/v1/completions`),
Responses (`/v1/responses`) and Anthropic Messages (`/v1/messages`), with V4.1's tool calls parsed
into standard tool-call items, its reasoning returned separately, images on every conversation
route, `n`, `seed` and `stop`, plus `/tokenize`, `/detokenize`, `/metrics`, `/health` and `/readyz`.

Not supported yet: `logprobs`, repetition, presence and frequency penalties, `response_format` JSON
modes, and Responses/Anthropic shapes the V4.1 chat template cannot represent (these return an error
rather than being ignored); and `ignore_eos` / `min_tokens`, which are ignored (generation stops at
the end-of-sequence token). Full route and limitation details: [docs/serving.md](docs/serving.md).

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
LICENSE                  Apache License, Version 2.0
NOTICE                   prior-art, architectural provenance and third-party attributions
```

## Acknowledgments & Prior Art

SparkFlash's Rust runtime and CUDA C++ (`.cu` / `.cuh`) kernels draw heavily from the architectural concepts introduced by existing open source works. See [NOTICE](NOTICE) for full provenance details:

- **[b12x](https://github.com/local-inference-lab/b12x)** (Local Inference Lab — **Luke Alonso ([@lukealonso](https://github.com/lukealonso) / `@lukea`)**, **[@original-el8](https://github.com/original-el8) (`@el8`)**, and contributors; Apache-2.0, now integrated into [FlashInfer `flashinfer/experimental/b12x`](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/experimental/b12x)):
  - **`RoCEnante`** (`b12x.comm.roce`, documented in [`docs/rocenante.md`](https://github.com/local-inference-lab/b12x/blob/2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88/docs/rocenante.md)) is the name and one-shot RoCEv2 collective architecture created by the `b12x` authors for multi-node DGX Spark tensor parallelism. SparkFlash's one-shot RDMA collective transport (`crates/sparkflash-comm`, `kernels/rocenante_oneshot_sm121.{cu,cuh}`, and `crates/sparkflash-kernels/src/roce_*.rs`) adopts the `RoCEnante` name and design: ping-pong pinned unified-memory slots registered across dual ConnectX-7 RoCEv2 rails, GPU-side staging with a system-scope fence, a dedicated CPU proxy polling a sequence doorbell and striping RDMA writes across rails with per-rail arrival flags, and deterministic canonical rank-order (`0 + 1 + 2 + 3`) f32 reduction with a single round-to-nearest-even bf16 store.
  - The **`b12x_*`** kernel naming prefix (`kernels/b12x_*.cu`) and the **`spark_b12x_*`** `32x32` weight micro-tile / `UE8M0` block-scale indexing helpers (`kernels/m4_common_sm121.cuh` and `crates/sparkflash-kernels`) are named after and inspired by `b12x`'s `SM120`/`SM121` block-scaled MXFP8/MXFP4/NVFP4 kernel and direct-I/O loader architecture.
- **[NVIDIA Flare (`flare-local-inference`)](https://github.com/NVIDIA/flare-local-inference)** (NVIDIA Corporation, **Luke Alonso ([@lukealonso](https://github.com/lukealonso))**, **[@original-el8](https://github.com/original-el8)**, **[@voipmonitor](https://github.com/voipmonitor)**, **[@eugr](https://github.com/eugr)**, and contributors; Apache-2.0):
  - Pioneered native Rust local inference serving (`Helios` runtime and `Ignition` compiler, built on `b12x` and FlashInfer) on `SM120`/`SM121` and multi-node DGX Spark hardware. SparkFlash draws architectural inspiration from Flare across its Rust integration of `RoCEnante` dual-rail one-shot RDMA collectives (`helios/src/execution/roce.rs`), bounded `O_DIRECT` / `io_uring` asynchronous NVMe row gathering for out-of-core Engram/PLE tables (`helios/src/engram/reader.rs`), pre-converted weight artifacts for fast cold boot, stateless seeded Gumbel-max sampling for draft/target agreement, and online calibrated acceptance profiles paired with a step cost model for adaptive speculative verification depth (`helios/src/scheduler/speculation.rs`, `cost.rs`, `learning.rs`).
- **[dgx-spark-networking](https://github.com/christopherowen/dgx-spark-networking)** (**Christopher Owen ([@christopherowen](https://github.com/christopherowen))**, [NVIDIA Forum #385049](https://forums.developer.nvidia.com/t/dgx-spark-networking-open-source-switchless-roce-collectives-nccl-profiles-and-fabric-tooling-for-multi-spark-inference/385049); Apache-2.0):
  - Packaged `b12x`'s `RoCEnante` one-shot collectives (`sparknet.oneshot`) with dual-backend payload-size dispatch (routing decode-sized payloads `<= 2 MiB` to `RoCEnante` one-shot RDMA and larger prefill payloads `> 2 MiB` to dual-rail NCCL), dual-rail ConnectX-7 discovery on GB10, fabric verification / collective doctor tooling, and documentation of the AArch64 NCCL send-path memory-ordering fix ([`NVIDIA/nccl#2393`](https://github.com/NVIDIA/nccl/pull/2393) by Stanislav Bardyuk).
- **[MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)** (MIT) and **[knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)** ([NVIDIA Forum #382897](https://forums.developer.nvidia.com/t/deepseek-v4-1-flash-552b-moe-on-4x-dgx-spark-tp4-77-2-tok-s-c1-on-peak-52-code-72-tok-s-on-a-warm-code-run-47-math-39-reasoning-23-prose/382897)):
  - Established the 4x DGX Spark TP=4 unified-memory budget analysis (showing why host-RAM CPU offload of the ~189 GiB Engram tables exhausts GB10's 128 GB unified memory pool), dual-rail RoCE fabric configurations, and the SGLang baseline measurements cited in [docs/performance.md](docs/performance.md).
- **[blaesus/dsgx10 (`spark-ds41f`)](https://github.com/blaesus/dsgx10)** ([NVIDIA Forum #385803](https://forums.developer.nvidia.com/t/squeezing-5-70-tok-s-over-sota-by-building-from-scratch-ds4-1flash-native-precision-on-4xgb10-switchless/385803)):
  - Concurrent community exploration of native C++/CUDA + Rust serving for DeepSeek-V4.1-Flash on 4x GB10 (`sm_121a`), DSpark multi-token drafting, and switchless ConnectX-7 ring topologies.
- **[DeepSeek AI](https://github.com/deepseek-ai)** ([`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [`deepseek-ai/deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe); MIT):
  - DeepSeek-V4.1-Flash model architecture, technical report (`arXiv:2609.19969`), Engram conditional memory paper (`arXiv:2601.07372`), upstream reference files under [`reference/`](reference/), and the `deepseek-recipe` Rust encoding, tokenization, and image preprocessing crates.

## Contributing

Contributions are welcome under the [Apache-2.0 license](LICENSE). Start with
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

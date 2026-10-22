# SparkFlash-RS

A purpose-built Rust inference server for **DeepSeek-V4.1-Flash** on a 4x NVIDIA DGX Spark
(GB10 Grace Blackwell, `sm_121a`) cluster.

**Current state:** TP=4 host path at **179.81 ms/step** (four-rank lean decode, 2026-10-22,
`cd233423`). The active track is the whole-layer device port on `win/layer-device`; no new speed
number is published until one full layer runs device-resident.

## Start here

| Document | What it is |
| :-- | :-- |
| [docs/RULES.md](docs/RULES.md) | The one rule set: metric, gates, measurement protocol, review, docs. Read this before doing work. |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Stable decision register (D1-D54), one row per decision. |
| [docs/architecture/](docs/architecture/) | Reference-model analysis: attention, mHC, MoE, Engram, DSpark, vision, encoding, KV footprint. |
| [docs/engine/01-forward-order.md](docs/engine/01-forward-order.md) | The reference forward, in execution order, with the tensor map. |
| [docs/build/](docs/build/) | Current engineering handbooks (device path, performance, kernels, serving, fabric, build/deploy, operations, evidence). |
| [docs/archive/investigations/](docs/archive/investigations/) | Deep dives on DeepInfer and `deepseek-recipe` that informed the decisions (archived, not maintained). |
| [reference/](reference/) | Upstream model card and encoding reference. |
| [deepseek_v41_flash_4x_dgx_spark_rust_spec.md](deepseek_v41_flash_4x_dgx_spark_rust_spec.md) | Source specification. |

Historical build records are in [docs/archive/](docs/archive/); they are provenance, not current
documentation.

## Design rules

1. **Every subsystem is testable without the 552B checkpoint.** Weight-independent logic (config
   parsing, TP sharding math, Engram hashing, KV/radix/SWA planning, DSpark confidence scheduling,
   prompt encoding, protocol conversion, collective routing) lives in pure functions with unit and
   property tests. Real-tensor paths use tiny synthetic fixtures.
2. **Hardware is optional at build time.** CUDA kernels compile only when `nvcc` is present
   (`sparkflash-kernels/build.rs`); every kernel has a CPU reference exercised on any machine, and an
   on-device parity test where a GPU exists.
3. **Determinism is a feature.** Collective reductions use a canonical rank order, so all four ranks
   produce bit-identical outputs for every finite value and both infinities. NaN payload/sign are not
   claimed (see [docs/DECISIONS.md](docs/DECISIONS.md) D21 and `sparkflash-comm`'s crate docs).

## Workspace layout

```
crates/
  sparkflash-trace/      zero-overhead scoped CPU (CNTVCT_EL0) + in-graph GPU (%globaltimer) tracepoints
  sparkflash-api/        OpenAI/Anthropic-compatible HTTP+SSE server, deepseek-recipe protocol layer
  sparkflash-scheduler/  continuous batcher, CED prefill bypass, DSpark confidence scheduler
  sparkflash-kv/         3-tier radix prefix cache, CSA2 page tables, SWA Bounded Replay planner
  sparkflash-engram/     tokenizer compression, n-gram XOR hashing, DRM/UMA tiered cache, io_uring NVMe store
  sparkflash-comm/       RoCEnante one-shot RDMA + NCCL dual-path collective transport
  sparkflash-kernels/    sm_121a CUDA kernels + CPU reference kernels
  sparkflash-engine/     reference engine, forward order, tensor plan, fixtures
  sparkflash-device/     device-resident layers, KV store, paged read, sessions
  sparkflash-cli/        `sparkflash` CLI: serve, doctor, TP=4 runner, windows
kernels/                 CUDA/PTX sources compiled by the workspace build script
scripts/                 window, gate, evidence and deployment tooling
```

## Build and test

```bash
cargo test --workspace              # no GPU, no weights required
cargo build --workspace --release
```

With a CUDA toolkit and an `sm_121a` device:

```bash
cargo test --workspace --features sparkflash-kernels/cuda
```

The four-rank build, deployment and window procedures are in
[docs/build/build-deploy.md](docs/build/build-deploy.md) and
[docs/build/operations.md](docs/build/operations.md).

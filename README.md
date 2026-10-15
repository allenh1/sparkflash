# SparkFlash-RS

A purpose-built Rust inference server for **DeepSeek-V4.1-Flash** on a 4x NVIDIA DGX Spark
(GB10 Grace Blackwell, `sm_121a`) cluster.

* Source specification: [`deepseek_v41_flash_4x_dgx_spark_rust_spec.md`](deepseek_v41_flash_4x_dgx_spark_rust_spec.md)
* Architecture notes (derived from the official reference implementation and the
  DeepSeek-V4.1 technical report): [`docs/architecture/`](docs/architecture/)

## Design rules

1. **Every subsystem is testable without the 552B checkpoint.** Weight-independent logic
   (config parsing, TP sharding math, Engram hashing, KV/radix/SWA planning, DSpark
   confidence scheduling, prompt encoding, protocol conversion, collective routing) lives in
   pure functions with unit + property tests. Anything that needs real tensors is fed from
   tiny synthetic fixtures generated in-test.
2. **Hardware is optional at build time.** CUDA kernels compile only when `nvcc` is present
   (`sparkflash-kernels/build.rs`); every kernel has a pure-Rust CPU reference implementation
   that the test-suite exercises on any machine, and an on-device parity test that runs where a
   GPU exists.
3. **Determinism is a feature.** Collective reductions use a canonical rank order so all four
   ranks produce bit-identical outputs.

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
kernels/                 CUDA/PTX sources compiled by the workspace build script
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

# Build, boot and deployment

Current-state handbook for getting a SparkFlash binary onto the four-node DGX Spark cluster and
running it as a service. Commands and contracts only; measurement policy is in `docs/RULES.md`.

## 1. Cluster and build host

| item | value | label |
| :-- | :-- | :-- |
| nodes | 4 x NVIDIA GB10 Grace (aarch64), rank order `192.0.2.80, .88, .59, .68` | MEASURED (deployment config; `serve-deployment.md`) |
| head node | `node0` = `192.0.2.80`, rail-2 address `.85` | MEASURED 2026-10-18 |
| dev/build host | x86_64, 251 GB RAM, 2.7 TB free disk; no root, no apt, no rustup initially | MEASURED 2026-10-16 |
| head-node memory | 124,610 MB total / 3,115 MB available while vLLM resident | MEASURED 2026-10-16 |
| driver / kernel | 580.173.02 / 6.17.0-1029-nvidia; CUDA 13.0; glibc 2.39 | MEASURED 2026-10-16 |
| rails | active f1 siblings at 200 Gb/s; MTU 9000 applied on all four nodes | MEASURED; MTU state updated 2026-10-22 |

The build host is the only place a Rust binary is compiled. The head node has no cargo/rustc and its
available memory is smaller than a workspace build peak, so compiling there endangers the inference
it serves. Plan against the host's ~3.1 GB available memory, not the container cgroup reading.

## 2. aarch64 toolchain and build

Pins are deliberate and checked on every run; a moved pin is a hard error.

| pin | value |
| :-- | :-- |
| Rust | `1.98.1`, user-local rustup, minimal profile |
| target | `aarch64-unknown-linux-gnu` (std added for that exact toolchain) |
| Zig | `0.17.0`, tarball sha256 `1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026` |
| toolchain root | `$HOME/opt`; override with `SPARKFLASH_TOOLCHAIN_DIR` |
| target dir | repository `target/`; override with `CARGO_TARGET_DIR` |

Zig is the cross C/C++ compiler, linker and archiver because it carries libc, libc++ and lld and needs
no sysroot package. Three translations are mandatory and live in the wrapper scripts: drop cargo's
`--target=<triple>` and pass `-target aarch64-linux-gnu`; map `-lstdc++` to `-lc++`; drop
`-Wl,--fix-cortex-a53-843419` (the GB10 is Neoverse-class, not Cortex-A53).

```sh
scripts/build-aarch64.sh                       # release sparkflash (sparkflash-cli)
scripts/build-aarch64.sh --workspace           # all workspace binaries
scripts/build-aarch64.sh --features=cuda       # CUDA-gated device arms
scripts/build-aarch64.sh --example=m7_layer0   # one kernels example
scripts/build-aarch64.sh --print-env           # resolved toolchain, then exit
```

Output: `<target-dir>/aarch64-unknown-linux-gnu/release/sparkflash`. The plain CLI artifact measured
39,565,400 bytes, sha256 `aee19f62...` (MEASURED 2026-10-16). The CUDA-feature binary at main
`0639413` measured 61,758,952 bytes and shipped with 19 cubins (16 kernels + 3 Engram) (MEASURED
2026-10-19). The current baseline is **27 kernels** (2026-10-22); the 19-cubin figure is the older
artifact, not the current pin set.

CUDA device code is compiled on the x86_64 host: `nvcc -arch=sm_121a` emits the same device binary
an aarch64 `nvcc` would, and a GPU is not needed to compile. The runtime driver is `dlopen`ed on the
node. Build invariant: keep cudarc on `fallback-dynamic-loading`; enabling `dynamic-linking` or
`static-linking` puts an x86_64-shaped link line under an aarch64 target.

Ops constraint: mount a worktree under `/home` when running the pinned CUDA image, because the docker
daemon cannot see an agent's private `/tmp` (empty root-owned mount). The pinned kernel-baseline
check is:

```sh
docker run --rm -v <worktree-under-/home>:/w -w /w nvidia/cuda:13.0.3-devel-ubuntu22.04 \
  sh -c 'SPARKFLASH_NVCC=/usr/local/cuda/bin/nvcc sh crates/sparkflash-kernels/scripts/check-kernel-baseline.sh'
```

Success line names toolkit `V13.0.88`. The Rust wrapper skips without `nvcc`, so a plain host run
is not evidence. Use a per-worktree `CARGO_TARGET_DIR`; a shared target dir can serve a binary or
kernel set from a different tree and produce a false green.

## 3. Boot conversion: pre-converted per-rank files

The boot path loads one contiguous per-rank file already in device layout, instead of walking
safetensors tensors one at a time. Conversion is offline; boot is sequential I/O into one slab.

```sh
sparkflash convert --checkpoint /ckpt/released --output-dir /nvme/converted --tp-world-size 4
sparkflash convert --inspect /nvme/converted/rank-0-of-4.bin
SPARKFLASH_CONVERTED_DIR=/nvme/converted sparkflash device-chain --upload ...
```

`convert` also accepts `--tp-rank R`, `--ranks 0,2`, `--no-pad-moe-640`, `--no-verify`. The
loader checks `SPARKFLASH_CONVERTED_DIR`, then beside the checkpoint; missing rank files fall back
to safetensors streaming with no behaviour change. `--verify-weights` re-digests every tensor;
`--mmap` selects `MmapRegistered` for a warm page-cache restart.

Converted file contract: `rank-{rank}-of-{world}.bin`, little-endian.

| offset | bytes | content |
| --: | --: | :-- |
| 0 | 8 | magic `SPKFRANK` |
| 8 | 4 | version `1` |
| 12 | 4 | rank |
| 16 | 4 | world |
| 20 | 4 | reserved |
| 24 | 8 | `index_len` |
| 32 | 8 | `data_offset` (4096-aligned) |
| 40 | 8 | `data_bytes` |
| 48 | 8 | `padded_data_bytes` (4096-aligned) |
| 56 | 8 | `header_hash` = FNV-1a 64 of the index bytes |
| 64 | `index_len` | JSON index |
| `data_offset` | `padded_data_bytes` | arena payload |

The JSON index (`ConvertedRankHeader`) carries `plan_hash`, `stored_bytes`, `payload_bytes`,
`allocated_bytes`, `file_digest`, one `ConvertedGroupEntry` per resident group and one
`ConvertedTensorEntry` per tensor (`name`, `group`, `slab_offset`, `group_offset`, `bytes`,
`stored_bytes`, `layout`, `dtype`, `stored_shape`, `shard_shape`, per-tensor FNV-1a
`digest`). `slab_offset` is the arena offset; dense FP8 is already `B12x`-tiled and routed
`w2` is already `MoePadded640` (320-byte rows, 20-byte scales); tensors and their scales are
adjacent. Every tensor offset is 256-aligned, so `O_DIRECT` and `cuMemHostRegister` are legal.

Boot loader order: (1) read the 64-byte preamble, check `st_size`, verify `header_hash`; (2) one
`UnifiedSingle` or `MmapRegistered` allocation; (3) parallel `O_DIRECT` reads, 16 MiB chunks,
8 in flight, directly into the slab; fall back to buffered I/O per chunk if alignment or the
filesystem rejects `O_DIRECT`; (4) optional re-digest. The 640-K MoE padding moves `+25.2
GiB/rank` from boot repack into file bytes (MEASURED design constant). Node boot wall-clock and
`O_DIRECT` throughput remain unmeasured; storage occupancy (f2fs at 94%) is the cap.

## 4. Container and four-node deployment

Image files: `Dockerfile`, `deploy/container-entrypoint.sh`, `.dockerignore`.

```sh
docker build -t sparkflash-serve .
docker build --build-arg SPARKFLASH_FEATURES="tp4-serve cuda" -t sparkflash-serve:tp4 .
docker run --rm -p 8000:8000 sparkflash-serve
docker run --rm -p 8000:8000 \
  -v /models/deepseek-v4.1-flash:/models/ckpt:ro \
  -e SPARKFLASH_SERVE_API_KEY=... \
  sparkflash-serve serve --checkpoint /models/ckpt \
    --tokenizer /models/ckpt/tokenizer.json --bind 0.0.0.0:8000
```

Image contracts: PID 1 is the server (entrypoint `exec`s the binary, so SIGTERM drains); UID/GID
10001, workdir `/opt/sparkflash`; tokenizer fallback `/opt/sparkflash/reference/tokenizer.json`;
`/health` is liveness, `/readyz` is readiness; API key only via `SPARKFLASH_SERVE_API_KEY`. The
default CMD serves the synthetic tiny plan and needs no checkpoint.

The four-node launcher is `scripts/serve-four-node.sh` (`print|launch|status|stop`); `print` is
offline and pins the four rank commands. Build with `cargo build --release --features tp4-serve -p
sparkflash-cli`, then set `SPARK_NODES`, `SPARKFLASH_CHECKPOINT`, `SPARKFLASH_TOKENIZER`. Each
rank gets `--tp4-rank R --tp4-world-size 4 --tp4-peers <rank order> --tp4-base-port 29601
--tp4-control-port 29701`; rank 0 binds `0.0.0.0:8000`, ranks 1-3 bind `127.0.0.1:0`. Device-token
window flags are rank-0 only.

Drain contract: SIGTERM/SIGINT flips `/readyz` to 503 and `sparkflash_serving` to 0; after
`SPARKFLASH_SERVE_DRAIN_DELAY_MS` (default 0) the listener closes; accepted connections finish
within `SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS` (default 30000) and the process exits 0, otherwise it
prints `drain_timeout forced_exit=1` and exits 1. Orchestrator grace must exceed delay + timeout.
The driver queue is bounded at 64; overflow is HTTP 503 `service_unavailable` (engine code
`refused`) and increments `sparkflash_requests_rejected_total`. Metrics scrape is public
`/metrics` on rank 0.

## 5. Host allocator tuning

The CLI calls `sparkflash_cli::allocator::tune()` at `main` entry (not in a library constructor).
On Linux it sets `M_MMAP_THRESHOLD` and `M_TRIM_THRESHOLD` to 1 GiB; the pair is required - neither
threshold alone retains the per-layer 134 MB / 168 MB buffers. Others are no-ops.

| knob | value |
| :-- | :-- |
| enable | default on |
| kill switch | `SPARKFLASH_ALLOCATOR_TUNING=0` |
| requested thresholds | `1073741824` bytes each (`REQUESTED_MMAP_THRESHOLD_BYTES`, `REQUESTED_TRIM_THRESHOLD_BYTES`) |
| evidence field | top-level `allocator_tuning` in `tp4-host-rR.json` |

Measured on base `1b23d6e` (MEASURED, host path): attention block 7.942 s -> 3.251 s per step;
four-rank wall 13.827 s -> 7.929 s per step; rank-0 VmHWM +0.20 GiB. The current product speed
number is device-path; do not compare these host numbers to it.

## 6. Open items

- Node boot wall-clock, `O_DIRECT` throughput and warm-`mmap` restart time are not measured.
- Storage is f2fs at 94% full; that caps boot improvements. Fix 4 (occupancy/`fio` baseline) is
  operational, not code.
- Container image and four-node launch are host-tested definitions, not cluster measurements.
- CUDA-enabled aarch64 build has been shipped (27 pins on the current tree); a same-tip manifest is
  required per row, because the binary sha does not carry across tips.
- `nvidia-peermem` is blocked on a root rebuild (2026-10-22); the device port currently uses NCCL.

## Archived sources

aarch64-pipeline.md; boot-convert.md; allocator-tuning.md; serve-deployment.md; window-runbook.md;
review-protocol.md; QUALIFICATION.md; d48-window-protocol.md.

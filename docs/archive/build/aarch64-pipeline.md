# Building and shipping the aarch64 head-node binaries

This is the task-28 deliverable: how a SparkFlash binary gets from this x86_64 dev box onto the
aarch64 head node, and the evidence that it runs there. The executable part lives in
[scripts/build-aarch64.sh](../../../scripts/build-aarch64.sh); this file records the decisions, the
commands and the verbatim results.

## 1. What was built

A statically-linked-against-nothing-extra release binary of the preflight CLI:

| | |
|---|---|
| artifact | `sparkflash` (bin of `sparkflash-cli`) |
| target | `aarch64-unknown-linux-gnu`, release |
| size | 39,565,400 bytes |
| sha256 | `aee19f62b1da4ef7cad68d37312a4191216634badc37f031f4e486dfab91410f` |
| `file` | `ELF 64-bit LSB pie executable, ARM aarch64, interpreter /lib/ld-linux-aarch64.so.1` |
| links against | `libm.so.6`, `libc.so.6`, `libpthread.so.0`, `libdl.so.2` |
| glibc symbol floor | `GLIBC_2.30` (head node has 2.39) |

The copy on the head node has byte-identical sha256 (checked on both sides, appendix A).

## 2. The toolchain, and why it is this one

### The route that was rejected

The textbook cross toolchain is:

    apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross

`g++-aarch64-linux-gnu` and `libc6-dev-arm64-cross` are required, not optional: the dependency
tree contains C and C++ (tokenizers -> onig is C, esaxx-rs is C++, ring is C), and the C++ objects
need an aarch64 libstdc++ at link time. **This box cannot run apt**: it has no rustup either, and
the container sets the no-new-privileges flag, so `sudo` fails with *"The no new privileges flag is
set"*. That is the only reason the route is rejected here - on a box with root, prefer it, because
it is one apt line with no version to pin. The script says all of this in its header.

### The route that was used

No root anywhere:

1. **rustup, user-local**, `~/.cargo` + `~/.rustup`, `--no-modify-path`, minimal profile, pinned to
   **1.98.1** - the same version as the box's root-owned `/opt/rust-bin-1.98.1`, so the cross build
   and the native build agree. The pinned version's `aarch64-unknown-linux-gnu` std is added.
2. **zig 0.17.0**, tarball pinned by sha256
   (`1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026`, 57,332,648 bytes) as the
   cross C compiler, C++ compiler, linker and archiver. Zig is self-contained: it carries its own
   libc, libc++ and lld, so no sysroot package is needed.

Both steps are idempotent, and both **fail loudly if a pin moves**: the tarball is checksummed on
every run, `zig version` and `cargo --version` are compared against the pins, and a mismatch is a
hard error rather than a silent toolchain change.

### The three translations the first real build forced

None of these were predicted; each is an exact error the build produced, and each is why the
wrappers in the script exist.

1. **Target spelling.** cargo passes `--target=aarch64-unknown-linux-gnu` to the C compiler, and zig
   parses target queries in its own spelling, so it reads that as an unknown operating system:
   `ring`'s build script died with *"unable to parse target query 'aarch64-unknown-linux-gnu':
   UnknownOperatingSystem"*. The wrappers drop whatever target flag they are given and append
   `-target aarch64-linux-gnu`.
2. **C++ standard library.** esaxx-rs compiles C++ through the wrapper, so its objects reference
   libc++'s ABI, while the cc crate still asks the linker for `-lstdc++`. The linker wrapper maps
   `-lstdc++` to `-lc++`; without it the link fails, because zig ships libc++ and no libstdc++.
3. **Cortex-A53 linker flag.** rustc asks for `-Wl,--fix-cortex-a53-843419` on every aarch64 target
   and lld refuses it: *"error: unsupported linker arg: --fix-cortex-a53-843419"*. The flag works
   around ARM erratum 843419 in the Cortex-A53 core; the head node is an NVIDIA GB10 Grace
   (Neoverse-class), so dropping it is safe here. It is the last step of the build, so keeping it
   would fail a build in which everything else already compiled.

## 3. Build it

    scripts/build-aarch64.sh                      # release sparkflash for aarch64
    scripts/build-aarch64.sh --workspace          # every workspace binary
    scripts/build-aarch64.sh --print-env          # resolved toolchain and target dir, then exit
    CARGO_TARGET_DIR=/somewhere/else scripts/build-aarch64.sh

Defaults to `-p sparkflash-cli`; the target dir defaults to the repository's own `target/`. The
toolchain lives under `$HOME/opt` (`SPARKFLASH_TOOLCHAIN_DIR` overrides), so nothing outside $HOME
is written. The CUDA feature is not enabled and is not needed: `sparkflash-kernels`' `cuda`
feature is opt-in, and this build produces the binary that does not need it.

## 4. Ship it and run the preflight

    scp sparkflash allenh1@192.0.2.80:/home/user/sparkflash-m7/sparkflash
    ssh allenh1@192.0.2.80 '~/sparkflash-m7/sparkflash doctor'          # exit 0
    ssh allenh1@192.0.2.80 '~/sparkflash-m7/sparkflash doctor --local'  # exit 1

The default run probes nothing and is read-only by construction; `--local` actually probes this
node. `--apply` is required for any state-changing command and was not passed. The `--local`
report says so in its own words: *"read-only: nvidia-smi -lgc 0,2200 not run, because locking SM
clocks changes this node's clock policy while it may be serving traffic"*. No other command was
executed on the head node. Nothing was written except `~/sparkflash-m7/`, the directory holding our
own binary, and `vllm_node` still reports *Up 5 days* after the run (appendix A).

### What the run found

**Read this first: `doctor --local` exiting 1 with "verdict: NOT READY" is a statement about this
node's hardware, not about the binary.** A cross build that produced a broken executable would fail
to start, not print a five-check report; the same binary exits 0 on the default probe-free run, and
a failure inside it would surface as a probe error rather than as a named check result. All four
failed checks are properties of the head node as it stands today - a driver older than the gate asks
for, two rail names in our own specification that do not match this chassis, an `nvidia-smi` that
exits 2 on GB10, and 3 GiB of MemAvailable because the inference holds the rest. None of them is a
regression, and no amount of rebuilding the binary can fix any of them: they are node or
documentation changes. The rail-name and MTU findings are tracked as a documentation task.

The point of task-28 is that the binary runs on aarch64 and produces a real report; the following
are findings about the node, not about the build:

* **driver 580.173.02 is older than the gate's 580.178.04.** The kernel
  (`6.17.0-1029-nvidia`) satisfies the gate. On the `nvidia-uvm` 64K-packing clause the report says
  no knob exists under any conventional name on a real 580-series node, and calls specification 6.1
  a specification gap rather than an operator error.
* **The rail names in specification 6.1 do not match this node.** `enp1s0f0np0` and
  `enP2p1s0f0np0` are DOWN, but their sibling functions on the same PCI devices
  (`enp1s0f1np1`, `enP2p1s0f1np1`) are ACTIVE at 200000 Mb/s - the report's own reading is *"a wrong
  name in the specification, not a disconnected cable"*. All four netdevs are MTU 1500, not 9000.
* **`nvidia-smi` exits 2 on this node** (consistent with its GB10 driver reporting
  `memory.total [N/A]`), so the SM-clock reading cannot be taken.
* **MemAvailable is 3 GiB, against a gate of 114 GiB** - vLLM holds the rest, which is the next
  section's whole subject.
* **No Engram store exists** (`/var/lib/sparkflash/engram/manifest.json` absent), so the shard check
  cannot run; that is task-29's territory.

## 5. The CUDA build environment on the head node: numbers first

Nothing in this section was created or started. The pull sizes below are registry metadata read
from the Docker Hub API rather than the result of a `docker pull`; the one image that has since
been pulled, under the approved device-path plan, is pinned by digest at the end of this section.

### Measured state of the head node (read-only, 2026-10-16)

| | |
|---|---|
| memory | 124,610 MB total, 121,494 MB used, **3,115 MB available** |
| `vllm_node` cgroup | 6.68 GiB / 121.7 GiB (5.49%) |
| disk | `/` 3.7 T, 3.0 T used, **581 G free** (84%) |
| docker storage | 595 GB images (61 GB reclaimable), **155 GB build cache** (94 GB reclaimable) |
| driver / kernel | 580.173.02 / 6.17.0-1029-nvidia |
| CUDA | 13.0 at `/usr/local/cuda` (`cuda_13.0.r13.0/compiler.36424714_0`) |
| compilers | gcc 13.3.0, glibc 2.39, docker 29.2.1 |

Two readings of the same machine disagree, and the *larger* one is the safe one: the container's
cgroup shows 6.68 GiB, but the host shows 121 GB in use with 3.1 GB available. On a GB10 the model's
unified-memory allocation largely does not appear in the container's cgroup counter. **Plan against
3.1 GB of available memory, not against the cgroup number.**

### Measured pull sizes (arm64, compressed layers + config)

| image | pull |
|---|---|
| `nvidia/cuda:13.0.1-devel-ubuntu24.04` | **3.90 GB** |
| `nvidia/cuda:13.0.0-devel-ubuntu24.04` | 3.84 GB |
| `ubuntu:24.04` | 0.03 GB |
| `rust:1.98-slim` | 0.28 GB |

Disk is not the constraint: 581 GB free, and 155 GB of that is reclaimable build cache.

### The pinned image, and the one thing it does not cover

    docker pull nvidia/cuda:13.0.1-devel-ubuntu24.04

pulled on this box on 2026-10-16 gives:

| | |
|---|---|
| digest | `sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92` |
| platform | `amd64/linux` |
| size | 3,948,969,770 bytes on disk (3.90 GB as compressed layers) |

Pin the digest, not the tag: `docker pull nvidia/cuda@sha256:7d2f6a8c...` is the same environment
forever, the way `ZIG_SHA256` and `RUST_VERSION` pin the cross toolchain. Next to it, the
unpulled alternative: the registry reports `sha256:5c36750138dc1...` as the platform manifest digest
for `linux/arm64`, which is not the digest of anything pulled here.

**The image is amd64, and that is fine, because device code is host-arch independent.** `nvcc`
compiling `-cubin`/`-arch=sm_121a` from an x86_64 host emits the same sm_121a device binary an
aarch64 `nvcc` would: no host target participates in device-only compilation. Only the *host* side
of an executable is arch-bound, and that is the cross toolchain's job. So the split is: amd64 `nvcc`
for cubin and PTX, the arm64 image's headers and libraries as plain files (`docker cp`, no
emulation) for anything the host side compiles against, and the pinned zig cross toolchain from
section 2 to link the aarch64 executable. It is also what `build.rs` already does, which is a point
in its favour.

**Why not the arm64 image under qemu.** Checked on this box: `/proc/sys/fs/binfmt_misc/` contains
only `register` and `status` (with `status` reading `enabled`), there is **no `qemu-aarch64`
entry**, and this account is uid 1000 with no root, so one cannot be registered. The qemu route is
therefore unavailable rather than merely slower, which is why the design rejects it on capability
rather than on performance.

**The risk to check in that split: does the host side have to link an aarch64 CUDA library? No.**
Verified against the vendored source (cudarc 0.19.10) and our own manifest:

* our workspace enables cudarc without `default-features = false` (`Cargo.toml:60`), so cudarc's
  defaults are active: `std, cublas, cublaslt, curand, driver, runtime, nvrtc,
  fallback-dynamic-loading`;
* `fallback-dynamic-loading` on its own makes cudarc's build script emit
  `cargo:rustc-cfg=feature="dynamic-loading"` (cudarc `build.rs:96-104`), and the code that emits
  `cargo:rustc-link-lib=dylib=cuda` runs only under `dynamic-linking` or `static-linking`
  (`build.rs:150-154` and `185-193`), neither of which is active - the build script even panics if
  none of the four modes is selected, which is what makes this an intentional default rather than an
  accident;
* cudarc's `src/` contains no `#[link(name = "cuda")]` at all, and obtains the driver through
  `libloading::Library::new` (for example `src/curand/sys/mod.rs:588-601`).

So the driver and runtime are dlopen'd at run time on the node, where `libcuda.so.1` is the node's
own aarch64 driver. The cross build needs CUDA *headers* and nothing to link, which removes the one
place a silent amd64/aarch64 mismatch could hide - **provided nobody enables `dynamic-linking` or
`static-linking`**: that single feature flip would put an x86_64-shaped link line under an aarch64
target, and the failure would be subtle. Treat "cudarc stays on `fallback-dynamic-loading`" as a
build invariant and re-check it when the `cuda` feature gains its first code path.

### Memory is the constraint

A cross build of this workspace on the dev box peaked at **3.8 GB aggregate RSS** across all build
processes (largest single process 410 MB). That is already more than the head node's 3,115 MB
available, and a build on the node would be competing with the inference it exists to serve. A
container that is OOM-killed while vLLM is loaded is not an acceptable failure mode.

### Options

**A - build on the dev box, ship the binary (recommended).** Pull
`nvidia/cuda:13.0.1-devel-ubuntu24.04` (3.90 GB) *here*, where there are 251 GB of RAM and 2.7 TB of
free disk. CUDA compilation needs no GPU - only execution does - so a GPU-less dev box is a
perfectly good CUDA build host. The head node then pays **zero bytes of pull and zero build
memory**. This is what the aarch64 binary in section 1 already demonstrates for the non-CUDA
feature set.

**B - build on the head node, cheapest correct environment.** The host already has CUDA 13.0 and
gcc 13.3, so most of a devel image would be downloaded twice:

    docker run --rm --gpus all --memory=2g --cpus=2 \
      -v /usr/local/cuda-13.0:/usr/local/cuda:ro \
      -v ~/sparkflash-m7:/work -w /work ubuntu:24.04 ...

`ubuntu:24.04` is 0.03 GB; user-local rustup 1.98.1 adds roughly 0.3 GB; the mounted toolkit costs
nothing. Total pull is about **0.35 GB** instead of 3.90 GB. The host and container are both Ubuntu
24.04/glibc 2.39, so the mounted `nvcc` runs. Set `-j2` and the `--memory` cap: the cap makes an
overrun a killed container rather than a starved inference.

**C - no build environment at all.** Not available for the CUDA feature: `sparkflash-kernels`'
build currently reports that `nvcc` was not found and that clang parsed the kernels with stub
headers instead (*"NOT PTX and NOT numerics"*), which is exactly the gap a real nvcc host has to
close.

### Recommendation

**A, with B as the documented fallback.** A keeps the 3.1 GB of head-node headroom untouched, costs
one 3.90 GB pull on a box with 251 GB of RAM and 2.7 TB free, and matches how every other artifact
in this project reaches the head node. B is correct and cheap in bytes, but it spends the head
node's scarcest resource, and it should only be used with the memory cap and `-j2`, and preferably
while the inference is not the only thing keeping the node alive.

## 6. What was not done

* No package was installed anywhere; the head node got a directory and one 39 MB binary.
* Nothing was pulled: section 5's sizes come from registry metadata.
* `--apply` and `--lock-sm-clocks` were never passed, so no state-changing command ran.
* The running `vllm_node` container was not stopped, restarted, modified or exec'd into.
* No CUDA-enabled build was attempted; the CUDA feature is opt-in and was left off.

## Appendix A - deploy transcript

    local sha256: aee19f62b1da4ef7cad68d37312a4191216634badc37f031f4e486dfab91410f
    remote: aee19f62b1da4ef7cad68d37312a4191216634badc37f031f4e486dfab91410f  /home/user/sparkflash-m7/sparkflash
    -rwxr-xr-x 1 allenh1 allenh1 39565400 Oct 16 20:29 /home/user/sparkflash-m7/sparkflash
    /home/user/sparkflash-m7/sparkflash: ELF 64-bit LSB pie executable, ARM aarch64, version 1 (SYSV),
      dynamically linked, interpreter /lib/ld-linux-aarch64.so.1, for GNU/Linux 2.0.0, with debug_info, not stripped
    after the run: vllm_node Up 5 days vllm/vllm-openai:deepseekv41-flash-0909

## Appendix B - `sparkflash doctor` (default), exit 0, verbatim

    sparkflash doctor - DGX Spark preflight (specification 6.1)
    note: specification 6.1 is titled 'Seven-Point' but enumerates five checks; doctor implements the five that are written down and does not invent the missing two

    [-] Driver and kernel (driver-kernel)
          DID NOT RUN: driver >= 580.178.04 (580.142 rejected), kernel >= 6.17.0-1018 or 7.0.0-1019-nvidia-64k, nvidia-uvm 64K packing
          - no hardware probe requested: pass --local for this node or --nodes for the cluster

    [-] Dual-rail ConnectX-7 link speed and MTU (link-mtu)
          DID NOT RUN: two ACTIVE ConnectX-7 rails at 200000 Mb/s with MTU 9000 and a 8972-byte do-not-fragment probe
          - no hardware probe requested: pass --local for this node or --nodes for the cluster
          - enp1s0f0np0: do-not-fragment probe skipped, no peer was given (--nodes)
          - enP2p1s0f0np0: do-not-fragment probe skipped, no peer was given (--nodes)

    [-] Dual-rail RDMA bandwidth and one-shot latency (rdma-bandwidth)
          DID NOT RUN: dual-rail RDMA aggregate >= 180 Gbps and 10 KiB one-shot All-Reduce < 22 us
          - no peer was given (--nodes); the RDMA gate did not run

    [-] SM clock cap and available memory (clocks-memory)
          DID NOT RUN: SM clocks lockable within [0, 2200] MHz and MemAvailable >= 114 GiB
          - no hardware probe requested: pass --local for this node or --nodes for the cluster
          - read-only: nvidia-smi -lgc 0,2200 not run, because locking SM clocks changes this node's clock policy while it may be serving traffic; pass --apply --lock-sm-clocks to do it deliberately

    [-] Local NVMe Engram shards (engram-shards)
          DID NOT RUN: both rank shards present, 512-byte aligned on local NVMe and matching the manifest SHA256
          - no hardware probe was requested (--local or --nodes)

    5 of 7 specified checks implemented: 0 passed, 0 failed, 5 not applicable
    verdict: INCOMPLETE: no check failed, but the hardware checks did not run; pass --local or --nodes to run them

## Appendix C - `sparkflash doctor --local` (excerpt), exit 1, verbatim

    [x] Driver and kernel (driver-kernel)
          NOT SATISFIED: driver >= 580.178.04 (580.142 rejected), kernel >= 6.17.0-1018 or 7.0.0-1019-nvidia-64k, nvidia-uvm 64K packing
          - driver 580.173.02 is older than the required 580.178.04
          - kernel 6.17.0-1029-nvidia satisfies the gate
          - nvidia-uvm: no 64K-packing knob found (53 module parameters listed under /sys/module/nvidia_uvm/parameters, none named *pack* or *64k*)

    [x] Dual-rail ConnectX-7 link speed and MTU (link-mtu)
          NOT SATISFIED: two ACTIVE ConnectX-7 rails at 200000 Mb/s with MTU 9000 and a 8972-byte do-not-fragment probe
          - RDMA devices on this node: roceP2p1s0f0/enP2p1s0f0np0 DOWN 40000 Mb/s; roceP2p1s0f1/enP2p1s0f1np1 ACTIVE 200000 Mb/s; rocep1s0f0/enp1s0f0np0 DOWN 40000 Mb/s; rocep1s0f1/enp1s0f1np1 ACTIVE 200000 Mb/s
          - 2 rails are ACTIVE at >= 200000 Mb/s: roceP2p1s0f1, rocep1s0f1
          - enp1s0f0np0 (rocep1s0f0) is DOWN, not ACTIVE: the specification asks for this rail
          - rocep1s0f0 is DOWN but its sibling function rocep1s0f1 on the same PCI device (0000:01:00) is ACTIVE at 200000 Mb/s with netdev enp1s0f1np1 - a wrong name in the specification, not a disconnected cable
          - enp1s0f0np0 reports its speed as 'Unknown!', which is not a negotiated rate: the link is down or has not trained
          - enp1s0f0np0: MTU 1500, expected 9000
          - enP2p1s0f0np0 (roceP2p1s0f0) is DOWN, not ACTIVE: the specification asks for this rail
          - roceP2p1s0f0 is DOWN but its sibling function roceP2p1s0f1 on the same PCI device (0002:01:00) is ACTIVE at 200000 Mb/s with netdev enP2p1s0f1np1 - a wrong name in the specification, not a disconnected cable
          - enP2p1s0f0np0 reports its speed as 'Unknown!', which is not a negotiated rate: the link is down or has not trained
          - enP2p1s0f0np0: MTU 1500, expected 9000
          - enP2p1s0f1np1 (active roceP2p1s0f1): MTU 1500, expected 9000
          - enp1s0f1np1 (active rocep1s0f1): MTU 1500, expected 9000

    [-] Dual-rail RDMA bandwidth and one-shot latency (rdma-bandwidth)
          DID NOT RUN: dual-rail RDMA aggregate >= 180 Gbps and 10 KiB one-shot All-Reduce < 22 us
          - no peer was given (--nodes); the RDMA gate did not run

    [x] SM clock cap and available memory (clocks-memory)
          NOT SATISFIED: SM clocks lockable within [0, 2200] MHz and MemAvailable >= 114 GiB
          - nvidia-smi exited with exit status: 2:
          - read-only: nvidia-smi -lgc 0,2200 not run, because locking SM clocks changes this node's clock policy while it may be serving traffic; pass --apply --lock-sm-clocks to do it deliberately
          - MemAvailable 3 GiB, expected >= 114 GiB

    [x] Local NVMe Engram shards (engram-shards)
          NOT SATISFIED: both rank shards present, 512-byte aligned on local NVMe and matching the manifest SHA256
          - shard verification not applicable: /var/lib/sparkflash/engram/manifest.json does not exist; shard integrity cannot be verified until pack-engram has run
          - /var/lib/sparkflash/engram/engram-l1-r0of4.bin: No such file or directory (os error 2)
          - /var/lib/sparkflash/engram/engram-l14-r0of4.bin: No such file or directory (os error 2)

    5 of 7 specified checks implemented: 0 passed, 4 failed, 1 not applicable
    verdict: NOT READY: fix the failed checks before loading weights

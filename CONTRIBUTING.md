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

# Contributing to SparkFlash

SparkFlash is an inference engine for large sparse models on small, memory-coherent GPUs. The
project would rather reject a change than merge a claim nobody can reproduce, so the rules below
are about evidence, not ceremony. Read them before you open a merge request.

## Ground rules

- **License.** SparkFlash is Apache-2.0. Every new source file starts with the copyright and
  license notice already used throughout the tree. By submitting a change you agree to license it
  under those terms.
- **Sign-off (DCO).** Every commit ends with
  `Signed-off-by: Your Name <you@example.com>`. Use `git commit -s`; the sign-off certifies that
  you wrote the change or have the right to submit it. Do not add co-authorship trailers for tools
  or assistants.
- **No deployment details.** Never commit hostnames, LAN or RoCE addresses, SSH keys, passwords,
  credentials, internal Git URLs, personal home directories, or notes about a particular cluster.
  Documentation examples use `node0`-`node3` and the TEST-NET ranges (`192.0.2.0/24`,
  `198.51.100.0/24`, `203.0.113.0/24`). A change that leaks any of these is rejected even if the code
  is correct.
- **One idea per commit.** Keep mechanical changes (formatting, renaming, license headers) separate
  from behavioural changes so a reviewer can see the behaviour.
- **Third-party material.** Preserve upstream notices. `reference/` is DeepSeek's MIT-licensed
  reference material and must not be relicensed or stamped with this project's copyright.

## Before you send a change

Run the same gates CI runs:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p sparkflash-trace --no-default-features
cargo test -p sparkflash-kernels --examples
python3 ci/check_features.py
```

CUDA changes additionally need a CUDA 13 toolkit and the compiled-artifact gates:

```bash
cargo test -p sparkflash-kernels --features cuda
cargo clippy -p sparkflash-kernels --all-targets --features cuda -- -D warnings
/bin/sh crates/sparkflash-kernels/scripts/check-kernel-baseline.sh
sh crates/sparkflash-kernels/scripts/regen-codec-fixture.sh
```

If a kernel's compiled form changes, regenerate the baseline with
`crates/sparkflash-kernels/scripts/regen-kernel-baseline.sh` and commit the new cubin sizes,
SHA-256 digests and PTX membar counts together with the kernel change. A baseline diff is a review
artifact, not a formality.

## Merge criteria for a new model

A model is a product, not a descriptor. A merge request that adds one is merged only when it
contains every item below. Missing evidence makes the request a draft; "it works on my machine" is
not evidence.

### 1. Tested and working on the stated configuration

State the exact configuration, then show the model running on it:

- hardware: GPU model and count, memory, interconnect, and any host requirements;
- software: OS and kernel version, GPU driver, CUDA toolkit, NCCL, Rust toolchain;
- topology: tensor/pipeline parallelism, node count, ports and collectives;
- context length, concurrency, quantization and memory footprint actually exercised;
- commands used to convert the checkpoint, launch the server and send requests;
- correctness evidence: greedy output against the reference implementation, logits or loss parity
  where available, and the refusal paths (unsupported context, unsupported dtype, missing tensors);
- known limitations: what is not implemented, what is slow, and which configurations are untested.

A model that only compiles, or that only runs on a host reference path while claiming the device
path, does not meet this criterion.

### 2. CUDA kernels clean and optimized

- Kernels compile without warnings and run without sanitizer findings, out-of-bounds accesses or
  uninitialised reads; document any deliberate divergence from the reference arithmetic.
- Determinism claims are backed by a test: the same inputs produce the same bits, and any
  non-determinism is named.
- Performance is engineered, not assumed: report occupancy, register and shared-memory use, the
  achieved fraction of memory bandwidth or tensor-core throughput, and the launch geometry. Explain
  why a custom kernel is needed instead of an existing library call.
- No hidden fallbacks. If a fast path can silently fall back to a host or reference path, the
  fallback must be observable in logs and counters and covered by a test.
- New kernels enter the baseline with a pinned `sm_121a` cubin (bytes, SHA-256 and PTX membar
  count) and, where applicable, the M4 ABI static-assert counts. Kernel source and baseline move in
  the same commit.

### 3. Benchmark numbers

Benchmarks must be reproducible from the repository, not typed into the pull request:

- commit the script or harness and the exact command line;
- report hardware, software and topology with the numbers;
- decode throughput (tokens/s per request and aggregate), time to first token, prefill throughput
  and memory use, at several context lengths and concurrency levels;
- state warmup, sample count, median and tail (p50/p99), and run-to-run variance;
- compare against at least one external baseline (for example vLLM or SGLang) using the same
  prompts, sampling parameters and measurement protocol;
- say what was not measured. A number without a protocol is a rumour.

### 4. Link to the weights

- the canonical Hugging Face repository URL and the exact revision (commit hash or tag);
- the license of the weights and whether redistribution and commercial use are permitted;
- the checkpoint size and the expected tensor layout, including any conversion this project needs;
- checksums or digests that let a reviewer confirm they downloaded the same bytes.

Model weights are not committed to this repository.

### 5. Technical paper or reference implementation

- link the paper, technical report, model card or specification that defines the architecture;
- link a reference implementation and name the revision you compared against;
- include a short reconciliation table: every place this implementation differs from the reference
  (missing features, fused operations, different rounding, unsupported modes), and the test that
  pins the difference. "Matches the paper" is a claim; the table is the evidence.

### 6. Copyright and licensing

- new files carry the Apache-2.0 copyright and license notice;
- every commit is signed off under the DCO;
- copied code, kernels or fixtures carry their original license and notice, and the model license
  is compatible with Apache-2.0 distribution;
- no weights, datasets or generated artifacts with unclear provenance are added.

### 7. Target platforms and operating systems

Name the supported matrix explicitly:

- CPU architecture (the reference target is aarch64; x86_64 is supported for host-side work);
- OS (DGX OS and Ubuntu are the reference distributions);
- CUDA toolkit and driver versions, GPU architecture (`sm_121a` is the device target);
- NCCL / RDMA requirements and the interconnect the model was tested on;
- unsupported combinations, which must fail with a clear error rather than silently misbehave.

If a model needs a different architecture, the build must select it explicitly and the CI coverage
must say what is and is not compiled.

### 8. Integration and documentation

- the model is registered through `sparkflash-model` with a `ModelDescriptor`, an engine selector
  and a capability set; no V4.1 geometry is assumed outside its model pack;
- tokenizer and prompt encoding go through the `PromptCodec` contract and are tested against the
  reference encoding;
- the API model card, `/v1/models` output and limits come from the descriptor, not from constants;
- `docs/` explains the architecture, the conversion path, the supported configuration and the
  benchmark method;
- the model has a smoke test that runs without a checkpoint, plus an ignored hardware test for the
  full path.

### 9. Review

At least one maintainer reviews the change. Reviewers check the evidence above, not just the diff.
CI must be green, including the CUDA job when kernels change. A merge request that changes a frozen
pin (M4 ABI line count, kernel baseline, golden fixtures) must explain the change in the commit
message and update every dependent artifact.

## Evidence checklist

Copy this into the merge request and fill it in:

```text
Model / revision:
Weights URL and revision:
Weight license:
Paper / reference implementation:
Target hardware:
OS / kernel / driver / CUDA / NCCL / Rust:
Topology and context length:
Conversion command:
Launch command:
Correctness evidence (reference comparison, tests):
Decode tok/s (per request, aggregate), TTFT, prefill tok/s:
Benchmark script and protocol:
Kernel changes and baseline update:
Known limitations:
Unsupported configurations:
```

## Commit and merge-request conventions

- Subject line: `area: imperative summary`, at most about 72 characters.
- Body: what changed, why, and the evidence. Include the command that produced a number.
- Reference the issue or decision record (`D<n>` in `docs/DECISIONS.md`) when the change depends on
  one.
- Keep the tree buildable at every commit; do not land generated binaries, cubins or checkpoint
  fragments.
- Squash fixups before review, then leave the history readable.

## Where to start

- Architecture and model notes: `docs/architecture/`
- Design and performance records: `docs/design.md`, `docs/performance.md`,
  `docs/DECISIONS.md`
- Serving and the HTTP surface: `docs/serving.md`
- Adding a second model: `docs/multi-model-plan.md`
- Kernel conventions and the compiled baseline: `crates/sparkflash-kernels/`

If something in this guide is wrong or a gate cannot be run on the hardware you have, say so in the
merge request. An honest gap is reviewable; a hidden one is not.

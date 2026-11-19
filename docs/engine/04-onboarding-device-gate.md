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

# 04-onboarding — device gate for the moved arms

The three arms that moved onto `Runtime<Walk>` (dense, Qwen, GLM) owe device evidence for the
behaviour changes listed in [04-onboarding-review.md](04-onboarding-review.md). This page records
the run at `09dda17a` on the CUDA host, the exact result of each command, and what remains
unmeasured.

## Host and tree

* Source: branch `onboarding/devv` at `09dda17a`, rsynced without `target`, `.git` or
  `bench/deepseek-v4.1-flash-1017/server` to `/home/allenh1/sf-gate-09dda17a`.
* Host: `gentoo-p16`, **aarch64**, one NVIDIA RTX Spark N1X (Blackwell), CUDA 13.4.59
  (`ptxas` V13.4.59), 419 GB free on `/`. This is a cross-architecture run: every test below
  was built and executed on aarch64, not on the x86_64 workstation.
* Build: `cargo build --workspace --all-targets` — **PASS** in 1m56s. The build embedded 13
  cubins (`qwen4_exp`, `qwen4_attn`, `qwen4_moe`, `qwen4_prefill`, `qwen4_scan`,
  `dflash2`, `glm_prefill`, `glm_attention`, `glm_moe`, `glm_decode`, `moe_plan`,
  `inkling`, `dory`). The only warning is the RDMA feature building without
  `infiniband/verbs.h` (one GPU; no verbs).
* `SPARKFLASH_KERNEL_DIR=/home/allenh1/sf-gate-09dda17a/kernel-dir` was assembled for the one
  test that reads it: all 53 `.ptx` files the build produced were passed through
  `ptxas -arch=sm_121a`, 53/53 succeeded. `decode_mma_device` needs this; no cubin was
  missing from the build.

Every command ran with `--test-threads=1`. Logs and exit codes are in
`/home/allenh1/sf-gate-09dda17a/logs/`.

## Results

| Test / command | Host | Result | What it proves, or what it needs |
|---|---|---|---|
| `cargo build --workspace --all-targets` | aarch64 + nvcc 13.4 | **PASS**, 1m56s, 13 cubins | The tree builds for `sm_121a` on the CUDA host; no source error. |
| `cargo test -p sparkflash-model-glm5-next -- --ignored` | aarch64 + GPU | **FAIL** (exit 101): 0 passed, 7 failed — the seven generated conformance checks | Not a code fault: the checks are stubs. First line: `the check needs a device: BackendError { code: "conformance", message: "alone = four at once needs sparkflash_model_glm5_next::walk::GlmWalk's device fixture; run scripts/onboard-gate.sh <arm>" }` at `walk.rs:1319`. Needs the arm's fixture. |
| `cargo test -p sparkflash-model-qwen4-exp -- --ignored` | aarch64 + GPU | **FAIL** (exit 101): 0 passed, 7 failed | Same seven stubs for `QwenWalk` at `device/walk.rs:930`. First: `"alone = four at once needs sparkflash_model_qwen4_exp::device::walk::QwenWalk's device fixture"`. Needs the arm's fixture. |
| `cargo test -p sparkflash-model-dense -- --ignored` | aarch64 + GPU | **FAIL** (exit 101): 0 passed, 7 failed | Same seven stubs for `DenseWalk` at `walk.rs:745`. Needs the arm's fixture. |
| `cargo test -p sparkflash-runtime -- --ignored` | aarch64 + GPU | **FAIL** (exit 101): 0 passed, 7 failed | Same seven stubs for the runtime's `ToyWalk` at `runtime.rs:1818`. The runtime's own `tick_trace` gate is a normal test and passes below. |
| `cargo test -p sparkflash-kernels -- --ignored` | aarch64 + GPU | **PASS**: 5 ignored tests ran and passed — 2 `moe_routed_device` device tests, `decode_mma_device`'s batch-invariance test, and both token-body tests | The tensor-core decode is batch-invariant and as accurate as the GEMV, and the two token bodies are the tile's rows. |
| `cargo test -p sparkflash-kernels --test token_bodies_device -- --ignored --nocapture` | aarch64 + GPU | **PASS**: 2 passed | `the_fp8_token_bodies_are_the_tiles_rows` and `the_packed_token_body_is_the_bf16_tiles_rows`: at 1, 2, 4 and 8 rows, on a covered shape and a 40-column one, the token body is the tile's bits; the packed copy unpacks to the BF16 rows word for word. |
| GLM `a_drafted_decode_is_the_bits_of_a_plain_one` | aarch64 + GPU | **PASS** (1 test, 3.47s) | A drafting session emits exactly the plain session's tokens and really drafted (`drafts_proposed > 0`, `verify_steps > 0`). This is the one-row `verify_batch` + `settle` path the move introduced. |
| GLM `drafting_members_verified_together_emit_the_plain_tokens` | aarch64 + GPU | **PASS** (1 test, 3.47s) | Two fused members verify together and each emits its plain session's tokens; both members drafted. |
| GLM `a_sequence_reports_the_prefix_it_holds` | aarch64 + GPU | **PASS** (1 test, 3.34s) | The session reports exactly the history it forwarded and an extension is a cache hit. The whole `serve_tiny` binary passed 7/7. |
| Qwen `batch_exactness` (`tests/batch_exactness.rs`) | aarch64 + GPU | **PASS**: 4 passed | A fused batch emits what its members emit alone; one-slot, two-slot, four-on-one-prompt and reopen-cold cases all agree. |
| Qwen `joint_off` (`tests/joint_off.rs`) | aarch64 + GPU | **PASS**: 1 passed | With `SPARKFLASH_JOINT_ROUNDS=off` each lane walks its round alone, drafts nothing, and still emits the plain tokens. |
| Dense device head: `tests::the_cuda_head_choice_is_the_host_samplers_token_on_the_toy_rows` | aarch64 + GPU | **PASS** (0.38s) | The device `q4e_argmax_slices` picks the host `Sampler`'s token on every toy row. `the_dense_tile_and_the_row_gemv_move_a_nine_row_projection_in_the_last_bits` also passed. |
| Full `cargo test -p sparkflash-model-glm5-next -- --test-threads=1 --no-fail-fast` | aarch64 + GPU | **FAIL**: lib 66 passed; `device_forward` 11 passed; `serve_tiny` 7 passed; `engine_builder` 2 passed, 1 failed | The failure is `a_small_checkpoint_opens_through_the_engine_builder` at `tests/engine_builder.rs:103`: `the serving boot check refuses 14 part instance(s) ... dense_mlp on sm121 at e4m3 ... hidden 256 does not equal 4096`. With real cubins embedded, the boot check (`crates/sparkflash-ops/src/parts_backend.rs:84`) resolves the 256-hidden tiny plan against the released image shapes and refuses it. Nothing else failed. |
| Full `cargo test -p sparkflash-model-qwen4-exp -- --test-threads=1 --no-fail-fast` | aarch64 + GPU | **FAIL**: every other target passed; `engine_builder` 4 passed, 1 failed | `a_tiny_checkpoint_opens_through_the_engine_builder` fails the same way at `tests/engine_builder.rs:103`: the 32-hidden/64-vocab tiny plan is refused by the boot check (`vocab 64 is not a multiple of 128`, `hidden 32 is not a multiple of 256`, ...). This is the GPU-build form of the tiny fixture, not a token difference. |
| Full `cargo test -p sparkflash-model-dense -- --test-threads=1` | aarch64 + GPU | **PASS**: 17 passed, 7 ignored | The dense arm's CPU tests and its two device tests pass; the seven ignored are the conformance stubs. |
| Full `cargo test -p sparkflash-runtime -- --test-threads=1` | aarch64 + GPU | **PASS**: 69 passed, 7 ignored; `tick_trace` 1 passed; doctests 2 passed | The runtime's scheduling and strategy tests, including the committed tick trace, pass on aarch64. |
| Full `cargo test -p sparkflash-kernels -- --test-threads=1 --no-fail-fast` | aarch64 + GPU | **FAIL**: one target, `kernel_baseline` 5 passed, 1 failed | `the_kernels_still_compile_to_the_recorded_baseline` at `tests/kernel_baseline.rs:88` refuses the toolkit: `baseline was generated with: release=V13.0.88` vs `this toolkit is: release=V13.4.59`. A pinned-toolkit fixture, not a source failure; regenerate the baseline or use CUDA 13.0.88. Every other kernels target passed. |

## Classification of the ignored failures

* The 7 failures in each of GLM, Qwen, dense and runtime are the **known conformance stub** (c):
  `sparkflash_runtime::conformance::run_check` is not wired to a fixture yet, so every generated
  `#[ignore]d` test panics with `code: "conformance"`. They check nothing until
  `scripts/onboard-gate.sh <arm>` has a fixture. The `tp_one_two_four` row additionally needs a
  TP group (four ranks for TP 4), which this single-GPU host cannot provide.
* `decode_mma_device` is **(a) a missing fixture** when `SPARKFLASH_KERNEL_DIR` is unset. Here
  the directory was assembled from the build's 53 PTX files and the test passed.
* The `engine_builder` failures are **genuine device-only failures** (b) of the tiny fixture
  against the real image boot check, at `crates/sparkflash-model-glm5-next/tests/engine_builder.rs:103`
  and `crates/sparkflash-model-qwen4-exp/tests/engine_builder.rs:103`; the refusal is raised at
  `crates/sparkflash-ops/src/parts_backend.rs:84`. They are independent of the three arm moves and
  are not fixed here.
* `kernel_baseline` is **(a) a fixture mismatch**: the committed baseline names CUDA 13.0.88 and
  the host has 13.4.59.

## What did not run

* **The served gate and the numerics reference.** No released checkpoint is on the gate host, so
  `scripts/onboard-gate.sh` / `served-gate.py` could not compare the moved arms bit-exactly with
  `numerics/registry.toml`. Every named device test above uses the arm's synthetic tiny fixture.
* **TP 2 and TP 4.** The host has one GPU. The `tp_one_two_four` conformance check needs a
  group and a fixture; no group was opened. The GLM, Qwen and dense runs here are TP 1.
* **A released draft head.** GLM's fixture drafts with its synthetic MTP layer; the released
  GLM/Qwen draft-head checkpoint was not present. The drafting paths (drafted = plain, joint
  verify, joint off) did run; the released drafter's bits did not.
* **Inkling.** It is still on its own shape at this tip; it is not part of this gate.
* **Any arm's kernel choice against a released weight file.** Only the token bodies and the
  decode MMA are checked against the tile, and they are checked with generated operands.

The device only changes the runner: the one-row GLM verify, the joint-member verify, Qwen's
joint-off lane walk, dense's head and the two token bodies are all confirmed. The arm-level
claims that need weights (registry-exact tokens, speed A/B) remain for a checkpoint-backed host.

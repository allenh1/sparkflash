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

# Adding a model arm

A model arm is configuration, a layer plan and typed op calls. It owns no CUDA and depends on no
other model crate. This guide is the bring-up path the parity harness assumes.

## 1. The one stack

Kernel sources live in `kernels/` and are built by `sparkflash-kernels/build.rs`. Every cubin is
exposed through `sparkflash_kernels::images`:

| image | accessor |
| :-- | :-- |
| Qwen token/attention/expert | `qwen4_exp_cubin()` |
| Qwen tensor-core bulk prefill | `qwen4_prefill_cubin()` |
| DFlash 2 block drafter | `dflash2_cubin()` |
| GLM block-scaled bulk prefill | `glm_prefill_cubin()` |

A model crate may not contain a `.cu` or `.cuh` file and may not carry a build script that
compiles one. The workspace test `sparkflash-ops/tests/one_stack.rs` enforces both, together with
the rule that no model crate depends on another model crate.

## 1a. The DeepSeek forward

DeepSeek's device forward is model code and lives in crates/sparkflash-deepseek-forward, not in
sparkflash-cli. The move kept the module names and crate:: paths, so it is numerically inert; the
CLI is a dispatcher over the forward crate and is under the 30k-line gate. The forward crate
depends on the V4.1 model pack and the shared kernel stack, never on the CLI.

## 2. Typed ops

`sparkflash-ops` is the model-facing surface. Every op is a plan built once at load:

- `gemm` — dense, tile-batched, split-K, grouped MoE, routing and combine.
- `attention` — MLA, grouped GQA, sparse/CSA and split-KV decode.
- `scan` — Gated DeltaNet and KDA.
- `epilogue` — norms, RoPE, activations, quantization, embeddings and Engram gathers.
- `collective` — one-shot, two-shot, reduce-scatter, all-gather and overlapped halves.

A plan fixes shapes, precision, layout and stream. It names the kernel it runs and returns a
`Launch`; it owns no buffers, allocates nothing at launch and never synchronizes the host. The
same plan therefore serves every session that reads the same weights.

```rust
use sparkflash_ops::gemm::DenseGemmPlan;
use sparkflash_ops::plan::{Layout, Matrix, OpPlan, Precision, StreamId};

let plan = DenseGemmPlan::new(
    Matrix::new(rows, columns),
    k,
    Precision::MxFp8,
    Layout::BlockScaled { block: 32 },
    StreamId(session_stream),
    1,
)?;
plan.validate()?;
let launch = plan.launch();
```

The shared launch helpers live beside the plans: `gemv` (tiled decode GEMV and its split rules),
`gdn` (scan geometry), `tp` (two-shot all-reduce) and `dflash` (the block drafter and its
loader). An arm implements `GemvDevice` for its stream and scratch; the drafter's
`DraftDevice` adds the profiling mark and the TP reduce.

## 3. Configuration

Every arm has one typed TOML profile:

| model | profile |
| :-- | :-- |
| Qwen3.8-Flash-Next | `crates/sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.toml` |
| GLM-5.3-Flash | `crates/sparkflash-model-glm5-next/config/glm53_flash.toml` |
| DeepSeek-V4.1-Flash | `crates/sparkflash-model-deepseek-v41/config/deepseek_v41_flash.toml` |

A profile opens with the registry model it configures, beside its family and weights:

```toml
model = "zai-org/GLM-5.3-Flash"
family = "glm5-next"
weights = "glm5.3-flash-fp8"
```

The declaration check reads `model` to know whose references a changed setting needs recaptured
([NUMERICS.md](NUMERICS.md) §5), and a `sparkflash-model` test holds it to the descriptor that
`weights` selects.

`sparkflash_model::profile::ModelProfile` parses with `deny_unknown_fields`, validates ranges and
prints the resolved values at boot. The environment is an override only; the complete override
table is `sparkflash_model::profile::ENV_OVERRIDES`. A value outside a field's domain is refused,
not silently ignored.

Rules for a new switch:

1. add a typed field to the profile;
2. add its environment alias to `ENV_OVERRIDES`, or do not read it from the environment at all;
3. document the field and alias in this guide;
4. never let a behaviour be reachable only through an undocumented variable.

## 4. Bring-up parity

`sparkflash_ops::parity` is the layer-by-layer gate. Build a `LayerPlan`, validate it, run the
device path, and compare each named buffer with the reference forward:

```rust
use sparkflash_ops::parity::{LayerPlan, ParityReport, Tolerance};
use sparkflash_ops::plan::OpKind;

let mut report = ParityReport::new("layer 0");
report.record("after_attention", OpKind::MlaAttention, &reference, &device, Tolerance::F32_ROUNDING)?;
assert!(report.passed(), "failed: {:?}", report.failures());
```

A case names the first element outside the tolerance, not just the layer. A new arm runs the same
harness with its host forward as the reference; the harness has no model-specific code.

## 5. Shared homes

Everything an arm needs outside its own forward already has one home. Add to that home, or move
the mechanism there; never copy it into the arm.

| mechanism | home |
| :-- | :-- |
| model id to checkpoint (the Hugging Face cache) | `sparkflash_weights::hub` |
| `--synthetic` prompts without a tokenizer | `sparkflash_ops::codec::ByteCodec` |
| `config.json` errors and a field that is one value or a list | `sparkflash_model::config::{ConfigError, OneOrMany}` |
| an arm that renders every text channel (tools, thinking, reasoning effort) | `Capabilities::ALL`; forced decoding is the driver's, which the host and TP drivers add |
| a tensor's byte length against its slot | `sparkflash_weights::weights::check_length` |
| walk-ordered rank files: placement, index, writer, read-back cursor | `sparkflash_weights::rank_file::{SlotPlan, SlotIndex, SlotFileWriter, SlotCursor, SlotRecords}` |
| `sparkflash convert` for such a pack (ranks, read-back, refusals) | `SlotPack` in `sparkflash-cli/src/convert.rs` |
| `sparkflash serve` at TP=1 and in a tensor-parallel group | `PackArm` in `sparkflash-cli/src/serve/pack.rs` |
| deterministic fixture weights and hash-table sizes | `sparkflash_weights::rng::{splitmix64, is_prime}` |

A served arm implements `PackArm` (its ids, fixture, configuration, descriptor, tokenizer, open
and response layer) and adds a `Pack` variant. The shared driver then finds the checkpoint
(`--checkpoint`, the hub cache or the fixture), derives the context on every rank of a group from
the same configuration, and refuses at parse time any option the arm does not apply
(`Pack::unapplied`): a group serves one request at a time, so `--concurrency`, `--fuse-batch` and
`--kv-budget-gib` are refused there, and an arm without a fused batch or a KV budget says so at
TP=1. A directory of rank files named by an environment variable resolves through
`converted_rank_file`, which refuses a directory that holds no file for the rank.

## 6. Checklist

- [ ] the config parses, validates, and its boot summary is in the startup log;
- [ ] nothing in section 5 is reimplemented in the arm;
- [ ] no `.cu`/`.cuh` and no kernel build script in the model crate;
- [ ] no dependency on another model crate;
- [ ] every op is a plan, validated before its first launch;
- [ ] the parity harness passes layer by layer against the host reference;
- [ ] new environment aliases are in `ENV_OVERRIDES` and documented here;
- [ ] the arm is registered in the model registry and has a descriptor;
- [ ] every profile override the arm adds is classified in `sparkflash_model::numerics` as
      arithmetic or neutral, and each configuration it serves has a captured reference in
      `numerics/registry.toml` ([NUMERICS.md](NUMERICS.md));
- [ ] the arm's profile names its registry model (`model`, section 3).

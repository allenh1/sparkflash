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

## 5. Checklist

- [ ] the config parses, validates, and its boot summary is in the startup log;
- [ ] no `.cu`/`.cuh` and no kernel build script in the model crate;
- [ ] no dependency on another model crate;
- [ ] every op is a plan, validated before its first launch;
- [ ] the parity harness passes layer by layer against the host reference;
- [ ] new environment aliases are in `ENV_OVERRIDES` and documented here;
- [ ] the arm is registered in the model registry and has a descriptor.

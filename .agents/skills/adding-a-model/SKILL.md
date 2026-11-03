---
name: adding-a-model
description: Add or change a SparkFlash model arm without a second kernel stack. Use when implementing a new model family, moving model CUDA, adding a typed op, changing a serving profile, or bringing up parity for an arm.
---

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

Read `docs/adding-a-model.md` first. This skill is the short, enforceable form.

## Before writing code

1. Find the model's existing host reference and descriptor. Do not start from the device arm.
2. Check `crates/sparkflash-ops/src` for a plan that already covers each operation.
3. Load the arm's TOML profile and its fields. If a behaviour has no typed field, add one first.

## Hard rules

- No `.cu` or `.cuh` under `crates/sparkflash-model-*`.
- No model crate depends on another model crate. Shared code belongs in `sparkflash-ops`,
  `sparkflash-kernels`, `sparkflash-model` or `sparkflash-weights`.
- A kernel source belongs in `kernels/`; its image comes from `sparkflash_kernels::images`.
- A new behaviour is a typed profile field. The environment may override it through
  `sparkflash_model::profile::ENV_OVERRIDES`; never read a new switch at the point of use.
- A plan fixes shapes, precision, layout and stream at load. No allocation or host
  synchronization at launch.

## Implementation order

1. Parse the released config into the model's typed config.
2. Plan the tensors and validate the byte identity against the checkpoint index.
3. Write the host reference forward.
4. Compose `LayerPlan` from typed ops, in forward order.
5. Implement the arm's `GemvDevice`; implement `DraftDevice` only if it drafts.
6. Run `sparkflash_ops::parity` layer by layer against the host reference before serving.
7. Register the descriptor and the profile; print `boot_summary()`.

## Verification

Run these before declaring the arm done:

```sh
cargo test -p sparkflash-ops
cargo test -p sparkflash-model
cargo check -p <model-crate> --all-targets
```

A missing cubin must skip the device test loudly, never fall back to a different arithmetic.

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

Read `docs/adding-a-model.md` first and run `sparkflash onboard ORG/REPO --emit DIR`; the
internals are in `docs/arm-reference.md`. This skill is the short, enforceable form.

## Before writing code

1. Run `sparkflash onboard ORG/REPO`. It names the parts the catalogue has, the parts that need a
   `PartExec`, the stored formats, the switches the parser does not model and the fit.
2. Teach `sparkflash_model::Architecture` every unmodelled switch the family needs before writing
   a part for it.
3. Run `sparkflash onboard ORG/REPO --emit DIR`. A catalogue-only family needs no crate; any
   other gets its skeleton and the list of edits outside it.

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

1. Fill each emitted `PartExec` stub: `state`, `split`, `rows`, `token`. Reuse the catalogue
   constructors through `OpenCx` where a part wraps one.
2. Write the tiny fixture checkpoint and the stated profile in `tests/conformance.rs`, so the
   generated checks run (`crates/sparkflash-model-dense/tests/onboard_example.rs` is the model).
3. Make the edits outside the crate that `--emit` printed; the tests named in
   `docs/adding-a-model.md` section 4 fail until each is made.
4. Run `scripts/onboard-gate.sh <arm>` for the served gate and the first numerics reference.

## Verification

Run these before declaring the arm done:

```sh
cargo test -p sparkflash-ops
cargo test -p sparkflash-model
cargo test -p <model-crate>
```

A missing cubin must skip the device test loudly, never fall back to a different arithmetic.
